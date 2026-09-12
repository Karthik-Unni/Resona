# -*- coding: utf-8 -*-
"""
RESONA Central Inference Pipeline
ONE function used by all input paths: upload, mic stream, demo WAV, edge agent.
"""
import sys
import io
import os
import time
import uuid
import json
import tempfile
import numpy as np
import torch
import librosa
import soundfile as sf
from datetime import datetime
# Force UTF-8 output on Windows to prevent charmap errors
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.cnn_classifier import SimpleCNN
from src.ood.detector import MahalanobisDetector
from src.audio.preprocess import extract_log_mel_spectrogram

# ─── Constants matching training ─────────────────────────────────────────────
SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
MAX_TIME_FRAMES = 313   # ~10 seconds at 16kHz, hop=512

ARTIFACTS_DIR = "artifacts/ood"
MODEL_GLOB = "results/models/RESONA_v*_RESONA_Replay.pth"

# ─── Module-level state ───────────────────────────────────────────────────────
_model = None
_detector = None
_model_version = None
_ood_calibrated = False


def load_model_and_detector():
    """Load latest trained CNN and OOD calibration artifacts."""
    global _model, _detector, _model_version, _ood_calibrated
    import glob
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = yaml.safe_load(open("config/model.yaml"))
    num_classes = model_config.get("num_classes", 2)
    ood_percentile = model_config.get("ood_threshold_percentile", 95)

    model = SimpleCNN(input_channels=1, num_classes=num_classes).to(device)

    model_files = sorted(glob.glob(MODEL_GLOB))
    version = "untrained"
    if model_files:
        latest = model_files[-1]
        version = os.path.basename(latest).replace('.pth', '')
        model.load_state_dict(torch.load(latest, map_location=device))
        print(f"[Inference] Loaded model: {latest}")
    else:
        print("[Inference] WARNING: No trained model found. Using random weights.")

    model.eval()
    _model = model
    _model_version = version

    # Load OOD calibration
    detector = MahalanobisDetector(threshold_percentile=ood_percentile)
    means_path = os.path.join(ARTIFACTS_DIR, "class_means.json")
    covinv_path = os.path.join(ARTIFACTS_DIR, "cov_inv.npy")
    thresh_path = os.path.join(ARTIFACTS_DIR, "threshold.json")

    if os.path.exists(means_path) and os.path.exists(covinv_path) and os.path.exists(thresh_path):
        with open(means_path) as f:
            means_raw = json.load(f)
        detector.class_means = {int(k): np.array(v) for k, v in means_raw.items()}

        detector.class_cov_inv = np.load(covinv_path)

        with open(thresh_path) as f:
            thresh_data = json.load(f)
        detector.threshold = thresh_data["threshold"]

        _ood_calibrated = True
        print(f"[Inference] OOD calibrated. Threshold: {detector.threshold:.4f}")
    else:
        print(f"[Inference] WARNING: OOD artifacts not found at {ARTIFACTS_DIR}/")
        print(f"[Inference] Run: python scripts/calibrate_ood.py")
        _ood_calibrated = False

    _detector = detector


def get_model_version():
    return _model_version or "unknown"


def is_ood_calibrated():
    return _ood_calibrated


def run_inference(audio_data: np.ndarray, sample_rate: int, machine_id: str = "unknown") -> dict:
    """
    Central inference function. Called by ALL input paths.

    Args:
        audio_data: 1D numpy float32 array of audio samples
        sample_rate: sample rate of audio_data
        machine_id: machine identifier string

    Returns:
        Full inference result dict
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model_and_detector() first.")

    device = next(_model.parameters()).device
    t_total_start = time.time()

    # ── 1. Resample if needed ─────────────────────────────────────────────────
    t_pre_start = time.time()
    if sample_rate != SR:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=SR)
        sample_rate = SR

    duration_sec = len(audio_data) / SR

    # ── 2. Validate audio ─────────────────────────────────────────────────────
    if len(audio_data) < SR:  # less than 1 second
        raise ValueError(f"Audio too short: {duration_sec:.2f}s (minimum 1s required)")

    # ── 3. Compute waveform for UI (downsampled to ~1000 points) ─────────────
    downsample_factor = max(1, len(audio_data) // 1000)
    waveform = audio_data[::downsample_factor].tolist()

    # ── 4. Log-Mel spectrogram (same params as training) ─────────────────────
    S = librosa.feature.melspectrogram(
        y=audio_data, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_dB = librosa.power_to_db(S, ref=np.max)  # (64, T)

    # ── 5. Normalize spectrogram for UI rendering ─────────────────────────────
    mel_min, mel_max = mel_dB.min(), mel_dB.max()
    mel_norm = (mel_dB - mel_min) / (mel_max - mel_min + 1e-8)  # [0, 1]
    spectrogram = mel_norm.tolist()  # 2D list: [mel_bins][time_frames]

    # ── 6. Pad/truncate for CNN (expects exactly MAX_TIME_FRAMES) ─────────────
    mel_for_cnn = mel_dB.copy()
    if mel_for_cnn.shape[1] < MAX_TIME_FRAMES:
        mel_for_cnn = np.pad(mel_for_cnn, ((0, 0), (0, MAX_TIME_FRAMES - mel_for_cnn.shape[1])))
    else:
        mel_for_cnn = mel_for_cnn[:, :MAX_TIME_FRAMES]

    # Add batch + channel dims: (1, 1, 64, 313)
    tensor = torch.tensor(mel_for_cnn, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    t_pre_end = time.time()
    preprocessing_latency_ms = (t_pre_end - t_pre_start) * 1000

    # ── 7. CNN Inference ──────────────────────────────────────────────────────
    t_inf_start = time.time()
    _model.eval()
    with torch.no_grad():
        logits, features = _model(tensor, return_features=True)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])

    features_np = features.cpu().numpy()  # (1, 128)
    t_inf_end = time.time()
    inference_latency_ms = (t_inf_end - t_inf_start) * 1000

    # ── 8. OOD Detection ──────────────────────────────────────────────────────
    if _ood_calibrated and _detector.threshold is not None and _detector.class_means:
        mahal_score = float(_detector.score_samples(features_np)[0])
        is_ood = bool(_detector.is_unknown(features_np)[0])
        ood_threshold = float(_detector.threshold)
    else:
        mahal_score = 0.0
        is_ood = False
        ood_threshold = None

    # ── 9. Decision ───────────────────────────────────────────────────────────
    # NORMAL = pred 0, in-distribution
    # ANOMALOUS = pred 1, in-distribution
    # UNKNOWN = OOD (out-of-distribution), regardless of CNN prediction
    if is_ood:
        decision = "UNKNOWN"
    elif pred == 1:
        decision = "ANOMALOUS"
    else:
        decision = "NORMAL"

    t_total_end = time.time()
    total_latency_ms = (t_total_end - t_total_start) * 1000

    event_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    return {
        "event_id": event_id,
        "timestamp": timestamp,
        "machine_id": machine_id,
        "prediction": pred,
        "probabilities": probs.tolist(),
        "confidence": confidence,
        "mahalanobis_distance": mahal_score,
        "ood_threshold": ood_threshold,
        "ood_status": "OOD" if is_ood else ("IN_DISTRIBUTION" if _ood_calibrated else "UNCALIBRATED"),
        "is_ood": is_ood,
        "ood_calibrated": _ood_calibrated,
        "decision": decision,
        "waveform": waveform,
        "spectrogram": spectrogram,
        "sample_rate": SR,
        "duration_sec": round(duration_sec, 2),
        "preprocessing_latency_ms": round(preprocessing_latency_ms, 1),
        "inference_latency_ms": round(inference_latency_ms, 1),
        "total_latency_ms": round(total_latency_ms, 1),
        "model_version": _model_version or "unknown"
    }
