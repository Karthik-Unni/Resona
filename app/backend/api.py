# -*- coding: utf-8 -*-
"""
RESONA Backend API — Ground-up rewrite
Single source of truth for all UI state.
"""
import sys
import io
# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import time
import uuid
import json
import re
import glob
import tempfile
import threading
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

# ─── Internal imports ─────────────────────────────────────────────────────────
from app.backend.db import (
    init_db, insert_event, upsert_alert, resolve_alert, acknowledge_alert,
    get_active_alerts, get_events, insert_review, get_pending_reviews,
    get_all_reviews, get_review, update_review, add_replay_sample, get_replay_count,
    insert_learning_job, update_learning_job, get_latest_learning_job,
    register_model_version, promote_model_version, get_active_model_version
)
from app.backend.inference import load_model_and_detector, run_inference, get_model_version, is_ood_calibrated

# ─── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='../frontend')
CORS(app)

# ─── Machine configuration (static) ──────────────────────────────────────────
MACHINES = {
    "pump_00": {"name": "Pump 00", "zone": "Zone A", "type": "pump", "machine_id_raw": "00"},
    "pump_02": {"name": "Pump 02", "zone": "Zone B", "type": "pump", "machine_id_raw": "02"},
    "pump_04": {"name": "Pump 04", "zone": "Zone D", "type": "pump", "machine_id_raw": "04"},
    "pump_06": {"name": "Pump 06 (OOD reference)", "zone": "Zone C", "type": "pump", "machine_id_raw": "06"},
}

# Machine ID to zone lookup
MACHINE_ZONES = {v["machine_id_raw"]: v["zone"] for v in MACHINES.values()}
MACHINE_NAMES = {v["machine_id_raw"]: v["name"] for v in MACHINES.values()}

# ─── Rolling audio buffer (for mic streaming) ─────────────────────────────────
_audio_buffer = np.zeros(160000, dtype=np.float32)  # 10s at 16kHz
_buffer_lock = threading.Lock()

# ─── Streaming simulation state ───────────────────────────────────────────────
_sim_thread = None
_sim_active = False

# ─── Startup ──────────────────────────────────────────────────────────────────
print("[RESONA] Starting up...")
init_db()
load_model_and_detector()
print(f"[RESONA] Model version: {get_model_version()}")
print(f"[RESONA] OOD calibrated: {is_ood_calibrated()}")


# ──────────────────────────────────────────────────────────────────────────────
# Static serving
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/')
def serve_desktop():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/mobile')
def serve_mobile():
    return send_from_directory(FRONTEND_DIR, 'mobile.html')

@app.route('/review/<review_id>')
def serve_review_page(review_id):
    return send_from_directory(FRONTEND_DIR, 'mobile.html')


# ──────────────────────────────────────────────────────────────────────────────
# Status endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/status', methods=['GET'])
def get_status():
    active_alerts = get_active_alerts()
    pending_reviews = get_pending_reviews()
    replay_count = get_replay_count()
    latest_job = get_latest_learning_job()
    active_model = get_active_model_version()

    return jsonify({
        "system": "OPERATIONAL",
        "model_version": get_model_version(),
        "ood_calibrated": is_ood_calibrated(),
        "ood_status": "CALIBRATED" if is_ood_calibrated() else "OOD CALIBRATION REQUIRED",
        "active_alerts_count": len(active_alerts),
        "pending_reviews_count": len(pending_reviews),
        "replay_buffer_count": replay_count,
        "latest_job": latest_job,
        "active_model": active_model,
        "timestamp": datetime.utcnow().isoformat()
    })


# ──────────────────────────────────────────────────────────────────────────────
# Machines
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/machines', methods=['GET'])
def get_machines():
    alerts = get_active_alerts()
    alert_by_machine = {}
    for a in alerts:
        alert_by_machine.setdefault(a['machine_id'], []).append(a)

    result = []
    for mid, info in MACHINES.items():
        raw_id = info['machine_id_raw']
        machine_alerts = alert_by_machine.get(raw_id, [])
        state = "normal"
        if any(a['decision'] == 'ANOMALOUS' for a in machine_alerts):
            state = "anomaly"
        elif any(a['decision'] == 'UNKNOWN' for a in machine_alerts):
            state = "unknown"
        result.append({
            "id": mid,
            "machine_id_raw": raw_id,
            "name": info["name"],
            "zone": info["zone"],
            "type": info["type"],
            "state": state,
            "active_alerts": len(machine_alerts)
        })
    return jsonify(result)


# ──────────────────────────────────────────────────────────────────────────────
# Alerts
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    alerts = get_active_alerts()
    for a in alerts:
        a['machine_name'] = MACHINE_NAMES.get(a['machine_id'], f"Machine {a['machine_id']}")
    return jsonify(alerts)

@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def ack_alert(alert_id):
    acknowledge_alert(alert_id)
    return jsonify({"success": True})


# ──────────────────────────────────────────────────────────────────────────────
# Reviews
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/reviews', methods=['GET'])
def get_reviews():
    reviews = get_all_reviews()
    for r in reviews:
        r['machine_name'] = MACHINE_NAMES.get(r['machine_id'], f"Machine {r['machine_id']}")
    return jsonify(reviews)

@app.route('/api/reviews/pending', methods=['GET'])
def get_pending():
    reviews = get_pending_reviews()
    for r in reviews:
        r['machine_name'] = MACHINE_NAMES.get(r['machine_id'], f"Machine {r['machine_id']}")
    return jsonify(reviews)

@app.route('/api/reviews/<review_id>', methods=['GET'])
def get_single_review(review_id):
    review = get_review(review_id)
    if not review:
        return jsonify({"error": "Review not found"}), 404
    review['machine_name'] = MACHINE_NAMES.get(review['machine_id'], f"Machine {review['machine_id']}")
    return jsonify(review)


# ──────────────────────────────────────────────────────────────────────────────
# Human feedback
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/human_feedback', methods=['POST'])
def human_feedback():
    data = request.json or {}
    review_id = data.get('review_id')
    human_label = data.get('label')  # "NORMAL", "ANOMALOUS", "NEW CONDITION"
    notes = data.get('notes', '')

    if not review_id or not human_label:
        return jsonify({"success": False, "error": "review_id and label required"}), 400

    review = get_review(review_id)
    if not review:
        return jsonify({"success": False, "error": "Review not found"}), 404
    if review['status'] != 'pending':
        return jsonify({"success": False, "error": "Review already resolved"}), 409

    # Update review
    update_review(review_id, human_label, notes)

    # Map label to numeric: NORMAL=0, ANOMALOUS=1, NEW CONDITION=2
    label_map = {"NORMAL": 0, "ANOMALOUS": 1, "NEW CONDITION": 2}
    label_int = label_map.get(human_label, 0)

    # Add to replay buffer
    add_replay_sample(
        machine_id=review['machine_id'],
        label=label_int,
        human_label=human_label,
        review_id=review_id
    )

    # Resolve the alert for UNKNOWN conditions
    resolve_alert(review['machine_id'], 'UNKNOWN')

    # Queue a learning job (does NOT train immediately)
    job_id = str(uuid.uuid4())
    insert_learning_job({
        "id": job_id,
        "created_at": datetime.utcnow().isoformat(),
        "status": "queued",
        "trigger_review_id": review_id
    })

    print(f"[HUMAN FEEDBACK] Machine={review['machine_id']} Label={human_label} -> Job queued: {job_id}")

    return jsonify({
        "success": True,
        "replay_count": get_replay_count(),
        "job_id": job_id,
        "job_status": "queued"
    })


# ──────────────────────────────────────────────────────────────────────────────
# Inference events log
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/events', methods=['GET'])
def get_inference_events():
    limit = int(request.args.get('limit', 50))
    events = get_events(limit)
    # Don't return large waveform/spectrogram arrays in event list
    for e in events:
        e.pop('waveform_json', None)
        e.pop('spectrogram_json', None)
    return jsonify(events)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation results
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
    result_files = {
        "RESONA_Replay": "results/results_RESONA_Replay.txt",
        "Joint": "results/results_Joint.txt",
        "Naive_FT": "results/results_Naive_FT.txt",
        "RESONA_NoReplay": "results/results_RESONA_NoReplay.txt",
    }
    results = {}
    for method, path in result_files.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
            results[method] = _parse_result_file(content)

    if not results:
        return jsonify({"available": False, "message": "No evaluation results found."})

    return jsonify({"available": True, "results": results})


def _parse_result_file(content: str) -> dict:
    """Parse a results_*.txt file into structured dict."""
    parsed = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if line.startswith("Method:"):
            parsed["method"] = line.split(":", 1)[1].strip()
        elif line.startswith("Average Final Accuracy:"):
            parsed["avg_final_accuracy"] = line.split(":", 1)[1].strip()
        elif line.startswith("Average Forgetting:"):
            parsed["avg_forgetting"] = line.split(":", 1)[1].strip()
        elif line.startswith("Accuracy Matrix:"):
            pass
        elif line.startswith("[[") or line.startswith(" ["):
            # Parse matrix rows
            if "matrix" not in parsed:
                parsed["matrix"] = []
            nums = re.findall(r'[\d.]+', line)
            if nums:
                parsed["matrix"].append([float(n) for n in nums])
    return parsed


# ──────────────────────────────────────────────────────────────────────────────
# WAV file upload — central inference
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"success": False, "error": "No audio file provided"}), 400

    file = request.files['audio']
    if not file.filename:
        return jsonify({"success": False, "error": "No filename"}), 400

    # Validate extension
    if not file.filename.lower().endswith('.wav'):
        return jsonify({"success": False, "error": "Only WAV files accepted"}), 400

    machine_id = request.form.get('machine_id', '00')
    if machine_id not in MACHINE_NAMES:
        machine_id = '00'

    # Save temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.wav')
    os.close(tmp_fd)
    try:
        file.save(tmp_path)

        # Load audio
        try:
            audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        except Exception as e:
            return jsonify({"success": False, "error": f"Could not read WAV: {e}"}), 422

        if len(audio) == 0:
            return jsonify({"success": False, "error": "WAV file is empty"}), 422

        # Run inference
        result = run_inference(audio, sr, machine_id=machine_id)

        # Persist
        _persist_inference_result(result)

        return jsonify({"success": True, "result": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Microphone audio chunk — rolling buffer
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/process_audio_chunk', methods=['POST'])
def process_audio_chunk():
    global _audio_buffer

    if 'audio' not in request.files:
        return jsonify({"success": False, "error": "No audio provided"}), 400

    machine_id = request.form.get('machine_id', '00')
    file = request.files['audio']

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.wav')
    os.close(tmp_fd)
    try:
        file.save(tmp_path)
        try:
            y_chunk, sr = librosa.load(tmp_path, sr=16000, mono=True)
        except Exception as e:
            return jsonify({"success": False, "error": f"Could not decode audio chunk: {e}"}), 422

        if len(y_chunk) == 0:
            return jsonify({"success": False, "error": "Empty audio chunk"}), 422

        # Update rolling buffer
        with _buffer_lock:
            _audio_buffer = np.roll(_audio_buffer, -len(y_chunk))
            _audio_buffer[-len(y_chunk):] = y_chunk[:len(_audio_buffer)]
            buffer_snapshot = _audio_buffer.copy()

        # Run inference on the current 10s window
        result = run_inference(buffer_snapshot, 16000, machine_id=machine_id)
        _persist_inference_result(result)

        return jsonify({"success": True, "result": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Demo / scenario WAV streaming (background thread)
# ──────────────────────────────────────────────────────────────────────────────
# Available scenario WAV paths
SCENARIO_WAVS = {
    1: {"path": "data/raw/pump/id_00/normal/00000000.wav", "machine_id": "00", "label": "Normal Pump Operation"},
    2: {"path": "data/raw/pump/id_02/abnormal/00000027.wav", "machine_id": "02", "label": "Known Anomaly"},
    3: {"path": "data/raw/pump/id_00/normal/00000001.wav", "machine_id": "00", "label": "Normal (0dB SNR)"},
    4: {"path": "data/raw/pump/id_02/abnormal/00000028.wav", "machine_id": "02", "label": "Noisy Anomaly"},
    5: {"path": "data/raw/pump/id_06/normal/00000000.wav", "machine_id": "06", "label": "Unknown (OOD) — Held-out machine"},
    6: {"path": "data/raw/pump/id_06/normal/00000001.wav", "machine_id": "06", "label": "Unknown — Human loop trigger"},
    7: {"path": "data/raw/pump/id_04/normal/00000000.wav", "machine_id": "04", "label": "Post-Learning Recognition"},
    8: {"path": "data/raw/pump/id_04/abnormal/00000000.wav", "machine_id": "04", "label": "Forgetting Eval — Anomaly"},
}

@app.route('/api/simulate', methods=['POST'])
def simulate():
    global _sim_thread, _sim_active

    data = request.json or {}
    scenario = data.get('test_case') or data.get('scenario')

    if scenario == 'stop':
        _sim_active = False
        return jsonify({"success": True, "message": "Simulation stopped."})

    if scenario not in SCENARIO_WAVS:
        return jsonify({"success": False, "error": f"Unknown scenario: {scenario}"}), 400

    wav_info = SCENARIO_WAVS[scenario]
    wav_path = wav_info['path']

    if not os.path.exists(wav_path):
        return jsonify({"success": False, "error": f"WAV file not found: {wav_path}"}), 404

    # Stop existing simulation
    _sim_active = False
    if _sim_thread and _sim_thread.is_alive():
        _sim_thread.join(timeout=2)

    def _stream_wav(path, machine_id):
        global _audio_buffer, _sim_active
        _sim_active = True
        try:
            y, sr = librosa.load(path, sr=16000, mono=True)
            chunk_size = 16000  # 1 second
            for i in range(0, len(y), chunk_size):
                if not _sim_active:
                    break
                chunk = y[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                with _buffer_lock:
                    _audio_buffer = np.roll(_audio_buffer, -chunk_size)
                    _audio_buffer[-chunk_size:] = chunk
                    buf = _audio_buffer.copy()

                try:
                    result = run_inference(buf, 16000, machine_id=machine_id)
                    _persist_inference_result(result)
                except Exception as e:
                    print(f"[SIM] Inference error: {e}")

                time.sleep(1.0)
        except Exception as e:
            print(f"[SIM] Stream error: {e}")
        finally:
            _sim_active = False
            print(f"[SIM] Simulation ended for {path}")

    _sim_thread = threading.Thread(
        target=_stream_wav,
        args=(wav_path, wav_info['machine_id']),
        daemon=True
    )
    _sim_thread.start()

    return jsonify({
        "success": True,
        "scenario": scenario,
        "label": wav_info["label"],
        "wav_path": wav_path,
        "machine_id": wav_info["machine_id"]
    })


# ──────────────────────────────────────────────────────────────────────────────
# Latest inference result (for UI polling)
# ──────────────────────────────────────────────────────────────────────────────
_latest_result = None
_latest_result_lock = threading.Lock()

@app.route('/api/latest', methods=['GET'])
def get_latest():
    with _latest_result_lock:
        if _latest_result is None:
            return jsonify({"available": False})
        return jsonify({"available": True, "result": _latest_result})


# ──────────────────────────────────────────────────────────────────────────────
# Persist helper
# ──────────────────────────────────────────────────────────────────────────────
def _persist_inference_result(result: dict):
    global _latest_result

    machine_id = result['machine_id']
    decision = result['decision']
    event_id = result['event_id']
    zone = MACHINE_ZONES.get(machine_id, "Unknown Zone")

    # Store in DB (compact: store spectrogram/waveform as JSON)
    db_event = dict(result)
    db_event['waveform_json'] = json.dumps(result['waveform'])
    db_event['spectrogram_json'] = json.dumps(result['spectrogram'])
    db_event['is_ood'] = int(result['is_ood'])
    db_event['ood_calibrated'] = int(result['ood_calibrated'])
    # remove the large lists before passing to DB
    db_event.pop('waveform', None)
    db_event.pop('spectrogram', None)
    db_event.pop('probabilities', None)
    # rename fields to match DB schema
    db_event['id'] = db_event.pop('event_id')

    try:
        insert_event(db_event)
    except Exception as e:
        print(f"[DB] Event insert error: {e}")

    # Alert logic (deduplication)
    if decision == 'ANOMALOUS':
        upsert_alert(machine_id, zone, 'ANOMALOUS', event_id)
        resolve_alert(machine_id, 'UNKNOWN')  # clear any prior unknown
    elif decision == 'UNKNOWN':
        alert_id, is_new = upsert_alert(machine_id, zone, 'UNKNOWN', event_id)
        # Create review only when the UNKNOWN alert is first created
        if is_new:
            _create_review(machine_id, event_id, result['mahalanobis_distance'], result['model_version'])
    elif decision == 'NORMAL':
        resolve_alert(machine_id, 'ANOMALOUS')
        resolve_alert(machine_id, 'UNKNOWN')

    # Update latest result for polling
    with _latest_result_lock:
        _latest_result = result

    print(f"[INFERENCE] {machine_id} -> {decision} "
          f"(conf={result['confidence']:.2%}, mahal={result['mahalanobis_distance']:.1f}, "
          f"latency={result['total_latency_ms']:.0f}ms)")


def _create_review(machine_id, event_id, ood_score, model_version):
    review_id = str(uuid.uuid4())
    review = {
        "id": review_id,
        "machine_id": machine_id,
        "event_id": event_id,
        "ood_score": ood_score,
        "model_version": model_version,
        "created_at": datetime.utcnow().isoformat(),
        "status": "pending"
    }
    insert_review(review)
    print(f"[REVIEW] Created review {review_id} for machine {machine_id}")


# ──────────────────────────────────────────────────────────────────────────────
# OOD info
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/ood_info', methods=['GET'])
def ood_info():
    if is_ood_calibrated():
        from app.backend.inference import _detector
        return jsonify({
            "calibrated": True,
            "threshold": _detector.threshold,
            "classes": list(_detector.class_means.keys()),
            "artifact_dir": "artifacts/ood/"
        })
    return jsonify({
        "calibrated": False,
        "message": "OOD CALIBRATION REQUIRED — run: python scripts/calibrate_ood.py"
    })


# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer stats
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/api/replay', methods=['GET'])
def replay_stats():
    count = get_replay_count()
    return jsonify({"count": count})


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')

