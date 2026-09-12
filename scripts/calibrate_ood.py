"""
RESONA OOD Calibration Script
Calibrates the MahalanobisDetector using VALIDATION split embeddings from training data.
Saves calibration artifacts to artifacts/ood/
Run ONCE after training: python scripts/calibrate_ood.py

Uses ONLY training/validation data — NEVER test data or OOD held-out data.
"""
import os
import sys
import json
import numpy as np
import glob
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.cnn_classifier import SimpleCNN
from src.ood.detector import MahalanobisDetector

ARTIFACTS_DIR = "artifacts/ood"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def get_embeddings_from_npy(model, device, npy_paths, labels):
    """Run forward pass on numpy files to get embeddings."""
    model.eval()
    all_features = []
    all_labels = []
    
    for npy_path, label in zip(npy_paths, labels):
        try:
            mel_spec = np.load(npy_path)  # (64, T)
            # Pad/truncate to 313 frames
            if mel_spec.shape[1] < 313:
                mel_spec = np.pad(mel_spec, ((0,0),(0, 313 - mel_spec.shape[1])))
            else:
                mel_spec = mel_spec[:, :313]
            
            # Add batch + channel dims
            tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, features = model(tensor, return_features=True)
            
            all_features.append(features.cpu().numpy()[0])
            all_labels.append(label)
        except Exception as e:
            print(f"  Skipped {npy_path}: {e}")
    
    return np.array(all_features), np.array(all_labels)


def main():
    print("=" * 60)
    print("RESONA OOD Calibration")
    print("=" * 60)
    
    model_config = yaml.safe_load(open("config/model.yaml"))
    cl_config = yaml.safe_load(open("config/continual_learning.yaml"))
    config = {**model_config, **cl_config}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load latest trained model
    model_files = sorted(glob.glob("results/models/RESONA_v*_RESONA_Replay.pth"))
    if not model_files:
        print("ERROR: No trained model found. Run training first.")
        sys.exit(1)
    
    latest_model_path = model_files[-1]
    model_version = os.path.basename(latest_model_path).replace('.pth', '')
    print(f"Loading model: {latest_model_path}")
    
    num_classes = config.get("num_classes", 2)
    model = SimpleCNN(input_channels=1, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.eval()
    print("Model loaded.")
    
    # Gather training/validation data from processed npy files
    # Use the known training machines (id 00, 02, 04) NOT the held-out OOD machine (id 06)
    processed_dir = "data/processed"
    
    # Get files for in-distribution machines only
    all_npy = glob.glob(os.path.join(processed_dir, "*.npy"))
    
    # Filter: use only known machines (00, 02, 04)
    # NPY filename pattern: {machine_id}_{condition}_{filename}.npy
    # e.g.: 00_normal_00000000.npy, 02_abnormal_00000027.npy
    in_dist_files = []
    in_dist_labels = []
    
    for f in all_npy:
        fname = os.path.basename(f)
        parts = fname.split('_')
        if len(parts) < 2:
            continue
        machine_id = parts[0]
        condition = parts[1]
        
        # Only use known machines for calibration
        if machine_id in ['00', '02', '04']:
            label = 0 if condition == 'normal' else 1
            in_dist_files.append(f)
            in_dist_labels.append(label)
    
    if not in_dist_files:
        print("ERROR: No processed .npy files found for known machines.")
        print(f"       Check data/processed/ directory.")
        sys.exit(1)
    
    print(f"\nFound {len(in_dist_files)} in-distribution samples.")
    
    # Use 70% of training data for calibration (validation split)
    np.random.seed(42)
    indices = np.random.permutation(len(in_dist_files))
    val_size = int(len(indices) * 0.30)  # use 30% as calibration set
    val_indices = indices[:val_size]
    
    cal_files = [in_dist_files[i] for i in val_indices]
    cal_labels = [in_dist_labels[i] for i in val_indices]
    
    print(f"Using {len(cal_files)} samples for calibration (30% split).")
    print("Extracting embeddings...")
    
    features, labels = get_embeddings_from_npy(model, device, cal_files, cal_labels)
    
    if len(features) == 0:
        print("ERROR: Failed to extract any embeddings.")
        sys.exit(1)
    
    print(f"Embeddings shape: {features.shape}")
    
    # Fit the detector
    detector = MahalanobisDetector(threshold_percentile=config.get("ood_threshold_percentile", 95))
    detector.fit(features, labels)
    
    print(f"\nCalibration complete:")
    print(f"  Threshold: {detector.threshold:.4f}")
    print(f"  Class means: {list(detector.class_means.keys())}")
    print(f"  Covariance shape: {detector.class_cov_inv.shape}")
    
    # Save artifacts
    # class_means: dict {label: np.array}
    means_dict = {str(k): v.tolist() for k, v in detector.class_means.items()}
    with open(os.path.join(ARTIFACTS_DIR, "class_means.json"), 'w') as f:
        json.dump(means_dict, f)
    
    np.save(os.path.join(ARTIFACTS_DIR, "cov_inv.npy"), detector.class_cov_inv)
    
    with open(os.path.join(ARTIFACTS_DIR, "threshold.json"), 'w') as f:
        json.dump({
            "threshold": float(detector.threshold),
            "threshold_percentile": config.get("ood_threshold_percentile", 95),
            "calibrated_with_n_samples": len(features),
            "model_version": model_version
        }, f, indent=2)
    
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}/")
    print("  class_means.json")
    print("  cov_inv.npy")
    print("  threshold.json")
    
    # Verify calibration works
    test_scores = detector.score_samples(features[:10])
    is_ood = detector.is_unknown(features[:10])
    print(f"\nVerification (first 10 training samples):")
    print(f"  Scores: {test_scores.round(2)}")
    print(f"  Is OOD (expect mostly False): {is_ood}")
    
    print("\nOOD calibration complete.")


if __name__ == '__main__':
    main()
