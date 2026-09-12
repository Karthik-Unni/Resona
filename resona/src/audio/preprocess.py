import os
import glob
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import yaml
import warnings

# Suppress librosa warnings about PySoundFile
warnings.filterwarnings('ignore', category=UserWarning)

def load_config():
    with open("config/dataset.yaml", "r") as f:
        return yaml.safe_load(f)

def extract_log_mel_spectrogram(file_path, sr=16000, n_mels=64, n_fft=1024, hop_length=512):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        # Convert to log scale
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Standardize the time frames to exactly ~313 for 10 seconds if needed, 
        # or we just return the raw matrix and handle padding/truncating in dataloader
        return S_dB
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_and_save_dataset():
    config = load_config()
    raw_dir = config["raw_dir"]
    out_dir = config["data_dir"]
    
    # E.g., data/raw/0_dB_pump/id_00/normal/*.wav
    # We will search for all wav files in raw_dir
    search_pattern = os.path.join(raw_dir, "**", "*.wav")
    wav_files = glob.glob(search_pattern, recursive=True)
    
    if not wav_files:
        print(f"No .wav files found in {raw_dir}.")
        return
        
    print(f"Found {len(wav_files)} wav files. Extracting features...")
    
    manifest_data = []
    
    os.makedirs(out_dir, exist_ok=True)
    
    for wav_file in tqdm(wav_files, desc="Processing Audio"):
        parts = wav_file.split(os.sep)
        
        # Expected structure: ... / raw / 0_dB_pump / id_00 / normal / 000000.wav
        try:
            # We assume parts[-3] is 'id_XX', parts[-2] is 'normal' or 'abnormal'
            # Let's find 'id_' in parts
            id_idx = [i for i, p in enumerate(parts) if p.startswith('id_')][0]
            machine_id = parts[id_idx].split('_')[1]
            condition = parts[id_idx + 1] # normal or abnormal
            
            # Use 0 for normal, 1 for abnormal
            label = 0 if condition == "normal" else 1
            
            mel_spec = extract_log_mel_spectrogram(
                wav_file, 
                sr=config["sample_rate"],
                n_mels=config["n_mels"],
                n_fft=config["n_fft"],
                hop_length=config["hop_length"]
            )
            
            if mel_spec is not None:
                out_filename = f"{machine_id}_{condition}_{os.path.basename(wav_file).replace('.wav', '.npy')}"
                out_path = os.path.join(out_dir, out_filename)
                np.save(out_path, mel_spec)
                
                manifest_data.append({
                    "file_path": out_path,
                    "machine_id": machine_id,
                    "condition": condition,
                    "label": label,
                    "original_wav": wav_file
                })
        except Exception as e:
            print(f"Error parsing path {wav_file}: {e}")
            continue

    df = pd.DataFrame(manifest_data)
    manifest_path = "data/manifest/full_manifest.csv"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"Dataset processing complete. Manifest saved to {manifest_path}")

if __name__ == "__main__":
    process_and_save_dataset()
