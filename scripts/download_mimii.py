import os
import urllib.request
import zipfile
import sys

URL = "https://zenodo.org/records/3384388/files/0_dB_pump.zip"
TARGET_ZIP = "data/raw/0_dB_pump.zip"
EXTRACT_DIR = "data/raw/"

def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"{target_path} already exists. Skipping download.")
        return

    print(f"Downloading {url} to {target_path}...")
    try:
        # We can add a simple progress bar
        def reporthook(count, block_size, total_size):
            if count % 1000 == 0:
                downloaded = count * block_size
                percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
                sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.2f} MB ({percent}%)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nError downloading file: {e}")
        sys.exit(1)

def extract_subset(zip_path, extract_dir, required_models=["00", "02", "04", "06"]):
    print(f"Extracting specific models from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        # Keep only members that match the required models
        to_extract = []
        for m in members:
            # Example path: 0_dB_pump/id_00/normal/00000000.wav
            parts = m.split('/')
            if len(parts) >= 2 and parts[1].startswith("id_"):
                model_id = parts[1].split("_")[1]
                if model_id in required_models:
                    to_extract.append(m)
        
        print(f"Extracting {len(to_extract)} files out of {len(members)}...")
        zf.extractall(path=extract_dir, members=to_extract)
        print("Extraction complete.")

if __name__ == "__main__":
    download_file(URL, TARGET_ZIP)
    extract_subset(TARGET_ZIP, EXTRACT_DIR)
    
    # Optional: remove zip file to save space if needed
    # os.remove(TARGET_ZIP)
