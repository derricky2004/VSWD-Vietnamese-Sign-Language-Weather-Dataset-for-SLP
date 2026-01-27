
import os
import sys
import requests
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import io

# Add parent directory to path to allow imports if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Add classifier directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "classifier_thumbnail"))

from inference import SignerClassifier

# --- CONFIGURATION ---
INPUT_ID_FILE = "/workspace/datdq/SignWeather/data_collection/ids_2020_2024.txt"
OUTPUT_FILE = "/workspace/datdq/SignWeather/data/lists/vtv_weather_filtered.txt"
LOCAL_THUMB_DIR = Path("/workspace/datdq/SignWeather/data/raw/thumbnails/origin")

def main():
    # 0. Load IDs
    if not os.path.exists(INPUT_ID_FILE):
        print(f"Error: Input file {INPUT_ID_FILE} not found.")
        return

    with open(INPUT_ID_FILE, 'r') as f:
        video_ids = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(video_ids)} IDs. Reversing order as requested...")
    video_ids = video_ids[::-1] # REVERSE ORDER
    
    # Check local thumbs dir
    if not LOCAL_THUMB_DIR.exists():
        print(f"Error: Local thumbnail directory {LOCAL_THUMB_DIR} not found.")
        return

    # 1. Init Classifier
    print("Loading Classifier...")
    try:
        classifier = SignerClassifier()
    except Exception as e:
        print(f"Failed to load classifier: {e}")
        return

    # 2. Setup Output File
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # We will APPEND to the file to preserve previous results if any, 
    # but the user might want a fresh run. 
    # Since we are reversing and scanning "fast", I assume this is the definitive run.
    # Let's clear it to be safe and clean.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("") 
    
    print(f"Output will be streamed to: {OUTPUT_FILE}")
    print("Starting Scan (Offline Mode)...")
    
    # Buffer writing
    with open(output_path, "a", encoding="utf-8", buffering=1) as f_out:
        for vid in tqdm(video_ids, desc="Scanning"):
            # Try load local file
            img_path = LOCAL_THUMB_DIR / f"{vid}.jpg"
            
            if not img_path.exists():
                # tqdm.write(f"Skip (Missing): {vid}")
                continue
                
            try:
                img = Image.open(img_path)
                
                # Predict
                has_signer, pos = classifier.predict(img)
                
                if has_signer:
                    # Write immediately
                    url = f"https://www.youtube.com/watch?v={vid}"
                    f_out.write(url + "\n")
                    
                    # Optional logging to simple stdout (tqdm handles it)
                    tqdm.write(f"✅ FOUND ({pos}): {vid}")
                        
            except Exception as e:
                # tqdm.write(f"Error processing {vid}: {e}")
                continue
            
    print(f"\nScanning Complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
