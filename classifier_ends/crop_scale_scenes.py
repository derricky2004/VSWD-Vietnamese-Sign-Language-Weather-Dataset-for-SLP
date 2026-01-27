import csv
import sys
import os
import subprocess
import cv2
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = "/workspace/datdq/SignWeather"
METADATA_DIR = f"{BASE_DIR}/data/metadata"
SCENE_METADATA_CSV = f"{METADATA_DIR}/scene_metadata_realtime.csv"
INPUT_SCENE_DIR = f"{BASE_DIR}/data/scene_videos"
OUTPUT_SCENE_DIR = f"{BASE_DIR}/data/scene_videos_cropped"

# Crop parameters (Relative)
CROP_PARAMS = {
    'x': 0.0816,
    'y': 0.6837,
    'w': 0.1525,
    'h': 0.2033
}
SCALE_FACTOR = 5

def get_video_dims(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def crop_and_scale_ffmpeg(input_path, output_path, crop_params, scale_factor):
    """
    Use ffmpeg to crop and scale video.
    Ensures H.264 encoding for better compatibility.
    """
    w_orig, h_orig = get_video_dims(input_path)
    if w_orig is None:
        raise ValueError(f"Cannot read video dimensions: {input_path}")
    
    # Calculate crop pixels
    crop_w = int(crop_params['w'] * w_orig)
    crop_h = int(crop_params['h'] * h_orig)
    crop_x = int(crop_params['x'] * w_orig)
    crop_y = int(crop_params['y'] * h_orig)
    
    # Calculate final scaled dims
    final_w = crop_w * scale_factor
    final_h = crop_h * scale_factor
    
    # Ensure even dimensions for ffmpeg/libx264
    if final_w % 2 != 0: final_w -= 1
    if final_h % 2 != 0: final_h -= 1
    
    # Construct Filter Chain
    # 1. crop=w:h:x:y
    # 2. scale=final_w:final_h
    vf_string = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={final_w}:{final_h}:flags=lanczos"
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', str(input_path),
        '-vf', vf_string,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', # Standard H.264 settings
        '-c:a', 'aac', # Re-encode audio to AAC to ensure compatibility
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True)

def main():
    if not os.path.exists(SCENE_METADATA_CSV):
        print(f"Error: Metadata file {SCENE_METADATA_CSV} not found.")
        return

    print(f"Reading metadata from {SCENE_METADATA_CSV}...")
    
    rows = []
    with open(SCENE_METADATA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    print(f"Found {len(rows)} scenes to process.")
    
    # Ensure output directory exists
    Path(OUTPUT_SCENE_DIR).mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    for row in tqdm(rows, desc="Cropping & Scaling (FFmpeg)"):
        # Rel path: "v001/scene_001.mp4"
        rel_path = row['path']
        
        input_path = Path(INPUT_SCENE_DIR) / rel_path
        output_path = Path(OUTPUT_SCENE_DIR) / rel_path
        
        # Ensure sub-directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            # error_count += 1
            # print(f"Warning: Input file not found {input_path}")
            continue
            
        try:
            crop_and_scale_ffmpeg(
                input_path=input_path,
                output_path=output_path,
                crop_params=CROP_PARAMS,
                scale_factor=SCALE_FACTOR
            )
            success_count += 1
        except Exception as e:
            # print(f"Error processing {rel_path}: {e}")
            error_count += 1
            
    print(f"\nProcessing Complete.")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output Directory: {OUTPUT_SCENE_DIR}")

if __name__ == "__main__":
    main()
