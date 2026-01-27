
import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "/workspace/datdq/SignWeather/data_collection/ids_2020_2024.txt"
OUTPUT_DIR = "/workspace/datdq/SignWeather/data/raw/thumbnails/origin"
MAX_WORKERS = 50  # High concurrency for I/O

def download_single(video_id):
    """
    Downloads thumbnail for a video_id. Tries maxresdefault first, then hqdefault.
    Returns (video_id, content_bytes, success_boolean)
    """
    urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    ]
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return video_id, resp.content, True
        except Exception:
            continue
            
    return video_id, None, False

def main():
    input_path = Path(INPUT_FILE)
    output_dir = Path(OUTPUT_DIR)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    # Read IDs
    with open(input_path, 'r') as f:
        video_ids = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(video_ids)} IDs.")
    print(f"Downloading images to: {output_dir}")
    
    success_count = 0
    fail_count = 0
    
    # Use ThreadPool for fast parallel downloading
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_vid = {executor.submit(download_single, vid): vid for vid in video_ids}
        
        for future in tqdm(as_completed(future_to_vid), total=len(video_ids), desc="Downloading"):
            vid, content, success = future.result()
            
            if success and content:
                try:
                    save_path = output_dir / f"{vid}.jpg"
                    with open(save_path, "wb") as f_out:
                        f_out.write(content)
                    success_count += 1
                except Exception as e:
                    print(f"Error saving {vid}: {e}")
                    fail_count += 1
            else:
                fail_count += 1
                
    print(f"\nDownload Completed.")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    main()
