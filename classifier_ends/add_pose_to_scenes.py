import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
BASE_DIR = Path("/workspace/datdq/SignWeather")
INPUT_DIR = BASE_DIR / "data/scene_videos_cropped"
OUTPUT_VIDEO_DIR = BASE_DIR / "data/scene_videos_pose"
OUTPUT_JSON_DIR = BASE_DIR / "data/scene_keypoints"
MAX_WORKERS = 2 

# Add BASE_DIR to path to allow import form utils
sys.path.append(str(BASE_DIR))

try:
    from utils.pose_detection import extract_pose_landmarks, visualize_pose_on_video
except ImportError as e:
    print(f"Error importing pose utils: {e}")
    print(f"Ensure {BASE_DIR}/utils/pose_detection.py exists.")
    sys.exit(1)

def process_single_video(file_path):
    """
    Process a single video file.
    file_path: Path object pointing to input video file
    """
    try:
        # Calculate relative path to maintain structure
        try:
            rel_path = file_path.relative_to(INPUT_DIR)
        except ValueError:
            rel_path = file_path.name
        
        # Output paths
        out_video_path = OUTPUT_VIDEO_DIR / rel_path
        json_rel_path = Path(rel_path).with_suffix('.json')
        out_json_path = OUTPUT_JSON_DIR / json_rel_path
        
        # Create parent directories
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if both exist? (Optional)
        # if out_video_path.exists() and out_json_path.exists():
        #    return {'status': 'skipped', 'path': str(rel_path)}

        # 1. Extract Landmarks
        landmarks_data = extract_pose_landmarks(file_path)
        
        # Save JSON
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(landmarks_data, f, indent=2)
            
        # 2. Visualize
        visualize_pose_on_video(file_path, out_video_path)
        
        return {'status': 'success', 'path': str(rel_path)}
        
    except Exception as e:
        return {'status': 'error', 'path': str(file_path.name), 'msg': str(e)}

def main():
    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}")
        return

    print(f"Scanning {INPUT_DIR} for videos...")
    
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = [
        p for p in INPUT_DIR.rglob("*") 
        if p.is_file() and p.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        print("No videos found.")
        return
        
    print(f"Found {len(video_files)} videos. Processing with {MAX_WORKERS} workers...")
    print(f"Output Video: {OUTPUT_VIDEO_DIR}")
    print(f"Output JSON: {OUTPUT_JSON_DIR}")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_video, vid_path): vid_path for vid_path in video_files}
        
        with tqdm(total=len(video_files), desc="Adding Pose") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result['status'] == 'success':
                    success_count += 1
                elif result['status'] == 'skipped':
                    skipped_count += 1
                else:
                    error_count += 1
                    tqdm.write(f"Error processing {result['path']}: {result.get('msg')}")
                
                pbar.update(1)
                
    print("\n--- Summary ---")
    print(f"Success: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Done.")

if __name__ == "__main__":
    main()
