import csv
import json
import os
import subprocess
from pathlib import Path
import math

# --- CONFIGURATION ---
VSWD_CSV = "/workspace/datdq/SignWeather/data/metadata/vswd_final_filtered.csv"
CLIP_MAPPING_CSV = "/workspace/datdq/SignWeather/data/metadata/clip_mapping_final.csv"
INPUT_VIDEO_DIR = "/workspace/datdq/SignWeather/data/cropped_videos"
LABEL_JSON_DIR = "/workspace/datdq/SignWeather/data/labeled_videos"
OUTPUT_SCENE_DIR = "/workspace/datdq/SignWeather/data/scene_videos"
OUTPUT_METADATA_CSV = "/workspace/datdq/SignWeather/data/metadata/scene_metadata.csv"

# Target configuration
TARGET_VIDEO_ID = "v004"
TARGET_JSON_FILE = "v004_labeled.json" # Corresponds to v004

def load_filtered_clips(csv_path, video_id_prefix):
    """Load valid clip IDs from vswd_final_filtered.csv"""
    valid_clips = set()
    clip_data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row['path']
            # path example: v003/v003_c005.mp4
            if path.startswith(video_id_prefix + "/"):
                clip_id = os.path.basename(path).replace(".mp4", "")
                valid_clips.add(clip_id)
                clip_data[clip_id] = row
    return valid_clips, clip_data

def load_clip_times(csv_path, video_id_prefix):
    """Load start/end times for ALL clips matching the video ID"""
    clip_times = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['new_video_id'] == video_id_prefix:
                clip_id = row['clip_id']
                start = float(row['start'])
                end = float(row['end'])
                clip_times[clip_id] = (start, end)
    return clip_times

def load_json_scenes(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['scenes']

def calculate_overlap(range1, range2):
    start1, end1 = range1
    start2, end2 = range2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start < overlap_end:
        return overlap_end - overlap_start
    return 0

def cut_video(input_path, output_path, start, end):
    """Cut video using ffmpeg with re-encoding for precision"""
    duration = end - start
    cmd = [
        'ffmpeg',
        '-y', # Overwrite
        '-i', str(input_path),
        '-ss', str(start),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        str(output_path)
    ]
    # Suppress output unless error
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error cutting video: {result.stderr}")
        return False
    return True

def main():
    print(f"Processing {TARGET_VIDEO_ID}...")
    
    # 1. Prepare Paths
    input_video_path = Path(INPUT_VIDEO_DIR) / f"{TARGET_VIDEO_ID}.mp4"
    json_path = Path(LABEL_JSON_DIR) / TARGET_JSON_FILE
    
    if not input_video_path.exists():
        print(f"Error: Input video not found: {input_video_path}")
        return
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return

    # 2. Load Data
    print("Loading metadata...")
    valid_clips, valid_clip_data = load_filtered_clips(VSWD_CSV, TARGET_VIDEO_ID)
    all_clip_times = load_clip_times(CLIP_MAPPING_CSV, TARGET_VIDEO_ID)
    
    # Filter to only use clips that exist in the filtered dataset
    filtered_clip_times = {k: v for k, v in all_clip_times.items() if k in valid_clips}
    
    scenes = load_json_scenes(json_path)
    print(f"Loaded {len(scenes)} scenes.")
    
    # 3. Match Clips to Scenes
    # Structure: scene_id -> list of clip_ids
    scene_assignments = {scene['scene_id']: [] for scene in scenes}
    
    for clip_id, (c_start, c_end) in filtered_clip_times.items():
        clip_range = (c_start, c_end)
        clip_duration = c_end - c_start
        
        best_scene_id = -1
        max_overlap_pct = 0.0
        
        for scene in scenes:
            scene_id = scene['scene_id']
            s_start = scene['start']
            s_end = scene['end']
            scene_range = (s_start, s_end)
            
            overlap = calculate_overlap(scene_range, clip_range)
            if overlap > 0:
                match_pct = (overlap / clip_duration) * 100
                if match_pct > max_overlap_pct:
                    max_overlap_pct = match_pct
                    best_scene_id = scene_id
        
        if best_scene_id != -1 and max_overlap_pct > 30:
            scene_assignments[best_scene_id].append({
                "clip_id": clip_id,
                "start": c_start, # For sorting
                "data": valid_clip_data[clip_id]
            })
            
    # 4. Process Scenes and Create New Mapping
    new_metadata_rows = []
    
    # Prepare output dir
    vid_output_dir = Path(OUTPUT_SCENE_DIR) / TARGET_VIDEO_ID
    vid_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nProcessing Scenes...")
    count_exported = 0
    
    for scene in scenes:
        scene_id = scene['scene_id']
        assignments = scene_assignments[scene_id]
        
        if not assignments:
            continue
            
        # Sort assignments by time
        assignments.sort(key=lambda x: x['start'])
        
        # Merge Metadata
        # If multiple clips, join text with space
        merged_text = " ".join([item['data']['text'] for item in assignments])
        
        # For other fields, we might take the first one or average or modal value
        # Let's take 'quality_level' from the first clip (assuming similarity)
        # or calculate 'average' score if numeric.
        first_clip_data = assignments[0]['data']
        quality_level = first_clip_data['quality_level']
        content_label = first_clip_data['content_label']
        
        # Calculate average thesis score
        scores = [float(item['data']['thesis_score']) for item in assignments]
        avg_score = sum(scores) / len(scores)
        
        # Generate New Filename
        # Format: v003/scene_001.mp4 (inside OUTPUT_SCENE_DIR)
        scene_filename = f"scene_{scene_id:03d}.mp4"
        scene_output_path = vid_output_dir / scene_filename
        
        # Cut Video
        # Use scene start/end from JSON
        print(f"  Exporting Scene {scene_id} ({len(assignments)} clips merged)...")
        if cut_video(input_video_path, scene_output_path, scene['start'], scene['end']):
            # Add to metadata list
            rel_path = f"{TARGET_VIDEO_ID}/{scene_filename}"
            new_metadata_rows.append({
                "path": rel_path,
                "text": merged_text,
                "quality_level": quality_level,
                "content_label": content_label,
                "thesis_score": int(avg_score),
                "original_clips": ";".join([x['clip_id'] for x in assignments])
            })
            count_exported += 1

    # 5. Write to CSV
    # If file exists, we append? Or overwrite? 
    # User said "create a .csv type ... to create new mapping"
    # Let's overwrite or create new for this video.
    # To be safe and since we process 1 video, let's just write this to a file
    # If we were processing bulk, we'd append.
    
    # Check if header needed
    file_exists = Path(OUTPUT_METADATA_CSV).exists()
    
    # For this demo, let's write to a fresh file or append if it exists
    mode = 'a' if file_exists else 'w'
    fieldnames = ["path", "text", "quality_level", "content_label", "thesis_score", "original_clips"]
    
    with open(OUTPUT_METADATA_CSV, mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_metadata_rows)
        
    print(f"\nCompleted! Exported {count_exported} scenes.")
    print(f"Metadata saved to {OUTPUT_METADATA_CSV}")
    print(f"Videos saved to {vid_output_dir}")

if __name__ == "__main__":
    main()
