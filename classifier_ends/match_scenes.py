import csv
import json
import os

# Paths
VSWD_CSV = "/workspace/datdq/SignWeather/data/metadata/vswd_final_filtered.csv"
CLIP_MAPPING_CSV = "/workspace/datdq/SignWeather/data/metadata/clip_mapping_final.csv"
JSON_FILE = "/workspace/datdq/SignWeather/data/labeled_videos/0Gw4diTa1xA_labeled.json"
TARGET_VIDEO_ID = "v003"  # 0Gw4diTa1xA corresponds to v003

def load_filtered_clips(csv_path, video_id_prefix):
    """Load valid clip IDs from vswd_final_filtered.csv"""
    valid_clips = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # path is like v003/v003_c005.mp4
            path = row['path']
            if path.startswith(video_id_prefix + "/"):
                clip_id = os.path.basename(path).replace(".mp4", "")
                valid_clips.add(clip_id)
    return valid_clips

def load_clip_times(csv_path, video_id_prefix):
    """Load start/end times for clips matching the video ID"""
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
    """Calculate overlap between two ranges (start, end)"""
    start1, end1 = range1
    start2, end2 = range2
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start < overlap_end:
        return overlap_end - overlap_start
    return 0

def main():
    print("Loading data...")
    valid_clips = load_filtered_clips(VSWD_CSV, TARGET_VIDEO_ID)
    print(f"Found {len(valid_clips)} valid clips in filtered CSV for {TARGET_VIDEO_ID}")
    
    all_clip_times = load_clip_times(CLIP_MAPPING_CSV, TARGET_VIDEO_ID)
    print(f"Found {len(all_clip_times)} total clips in mapping CSV for {TARGET_VIDEO_ID}")
    
    # Filter times to include only valid clips
    valid_clip_times = {k: v for k, v in all_clip_times.items() if k in valid_clips}
    print(f"Processing {len(valid_clip_times)} clips after filtering")
    
    scenes = load_json_scenes(JSON_FILE)
    print(f"Loaded {len(scenes)} scenes from JSON")
    
    print("\nMatching Results (1 Clip -> 1 Scene):")
    print("-" * 60)
    
    # Data structure to hold matches: scene_id -> list of (clip_id, match_pct)
    scene_matches = {scene['scene_id']: [] for scene in scenes}
    
    # Track assigned clips to ensure they aren't double counted (though logic ensures max overlap)
    assigned_clips = set()
    
    for clip_id, (c_start, c_end) in valid_clip_times.items():
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
                # Calculate percentage of the CLIP that is covered by the scene
                match_pct = (overlap / clip_duration) * 100
                
                if match_pct > max_overlap_pct:
                    max_overlap_pct = match_pct
                    best_scene_id = scene_id
        
        # Threshold for validation (e.g. at least 30% of clip must be in scene)
        if best_scene_id != -1 and max_overlap_pct > 30:
            scene_matches[best_scene_id].append((clip_id, max_overlap_pct, c_start, c_end))
            assigned_clips.add(clip_id)
            
    # Print Results by Scene
    for scene in scenes:
        scene_id = scene['scene_id']
        s_start = scene['start']
        s_end = scene['end']
        
        matches = scene_matches[scene_id]
        # Sort matches by clip start time within the scene
        matches.sort(key=lambda x: x[2])
        
        print(f"Scene {scene_id} ({s_start:.2f}s - {s_end:.2f}s):")
        if matches:
            for m in matches:
                # m = (clip_id, match_pct, c_start, c_end)
                print(f"  -> {m[0]} ({m[2]:.2f}s - {m[3]:.2f}s) [Covered: {m[1]:.1f}%]")
        else:
            print("  -> (No clips assigned)")
        print("-" * 60)

    # Check for missed clips
    missed_clips = valid_clips - assigned_clips
    if missed_clips:
        print("\nValid clips NOT matched to any scene:")
        for clip in sorted(missed_clips):
             times = valid_clip_times.get(clip, ("?", "?"))
             print(f"  {clip} ({times[0]}s - {times[1]}s)")

if __name__ == "__main__":
    main()
