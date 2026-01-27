import csv
import os
import re
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
BASE_DIR = "/workspace/datdq/SignWeather"
METADATA_DIR = f"{BASE_DIR}/data/metadata"
SCENE_METADATA_CSV = f"{METADATA_DIR}/scene_metadata_realtime.csv"
CLIP_MAPPING_CSV = f"{METADATA_DIR}/clip_mapping_final.csv"
SCENE_VIDEO_DIR = f"{BASE_DIR}/data/scene_videos"
RAW_VIDEO_DIR = f"{BASE_DIR}/data/raw_videos"

def load_clip_mapping(csv_path):
    """
    Load clip details: start, end, text, quality, etc.
    Returns dict: clip_id -> {start, end, raw_video_path, ...}
    """
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row['clip_id']
            # Reconstruct raw video path implicitly or explicit
            # original_video_id is '1-iUEsz_srY' -> raw_videos/1-iUEsz_srY.mp4
            orig_id = row['original_video_id']
            raw_path = os.path.join(RAW_VIDEO_DIR, f"{orig_id}.mp4")
            
            mapping[clip_id] = {
                'start': float(row['start']),
                'end': float(row['end']),
                'text': row['final_text'] if row.get('final_text') else row.get('text', ''),
                'quality': row['quality_level'],
                'label': row['content_label'],
                'score': float(row['thesis_score']),
                'raw_video_path': raw_path, # Mapping back to raw file
                'original_video_id': orig_id
            }
    return mapping

def parse_clip_idx(clip_id):
    # v001_c005 -> 5
    match = re.search(r'_c(\d+)$', clip_id)
    if match:
        return int(match.group(1))
    return -1

def split_clip_groups(clip_ids):
    """
    Split a list of clip IDs into consecutive sub-groups.
    ['c000', 'c002', 'c004', 'c005'] -> [['c000'], ['c002'], ['c004', 'c005']]
    """
    if not clip_ids:
        return []
    
    # Sort by index just in case
    clip_ids.sort(key=parse_clip_idx)
    
    groups = []
    current_group = [clip_ids[0]]
    
    for i in range(1, len(clip_ids)):
        curr_id = clip_ids[i]
        prev_id = clip_ids[i-1]
        
        curr_idx = parse_clip_idx(curr_id)
        prev_idx = parse_clip_idx(prev_id)
        
        if curr_idx == prev_idx + 1:
            current_group.append(curr_id)
        else:
            groups.append(current_group)
            current_group = [curr_id]
    
    groups.append(current_group)
    return groups

def cut_video(input_path, output_path, start, end):
    duration = end - start
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', str(input_path),
        '-ss', str(start),
        '-t', str(duration),
        '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

def main():
    if not os.path.exists(SCENE_METADATA_CSV):
        print(f"Error: {SCENE_METADATA_CSV} not found.")
        return

    print("Loading Clip Mapping...")
    clip_db = load_clip_mapping(CLIP_MAPPING_CSV)
    
    # Read current scene metadata
    print(f"Reading {SCENE_METADATA_CSV}...")
    scenes_to_process = []
    with open(SCENE_METADATA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenes_to_process.append(row)
            
    # Group by Video ID (v001, v003...) to handle re-indexing
    # video_id -> list of rows
    video_groups = defaultdict(list)
    for row in scenes_to_process:
        # Extract video_id from path "v003/scene_001.mp4"
        path = row['path']
        vid_id = path.split('/')[0]
        video_groups[vid_id].append(row)
        
    new_metadata_rows = []
    
    # Backup original csv
    shutil.copy(SCENE_METADATA_CSV, SCENE_METADATA_CSV + ".bak")
    print(f"Backup created at {SCENE_METADATA_CSV}.bak")
    
    print("re-processing Scenes...")
    
    # Iterate over each video group
    for vid_id, rows in tqdm(video_groups.items(), desc="Processing Videos"):
        
        output_dir = Path(SCENE_VIDEO_DIR) / vid_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # New scene counter for this video
        scene_counter = 1
        
        # Sort rows conceptually? Actually original order implies time order usually.
        # But we need to re-sort after splitting.
        # Let's collect ALL split groups for this video first, then sort by start_time, then cut.
        
        all_split_groups = []
        
        for row in rows:
            orig_clips_str = row['original_clips']
            if not orig_clips_str or orig_clips_str.strip() == "":
                continue
                
            clip_ids = orig_clips_str.split(';')
            
            # Split logic
            sub_groups = split_clip_groups(clip_ids)
            
            for grp in sub_groups:
                # Calculate info for this group
                try:
                    start_time = clip_db[grp[0]]['start']
                    end_time = clip_db[grp[-1]]['end']
                    
                    # Merge text
                    texts = [clip_db[c]['text'] for c in grp]
                    merged_text = " ".join(texts)
                    
                    # Avg score
                    scores = [clip_db[c]['score'] for c in grp]
                    avg_score = int(sum(scores) / len(scores))
                    
                    # Quality
                    quality = clip_db[grp[0]]['quality']
                    label = clip_db[grp[0]]['label']
                    
                    raw_video = clip_db[grp[0]]['raw_video_path']
                    
                    all_split_groups.append({
                        'clips': grp,
                        'start': start_time,
                        'end': end_time,
                        'text': merged_text,
                        'quality': quality,
                        'label': label,
                        'score': avg_score,
                        'raw_video': raw_video,
                        'vid_id': vid_id
                    })
                except KeyError as e:
                    print(f"Warning: Clip ID {e} matching failed. Skipping group.")
                    
        # Sort all groups by start time
        all_split_groups.sort(key=lambda x: x['start'])
        
        # Now Generate Video and Metadata
        for scene_data in all_split_groups:
            scene_filename = f"scene_{scene_counter:03d}.mp4"
            scene_out_path = output_dir / scene_filename
            scene_rel_path = f"{vid_id}/{scene_filename}"
            
            # Retrieve clip IDs for CSV
            clips_str = ";".join(scene_data['clips'])
            
            # CUT VIDEO
            try:
                cut_video(scene_data['raw_video'], scene_out_path, scene_data['start'], scene_data['end'])
                
                # Add to new metadata
                new_metadata_rows.append({
                    "path": scene_rel_path,
                    "text": scene_data['text'],
                    "quality_level": scene_data['quality'],
                    "content_label": scene_data['label'],
                    "thesis_score": scene_data['score'],
                    "original_clips": clips_str
                })
                
                scene_counter += 1
            except Exception as e:
                print(f"Error cutting {scene_rel_path}: {e}")

    # Write Final CSV
    print(f"Writing updated metadata to {SCENE_METADATA_CSV}...")
    fieldnames = ["path", "text", "quality_level", "content_label", "thesis_score", "original_clips"]
    with open(SCENE_METADATA_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_metadata_rows)
        
    print("Done.")

if __name__ == "__main__":
    main()
