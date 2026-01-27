import csv
import os
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = "/workspace/datdq/SignWeather"
METADATA_DIR = f"{BASE_DIR}/data/metadata"
SCENE_METADATA_CSV = f"{METADATA_DIR}/scene_metadata.csv"
CLIP_MAPPING_CSV = f"{METADATA_DIR}/clip_mapping_final.csv"
VIDEO_ID_MAPPING_CSV = f"{METADATA_DIR}/mapping/video_id_mapping.csv"

def get_unique_vid_ids_from_scene_metadata(csv_path):
    vid_ids = set()
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return vid_ids
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get('path', '')
            if '/' in path:
                vid_id = path.split('/')[0]
                vid_ids.add(vid_id)
    return vid_ids

def get_mapping_from_clip_mapping(csv_path):
    mapping = {} # new_video_id -> original_video_id
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return mapping
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_id = row.get('new_video_id')
            orig_id = row.get('original_video_id')
            if new_id and orig_id:
                mapping[new_id] = orig_id
    return mapping

def main():
    print("Gathering unique video IDs from scene_metadata.csv...")
    scene_vid_ids = get_unique_vid_ids_from_scene_metadata(SCENE_METADATA_CSV)
    print(f"Found {len(scene_vid_ids)} unique video IDs in scene_metadata.csv")

    print("Loading global mapping from clip_mapping_final.csv...")
    global_mapping = get_mapping_from_clip_mapping(CLIP_MAPPING_CSV)
    print(f"Loaded {len(global_mapping)} mappings from clip_mapping_final.csv")

    # Build the final mapping list
    final_rows = []
    # Sort IDs numerically (v000, v001...)
    def sort_key(vid):
        try:
            return int(vid[1:])
        except:
            return vid
            
    sorted_ids = sorted(list(scene_vid_ids), key=sort_key)
    
    for vid_id in sorted_ids:
        if vid_id in global_mapping:
            final_rows.append({
                'original_video_id': global_mapping[vid_id],
                'new_video_id': vid_id
            })
        else:
            print(f"Warning: Video ID {vid_id} found in scene_metadata.csv but MISSING in clip_mapping_final.csv")

    # Save to mapping/video_id_mapping.csv
    print(f"Writing corrected mapping to {VIDEO_ID_MAPPING_CSV}...")
    
    # Backup
    if os.path.exists(VIDEO_ID_MAPPING_CSV):
        shutil_backup = VIDEO_ID_MAPPING_CSV + ".bak_" + str(int(os.path.getmtime(VIDEO_ID_MAPPING_CSV)))
        import shutil
        shutil.copy(VIDEO_ID_MAPPING_CSV, shutil_backup)
        print(f"Original mapping backed up to {shutil_backup}")

    with open(VIDEO_ID_MAPPING_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['original_video_id', 'new_video_id'])
        writer.writeheader()
        writer.writerows(final_rows)

    print("Sync completed successfully.")

if __name__ == "__main__":
    main()
