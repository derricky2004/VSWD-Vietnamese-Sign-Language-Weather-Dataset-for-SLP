import csv
import os
import re
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = "/workspace/datdq/SignWeather"
METADATA_DIR = f"{BASE_DIR}/data/metadata"
SCENE_METADATA_CSV = f"{METADATA_DIR}/scene_metadata_realtime.csv"
BACKUP_CSV = f"{SCENE_METADATA_CSV}.bak_sort"

def sort_key_func(row):
    """
    Extract video ID and scene number for sorting.
    Path format: "v001/scene_001.mp4"
    """
    path = row.get('path', '')
    # Match video ID (v\d+) and scene number (scene_\d+)
    match = re.search(r'(v\d+)/scene_(\d+)', path)
    if match:
        video_id = match.group(1)
        scene_num = int(match.group(2))
        return (video_id, scene_num)
    return (path, 0) # Fallback

def main():
    if not os.path.exists(SCENE_METADATA_CSV):
        print(f"Error: {SCENE_METADATA_CSV} not found.")
        return

    print(f"Reading {SCENE_METADATA_CSV}...")
    rows = []
    fieldnames = []
    with open(SCENE_METADATA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Found {len(rows)} rows. Sorting...")
    
    # Sort rows
    rows.sort(key=sort_key_func)

    # Backup original
    print(f"Creating backup at {BACKUP_CSV}...")
    if os.path.exists(SCENE_METADATA_CSV):
        import shutil
        shutil.copy(SCENE_METADATA_CSV, BACKUP_CSV)

    # Write back
    print(f"Writing sorted metadata back to {SCENE_METADATA_CSV}...")
    with open(SCENE_METADATA_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Sorting completed successfully.")

if __name__ == "__main__":
    main()
