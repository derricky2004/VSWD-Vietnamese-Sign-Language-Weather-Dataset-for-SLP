import csv
import json
import os
import subprocess
import cv2
import threading
import time
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rule_based_classifier import RuleBasedClassifier
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = "/workspace/datdq/SignWeather"
RAW_VIDEO_DIR = f"{BASE_DIR}/data/raw_videos"
SCENE_VIDEO_DIR = f"{BASE_DIR}/data/scene_videos" # Fixed path from _rule to normal
LABELED_JSON_DIR = f"{BASE_DIR}/data/labeled_videos" # Fixed path to keep it standard
METADATA_DIR = f"{BASE_DIR}/data/metadata"

CLIP_MAPPING_CSV = f"{METADATA_DIR}/clip_mapping_final.csv"
VSWD_CSV = f"{METADATA_DIR}/vswd_final_filtered.csv"
OUTPUT_METADATA_CSV = f"{METADATA_DIR}/scene_metadata_realtime.csv"

# Inference Config
CONFIDENCE_THRESHOLD = 0.20
MIN_EVENT_FRAMES = 5
MAX_WORKERS = 2 

# Globals
csv_lock = threading.Lock()
console_lock = threading.Lock()

# --- HELPER CLASSES & FUNCTIONS ---

def log(msg):
    with console_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

class PipelineUtils:
    @staticmethod
    def get_video_mapping(csv_path):
        """Map new_video_id (v003) to original_video_id (0Gw4diTa1xA)"""
        mapping = {}
        if not os.path.exists(csv_path):
            return mapping
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = row.get('new_video_id')
                oid = row.get('original_video_id')
                if nid and oid and nid not in mapping:
                    mapping[nid] = oid
        return mapping

    @staticmethod
    def get_valid_videos_from_vswd(csv_path):
        valid_ids = set()
        if not os.path.exists(csv_path):
            return valid_ids
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get('path', '')
                if '/' in path:
                    vid_id = path.split('/')[0]
                    valid_ids.add(vid_id)
        return valid_ids

    @staticmethod
    def load_clip_data(csv_path, vswd_path):
        valid_clips = set()
        vswd_data = {}
        if os.path.exists(vswd_path):
            with open(vswd_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    path = row['path']
                    clip_id = os.path.basename(path).replace(".mp4", "")
                    valid_clips.add(clip_id)
                    vswd_data[clip_id] = row

        clip_times = {}
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    clip_id = row['clip_id']
                    if clip_id in valid_clips:
                        clip_times[clip_id] = {
                            'start': float(row['start']),
                            'end': float(row['end']),
                            'data': vswd_data.get(clip_id)
                        }
        return clip_times

# --- STEP 1: SEGMENTATION (INFERENCE) ---

def analyze_scenes(yes_indices, total_frames, fps, output_json_path, min_event_frames=5):
    if not yes_indices:
        return

    events = []
    merge_threshold = int(1.0 * fps)
    
    curr_start = yes_indices[0]
    curr_end = yes_indices[0]
    
    for i in range(1, len(yes_indices)):
        frame = yes_indices[i]
        if frame - curr_end <= merge_threshold:
            curr_end = frame
        else:
            if (curr_end - curr_start + 1) >= min_event_frames:
                events.append((curr_start, curr_end))
            curr_start = frame
            curr_end = frame
    if (curr_end - curr_start + 1) >= min_event_frames:
        events.append((curr_start, curr_end))
    
    scenes = []
    curr_frame_idx = 0
    scene_counter = 1
    
    for event_start, event_end in events:
        scene_end = event_start - 1
        if scene_end > curr_frame_idx: 
            scene_duration_frames = scene_end - curr_frame_idx + 1
            if scene_duration_frames > 5:
                scenes.append({
                    "scene_id": scene_counter,
                    "start": round(curr_frame_idx / fps, 3),
                    "end": round(scene_end / fps, 3),
                    "duration": round(scene_duration_frames / fps, 3),
                    "frame_start": curr_frame_idx,
                    "frame_end": scene_end
                })
                scene_counter += 1
        curr_frame_idx = event_end + 1
        
    if curr_frame_idx < total_frames - 1:
        scene_end = total_frames - 1
        scene_duration_frames = scene_end - curr_frame_idx + 1
        if scene_duration_frames > 5:
            scenes.append({
                "scene_id": scene_counter,
                "start": round(curr_frame_idx / fps, 3),
                "end": round(scene_end / fps, 3),
                "duration": round(scene_duration_frames / fps, 3),
                "frame_start": curr_frame_idx,
                "frame_end": scene_end
            })

    with open(output_json_path, 'w') as f:
        json.dump({
            "total_yes_events": len(events),
            "total_scenes": len(scenes),
            "scenes": scenes
        }, f, indent=4)

def run_inference(input_path, output_json_path):
    # RuleBased logic
    classifier = RuleBasedClassifier(scale_factor=1.5)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise Exception(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    yes_frames_indices = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        is_yes, confidence = classifier.predict(pil_img, do_crop=True, threshold=CONFIDENCE_THRESHOLD)
        if is_yes:
            yes_frames_indices.append(frame_idx)
            
        frame_idx += 1
            
    cap.release()
    analyze_scenes(yes_frames_indices, total_frames, fps, output_json_path, MIN_EVENT_FRAMES)

# --- STEP 2: MATCHING & CUTTING ---

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

def append_to_csv(rows, csv_path, fieldnames):
    if not rows:
        return
    with csv_lock:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
            f.flush()

def process_single_video_pipeline(new_id, original_id, all_clip_times):
    try:
        raw_vid_path = Path(RAW_VIDEO_DIR) / f"{original_id}.mp4"
        labeled_json_path = Path(LABELED_JSON_DIR) / f"{original_id}_labeled.json"
        
        # Ensure directories exist
        Path(LABELED_JSON_DIR).mkdir(parents=True, exist_ok=True)
        
        if not raw_vid_path.exists():
            log(f"[{new_id}] Skipped: Raw video missing ({raw_vid_path})")
            return 0

        # 1. Inference
        if not labeled_json_path.exists():
            log(f"[{new_id}] Generating JSON (Inference)...")
            try:
                run_inference(raw_vid_path, labeled_json_path)
                log(f"[{new_id}] JSON generated.")
            except Exception as e:
                log(f"[{new_id}] Inference Failed: {e}")
                return 0
        else:
            pass

        # 2. Match & Process Scenes
        with open(labeled_json_path) as f:
            scenes = json.load(f).get('scenes', [])

        video_clips = {k: v for k, v in all_clip_times.items() if k.startswith(f"{new_id}_")}
        if not video_clips:
            log(f"[{new_id}] No clips found in VSWD.")
            return 0

        scene_matches = {scene['scene_id']: [] for scene in scenes}
        
        for clip_id, info in video_clips.items():
            c_start, c_end = info['start'], info['end']
            clip_dur = c_end - c_start
            
            best_scene = -1
            max_overlap_pct = 0.0
            
            for scene in scenes:
                # Intersection logic
                inter_start = max(scene['start'], c_start)
                inter_end = min(scene['end'], c_end)
                overlap = max(0, inter_end - inter_start)
                
                if overlap > 0:
                    pct = (overlap / clip_dur) * 100
                    if pct > max_overlap_pct:
                        max_overlap_pct = pct
                        best_scene = scene['scene_id']
            
            if best_scene != -1 and max_overlap_pct > 30:
                scene_matches[best_scene].append(info)

        output_dir = Path(SCENE_VIDEO_DIR) / new_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_buffer = []
        scenes_processed = 0
        
        for scene in scenes:
            sid = scene['scene_id']
            matches = scene_matches[sid]
            if not matches:
                continue

            matches.sort(key=lambda x: x['start'])
            
            merged_text = " ".join([m['data']['text'] for m in matches])
            first = matches[0]['data']
            scores = [float(m['data']['thesis_score']) for m in matches]
            avg_score = int(sum(scores) / len(scores))
            orig_clips = ";".join([os.path.basename(m['data']['path']).replace(".mp4","") for m in matches])
            
            scene_filename = f"scene_{sid:03d}.mp4"
            scene_out = output_dir / scene_filename
            
            try:
                cut_video(raw_vid_path, scene_out, scene['start'], scene['end'])
                
                row = {
                    "path": f"{new_id}/{scene_filename}",
                    "text": merged_text,
                    "quality_level": first['quality_level'],
                    "content_label": first['content_label'],
                    "thesis_score": avg_score,
                    "original_clips": orig_clips
                }
                metadata_buffer.append(row)
                scenes_processed += 1
            except Exception as e:
                log(f"[{new_id}] Error cutting scene {sid}: {e}")
        
        fieldnames = ["path", "text", "quality_level", "content_label", "thesis_score", "original_clips"]
        append_to_csv(metadata_buffer, OUTPUT_METADATA_CSV, fieldnames)
        
        log(f"[{new_id}] Completed. {scenes_processed} scenes exported.")
        return scenes_processed

    except Exception as e:
        log(f"[{new_id}] CRITICAL FAIL: {e}")
        return 0

# --- MAIN ---

def main():
    log("=== STARTING MULTI-WORKER PIPELINE (RULE-BASED) ===")
    log(f"Outputs will be streamed to: {OUTPUT_METADATA_CSV}")
    
    # 0. Load Data
    video_map = PipelineUtils.get_video_mapping(CLIP_MAPPING_CSV)
    valid_new_ids = PipelineUtils.get_valid_videos_from_vswd(VSWD_CSV)
    all_clip_times = PipelineUtils.load_clip_data(CLIP_MAPPING_CSV, VSWD_CSV)
    
    # Initialize CSV with header
    fieldnames = ["path", "text", "quality_level", "content_label", "thesis_score", "original_clips"]
    with open(OUTPUT_METADATA_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # Prepare Tasks
    tasks = []
    # Prioritize v003, v004
    sorted_ids = sorted(list(valid_new_ids))
    for special in ['v004', 'v003']:
        if special in sorted_ids:
            sorted_ids.insert(0, sorted_ids.pop(sorted_ids.index(special)))
    
    log(f"Queueing {len(sorted_ids)} videos with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for new_id in sorted_ids:
            original_id = video_map.get(new_id)
            if not original_id:
                continue
            
            future = executor.submit(process_single_video_pipeline, new_id, original_id, all_clip_times)
            futures[future] = new_id
            
        completed_count = 0
        total_count = len(futures)
        
        for future in as_completed(futures):
            new_id = futures[future]
            try:
                count = future.result()
                completed_count += 1
            except Exception as e:
                log(f"[{new_id}] Unhandled Exception: {e}")

    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
