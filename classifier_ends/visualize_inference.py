
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from inference import EndClassifier

def visualize_video(input_path, output_path, model_path, confidence_threshold=0.5, min_event_frames=5):
    print(f"Processing: {input_path}")
    print(f"Config: Threshold={confidence_threshold}, Min Event Frames={min_event_frames}")
    
    # Initialize Classifier
    classifier = EndClassifier(model_path=model_path)
    print("Model loaded.")

    # Open Video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return

    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output Writer Replaced by FFmpeg Assembly
    # Create temp directory for frames
    import tempfile
    import shutil
    import subprocess
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp dir: {temp_dir}")
    
    # Crop Params for Visualization
    crop_x = int(classifier.crop_rel_x * width)
    crop_y = int(classifier.crop_rel_y * height)
    crop_w = int(classifier.crop_rel_w * width)
    crop_h = int(classifier.crop_rel_h * height)
    
    # Clamp
    crop_x = max(0, min(crop_x, width-1))
    crop_y = max(0, min(crop_y, height-1))
    
    print(f"Crop Region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
    print(f"Processing video ({total_frames} / {total_frames} frames)...")
    
    frame_idx = 0
    limit_frames = total_frames
    
    yes_frames_indices = []
    
    try:
        while True:
            if frame_idx >= limit_frames:
                print(f"\nReached limit ({frame_idx} frames). Stopping capture.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            is_yes, confidence = classifier.predict(pil_img, do_crop=True, threshold=confidence_threshold)
            
            if is_yes:
                yes_frames_indices.append(frame_idx)

            label_text = "YES" if is_yes else "NO"
            color = (0, 255, 0) if is_yes else (0, 0, 255)
            
            # 1. Draw Crop Rectangle
            cv2.rectangle(frame, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), (255, 255, 0), 2)
            
            # 2. Draw Label Background
            text = f"{label_text} ({confidence:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            cv2.rectangle(frame, (20, 20), (20 + text_w + 20, 20 + text_h + 20), (0, 0, 0), -1)
            
            # 3. Draw Label Text
            cv2.putText(frame, text, (30, 30 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Save frame
            frame_path = f"{temp_dir}/frame_{frame_idx:06d}.png"
            cv2.imwrite(frame_path, frame)
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{limit_frames} frames...", end='\r')
                
        cap.release()
        
        print(f"\nEncoding video with ffmpeg...")
        # Use ffmpeg to create video from frames
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', f'{temp_dir}/frame_%06d.png',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
        else:
            print(f"Done! Video saved to {output_path}")
            
        # Analyze Scenes
        json_path = output_path.with_suffix('.json')
        analyze_scenes(yes_frames_indices, total_frames, fps, json_path, min_event_frames)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def analyze_scenes(yes_indices, total_frames, fps, output_json_path, min_event_frames=5):
    import json
    
    # 1. Group YES frames into Events
    if not yes_indices:
        print("No YES signals found.")
        # If no signals, the whole video is one scene? Or no scene?
        # Let's save one big scene?
        # scenes = [{ "scene_id": 1, "start": 0, "end": total_frames/fps, ... }]
        # For now, let's keep original behavior -> No events.
        return

    events = []
    
    # Algorithm: Merge close frames
    merge_threshold = int(1.0 * fps) # Merge if within 1 second
    
    curr_start = yes_indices[0]
    curr_end = yes_indices[0]
    
    for i in range(1, len(yes_indices)):
        frame = yes_indices[i]
        if frame - curr_end <= merge_threshold:
            curr_end = frame
        else:
            # Check length before adding
            if (curr_end - curr_start + 1) >= min_event_frames:
                events.append((curr_start, curr_end))
            curr_start = frame
            curr_end = frame
            
    # Add last event
    if (curr_end - curr_start + 1) >= min_event_frames:
        events.append((curr_start, curr_end))
    
    print(f"Found {len(events)} YES events (after filtering < {min_event_frames} frames).")
    
    # 2. Extract Scenes BETWEEN events (n-1 scenes)
    scenes = []
    
    # We define scenes relative to the YES events.
    # Scene 1: Start of video -> Start of Event 1 ?
    # Typically: Content -> YES -> Content -> YES
    # So we should have a scene BEFORE the first event too.
    
    curr_frame_idx = 0
    scene_counter = 1
    
    # Loop through events to create scenes before them
    for event_start, event_end in events:
        scene_end = event_start - 1
        
        if scene_end > curr_frame_idx: # Valid scene
            scene_duration_frames = scene_end - curr_frame_idx + 1
            if scene_duration_frames > 5: # Minimal scene length filter too?
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
        
    # Check for scene AFTER last event
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
            
    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump({
            "total_yes_events": len(events),
            "total_scenes": len(scenes),
            "scenes": scenes
        }, f, indent=4)
    print(f"Saved scene analysis to {output_json_path}")


if __name__ == "__main__":
    MODEL_PATH = "/workspace/datdq/SignWeather/classifier_ends/classifier_ends.pth"
    INPUT_VIDEO = "/workspace/datdq/SignWeather/data/raw_videos/1-iUEsz_srY.mp4"
    OUTPUT_DIR = Path("/workspace/datdq/SignWeather/data/labeled_videos")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    OUTPUT_VIDEO = OUTPUT_DIR / "1-iUEsz_srY_labeled.mp4"
    
    # CONFIG
    # Lower threshold to catch missing events
    # Increase min_event_frames to avoid noise
    THRESH = 0.20 
    MIN_FRAMES = 5 # ~0.2s at 25fps
    
    if not Path(INPUT_VIDEO).exists():
        print(f"File not found: {INPUT_VIDEO}")
    else:
        visualize_video(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH, confidence_threshold=THRESH, min_event_frames=MIN_FRAMES)
