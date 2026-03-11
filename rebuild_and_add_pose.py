import pandas as pd
import os
import subprocess
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import shutil

# Paths
SCENE_METADATA_BACKUP = "/workspace/datdq/SignWeather/data/metadata/scene_metadata.csv.bak"
SCENE_METADATA_TARGET = "/workspace/datdq/SignWeather/data/metadata/scene_metadata.csv"
CLIP_MAPPING_PATH = "/workspace/datdq/SignWeather/data/metadata/clip_mapping_final.csv"

# Directories
POSE_ROOT = "/workspace/datdq/SignWeather/data/scene_videos_pose"
ORIGINAL_ROOT = "/workspace/datdq/SignWeather/data/scene_videos_orginal"

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_video_duration(file_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except:
        return None

def apply_pose_and_save_frames(input_path, output_video_path):
    """
    Reads video, applies MediaPipe Holistic, saves frames to temp dir, converts to video using ffmpeg.
    Modeled after SignWeather/utils/pose_detection.py
    """
    if not os.path.exists(input_path):
        return False

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS) # Keep float for precise fps
    if fps <= 0: fps = 25.0
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed_count = 0
    
    try:
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Enhance Contrast (Logic from pose_detection.py)
                try:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l2 = clahe.apply(l)
                    lab = cv2.merge((l2, a, b))
                    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                except:
                    enhanced_frame = frame
                
                # Process
                image = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                
                # Draw
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
                
                # Save frame
                frame_path = f"{temp_dir}/frame_{processed_count:06d}.png"
                cv2.imwrite(frame_path, image)
                processed_count += 1
        
        cap.release()
        
        if processed_count == 0:
            print("Error: No frames processed.")
            return False

        # Use ffmpeg to create video from frames (Robust method)
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', f'{temp_dir}/frame_%06d.png',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-loglevel', 'error',
            output_video_path
        ]
        subprocess.run(cmd, check=True)
        return True
        
    except Exception as e:
        print(f"Error in apply_pose: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def process_single_clip(original_source, start_time, duration, final_output_path):
    temp_raw = final_output_path + ".raw.mp4"
    temp_pose_video = final_output_path + ".pose_only.mp4"
    
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    
    try:
        # 1. Cut Raw Video (video+audio)
        cmd_cut = [
            'ffmpeg', '-y', '-i', original_source,
            '-ss', str(start_time), '-t', str(duration),
            '-c:v', 'libx264', '-c:a', 'aac', '-loglevel', 'error',
            temp_raw
        ]
        subprocess.run(cmd_cut, check=True)
        
        # 2. Apply Pose (Generates temp_pose_video from frames)
        if not apply_pose_and_save_frames(temp_raw, temp_pose_video):
            print(f"Failed to generate pose video for {final_output_path}")
            return False
            
        # 3. Mux: Pose Video + Raw Audio
        cmd_mux = [
            'ffmpeg', '-y',
            '-i', temp_pose_video,
            '-i', temp_raw,
            '-c:v', 'copy', # Video is already encoded by ffmpeg in step 2
            '-c:a', 'aac', # Re-encode audio to be safe
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest',
            '-loglevel', 'error',
            final_output_path
        ]
        subprocess.run(cmd_mux, check=True)
        
        return os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 1000
        
    except Exception as e:
        print(f"Error processing clip {final_output_path}: {e}")
        return False
    finally:
        if os.path.exists(temp_raw): os.remove(temp_raw)
        if os.path.exists(temp_pose_video): os.remove(temp_pose_video)

def main():
    print("Loading backup metadata...")
    if not os.path.exists(SCENE_METADATA_BACKUP):
        return

    scene_df = pd.read_csv(SCENE_METADATA_BACKUP)
    clip_df = pd.read_csv(CLIP_MAPPING_PATH)
    clip_info_map = clip_df.set_index('clip_id').to_dict('index')
    clip_duration_map = dict(zip(clip_df['clip_id'], clip_df['duration']))

    new_rows = []
    processed_count = 0
    
    # Process Logic
    for idx, row in scene_df.iterrows():
        original_clips_str = str(row['original_clips']) if pd.notna(row['original_clips']) else ""
        clips = [c.strip() for c in original_clips_str.split(';') if c.strip()]
        
        if len(clips) <= 1:
            new_rows.append(row.to_dict())
            continue
            
        scene_rel_path = row['path']
        original_source = os.path.join(ORIGINAL_ROOT, scene_rel_path)
        
        if not os.path.exists(original_source):
             print(f"Missing original source: {original_source}")
             continue
             
        physical_duration = get_video_duration(original_source)
        if not physical_duration: continue
            
        clip_durations = [clip_duration_map.get(c, 0) for c in clips]
        total_mapped = sum(clip_durations)
        current_start = 0.0
        
        scene_dir = os.path.dirname(scene_rel_path)
        scene_stem = os.path.splitext(os.path.basename(scene_rel_path))[0]
        
        for i, clip_id in enumerate(clips):
            ratio = clip_durations[i] / total_mapped if total_mapped > 0 else 0
            actual_duration = ratio * physical_duration
            
            new_filename = f"{scene_stem}_{clip_id}.mp4"
            new_rel_path = os.path.join(scene_dir, new_filename)
            final_path = os.path.join(POSE_ROOT, new_rel_path)
            
            print(f"Processing {clip_id} ({actual_duration:.2f}s)...")
            if process_single_clip(original_source, current_start, actual_duration, final_path):
                c_data = clip_info_map.get(clip_id, {})
                new_row = {
                    'path': new_rel_path,
                    'text': c_data.get('text_final', row['text']),
                    'quality_level': c_data.get('quality_level', row['quality_level']),
                    'content_label': c_data.get('content_label', row['content_label']),
                    'thesis_score': c_data.get('thesis_score', row['thesis_score']),
                    'original_clips': clip_id
                }
                new_rows.append(new_row)
                processed_count += 1
            
            current_start += actual_duration

    final_df = pd.DataFrame(new_rows)
    final_df.to_csv(SCENE_METADATA_TARGET, index=False)
    print(f"Done. Processed {processed_count} clips.")

if __name__ == "__main__":
    main()
