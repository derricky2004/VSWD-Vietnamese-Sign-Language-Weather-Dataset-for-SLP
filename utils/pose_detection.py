import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_pose_landmarks(video_path, min_detection_confidence=0.3, min_tracking_confidence=0.7):
    """
    Extract pose landmarks from video using MediaPipe Holistic.
    
    Args:
        video_path: Path to input video
        min_detection_confidence: Detection confidence threshold
        min_tracking_confidence: Tracking confidence threshold
    
    Returns:
        List of frame data with pose, face, and hand landmarks
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    all_frames_data = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=2
    ) as holistic:
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            image = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            frame_data = {
                "frame": frame_count,
                "pose": [],
                "face": [],
                "left_hand": [],
                "right_hand": []
            }
            
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_data["pose"].append([lm.x, lm.y, lm.z, lm.visibility])
            
            if results.face_landmarks:
                for lm in results.face_landmarks.landmark:
                    frame_data["face"].append([lm.x, lm.y, lm.z])
            
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_data["left_hand"].append([lm.x, lm.y, lm.z])
            
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    frame_data["right_hand"].append([lm.x, lm.y, lm.z])
            
            all_frames_data.append(frame_data)
    
    cap.release()
    return all_frames_data

def visualize_pose_on_video(input_path, output_path, min_detection_confidence=0.3, min_tracking_confidence=0.7):
    """
    Process video and draw pose landmarks on frames.
    Uses ffmpeg for final encoding to ensure compatibility.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video with pose visualization
        min_detection_confidence: Detection confidence threshold
        min_tracking_confidence: Tracking confidence threshold
    
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    import tempfile
    import shutil
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp()
    
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        with mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        ) as holistic:
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l2 = clahe.apply(l)
                lab = cv2.merge((l2, a, b))
                enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                image = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # cv2.putText(image, f"Frame: {frame_count}/{total_frames}", 
                #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save frame as image
                frame_path = f"{temp_dir}/frame_{frame_count:06d}.png"
                cv2.imwrite(frame_path, image)
        
        cap.release()
        
        # Use ffmpeg to create video from frames
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', f'{temp_dir}/frame_%06d.png',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return False
        
        return True
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def enhance_frame_contrast(frame):
    """
    Enhance frame contrast using CLAHE.
    
    Args:
        frame: Input BGR frame
    
    Returns:
        Enhanced BGR frame
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
