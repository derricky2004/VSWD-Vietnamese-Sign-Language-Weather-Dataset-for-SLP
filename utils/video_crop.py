import cv2
from pathlib import Path

def crop_video(input_path, output_path, crop_params):
    """
    Crop video based on relative coordinates (0-1 range).
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        crop_params: Dict with keys 'x', 'y', 'w', 'h' (all in 0-1 range)
    
    Returns:
        frame_count: Number of frames processed
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    x = int(crop_params['x'] * orig_w)
    y = int(crop_params['y'] * orig_h)
    w = int(crop_params['w'] * orig_w)
    h = int(crop_params['h'] * orig_h)
    
    if w % 2 != 0: w -= 1
    if h % 2 != 0: h -= 1
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        crop_frame = frame[y:y+h, x:x+w]
        out.write(crop_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return frame_count

def get_crop_dimensions(video_path, crop_params):
    """
    Calculate crop dimensions without processing video.
    
    Returns:
        Dict with original and cropped dimensions
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    crop_w = int(crop_params['w'] * orig_w)
    crop_h = int(crop_params['h'] * orig_h)
    
    if crop_w % 2 != 0: crop_w -= 1
    if crop_h % 2 != 0: crop_h -= 1
    
    return {
        'original': {'width': orig_w, 'height': orig_h},
        'cropped': {'width': crop_w, 'height': crop_h}
    }
