import cv2
from pathlib import Path

def scale_video(input_path, output_path, scale_factor, interpolation=cv2.INTER_CUBIC):
    """
    Scale video by a factor.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        scale_factor: Scale multiplier (e.g., 5 for 500%)
        interpolation: OpenCV interpolation method (default: INTER_CUBIC)
    
    Returns:
        frame_count: Number of frames processed
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    new_w = orig_w * scale_factor
    new_h = orig_h * scale_factor
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_w, new_h))
    
    if not out.isOpened():
        print("mp4v codec failed, trying avc1...")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_w, new_h))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
        out.write(scaled_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return frame_count

def crop_and_scale_video(input_path, output_path, crop_params, scale_factor, interpolation=cv2.INTER_CUBIC):
    """
    Crop and scale video in one pass.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        crop_params: Dict with keys 'x', 'y', 'w', 'h' (all in 0-1 range)
        scale_factor: Scale multiplier
        interpolation: OpenCV interpolation method
    
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
    
    final_w = w * scale_factor
    final_h = h * scale_factor
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (final_w, final_h))
    
    if not out.isOpened():
        print("mp4v codec failed, trying avc1...")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (final_w, final_h))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        crop_frame = frame[y:y+h, x:x+w]
        scaled_frame = cv2.resize(crop_frame, (final_w, final_h), interpolation=interpolation)
        out.write(scaled_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return frame_count
