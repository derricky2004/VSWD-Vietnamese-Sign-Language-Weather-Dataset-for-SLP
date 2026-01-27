from pathlib import Path
from utils.common import run_cmd, ensure_dir_exists
from config import Config

def extract_audio_to_wav(video_path: Path, audio_path: Path) -> None:
    ensure_dir_exists(audio_path.parent)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", str(Config.AUDIO_CHANNELS),
        "-ar", str(Config.AUDIO_SAMPLE_RATE),
        str(audio_path)
    ]
    
    print(f"Extracting audio: {video_path.name} -> {audio_path.name}")
    run_cmd(cmd)

def cut_video_segment(video_path: Path, start: float, end: float, output_path: Path) -> None:
    ensure_dir_exists(output_path.parent)
    
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path)
    ]
    
    run_cmd(cmd)
