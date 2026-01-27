import re
from pathlib import Path
from typing import Optional
from utils.common import run_cmd, ensure_dir_exists
from config import Config


def get_video_urls(url: str, out_dir: Path = Config.RAW_VIDEOS_DIR) -> list[str]:
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(url)s",
        url
    ]
    
    try:
        run_cmd(cmd)
        return output_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def parse_video_id(url: str) -> Optional[str]:
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)',
        r'youtube\.com/embed/([a-zA-Z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_video(url: str, out_dir: Path = Config.RAW_VIDEOS_DIR) -> Optional[Path]:
    ensure_dir_exists(out_dir)
    video_id = parse_video_id(url)
    if not video_id:
        print(f"Cannot parse video ID from URL: {url}")
        return None
    
    final_mp4 = out_dir / f"{video_id}.mp4"
    if final_mp4.exists():
        if final_mp4.stat().st_size > 0:
            print(f"Video already exists: {final_mp4}")
            return final_mp4
        else:
             final_mp4.unlink()
    
    
    print(f"Downloading (VSWD style): {url}")

    temp_pattern = str(out_dir / f"{video_id}.%(ext)s")
    
    cmd_dl = [
        "yt-dlp",
        "-f", "bv*+ba/b",
        "-o", temp_pattern,
        "--no-playlist",
        "--no-check-certificates",
        url
    ]
    
    try:
        run_cmd(cmd_dl)
        
        # Tìm file vừa down
        # Exclude .part and .ytdl which are temporary
        candidates = [p for p in out_dir.glob(f"{video_id}.*") 
                      if not p.name.endswith(".part") and not p.name.endswith(".ytdl")]
        
        # Filter out the temp raw file we might have created in a previous failed run
        candidates = [p for p in candidates if not p.name.endswith("_temp_raw.mp4")]
        
        # If final MP4 exists in candidates (maybe from a previous partial run?)
        # we can't trust it unless we verify it. But logic start checked final_mp4 existence.
        # So here candidates are likely the raw download.
        
        if not candidates:
            print(f"Error: Downloaded file not found per ID {video_id}")
            # Debug: List what IS there
            all_files = list(out_dir.glob(f"{video_id}*"))
            print(f"   Debug: Files found match video_id: {[f.name for f in all_files]}")
            return None
            
        # Pick the most likely video file (largest size usually)
        raw_file = max(candidates, key=lambda p: p.stat().st_size)
        
        if raw_file.stat().st_size < 1024:
            print(f"Error: Downloaded file too small ({raw_file.stat().st_size} bytes): {raw_file.name}")
            return None

        # Chuẩn hóa sang MP4 (AAC + H264)
        # Trường hợp raw_file trùng tên final_mp4 (do yt-dlp tự ra mp4)
        if raw_file.resolve() == final_mp4.resolve():
            temp_name = out_dir / f"{video_id}_temp_raw.mp4"
            # Move raw to temp to clear the path for final output
            raw_file.rename(temp_name)
            raw_file = temp_name
            
        print(f"Converting/Standardizing to MP4: {final_mp4.name} (Source: {raw_file.name})")
        
        cmd_convert = [
            "ffmpeg", "-y",
            "-i", str(raw_file),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            str(final_mp4)
        ]
        
        run_cmd(cmd_convert)
        
        # Clean up raw temp file
        if raw_file.exists() and raw_file.resolve() != final_mp4.resolve():
             raw_file.unlink()
             
        if final_mp4.exists() and final_mp4.stat().st_size > 0:
            print(f"Success: {final_mp4}")
            return final_mp4
        else:
            return None

    except Exception as e:
        print(f"Failed to download/convert {url}: {e}")
        return None

from utils.title_filter import is_weather_related

def download_all_from_links_file(links_file: Path = Config.LINKS_FILE) -> list[Path]:
    if not links_file.exists():
        print(f"Links file not found: {links_file}")
        return []
    
    with open(links_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    downloaded = []
    print(f"Found {len(urls)} links. Starting filter & download process...")

    for url in urls:
        if not is_weather_related(url):
            print(f"Skipping non-weather video: {url}")
            continue

        video_path = download_video(url)
        if video_path:
            downloaded.append(video_path)
    
    return downloaded
