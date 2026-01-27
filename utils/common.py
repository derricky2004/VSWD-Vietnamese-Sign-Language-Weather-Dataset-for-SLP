import subprocess
from pathlib import Path
from typing import Optional

def run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result

def ensure_file_exists(file_path: Path) -> bool:
    return file_path.exists() and file_path.is_file()

def ensure_dir_exists(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)

def get_file_size(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    return file_path.stat().st_size

def normalize_text(text: str) -> str:
    text = text.strip()
    text = ' '.join(text.split())
    return text
