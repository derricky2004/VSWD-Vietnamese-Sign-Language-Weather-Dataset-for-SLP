import json
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from config import Config
from utils.common import ensure_dir_exists

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def run_whisper_verbose(input_wav: Path, out_json: Path) -> dict:
    ensure_dir_exists(out_json.parent)
    print("Run Whisper verbose_json...")
    
    with open(input_wav, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )
    
    try:
        data = resp.model_dump()
    except Exception:
        data = resp
        
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return data

def extract_segments(whisper_json: dict) -> List[dict]:
    segs = whisper_json.get("segments", []) or []
    segs = sorted(segs, key=lambda s: s["start"])

    results = []
    prev_end = 0.0

    for idx, s in enumerate(segs):
        start = float(s["start"])
        end = float(s["end"])
        text = (s.get("text") or "").strip()

        if start < prev_end:
            start = prev_end

        results.append({
            "id": idx,
            "start": start,
            "end": end,
            "duration": end - start,
            "text_len": len(text),
            "text": text,
        })
        prev_end = end

    return results
