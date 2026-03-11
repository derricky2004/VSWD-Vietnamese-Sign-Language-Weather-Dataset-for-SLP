#!/usr/bin/env python3
"""
Prepare keypoints & mapping for evaluation.

- Reads scene metadata CSV (default: data/metadata/scene_metadata.csv)
- Looks for keypoint JSONs under data/scene_keypoints/ (flexible matching)
- Copies found keypoint JSONs into output_dir/<split>/ and writes mapping CSVs per split
- Marks missing keypoints as invalid in the mapping CSV
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration defaults ---
DEFAULT_METADATA = "data/metadata/scene_metadata.csv"
DEFAULT_SCENE_KEYPOINTS = "data/scene_keypoints"
DEFAULT_SCENE_VIDEOS = "data/scene_videos_pose"
DEFAULT_OUTPUT = "data/keypoints_processed"


def find_keypoint_file(scene_basename: str, keypoints_root: Path):
    """
    Try to locate a keypoint JSON file for a given scene basename.
    Matching strategy (in order):
      1) keypoints_root / <scene_basename>.json
      2) extract scene number and try scene_XXX.json pattern
      3) recursive glob for "**/*<scene_basename>*.json"
    Returns Path or None.
    """
    cand = keypoints_root / f"{scene_basename}.json"
    if cand.exists():
        return cand

    # Try extracting scene number (e.g., scene_001_v008_c009 -> scene_001)
    import re
    match = re.match(r"(scene_\d+)", scene_basename)
    if match:
        scene_num_basename = match.group(1)
        cand2 = keypoints_root / f"{scene_num_basename}.json"
        if cand2.exists():
            return cand2
        # Try recursive with scene number
        pattern = f"**/{scene_num_basename}.json"
        matches = list(keypoints_root.glob(pattern))
        if matches:
            return matches[0]

    matches = list(keypoints_root.glob(f"**/*{scene_basename}*.json"))
    if matches:
        for m in matches:
            if m.stem == scene_basename:
                return m
        return matches[0]
    return None


def infer_scene_basename_from_row(row: pd.Series):
    """
    Infer a scene basename from common metadata columns.
    Tries: 'scene', 'scene_id', 'scene_filename', 'scene_video',
           'path', 'relative_path', 'video', 'scene_path', 'scene_file', 'file'
    Returns basename (no extension) or None.
    """
    possible_cols = [
        "scene", "scene_id", "scene_filename", "scene_video",
        "path", "relative_path", "video", "scene_path", "scene_file", "file"
    ]
    for c in possible_cols:
        if c in row.index and pd.notna(row[c]):
            val = str(row[c]).strip()
            base = os.path.splitext(os.path.basename(val))[0]
            if base:
                return base
    if isinstance(row.name, str):
        return os.path.splitext(os.path.basename(row.name))[0]
    return None


def load_keypoints_array(json_path: Path):
    """Load keypoints JSON and convert to numpy array (object-safe)."""
    if not json_path.exists():
        return None, 0

    try:
        with open(json_path, "r", encoding="utf8") as f:
            j = json.load(f)
    except Exception:
        return None, 0

    num_frames = 0
    if isinstance(j, dict):
        if "frames" in j and isinstance(j["frames"], list):
            frames = j["frames"]
            num_frames = len(frames)
            return np.array(frames, dtype=object), num_frames
        if "keypoints" in j and isinstance(j["keypoints"], list):
            frames = j["keypoints"]
            num_frames = len(frames)
            return np.array(frames, dtype=object), num_frames
        # Fallback: try to find the longest list value
        list_values = [v for v in j.values() if isinstance(v, list)]
        if list_values:
            frames = max(list_values, key=len)
            num_frames = len(frames)
            return np.array(frames, dtype=object), num_frames
        return None, 0
    if isinstance(j, list):
        num_frames = len(j)
        return np.array(j, dtype=object), num_frames

    return None, 0


def prepare(args):
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: metadata CSV not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    scene_kp_root = Path(args.scene_keypoints)
    scene_videos_root = Path(args.scene_videos)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    total = len(df)
    print(f"📂 Output folder: {out_root.resolve()}")
    print(f"Tổng scenes trong metadata: {total}")

    split_col = args.split_col
    if split_col not in df.columns:
        print(f"⚠️ Split column '{split_col}' not found in metadata -> assigning all to 'all'")
        df["split"] = "all"
        split_col = "split"

    splits = df[split_col].fillna("all").unique().tolist()
    print(f"Các split có trong metadata: {splits}\n")

    summary = {}

    for split in splits:
        df_split = df[df[split_col] == split].reset_index(drop=True)
        n = len(df_split)
        print(f"Xử lý split '{split}': {n} scenes")
        split_out_dir = out_root / str(split)
        split_out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for idx, row in tqdm(df_split.iterrows(), total=n, desc=str(split), unit="scene"):
            scene_basename = infer_scene_basename_from_row(row)
            if "scene_id" in row.index and pd.notna(row["scene_id"]):
                scene_id = str(row["scene_id"])
            else:
                scene_id = scene_basename or f"scene_{idx}"

            kp_path = None
            valid = False
            num_frames = 0

            found_kp = find_keypoint_file(scene_basename, scene_kp_root) if scene_basename else None
            if found_kp:
                dst_name = f"{scene_id}_keypoints.npy"
                dst = split_out_dir / dst_name
                try:
                    kp_array, num_frames = load_keypoints_array(found_kp)
                    if kp_array is None:
                        raise ValueError("Invalid keypoints JSON")
                    np.save(dst, kp_array)
                    kp_path = str(dst.resolve())
                    valid = True
                except Exception:
                    valid = False
                    kp_path = None

            scene_video = ""
            for c in ("scene_video", "scene_filename", "path", "relative_path", "video"):
                if c in row.index and pd.notna(row[c]):
                    scene_video = str(row[c])
                    break
            if not scene_video and scene_basename:
                candidates = list(scene_videos_root.glob(f"**/*{scene_basename}*"))
                scene_video = str(candidates[0]) if candidates else ""

            rows.append({
                "scene_id": scene_id,
                "scene_basename": scene_basename or "",
                "scene_video": scene_video or "",
                "kp_file": kp_path or "",
                "valid": bool(valid),
                "num_frames": int(num_frames),
            })

        df_map = pd.DataFrame(rows)
        map_csv = out_root / f"{split}_mapping.csv"
        df_map.to_csv(map_csv, index=False)
        valid_count = int(df_map["valid"].sum())
        missing = n - valid_count
        print(f"→ {split}: {valid_count} clips hợp lệ, {missing} clips thiếu keypoints")
        print(f"Đã lưu: {map_csv.name} với {len(df_map)} scenes\n")
        summary[split] = {"total": n, "valid": valid_count, "missing": int(missing)}

    print("\n============================================================")
    print("TỔNG HỢP THỐNG KÊ:")
    print("============================================================")
    for s, info in summary.items():
        total = info["total"]
        ratio = (info["valid"] / total * 100) if total else 0
        print(f"{s:<10} → {info['valid']:4d} / {total:4d} scenes hợp lệ ({ratio:.1f}%)")
    print("\n============================================================")
    print("✅ Hoàn tất chuẩn bị keypoints cho tất cả splits!")
    print("============================================================\n")
    print("Cấu trúc output:")
    print(f"{out_root.resolve()}/")
    for s in splits:
        print(f"├── {s}/")
        print("│   ├── <scene_id>_keypoints.json")
    print("├── <split>_mapping.csv")
    print("\nFiles mapping chứa: scene_id, scene_basename, scene_video, kp_file, valid, num_frames")


def main():
    p = argparse.ArgumentParser(description="Prepare keypoints for eval (scene-based).")
    p.add_argument("--metadata", type=str, default=DEFAULT_METADATA, help="Path to scene metadata CSV")
    p.add_argument("--scene_keypoints", type=str, default=DEFAULT_SCENE_KEYPOINTS, help="Root folder of scene keypoint JSONs")
    p.add_argument("--scene_videos", type=str, default=DEFAULT_SCENE_VIDEOS, help="Root folder of scene videos (optional)")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output folder for processed keypoints and mapping CSVs")
    p.add_argument("--split-col", dest="split_col", type=str, default="split", help="Column name defining split (train/val/test). If missing, all -> 'all'")
    args = p.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()