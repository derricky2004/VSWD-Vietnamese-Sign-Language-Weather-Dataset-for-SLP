import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def _stack_frame_arrays(values):
    frames = []
    for value in values:
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size % 3 == 0:
            width = 4 if arr.size % 4 == 0 else 3
            arr = arr.reshape(-1, width)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
            continue
        frames.append(arr.astype(np.float32))

    if not frames:
        raise ValueError('No valid frame keypoints found to stack')

    min_joints = min(frame.shape[0] for frame in frames)
    min_dims = min(frame.shape[1] for frame in frames)
    min_dims = max(3, min_dims)
    norm = [frame[:min_joints, :min_dims] for frame in frames]
    return np.stack(norm, axis=0)


def _load_array(path: Path):
    suffix = path.suffix.lower()
    if suffix == '.npy':
        arr = np.load(path, allow_pickle=True)
        return np.asarray(arr)
    if suffix == '.npz':
        data = np.load(path, allow_pickle=True)
        for key in ['pose', 'poses', 'keypoints', 'kpts', 'arr_0']:
            if key in data:
                return np.asarray(data[key])
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]])
        raise ValueError(f'Cannot infer pose array key from npz: {path}')
    if suffix == '.json':
        obj = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            for key in ['pose', 'poses', 'keypoints', 'kpts', 'frames']:
                if key in obj[0]:
                    return _stack_frame_arrays([frame.get(key, []) for frame in obj])
        if isinstance(obj, dict):
            for key in ['pose', 'poses', 'keypoints', 'kpts', 'frames']:
                if key in obj:
                    values = obj[key]
                    if isinstance(values, list) and values and isinstance(values[0], (list, dict)):
                        return _stack_frame_arrays(values)
                    return np.asarray(values)
        if isinstance(obj, list):
            return _stack_frame_arrays(obj)
        return np.asarray(obj)
    raise ValueError(f'Unsupported file type: {path}')


def _normalize_pose_shape(arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        dim = arr.shape[-1]
        if dim % 3 != 0:
            raise ValueError(f'2D array last dim must be multiple of 3, got {arr.shape}')
        arr = arr.reshape(arr.shape[0], dim // 3, 3)
    if arr.ndim != 3:
        raise ValueError(f'Expected pose shape [T, J, 3], got {arr.shape}')
    if arr.shape[-1] < 3:
        raise ValueError(f'Last dim must be >= 3, got {arr.shape}')
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    target_joints = 53
    joints = arr.shape[1]
    if joints < target_joints:
        pad = np.zeros((arr.shape[0], target_joints - joints, 3), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
    elif joints > target_joints:
        arr = arr[:, :target_joints, :]
    return arr.astype(np.float32)


def build(manifest_csv: Path, output_root: Path, split: str):
    rows = list(csv.DictReader(manifest_csv.open(encoding='utf-8')))
    if not rows:
        raise ValueError('Manifest is empty')

    required = {'sample_id', 'text', 'pose_path'}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f'Manifest must contain columns: {sorted(required)}; missing: {sorted(missing)}')

    samples_pose = []
    samples_label = []
    skipped = []

    for r in rows:
        pose_path = Path(r['pose_path'])
        if not pose_path.is_absolute():
            pose_path = (manifest_csv.parent / pose_path).resolve()
        try:
            arr = _normalize_pose_shape(_load_array(pose_path))
        except Exception as exc:
            skipped.append((r.get('sample_id', ''), str(pose_path), str(exc)))
            continue
        samples_pose.append(arr)
        samples_label.append(r['text'].strip())

    if not samples_pose:
        raise ValueError('No valid samples were converted from manifest')

    pkl_root = output_root / 'gloss_pkl_100' / split
    proj_root = output_root / 'gloss_projection'
    pkl_root.mkdir(parents=True, exist_ok=True)
    proj_root.mkdir(parents=True, exist_ok=True)

    with (pkl_root / 'samples_pose.pkl').open('wb') as f:
        pickle.dump(samples_pose, f)
    with (pkl_root / 'samples_label.pkl').open('wb') as f:
        pickle.dump(samples_label, f)

    unique_actions = sorted(set(samples_label))
    action2label = {a: i for i, a in enumerate(unique_actions)}
    label2action = {i: a for i, a in enumerate(unique_actions)}

    (proj_root / 'h4w_gloss_100.txt').write_text('\n'.join(unique_actions) + '\n', encoding='utf-8')
    (proj_root / 'train_action2label_h4w_100.txt').write_text(json.dumps(action2label, ensure_ascii=False), encoding='utf-8')
    (proj_root / 'train_label2action_h4w_100.txt').write_text(json.dumps(label2action, ensure_ascii=False), encoding='utf-8')

    print(f'Wrote: {pkl_root / "samples_pose.pkl"}')
    print(f'Wrote: {pkl_root / "samples_label.pkl"}')
    print(f'Wrote projection files in: {proj_root}')
    print(f'Samples: {len(samples_pose)}, unique text labels: {len(unique_actions)}')
    if skipped:
        print(f'Skipped invalid samples: {len(skipped)}')
        for sample_id, pose_path, err in skipped[:10]:
            print(f'  - {sample_id}: {pose_path} ({err})')


def main():
    parser = argparse.ArgumentParser(description='Build wSignGen-compatible PKL data from a manifest CSV')
    parser.add_argument('--manifest', required=True, help='CSV with columns: sample_id,text,pose_path')
    parser.add_argument('--output-root', required=True, help='Target ASLGloss100 root (contains gloss_pkl_100/gloss_projection)')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'val', 'validation'])
    args = parser.parse_args()

    split = 'validation' if args.split == 'val' else args.split
    build(Path(args.manifest).resolve(), Path(args.output_root).resolve(), split)


if __name__ == '__main__':
    main()
