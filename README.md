# VSWD: Vietnamese Sign Language Weather Dataset for Sign Language Processing

A dataset-centric repository for building, curating, and benchmarking Vietnamese Sign Language weather clips for SLP research.

Language: **English** | [Tiếng Việt](README.vi.md)

---

## 1) Overview

This repository provides an end-to-end workflow for:

- constructing a scene-level Vietnamese weather sign dataset from raw videos,
- extracting and organizing pose/keypoint data,
- producing train/val/test splits with reproducible metadata,
- generating statistical and quality reports,
- evaluating dataset utility for retrieval and production-style SLP benchmarks.

The codebase is designed for **research reproducibility** and **thesis/paper reporting**.

---

## 2) What This Repo Contains

### Dataset pipeline

- Scene segmentation and scene-level clip creation.
- Rule-based sign scene filtering.
- Metadata synchronization and refinement.
- Pose/keypoint preparation for downstream experiments.

### Analysis and reporting

- Dataset statistics and quality summaries.
- Train/val/test split generation with stratification.
- Plot scripts for final report figures.
- Final benchmark and experiment artifacts under `results/`.

### Research outputs

- Retrieval benchmark reports.
- Production benchmark reports and summary tables.
- Final integrated report for thesis/paper usage.

---

## 3) Dataset Access

The dataset files are provided via Google Drive:

- **VSWD dataset folder**: https://drive.google.com/drive/folders/1c45THLWH5vTPxlbrOdU4tlgACAi4rRFG?usp=sharing

After downloading, place the data into this repository structure:

```text
SignWeather/
└── data/
    ├── raw_videos/
    ├── scene_videos_orginal/
    ├── scene_videos_pose/
    ├── scene_keypoints/
    ├── metadata/
    └── lists/
```

Minimum metadata file expected by most scripts:

- `data/metadata/scene_metadata.csv`

Recommended key data files:

- `data/metadata/vswd_final_filtered.csv`
- `data/metadata/vswd_final_split.csv`
- `data/lists/train.csv`, `val.csv`, `test.csv`

---

## 4) Repository Structure

```text
SignWeather/
├── README.md
├── README.vi.md
├── requirements.txt
├── rebuild_and_add_pose.py
├── classifier_ends/
│   ├── run_full_pipeline.py
│   ├── refine_scenes.py
│   ├── crop_scale_scenes.py
│   ├── sort_metadata.py
│   ├── sync_mapping.py
│   └── ...
├── stats_and_eval/
│   └── stats/
│       ├── data_stats.py
│       ├── quality_metrics.py
│       ├── train_val_split.py
│       ├── prepare_eval.py
│       └── plot_data_distribution.py
├── data/
│   ├── raw_videos/
│   ├── scene_videos_orginal/
│   ├── scene_videos_pose/
│   ├── scene_keypoints/
│   ├── metadata/
│   └── lists/
├── docs/
└── results/
```

---

## 5) Environment Setup

## 5.1 Prerequisites

- Linux (recommended)
- Python 3.8+
- FFmpeg installed on system

Install FFmpeg (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## 5.2 Python setup

```bash
git clone https://github.com/derricky2004/VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP.git
cd VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP/SignWeather
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use Conda:

```bash
conda create -n vswd python=3.10 -y
conda activate vswd
pip install -r requirements.txt
```

---

## 6) Quick Start (End-to-End)

### Step 1 — Prepare data folders

```bash
mkdir -p data/raw_videos data/scene_videos_orginal data/scene_videos_pose data/scene_keypoints data/metadata data/lists
```

### Step 2 — Put dataset files from Drive into `data/`

Use the dataset folder link above and copy corresponding files into the expected directories.

### Step 3 — Run the main scene pipeline

```bash
python classifier_ends/run_full_pipeline.py
```

### Step 4 — Rebuild pose-rendered scene videos (if needed)

```bash
python rebuild_and_add_pose.py
```

### Step 5 — Generate train/val/test splits

```bash
python stats_and_eval/stats/train_val_split.py \
  --metadata scene_metadata.csv \
  --output vswd_final_split.csv \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --random-seed 42
```

### Step 6 — Prepare keypoints for evaluation

```bash
python stats_and_eval/stats/prepare_eval.py \
  --metadata data/metadata/scene_metadata.csv \
  --scene_keypoints data/scene_keypoints \
  --scene_videos data/scene_videos_pose \
  --output data/keypoints_processed
```

### Step 7 — Compute dataset statistics

```bash
python stats_and_eval/stats/data_stats.py
```

Output report:

- `docs/dataset_full_stats.md`

---

## 7) Detailed Pipeline Commands

### 7.1 Scene refinement utilities

```bash
python classifier_ends/refine_scenes.py
python classifier_ends/crop_scale_scenes.py
python classifier_ends/sort_metadata.py
python classifier_ends/sync_mapping.py
```

### 7.2 Visualization/check scripts

```bash
python classifier_ends/visualize_inference.py
python stats_and_eval/stats/plot_data_distribution.py
```

### 7.3 Quality scoring script

```bash
python stats_and_eval/stats/quality_metrics.py
```

Notes:

- This script may require optional packages/API setup depending on the scoring path you use.
- Validate configuration before running expensive quality scoring.

---

## 8) Dataset Usage in Your Own Project

Typical usage flow:

1. Use `data/metadata/vswd_final_split.csv` as source metadata.
2. Read split-specific files under `data/lists/` for data loaders.
3. Resolve `path` to corresponding scene video under `data/scene_videos_pose/` or `data/scene_videos_orginal/`.
4. Load keypoints from `data/keypoints_processed/<split>/` (or from original keypoint JSON source).

Recommended minimum columns in metadata:

- `path`
- `text`
- `quality_level`
- `content_label`
- `split` (for pre-split metadata)

---

## 9) Reproducibility Guidelines

To keep experiments reproducible:

- Fix random seed (`42` by default in split script).
- Keep train/val/test ratios unchanged for official benchmarks.
- Do not mix metadata versions in a single experiment.
- Store generated reports under timestamped files in `results/`.
- Record Python package versions with each benchmark run.

---

## 10) Results and Reports

Primary final outputs are stored in:

- `results/final_reports/`
- `results/retrieval_parallel_methods/`
- `results/production_parallel_methods/`

Important files (current repository snapshot):

- `results/final_reports/final_experiment_report_retrieval_production_2026-03-05.md`
- `results/final_reports/final_benchmark_table_2026-03-05.csv`
- `results/production_parallel_methods/cross_model_benchmark_clean_2026-03-05.csv`
- `results/production_parallel_methods/cross_model_benchmark_stats_2026-03-05.csv`

---

## 11) Citation

If you use this repository or dataset in academic work, please cite:

```bibtex
@misc{vswd2026,
  title={VSWD: Vietnamese Sign Language Weather Dataset for Sign Language Processing},
  author={Quoc Dat Do},
  year={2026},
  howpublished={\url{https://github.com/derricky2004/VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP}}
}
```

---

## 12) License and Responsible Use

- Use this repository for research and educational purposes.
- Respect privacy, broadcaster rights, and local legal constraints when redistributing media.
- For production deployment in high-stakes use cases, conduct independent validation and ethics review.

---

## 13) Contact

For collaboration, issue reports, or research discussion, open a GitHub issue in:

- https://github.com/derricky2004/VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP
