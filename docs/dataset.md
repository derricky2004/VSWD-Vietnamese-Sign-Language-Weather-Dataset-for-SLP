# VSWD Dataset Specification

> Technical specification for the Vietnamese Sign Language Weather Dataset (VSWD), including schema, splits, quality protocol, and benchmark-facing interfaces.

---

## 1. Scope

VSWD is a scene-level multimodal dataset designed for Sign Language Processing (SLP) in Vietnamese weather domain content. The dataset is intended to support:

1. **Discriminative tasks** (retrieval and semantic matching).
2. **Generative tasks** (text-to-pose production).

The dataset is released together with a reproducible construction pipeline and benchmark artifacts.

---

## 2. Final Dataset Snapshot

Source of record: `docs/dataset_full_stats.md`.

- Valid clips: **3,139**
- Total duration: **4.42 hours**
- Mean clip length: **5.07 seconds**
- Unique sentences: **3,116**
- Vocabulary size: **2,908**
- Mean thesis quality score: **77.99 / 100**
- HIGH quality ratio: **71.46%**

These statistics reflect the finalized corpus used for reporting in the final experiment package.

---

## 3. Data Modalities

Each scene sample is represented by the following modalities:

1. **Video clip** (scene-level segment)
2. **Pose/keypoint sequence** (frame-level landmarks)
3. **Text annotation** (Vietnamese sentence/transcript)
4. **Metadata fields** (quality labels, content labels, IDs, optional timing fields)

---

## 4. Physical Layout

```text
SignWeather/data/
├── raw_videos/                  # source broadcast videos (local only)
├── scene_videos_orginal/        # segmented raw scene clips
├── scene_videos_pose/           # scene clips with pose overlay
├── scene_keypoints/             # frame-level keypoints (JSON)
├── keypoints_processed/         # optional processed keypoint format
├── asr/                         # transcript artifacts (if enabled)
├── audio/                       # extracted audio
├── metadata/
│   ├── scene_metadata.csv
│   ├── vswd_final_filtered.csv
│   ├── vswd_final_split.csv
│   ├── scene_metadata_with_duration.csv
│   └── scene_metadata_with_gloss.csv
└── lists/
    ├── train.csv / val.csv / test.csv
    ├── train_paths.txt / val_paths.txt / test_paths.txt
    └── (optional URL/ID list files)
```

Notes:
- Large binary media are local artifacts and should not be versioned.
- Metadata/list files are the canonical lightweight exchange layer.

---

## 5. Canonical Metadata Schema

Typical columns used across final metadata files:

- `path`: relative path to scene clip.
- `text`: sentence-level Vietnamese annotation.
- `quality_level`: categorical quality tag (e.g., HIGH, MEDIUM).
- `content_label`: semantic category (e.g., WEATHER_CORE, WEATHER_SUPPORT).
- `thesis_score`: numeric quality score.
- `split`: split assignment (`train`, `val`, `test`) when applicable.

Optional columns may include clip identifiers and timing boundaries.

---

## 6. Split Protocol

VSWD is used with train/validation/test partitioning stored in:

- `data/metadata/vswd_final_split.csv`
- `data/lists/train.csv`, `data/lists/val.csv`, `data/lists/test.csv`
- `data/lists/*_paths.txt`

The split artifacts are intended to be deterministic for reproducibility of benchmark outputs.

---

## 7. Quality Control Protocol

### 7.1 Quality dimensions

VSWD quality filtering and reporting are centered around:

1. Semantic relevance to weather-domain signing.
2. Temporal coherence at scene level.
3. Pose extraction stability.
4. Annotation consistency.

### 7.2 Quality labels and scores

- `quality_level`: coarse categorical control.
- `thesis_score`: continuous score used for ranking/filtering and audit summaries.

### 7.3 Supporting documents

- `docs/quality_report.md`
- `docs/quality_report_pseudo.csv`

---

## 8. Benchmark Interfaces

VSWD supports two benchmark families.

### 8.1 Retrieval interface

- Input: text query
- Candidate pool: indexed train corpus
- Output: ranked scene clips + pose comparison outputs
- Key metrics: Recall@k, mAP@10, BLEU, WER, DTW, Approx FID

Benchmark artifacts:
- `results/retrieval_parallel_methods/*/retrieval_benchmark_report.md`

### 8.2 Production interface

- Input: text prompt
- Output: generated pose sequence
- Key metrics: Accuracy, FID, DTW norm, Diversity

Benchmark artifacts:
- `results/production_parallel_methods/cross_model_benchmark_clean_2026-03-05.csv`
- `results/production_parallel_methods/cross_model_benchmark_stats_2026-03-05.csv`
- `results/production_parallel_methods/wsigngen_seed_stats_2026-03-05.csv`

Integrated final report:
- `results/final_reports/final_experiment_report_retrieval_production_2026-03-05.md`

---

## 9. Data Conversion Utility for Production

For model-specific format adaptation, use:

- `results/production_parallel_methods/build_wsigngen_pkl_from_manifest.py`

Expected input manifest columns:
- `sample_id`
- `text`
- `pose_path`

Expected outputs (wSignGen-compatible):
- `gloss_pkl_100/<split>/samples_pose.pkl`
- `gloss_pkl_100/<split>/samples_label.pkl`
- `gloss_projection/*.txt`

---

## 10. Reproducibility Requirements

To reproduce final benchmark-facing artifacts:

1. Preserve split files in `data/lists` and `data/metadata`.
2. Use deterministic preprocessing scripts in `classifier_ends/`.
3. Report results with statistical aggregation (`mean`, `variance`, `std`) where applicable.
4. Keep final tables in `results/final_reports/` and `results/production_parallel_methods/`.

---

## 11. Limitations

- Domain coverage is weather-focused; cross-domain generalization is not guaranteed.
- Scene segmentation quality is tied to source video conditions and pose visibility.
- Some benchmark rows are comparative references and should be updated with runtime logs if re-executed.

---

## 12. Ethical and Legal Notes

- Use only under applicable broadcast/data-use constraints.
- Avoid releasing personally sensitive content.
- Intended for research and educational usage.

---

## 13. Citation

```bibtex
@misc{vswd2026,
  title={VSWD: Vietnamese Sign Language Weather Dataset for Sign Language Processing},
  author={Quoc Dat Do},
  year={2026},
  publisher={GitHub}
}
```
