# VSWD: Vietnamese Sign Language Weather Dataset for Sign Language Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper-green.svg)](https://openai.com/research/whisper)

## Overview

The **VSWD (Vietnamese Sign Language Weather Dataset)** is a comprehensive dataset and processing pipeline designed for Sign Language Processing (SLP) research, specifically targeting Vietnamese weather-related sign language videos. This project provides automated tools to collect, process, and annotate raw weather broadcast videos into structured datasets suitable for training and evaluating sign language recognition and production models.

The dataset focuses on weather forecasting content from Vietnamese television (e.g., VTV), capturing sign language gestures for meteorological phenomena such as rain, storms, temperature, and natural disasters. The processing pipeline employs rule-based pose detection using MediaPipe to segment videos into meaningful scenes, extract keypoints, and generate multimodal annotations (video, audio transcripts, pose data).

**Key Contributions:**
- Automated pipeline for video segmentation and pose extraction.
- Multimodal dataset with video clips, pose keypoints, and ASR transcripts.
- Rule-based classifier for detecting sign language gestures (e.g., "clasped hands" as scene delimiters).
- Tools for data auditing, filtering, and quality assurance using GPT models.

## Dataset Description

The VSWD dataset consists of processed video clips from Vietnamese weather broadcasts, annotated with pose keypoints, audio transcripts, and metadata. It is designed to support tasks in sign language recognition, gesture analysis, and multimodal learning.

- **Total Videos:** 3,680 video clips
- **Duration:** Approximately 13.25 hours
- **Annotations:** Pose keypoints (MediaPipe Holistic), ASR transcripts (OpenAI Whisper), scene metadata.
- **Format:** MP4 videos, JSON keypoints, CSV metadata.
- **Language:** Vietnamese (sign language and spoken transcripts).
- **Source:** Public weather broadcast videos from VTV and similar channels.

**Download the Dataset:**
- [Link to dataset download - please insert here after uploading]

For more details on dataset statistics, annotation format, and usage, see the [Dataset Documentation](./docs/dataset.md).

## Features

- **Automated Video Processing Pipeline:** End-to-end processing from raw videos to annotated clips.
- **Pose-Based Segmentation:** Uses MediaPipe Holistic to detect key gestures for scene cutting.
- **Multimodal Annotations:** Combines video, pose, and audio data.
- **Quality Assurance:** GPT-powered auditing for content relevance and naturalness.
- **Scalable and Extensible:** Supports batch processing with multi-threading.
- **Open-Source Tools:** All code is released under MIT license for research use.

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for video processing)
- yt-dlp (for video downloading, optional)

### Step-by-Step Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/derricky2004/VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP.git
   cd VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install System Dependencies:**
   - **FFmpeg:** Required for video cropping, scaling, and audio extraction.
     - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
     - macOS: `brew install ffmpeg`
     - Windows: Download from [FFmpeg official site](https://ffmpeg.org/download.html) and add to PATH.
   - **yt-dlp:** For downloading videos from YouTube or similar platforms.
     - `pip install yt-dlp` or install system-wide.

4. **Set Up API Keys (Optional, for GPT/Whisper):**
   - Obtain an OpenAI API key from [OpenAI Platform](https://platform.openai.com/).
   - Create a `config.py` file in the root directory with:
     ```python
     class Config:
         OPENAI_API_KEY = "your-api-key-here"
         OPENAI_MODEL_MINI = "gpt-4o-mini"
         RAW_VIDEOS_DIR = "data/raw_videos"
         AUDIO_CHANNELS = 1
         AUDIO_SAMPLE_RATE = 16000
     ```
   - Note: GPT/Whisper features are optional; the core pipeline runs without them.

## Usage

### Quick Start

1. **Prepare Raw Videos:**
   Place raw weather videos in `data/raw_videos/`. Ensure metadata files (e.g., `clip_mapping_final.csv`, `vswd_final_filtered.csv`) are in `data/metadata/`.

2. **Run the Full Pipeline:**
   ```bash
   python classifier_ends/run_full_pipeline.py
   ```
   This processes videos, segments scenes, crops/scales, and extracts poses.

3. **Rebuild and Add Pose (if scenes need pose re-extraction):**
   ```bash
   python rebuild_and_add_pose.py
   ```
   This script reconstructs pose videos from original scenes by:
   - Reading backup metadata
   - Extracting clips from source videos
   - Applying MediaPipe Holistic to draw skeleton overlays
   - Muxing pose video with original audio
   - Outputs to `data/scene_videos_pose/`

4. **Optional Refinements:**
   - Refine scenes: `python classifier_ends/refine_scenes.py`
   - Crop and scale: `python classifier_ends/crop_scale_scenes.py`
   - Sort metadata: `python classifier_ends/sort_metadata.py`

5. **Visualize Results:**
   Use `classifier_ends/visualize_inference.py` to inspect processed videos and keypoints.

### Advanced Usage

- **Custom Configuration:** Modify thresholds in `run_full_pipeline.py` (e.g., `CONFIDENCE_THRESHOLD`, `MIN_EVENT_FRAMES`).
- **Batch Processing:** Adjust `MAX_WORKERS` for parallel processing.
- **ASR and Auditing:** Run `utils/whisper_utils.py` for transcripts and `utils/audit.py` for quality checks (requires OpenAI API).

### Directory Structure

```
VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP/
├── .gitignore                      # Quy tắc ignore file (bỏ qua data lớn)
├── README.md                       # Tài liệu này
├── requirements.txt                # Danh sách thư viện Python
├── rebuild_and_add_pose.py         # Script rebuild pose videos từ scenes
├── classifier_ends/                # Scripts chính cho pipeline
│   ├── run_full_pipeline.py        # Pipeline chính: cắt scene từ video thô
│   ├── rule_based_classifier.py    # Phân loại pose dựa trên luật
│   ├── add_pose_to_scenes.py       # Trích xuất pose và vẽ skeleton
│   ├── crop_scale_scenes.py        # Crop và scale video scene
│   ├── refine_scenes.py            # Tinh chỉnh scene (tách clip rời rạc)
│   ├── sort_metadata.py            # Sắp xếp metadata CSV
│   ├── sync_mapping.py             # Đồng bộ mapping ID video
│   ├── process_video_scenes.py     # Xử lý logic cắt ghép
│   ├── match_scenes.py             # Module hỗ trợ matching
│   └── visualize_inference.py      # Trực quan hóa kết quả
├── utils/                          # Thư viện hàm hỗ trợ
│   ├── __init__.py
│   ├── audit.py                    # Kiểm tra tính toàn vẹn dữ liệu
│   ├── classification.py           # Hàm phân loại chung
│   ├── common.py                   # Hàm tiện ích chung
│   ├── download_raw_thumbnails.py  # Tải thumbnail
│   ├── ffmpeg_utils.py             # Hàm xử lý FFmpeg
│   ├── filter_urls.py              # Lọc URL
│   ├── gpt_utils.py                # Tích hợp GPT
│   ├── pose_detection.py           # Trích xuất pose với MediaPipe
│   ├── title_filter.py             # Lọc tiêu đề
│   ├── video_crop.py               # Hàm crop video
│   ├── video_download.py           # Tải video
│   ├── video_scale.py              # Hàm scale video
│   └── whisper_utils.py            # Tích hợp Whisper (ASR)
├── docs/                           # Tài liệu
│   └── dataset.md                  # Tài liệu chi tiết về dataset
├── data/                           # Dữ liệu (bị ignore trong Git, trừ CSV)
│   ├── raw_videos/                 # Video gốc tải về
│   ├── scene_videos_orginal/       # Video scenes được cắt (nguyên gốc, không pose)
│   ├── scene_videos_pose/          # Video scene với skeleton pose (đầu ra cuối)
│   ├── scene_keypoints/            # File JSON keypoints pose
│   ├── labeled_videos/             # JSON kết quả phân đoạn
│   ├── metadata/                   # File CSV metadata
│   │   ├── scene_metadata.csv      # Metadata scene chính
│   │   ├── clip_mapping_final.csv  # Mapping thời gian clip
│   │   ├── vswd_final_filtered.csv # Danh sách clip lọc (3,680)
│   │   └── mapping/                # Folder mapping ID video
│   ├── asr/                        # Dữ liệu ASR (audio transcripts)
│   ├── audio/                      # File audio trích xuất
│   ├── lists/                      # Danh sách URL/video
│   ├── train_ends/                 # Dữ liệu cho training
│   └── train_v2/                   # Dữ liệu training v2
└── legacy/                         # Mã nguồn cũ (không dùng)
    ├── classifier_ends/            # Scripts cũ
    ├── classifier_thumbnail/       # Phân loại thumbnail cũ
    └── data_processing/            # Xử lý dữ liệu cũ
```

## Methodology

### Video Segmentation
- **Rule-Based Classifier:** Detects "clasped hands" gesture using wrist positions and distances from MediaPipe landmarks.
- **Scene Cutting:** Splits videos at gesture detections, ensuring coherent sign language segments.

### Pose Extraction
- Utilizes MediaPipe Holistic for 33 pose keypoints, 21 per hand, and face landmarks.
- Outputs JSON files with frame-wise keypoints for model training.

### Audio Processing
- Extracts audio using FFmpeg.
- Transcribes using OpenAI Whisper for spoken content alignment.

### Quality Control
- GPT models audit transcripts for weather relevance, naturalness, and redundancy.

## Evaluation

To evaluate models on VSWD:
- Use pose keypoints for gesture recognition.
- Align with ASR transcripts for multimodal tasks.
- Metrics: Accuracy, BLEU for transcripts, pose estimation errors.

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@misc{vswd2026,
  title={VSWD: Vietnamese Sign Language Weather Dataset for Sign Language Processing},
  author={Quoc Dat Do},
  year={2026},
  publisher={GitHub},
  url={https://github.com/derricky2004/VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open issues or pull requests for bugs, features, or improvements.

## Acknowledgments

- MediaPipe for pose detection.
- OpenAI for Whisper and GPT models.
- VTV for public weather broadcasts.

## Contact

For questions or collaborations: [Your Email or GitHub Issues]

---

*This dataset and pipeline are intended for research purposes in sign language processing and should be used ethically.*
