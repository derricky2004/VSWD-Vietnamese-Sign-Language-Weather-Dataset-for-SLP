# VSWD Dataset Documentation

## Overview

The VSWD (Vietnamese Sign Language Weather Dataset) is a multimodal dataset for Sign Language Processing (SLP), focusing on Vietnamese weather-related sign language videos. It includes video clips, pose keypoints, audio transcripts, and metadata for training and evaluating sign language recognition models.

## Dataset Statistics

- **Number of Clips:** 3,680 video clips
- **Total Duration:** Approximately 13.25 hours
- **Number of Source Videos:** 273 original videos (mapped to v000-v272)
- **Annotations:**
  - Pose keypoints: 3,680 JSON files (MediaPipe Holistic)
  - Audio transcripts: Available in `data/asr/` (OpenAI Whisper)
  - Metadata: CSV files with quality labels, content categories, and thesis scores

## Data Structure

The dataset is organized as follows:

```
data/
├── scene_videos_pose/        # Final video clips with pose overlays (MP4)
│   ├── v000/                 # Clips from video v000
│   ├── v001/
│   └── ...
├── scene_keypoints/          # Pose keypoints (JSON)
│   ├── v000/
│   └── ...
├── asr/                      # Audio transcripts (JSON/verbose JSON)
├── audio/                    # Extracted audio (WAV)
├── metadata/
│   ├── scene_metadata.csv    # Clip metadata (text, quality, labels)
│   ├── clip_mapping_final.csv # Detailed clip mappings with duration
│   └── vswd_final_filtered.csv # Filtered dataset list
└── lists/                    # Video lists for splits
```

## Annotation Details

### Video Clips
- **Format:** MP4, H.264 encoded
- **Resolution:** Cropped and scaled to focus on signer
- **Content:** Weather-related sign language segments from VTV broadcasts

### Pose Keypoints
- **Format:** JSON per clip
- **Landmarks:** MediaPipe Holistic (pose, face, hands)
- **Structure:**
  ```json
  {
    "frame_0": {
      "pose": [[x, y, z, visibility], ...],
      "face": [...],
      "left_hand": [...],
      "right_hand": [...]
    },
    ...
  }
  ```

### Audio Transcripts
- **Tool:** OpenAI Whisper (verbose_json)
- **Content:** Spoken Vietnamese transcripts aligned with video
- **Format:** JSON with timestamps, text, and confidence

### Metadata
- **scene_metadata.csv:** Columns include path, text, quality_level, content_label, thesis_score
- **clip_mapping_final.csv:** Detailed mappings with start/end times, duration, refined text

## Splits

- **Train/Test Splits:** Available in `data/train_ends/` and `data/train_v2/` (custom splits based on quality and content)
- **Recommended Split:** Use 80/20 train/test based on video IDs

## Usage Examples

### Loading Pose Data
```python
import json
with open('data/scene_keypoints/v000/scene_001.json', 'r') as f:
    keypoints = json.load(f)
# Access pose for frame 0
pose = keypoints['frame_0']['pose']
```

### Loading Metadata
```python
import pandas as pd
metadata = pd.read_csv('data/metadata/scene_metadata.csv')
print(metadata.head())
```

### Training Example (Pseudocode)
```python
# Load clips and keypoints
clips = load_videos('data/scene_videos_pose/')
keypoints = load_keypoints('data/scene_keypoints/')

# Train SLP model
model = SignLanguageModel()
model.train(clips, keypoints)
```

## Quality Assurance

- **Quality Levels:** HIGH, MEDIUM, LOW based on GPT auditing
- **Content Labels:** WEATHER_CORE, WEATHER_SUPPORT, NON_WEATHER
- **Thesis Scores:** 0-100 indicating information content

## Ethical Considerations

- Dataset sourced from public broadcasts.
- Ensure compliance with copyright and privacy laws.
- Use for research purposes only.

## Citation

If using this dataset, cite:

```bibtex
@dataset{vswd2024,
  title={VSWD: Vietnamese Sign Language Weather Dataset for Sign Language Processing},
  author={Derrick Nguyen},
  year={2024},
  url={https://github.com/derricky2004/VSWD-Vietnamese-Sign-Language-Weather-Dataset-for-SLP}
}
```

## Contact

For issues or questions: Open GitHub issues or contact the maintainer.