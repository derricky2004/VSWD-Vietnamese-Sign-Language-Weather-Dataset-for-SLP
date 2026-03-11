import pandas as pd
import numpy as np
from collections import Counter
import subprocess
try:
    import cv2
except Exception:
    cv2 = None
import os
from tqdm import tqdm

# === HARD-CODE ĐƯỜNG DẪN ===
# Use final scene metadata as source
CSV_PATH = "/workspace/datdq/SignWeather/data/metadata/scene_metadata.csv"
# Try scene_videos_pose first, fallback to scene_videos_orginal if not found
VIDEO_ROOT_DIRS = [
    "/workspace/datdq/SignWeather/data/scene_videos_pose",
    "/workspace/datdq/SignWeather/data/scene_videos_orginal",
    "/workspace/datdq/SignWeather/data/raw_videos",
]
OUTPUT_MD = "/workspace/datdq/SignWeather/docs/dataset_full_stats.md"
# =================================

def get_video_duration(video_path):
    """Trả về thời lượng video (giây) bằng OpenCV. Thử nhiều thư mục gốc."""
    for root in VIDEO_ROOT_DIRS:
        full_path = os.path.join(root, video_path)
        if not os.path.exists(full_path):
            continue

        # If OpenCV available, use it (fast). Otherwise, try ffprobe as fallback.
        if cv2 is not None:
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                cap.release()
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps <= 0 or frame_count <= 0:
                cap.release()
                continue

            duration = frame_count / fps
            cap.release()
            return duration
        else:
            # Use ffprobe to get duration
            try:
                cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', full_path
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out = result.stdout.strip()
                if out:
                    return float(out)
            except Exception:
                pass

    return np.nan


def calculate_full_stats():
    print("Đang đọc metadata...")
    if not os.path.exists(CSV_PATH):
        print(f"Không tìm thấy file CSV: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Tổng clips ban đầu: {len(df)}")

    # Debug: In ra 5 path đầu tiên từ CSV để kiểm tra
    print("\nDebug: 5 path đầu tiên trong CSV:")
    print(df['path'].head(5))

    # Thêm cột duration nếu chưa có
    if 'duration' not in df.columns:
        # Nếu có clip mapping, lấy duration từ đó (nhiều file đã có thông tin)
        mapping_path = os.path.join(os.path.dirname(CSV_PATH), 'clip_mapping_final.csv')
        clip_dur_map = {}
        if os.path.exists(mapping_path):
            try:
                cm = pd.read_csv(mapping_path)
                if 'clip_id' in cm.columns and 'duration' in cm.columns:
                    clip_dur_map = dict(zip(cm['clip_id'].astype(str), cm['duration'].astype(float)))
            except Exception:
                clip_dur_map = {}

        print("\nĐang gán duration từ clip_mapping_final (nếu có), hoặc thử đọc file video...")
        durations = []
        missing_videos = 0

        for path in tqdm(df['path'], desc="Computing durations"):
            # derive clip_id from basename without extension
            clip_id = os.path.splitext(os.path.basename(str(path)))[0]
            dur = clip_dur_map.get(clip_id, np.nan)
            if np.isnan(dur):
                # fallback: try reading the video file
                dur = get_video_duration(path)
            if np.isnan(dur):
                missing_videos += 1
            durations.append(dur)

        df['duration'] = durations

        output_csv = CSV_PATH.replace('.csv', '_with_duration.csv')
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Đã lưu metadata có cột duration vào: {output_csv}")
        if missing_videos > 0:
            print(f"Cảnh báo: Có {missing_videos} video không đọc được (duration = NaN)")
    else:
        print("Đã có cột duration sẵn, bỏ qua tính lại")

    # Loại bỏ clip không có duration
    df_valid = df.dropna(subset=['duration'])
    print(f"Số clip hợp lệ sau khi đọc video: {len(df_valid)}")

    if len(df_valid) == 0:
        print("Không có clip nào hợp lệ → debug thêm:")
        print(f"  VIDEO_ROOT_DIRS = {VIDEO_ROOT_DIRS}")
        print(f"  Ví dụ path trong CSV: {df['path'].iloc[0]}")
        print(f"  Đường dẫn đầy đủ code đang thử: {os.path.join(VIDEO_ROOT_DIRS[0], df['path'].iloc[0])}")
        print("\nChạy lệnh này trong terminal để kiểm tra file tồn tại:")
        print(f"ls {os.path.join(VIDEO_ROOT_DIR, df['path'].iloc[0])}")
        print("\nNếu file tên 'scene_001.mp4' thay vì 'v000_c000.mp4', thì CSV có tên file sai. Hãy kiểm tra và báo mình tên file thật trong folder v000/.")
        return

    # Thống kê chính
    total_clips = len(df_valid)
    total_duration_sec = df_valid['duration'].sum()
    total_duration_hours = total_duration_sec / 3600
    avg_duration = df_valid['duration'].mean()

    unique_sentences = df_valid['text'].nunique()
    all_words = ' '.join(df_valid['text'].astype(str)).split()
    vocab_size = len(set(all_words))
    top_words = Counter(all_words).most_common(20)

    quality_counts = df_valid['quality_level'].value_counts().to_dict()
    label_counts = df_valid['content_label'].value_counts().to_dict()
    avg_score = df_valid['thesis_score'].mean()
    high_quality_ratio = (df_valid['quality_level'] == 'HIGH').mean() * 100

    # Tạo bảng Markdown
    stats_md = f"""
# VSWD Dataset Statistics (Final - {pd.Timestamp.now().strftime('%Y-%m-%d')})

| Chỉ số                              | Giá trị                              |
|-------------------------------------|--------------------------------------|
| Tổng số clips hợp lệ                | {total_clips:,}                      |
| Tổng thời lượng                     | {total_duration_hours:.2f} giờ ≈ {total_duration_sec/60:.1f} phút |
| Thời lượng trung bình mỗi clip      | {avg_duration:.2f} giây              |
| Số câu unique                       | {unique_sentences:,}                 |
| Vocabulary size                     | {vocab_size:,}                       |
| Điểm thesis_score trung bình        | {avg_score:.2f} / 100                |
| Tỷ lệ HIGH quality                  | {high_quality_ratio:.2f}%            |
| Phân bố quality_level               | {quality_counts}                     |
| Phân bố content_label               | {label_counts}                       |

**Top 20 từ phổ biến nhất**:
{', '.join([word for word, cnt in top_words])}

**Ghi chú**:
- Thời lượng được tính chính xác từ file video bằng OpenCV.
- Các clip không đọc được video đã bị loại khỏi thống kê.
"""

    print("\n" + "="*70)
    print(stats_md)
    print("="*70)

    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(stats_md)

    print(f"\nKết quả đã được lưu vào file: {OUTPUT_MD}")


if __name__ == "__main__":
    calculate_full_stats()