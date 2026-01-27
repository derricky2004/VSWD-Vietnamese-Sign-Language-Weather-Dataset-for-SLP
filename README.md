# SignWeather Project

Dự án xử lý và phân tích dữ liệu video thời tiết, tập trung vào việc cắt tách cảnh (scene), cắt (crop), thay đổi kích thước (scale) và trích xuất tư thế (pose).

## Cấu trúc thư mục

- **classifier_ends/**: Chứa các script chính để chạy quy trình xử lý dữ liệu (pipeline).
  - `run_full_pipeline.py`: Pipeline chính, thực hiện cắt scene ban đầu dựa trên bộ phân loại dựa trên luật (rule-based classifier - phân loại kiểu tay).
  - `rule_based_classifier.py`: Định nghĩa lớp classifier dựa trên luật (vị trí cổ tay, khoảng cách) sử dụng MediaPipe.
  - `refine_scenes.py`: Script hậu kỳ, dùng để tách các scene bị gộp (nhiều clip rời rạc) thành các scene nhỏ hơn liên tiếp.
  - `crop_scale_scenes.py`: Cắt (crop) và phóng to (scale) video scene theo tham số định sẵn, sử dụng FFmpeg để đảm bảo chuẩn nén H.264.
  - `add_pose_to_scenes.py`: Trích xuất điểm mốc cơ thể (keypoints pose) từ video và vẽ khung xương lên video, xuất ra file JSON và video mới.
  - `sort_metadata.py`: Sắp xếp file metadata CSV theo thứ tự video ID và scene ID.
  - `sync_mapping.py`: Đồng bộ và cập nhật file ánh xạ (mapping) ID video gốc sang ID mới (vXXX).
  - `process_video_scenes.py`, `match_scenes.py`: Các module hỗ trợ xử lý logic cắt và ghép clip.

- **utils/**: Thư viện các hàm hỗ trợ chung.
  - `pose_detection.py`: Các hàm liên quan đến MediaPipe Holistic để trích xuất pose.
  - `video_scale.py`: Các hàm xử lý cắt và scale video (OpenCV).
  - Các file khác như `video_download.py`, `ffmpeg_utils.py`,...

- **data/**: Nơi lưu trữ toàn bộ dữ liệu.
  - **raw_videos/**: Video gốc tải về từ nguồn.
  - **scene_videos/**: Video scene được cắt ra từ pipeline chính (chưa crop).
  - **scene_videos_cropped/**: Video scene đã được crop và scale.
  - **scene_videos_pose/**: Video scene đã được vẽ khung xương pose.
  - **scene_keypoints/**: File JSON chứa tọa độ keypoints của từng scene.
  - **labeled_videos/**: File JSON kết quả phân đoạn từ classifier.
  - **metadata/**: Chứa các file CSV thông tin.
    - `scene_metadata_realtime.csv`: Metadata chính của các scene (đường dẫn, văn bản, điểm số, clips...).
    - `clip_mapping_final.csv`: Mapping chi tiết thời gian bắt đầu/kết thúc của từng clip nhỏ trong video gốc.
    - `vswd_final_filtered.csv`: Tập dữ liệu lọc ban đầu.
    - `mapping/video_id_mapping.csv`: Bảng ánh xạ ID video giống Youtube sang ID nội bộ (v001, v002...).

- **legacy/**: Chứa mã nguồn và dữ liệu cũ, không còn được sử dụng trong luồng xử lý hiện tại (bao gồm model train cũ, các script xử lý dữ liệu cũ).

## Luồng xử lý dữ liệu (Pipeline)

1. **Chạy Pipeline Chính**: Sử dụng `run_full_pipeline.py` để quét video thô (raw), phân loại frame (tay nắm vào nhau - clasped hands), và cắt thành các video scene cơ bản.
2. **Tinh Chỉnh Scene**: (Tùy chọn) Dùng `refine_scenes.py` để tách các scene chứa clip không liên tiếp.
3. **Crop & Scale**: Dùng `crop_scale_scenes.py` để cắt lấy phần người dẫn chương trình và scale lên, chuyển định dạng về H.264.
4. **Trích Xuất Pose**: Dùng `add_pose_to_scenes.py` để tạo video pose và file JSON keypoints.
5. **Quản Lý Metadata**: Dùng `sort_metadata.py` và `sync_mapping.py` để đảm bảo tính nhất quản của dữ liệu metadata.
