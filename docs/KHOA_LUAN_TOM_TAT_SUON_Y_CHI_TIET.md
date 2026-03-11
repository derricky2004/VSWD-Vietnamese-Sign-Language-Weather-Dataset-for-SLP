# TÓM TẮT VÀ SƯỜN Ý CHI TIẾT KHOÁ LUẬN TỐT NGHIỆP
## Đề tài: Xây dựng pipeline và bộ dữ liệu đa phương thức cho ngôn ngữ ký hiệu tiếng Việt miền thời tiết

---

## 1) BẢN TÓM TẮT MẪU (DÙNG TRỰC TIẾP)

Khoá luận tập trung giải quyết bài toán thiếu dữ liệu có cấu trúc cho nghiên cứu ngôn ngữ ký hiệu tiếng Việt trong miền thời tiết. Thay vì chỉ thu thập video thô, nghiên cứu xây dựng một pipeline dữ liệu có khả năng tái lập, bao gồm: phân đoạn cảnh dựa trên landmark hình thể, trích xuất keypoints theo frame, tạo transcript theo mốc thời gian, chuẩn hoá văn bản và hợp nhất metadata đa nguồn. Kết quả thu được bộ dữ liệu scene-level gồm 3,680 clips tương ứng 3,680 câu unique, vocabulary size 3,128 và tổng thời lượng xấp xỉ 6 giờ. Bộ dữ liệu được tổ chức theo chuẩn train/validation/test với tỷ lệ 80/10/10, phục vụ các tác vụ nhận dạng, truy hồi và học đa phương thức. Đóng góp chính của khoá luận nằm ở hai mặt: (i) tạo ra tài nguyên dữ liệu có giá trị thực tiễn cho tiếng Việt, và (ii) đề xuất quy trình kỹ thuật có kiểm soát chất lượng, giúp tăng tính tái lập và tính thuyết phục học thuật cho các nghiên cứu SLP tiếp theo.

---

## 2) PHÁT BIỂU VẤN ĐỀ (PROBLEM STATEMENT)

- Bài toán cốt lõi: thiếu bộ dữ liệu ngôn ngữ ký hiệu tiếng Việt có chú thích đa phương thức, đủ sạch và đủ chuẩn để phục vụ nghiên cứu nghiêm túc.
- Khó khăn chính:
  - Video phát sóng thực tế có nhiễu cao, không chia sẵn theo câu/cảnh.
  - Chú thích văn bản và tín hiệu hình ảnh dễ lệch thời gian nếu xử lý thủ công.
  - Thiếu quy trình chuẩn hoá khiến dữ liệu khó tái lập giữa các nhóm nghiên cứu.
- Nhu cầu khoa học: cần một pipeline có mô tả rõ thuật toán, tham số, tiêu chuẩn lọc và cấu trúc đầu ra.

---

## 3) MỤC TIÊU, CÂU HỎI, GIẢ THUYẾT

## 3.1 Mục tiêu tổng quát

Xây dựng pipeline dữ liệu end-to-end để chuyển đổi video bản tin thời tiết thành bộ dữ liệu scene-level đa phương thức, có thể dùng trực tiếp cho thực nghiệm SLP.

## 3.2 Mục tiêu cụ thể

- Thiết kế và triển khai pipeline phân đoạn cảnh tự động từ video thô.
- Tạo annotation văn bản có timestamp và kiểm soát chất lượng.
- Trích xuất keypoints Holistic theo frame cho từng clip.
- Chuẩn hoá metadata và chia train/val/test theo tỷ lệ 80/10/10.
- Thiết lập baseline đánh giá phục vụ nghiên cứu nối tiếp.

## 3.3 Câu hỏi nghiên cứu

- RQ1: Pipeline phân đoạn dựa trên landmark có ổn định để tạo scene-level dataset không?
- RQ2: Chuỗi ASR + refinement + quality filtering có cải thiện tính nhất quán annotation không?
- RQ3: Dữ liệu keypoints trích xuất có đủ chất lượng cho baseline retrieval/recognition không?

## 3.4 Giả thuyết

- H1: Segmentation có kiểm soát ngưỡng làm giảm nhiễu biên cảnh so với cắt thủ công không có tiêu chí.
- H2: Hậu xử lý văn bản giúp tăng chất lượng metadata cho downstream tasks.
- H3: Biểu diễn keypoints scene-level tạo nền tảng khả thi cho benchmark ban đầu.

---

## 4) ĐÓNG GÓP DỰ KIẾN (ĐỂ GHI RÕ TRONG CHƯƠNG 1)

- Đóng góp 1: Một pipeline xây dựng dữ liệu đa phương thức có khả năng tái lập từ raw video broadcast.
- Đóng góp 2: Một bộ dữ liệu scene-level tiếng Việt miền thời tiết với 3,680 mẫu, 3,680 câu unique, vocab 3,128, tổng thời lượng ~6 giờ.
- Đóng góp 3: Khung đánh giá ban đầu (text + pose) và kế hoạch ablation phục vụ phát triển nghiên cứu tiếp theo.

---

## 5) THỐNG KÊ DATASET CẦN TRÌNH BÀY

## 5.1 Chỉ số chính

- Tổng clips: 3,680
- Unique sentences: 3,680
- Vocabulary size: 3,128
- Tổng thời lượng: ~6 giờ (~21,600 giây)
- Thời lượng trung bình mỗi clip: ~5.87 giây

## 5.2 Chia tập theo tỷ lệ 80/10/10

- Train: 2,944 clips (~4.8 giờ)
- Validation: 368 clips (~0.6 giờ)
- Test: 368 clips (~0.6 giờ)

## 5.3 Bảng mô tả gợi ý đưa vào khoá luận

| Chỉ số | Giá trị |
|---|---:|
| Tổng số mẫu | 3,680 |
| Train/Val/Test | 2,944 / 368 / 368 |
| Unique sentence | 3,680 |
| Vocabulary size | 3,128 |
| Tổng thời lượng | ~6 giờ |
| Thời lượng trung bình | ~5.87 giây/clip |

---

## 6) SƯỜN Ý CHI TIẾT THEO CHƯƠNG

## Chương 1. Giới thiệu

### 1.1 Bối cảnh
- Vai trò của dữ liệu trong SLP.
- Thực trạng thiếu dữ liệu tiếng Việt.

### 1.2 Động lực chọn đề tài
- Khoảng trống nghiên cứu.
- Lý do chọn miền thời tiết.
- Ý nghĩa khoa học và thực tiễn.

### 1.3 Mục tiêu, phạm vi, phương pháp tiếp cận
- Mục tiêu tổng quát và cụ thể.
- Phạm vi dữ liệu và giới hạn đề tài.

### 1.4 Đóng góp
- Liệt kê 2–4 đóng góp ngắn, đo được.

### 1.5 Cấu trúc khoá luận
- Tóm tắt nội dung từng chương.

## Chương 2. Cơ sở lý thuyết và nghiên cứu liên quan

### 2.1 Cơ sở lý thuyết
- Sign Language Processing.
- Pose representation, temporal segmentation.
- ASR và hậu xử lý văn bản.

### 2.2 Công trình liên quan
- Nhóm công trình về dataset.
- Nhóm công trình về segmentation/pose.
- Nhóm công trình về multimodal benchmark.

### 2.3 Khoảng trống nghiên cứu
- Chỉ ra khoảng trống rõ ràng mà khoá luận giải quyết.

## Chương 3. Phương pháp và kiến trúc hệ thống

### 3.1 Tổng quan hệ thống
- Data flow end-to-end.
- Sơ đồ pipeline.

### 3.2 Phân đoạn cảnh
- Mô tả thuật toán phát hiện boundary.
- Tham số và ràng buộc thời gian.
- Ưu/nhược điểm.

### 3.3 ASR và text pipeline
- Trích âm thanh, ASR timestamp.
- Chuẩn hoá/refinement/filtering.

### 3.4 Trích xuất keypoints
- Thành phần pose/face/hands.
- Dạng dữ liệu đầu ra và cấu trúc lưu trữ.

### 3.5 Hợp nhất metadata và tạo split
- Logic map scene-clip.
- Tạo train/val/test.

## Chương 4. Xây dựng bộ dữ liệu và phân tích thống kê

### 4.1 Quy trình xây dựng dữ liệu
- Nguồn dữ liệu đầu vào.
- Các bước xử lý và kiểm tra chất lượng.

### 4.2 Thống kê dataset
- Bảng số liệu chính.
- Phân tích độ dài câu, độ dài clip, phân bố từ vựng.

### 4.3 Chất lượng dữ liệu
- Tiêu chí quality.
- Các lỗi phổ biến và cách xử lý.

## Chương 5. Thực nghiệm và đánh giá

### 5.1 Thiết lập thực nghiệm
- Môi trường chạy.
- Baselines.
- Dữ liệu và split.

### 5.2 Metrics
- BLEU, WER, DTW, FID-approx (nếu dùng).

### 5.3 Kết quả
- Bảng chính.
- So sánh baseline.

### 5.4 Ablation
- Ảnh hưởng của segmentation thresholds.
- Ảnh hưởng của refinement/filtering.

### 5.5 Thảo luận
- Phân tích điểm mạnh/yếu.
- Tác động của chất lượng dữ liệu đến kết quả.

## Chương 6. Kết luận và hướng phát triển

### 6.1 Kết luận
- Tóm tắt đóng góp.
- Trả lời lại các RQ.

### 6.2 Hạn chế
- Hạn chế thuật toán và dữ liệu.

### 6.3 Hướng mở rộng
- Learned segmentation.
- Mở rộng domain.
- Chuẩn benchmark mới.

---

## 7) KHUNG NỘI DUNG MỖI CHƯƠNG (MẸO VIẾT NHANH)

Mỗi chương nên có đúng 4 phần để dễ viết và dễ đọc:
1. Mục tiêu chương.
2. Nội dung chính.
3. Kết quả/rút ra của chương.
4. Tiểu kết chương (3–5 dòng).

Công thức viết đoạn chuẩn học thuật:
- Câu 1: Nêu vấn đề.
- Câu 2: Nêu cách giải quyết.
- Câu 3: Nêu kết quả hoặc ý nghĩa.

---

## 8) CHECKLIST NỘP BẢN NHÁP KHOÁ LUẬN

## 8.1 Nội dung
- Có Problem Statement rõ ràng.
- Có ít nhất 3 đóng góp cụ thể.
- Có sơ đồ pipeline tổng thể.
- Có bảng thống kê dataset chuẩn.
- Có phần thảo luận hạn chế.

## 8.2 Kỹ thuật
- Mô tả được quy trình tái lập dữ liệu.
- Nêu được các tham số chính của segmentation.
- Có split train/val/test nhất quán với thống kê.

## 8.3 Trình bày
- Thuật ngữ nhất quán: clip/scene/segment.
- Bảng và hình có caption rõ.
- Không dùng văn phong cảm tính, không khẳng định quá mức.

---

## 9) LỘ TRÌNH VIẾT KHOÁ LUẬN 4 TUẦN (THỰC CHIẾN)

## Tuần 1
- Hoàn thành Chương 1 + Chương 2.
- Chốt sơ đồ pipeline và bảng thuật ngữ.

## Tuần 2
- Viết Chương 3 (phương pháp) hoàn chỉnh.
- Hoàn thiện bảng tham số và mô tả thuật toán.

## Tuần 3
- Viết Chương 4 + Chương 5.
- Chèn bảng thống kê và kết quả thực nghiệm.

## Tuần 4
- Viết Chương 6.
- Rà soát toàn văn, chuẩn hoá tài liệu tham khảo, sửa lỗi trình bày.

---

## 10) ĐOẠN MỞ ĐẦU ẤN TƯỢNG (DÙNG CHO PHẦN GIỚI THIỆU)

Trong nghiên cứu ngôn ngữ ký hiệu tiếng Việt, thách thức lớn nhất hiện nay không chỉ đến từ mô hình mà đến từ dữ liệu: dữ liệu thiếu chuẩn hoá, thiếu đồng bộ đa phương thức và thiếu khả năng tái lập. Khoá luận này tiếp cận bài toán theo hướng dataset-centric, xây dựng một pipeline kỹ thuật đầy đủ từ video phát sóng thô đến bộ dữ liệu scene-level có thể dùng trực tiếp cho thực nghiệm SLP. Trên cơ sở đó, đề tài cung cấp một bộ dữ liệu 3,680 mẫu cùng khung xử lý có kiểm soát chất lượng, góp phần tạo nền tảng thực chứng cho các nghiên cứu tiếp theo trong cộng đồng AI tiếng Việt.

---

## 11) GỢI Ý CÁCH DÙNG FILE NÀY

- Dùng Mục 1 làm tóm tắt nộp đề cương.
- Dùng Mục 6 làm mục lục chi tiết chính thức.
- Dùng Mục 8 làm checklist trước mỗi lần nộp cho giảng viên.
- Dùng Mục 9 để tự quản tiến độ theo tuần.

Hoàn thành file này đồng nghĩa bạn đã có xương sống nội dung để viết khoá luận đầy đủ mà không bị lạc hướng.
