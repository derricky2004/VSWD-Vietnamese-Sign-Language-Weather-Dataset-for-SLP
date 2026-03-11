# BÁO CÁO KHOÁ LUẬN TỐT NGHIỆP (DRAFT)
## Đề tài: Xây dựng bộ dữ liệu đa phương thức cho nhận dạng ngôn ngữ ký hiệu tiếng Việt miền thời tiết từ video phát sóng

---

## Thông tin sử dụng trong bản này

- Tổng số mẫu: **3,680**
- Số clips: **3,680**
- Số câu unique: **3,680**
- Vocabulary size: **3,128**
- Tổng thời lượng: **~6 giờ** (xấp xỉ 21,600 giây)

Các thông số chia tách trong báo cáo được tính theo tỷ lệ chuẩn **80/10/10** (train/val/test).

---

## TÓM TẮT

Khoá luận tập trung xây dựng một pipeline dữ liệu có khả năng tái lập cho bài toán xử lý ngôn ngữ ký hiệu tiếng Việt trong ngữ cảnh bản tin thời tiết. Quy trình bao gồm thu thập video phát sóng, phân đoạn cảnh bằng tín hiệu hình thể, trích xuất keypoints toàn thân/tay/mặt, tạo transcript theo mốc thời gian và hợp nhất thành bộ dữ liệu đa phương thức ở mức scene-level. Bộ dữ liệu cuối cùng gồm 3,680 clips, tương ứng 3,680 câu unique, vocabulary 3,128 và tổng thời lượng khoảng 6 giờ. Kết quả của khoá luận không chỉ dừng ở sản phẩm dữ liệu mà còn đưa ra một quy trình kỹ thuật có kiểm soát chất lượng, có thể dùng làm nền tảng cho các tác vụ SLP (Sign Language Processing), retrieval và học đa phương thức trong tiếng Việt.

---

## CHƯƠNG 1. MỞ ĐẦU

## 1.1 Bối cảnh

Bài toán ngôn ngữ ký hiệu tiếng Việt hiện vẫn gặp nút thắt lớn ở tầng dữ liệu: thiếu dữ liệu có cấu trúc, thiếu nhãn đồng bộ đa phương thức và thiếu quy trình chuẩn hoá để tái lập trong nghiên cứu. Trong khi đó, dữ liệu phát sóng bản tin thời tiết có đặc điểm rất phù hợp cho xây dựng dataset: ngữ cảnh rõ ràng, tín hiệu ký hiệu lặp theo chủ đề chuyên biệt, và có đồng thời kênh hình + kênh tiếng nói.

## 1.2 Động lực và lý do chọn đề tài (có căn cứ)

### (A) Khoảng trống dữ liệu tiếng Việt cho SLP ở miền chuyên ngành

- Nút thắt chính trong SLP tiếng Việt không chỉ là mô hình, mà là **dữ liệu huấn luyện có chất lượng và tái lập được**.
- Miền thời tiết có lợi thế về tính nhất quán ngữ cảnh (nhiệt độ, mưa, bão, vùng địa lý, cảnh báo), giúp giảm nhiễu ngữ nghĩa khi xây dựng dữ liệu ban đầu.
- Căn cứ kỹ thuật: cấu trúc dự án cho thấy đã phải thiết kế pipeline chuyên dụng để đi từ raw video đến scene-level metadata, thay vì sử dụng trực tiếp video thô.

### (B) Nhu cầu khoa học: từ “data collection” sang “data engineering có kiểm chứng”

- Nhiều bộ dữ liệu thất bại ở khả năng tái lập do thiếu mô tả pipeline, thiếu tham số cắt/lọc, thiếu log chất lượng.
- Đề tài chọn hướng đóng góp vào **phương pháp xây dựng dataset**: có thuật toán segmentation, có chuẩn hóa transcript, có phân tầng chất lượng/nội dung, và có split rõ ràng cho đánh giá.
- Căn cứ hiện hữu trong mã nguồn:
  - Pipeline phân đoạn và cắt cảnh: [SignWeather/classifier_ends/run_full_pipeline.py](SignWeather/classifier_ends/run_full_pipeline.py)
  - Luật hình học phát hiện boundary: [SignWeather/classifier_ends/rule_based_classifier.py](SignWeather/classifier_ends/rule_based_classifier.py)
  - Trích xuất keypoints Holistic: [SignWeather/utils/pose_detection.py](SignWeather/utils/pose_detection.py)
  - ASR và xử lý văn bản: [SignWeather/utils/whisper_utils.py](SignWeather/utils/whisper_utils.py)

### (C) Tính đóng góp học thuật và khả năng công bố

- Đề tài tạo ra cả **artifact dữ liệu** (dataset) và **artifact phương pháp** (pipeline tái lập).
- Có thể xây dựng câu chuyện paper rõ ràng: bài toán thiếu dữ liệu → pipeline đa bước có kiểm soát chất lượng → benchmark baseline và ablation.
- Đây là hướng đóng góp phù hợp với các hội nghị AI/CV chú trọng reproducibility và dataset-centric research.

### (D) Ý nghĩa ứng dụng xã hội

- Dữ liệu ngôn ngữ ký hiệu tiếng Việt chất lượng cao là nền tảng cho các hệ thống hỗ trợ tiếp cận thông tin, đặc biệt trong ngữ cảnh cảnh báo thời tiết/thiên tai.
- Đề tài mang giá trị kép: đóng góp học thuật + tiềm năng ứng dụng thực tế.

## 1.3 Phát biểu bài toán

Xây dựng một pipeline tự động, có thể tái lập và kiểm chứng được, nhằm chuyển đổi video bản tin thời tiết tiếng Việt thành bộ dữ liệu đa phương thức scene-level phục vụ nghiên cứu SLP, bảo đảm tính nhất quán giữa tín hiệu hình ảnh, keypoints và transcript.

## 1.4 Mục tiêu nghiên cứu

- Thiết kế pipeline xử lý dữ liệu từ raw video đến dataset cuối.
- Chuẩn hoá annotation đa phương thức (video, keypoints, text, metadata).
- Xây dựng bộ dữ liệu 3,680 mẫu với thống kê rõ ràng, sẵn sàng train/val/test.
- Đề xuất bộ chỉ số và quy trình đánh giá phục vụ nghiên cứu tiếp theo.

## 1.5 Câu hỏi nghiên cứu

- RQ1: Có thể tự động phân đoạn scene từ video phát sóng bằng luật hình học landmark với độ ổn định đủ dùng cho dataset construction không?
- RQ2: Chuỗi ASR + refinement + quality filtering có làm tăng tính nhất quán annotation văn bản không?
- RQ3: Keypoints Holistic theo scene-level có đủ làm đầu vào cho các baseline retrieval/recognition không?

## 1.6 Phạm vi

- Miền dữ liệu: bản tin thời tiết tiếng Việt.
- Đơn vị mẫu: scene clip.
- Thành phần annotation: transcript, quality/content labels, keypoints frame-level.
- Không bao gồm: huấn luyện mô hình sinh video ký hiệu hoàn chỉnh ở quy mô production.

---

## CHƯƠNG 2. CƠ SỞ KỸ THUẬT VÀ PHƯƠNG PHÁP

## 2.1 Tổng quan pipeline

1. Thu thập video thô.
2. Trích audio và chạy ASR.
3. Chuẩn hóa/refine transcript.
4. Phát hiện boundary dựa trên landmark pose.
5. Cắt scene và match scene với clip metadata theo overlap thời gian.
6. Trích keypoints Holistic theo frame.
7. Hợp nhất metadata, lọc chất lượng, tạo split train/val/test.

## 2.2 Segmentation (trọng tâm)

- Dựa trên luật hình học wrist–hip để phát hiện frame boundary.
- Gom frame boundary thành event theo ràng buộc thời gian.
- Scene được lấy là đoạn giữa các event và lọc đoạn quá ngắn.
- Ưu điểm: diễn giải được, rẻ tính toán, dễ tái lập.
- Hạn chế: phụ thuộc chất lượng landmark và domain gesture.

## 2.3 ASR & Text refinement

- ASR tạo transcript có timestamp.
- LLM refinement và quality/content labeling để tăng độ sạch annotation.
- Kết quả text cuối cùng được gắn vào scene metadata.

## 2.4 Keypoint extraction

- Dùng MediaPipe Holistic cho pose/face/hands.
- Lưu JSON theo frame, đồng bộ với clip scene-level.

---

## CHƯƠNG 3. BỘ DỮ LIỆU VÀ THỐNG KÊ

## 3.1 Thống kê chính (đã chốt)

| Chỉ số | Giá trị |
|---|---:|
| Tổng số mẫu | 3,680 |
| Tổng số clips | 3,680 |
| Unique sentences | 3,680 |
| Vocabulary size | 3,128 |
| Tổng thời lượng | ~6 giờ |
| Tổng thời lượng (giây) | ~21,600 |
| Thời lượng trung bình mỗi clip | ~5.87 giây |

Công thức:
- Thời lượng trung bình/clip = 21,600 / 3,680 ≈ 5.87 giây.

## 3.2 Chia tập theo tỷ lệ 80/10/10

| Tập | Tỷ lệ | Số clips | Số câu unique | Thời lượng xấp xỉ |
|---|---:|---:|---:|---:|
| Train | 80% | 2,944 | 2,944 | ~4.8 giờ (~17,280 giây) |
| Validation | 10% | 368 | 368 | ~0.6 giờ (~2,160 giây) |
| Test | 10% | 368 | 368 | ~0.6 giờ (~2,160 giây) |
| Tổng | 100% | 3,680 | 3,680 | ~6.0 giờ |

## 3.3 Thông số dẫn xuất theo tỷ lệ

- Mật độ unique sentence = 3,680 / 3,680 = **1 câu/clip**.
- Tỷ lệ vocab trên tổng câu = 3,128 / 3,680 ≈ **0.85**.
- Tỷ lệ vocab trên tổng clip = 3,128 / 3,680 ≈ **0.85**.

Ghi chú học thuật:
- Vocabulary theo từng split không nhất thiết chia tuyến tính do giao nhau từ vựng; nếu cần báo cáo chính xác, phải tính trực tiếp trên file split thực tế.

---

## CHƯƠNG 4. THIẾT KẾ THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1 Baseline đề xuất

- Text retrieval baseline (bi-encoder tiếng Việt).
- Pose alignment baseline dùng DTW.
- (Tuỳ chọn) fusion baseline text+pose.

## 4.2 Chỉ số

- Text: BLEU, WER.
- Pose: DTW normalized.
- Distribution: FID-approx trên feature keypoints.
- Data quality: tỷ lệ mẫu hợp lệ, missing keypoints, duplicate ratio.

## 4.3 Ablation

- Bật/tắt refinement text.
- So sánh các ngưỡng segmentation.
- Bật/tắt quality/content filtering.

---

## CHƯƠNG 5. ĐÓNG GÓP, HẠN CHẾ, VÀ HƯỚNG PHÁT TRIỂN

## 5.1 Đóng góp

- Hoàn thiện pipeline xây dựng dataset đa phương thức từ video phát sóng.
- Cung cấp bộ dữ liệu 3,680 mẫu với cấu trúc rõ ràng và thông số đủ dùng cho nghiên cứu.
- Đưa ra khung đánh giá và thực nghiệm để phục vụ nghiên cứu nối tiếp.

## 5.2 Hạn chế

- Segmentation hiện tại dựa rule nên độ tổng quát có thể giảm trên domain ngoài thời tiết.
- Chất lượng transcript phụ thuộc ASR/LLM và điều kiện audio.
- Một số chỉ số semantic cần đánh giá bổ sung bằng human audit.

## 5.3 Hướng phát triển

- Chuyển sang learned boundary detector.
- Mở rộng sang thêm domain ký hiệu khác.
- Chuẩn hóa benchmark đa tác vụ (recognition, retrieval, generation).

---

## KẾT LUẬN

Khoá luận giải quyết bài toán thiếu dữ liệu ngôn ngữ ký hiệu tiếng Việt theo hướng dataset-centric có tính hệ thống. Thay vì chỉ thu thập dữ liệu thô, đề tài xây dựng một quy trình kỹ thuật có khả năng tái lập, có kiểm soát chất lượng và có thể tích hợp trực tiếp vào chuỗi nghiên cứu học thuật. Với bộ dữ liệu 3,680 clips, 3,680 câu unique, vocab 3,128 và tổng thời lượng khoảng 6 giờ, kết quả đạt được tạo nền tảng thực nghiệm vững chắc cho các nghiên cứu SLP tiếng Việt trong giai đoạn tiếp theo.

---

## PHỤ LỤC A. ĐOẠN NÊU ĐỘNG LỰC (DÙNG TRỰC TIẾP TRONG THUYẾT TRÌNH)

Động lực của đề tài xuất phát từ một thực tế khoa học: trong xử lý ngôn ngữ ký hiệu tiếng Việt, nút thắt lớn nhất không nằm ở việc thiếu mô hình, mà nằm ở việc thiếu dữ liệu có cấu trúc, có độ tin cậy và có khả năng tái lập. Dữ liệu phát sóng bản tin thời tiết là nguồn dữ liệu giàu ngữ nghĩa nhưng có nhiễu cao, đòi hỏi một pipeline xử lý đa bước thay vì gắn nhãn thủ công đơn lẻ. Vì vậy, đề tài tập trung xây dựng một quy trình đầy đủ từ phân đoạn cảnh, đồng bộ transcript theo thời gian, trích xuất keypoints đến hợp nhất metadata, với mục tiêu tạo ra bộ dữ liệu chuẩn cho nghiên cứu. Đóng góp của đề tài là cung cấp cả một tài sản dữ liệu cụ thể (3,680 mẫu) và một phương pháp xây dựng dữ liệu có kiểm chứng, từ đó tăng tính tái lập và tính thuyết phục của các nghiên cứu tiếp theo trong lĩnh vực SLP tiếng Việt.
