# SignWeather Master Guide
## Định hướng khoá luận tốt nghiệp và paper khoa học

---

## 0) Mục đích tài liệu

Tài liệu này đóng vai trò **bản thiết kế tổng thể** cho dự án SignWeather, giúp:
- Mô tả đầy đủ dự án theo chuẩn học thuật.
- Định hình câu chuyện nghiên cứu (research story) từ bài toán đến đóng góp.
- Chuyển trực tiếp thành dàn ý khoá luận và paper conference (CV/AI).
- Chuẩn hoá quy trình thực nghiệm, đánh giá, và báo cáo kết quả.

Tài liệu được viết để dùng theo 3 chế độ:
1. **Proposal mode**: dùng cho đề cương, seminar giữa kỳ.
2. **Thesis mode**: dùng cho khoá luận đầy đủ.
3. **Paper mode**: dùng để nén thành manuscript 6–8 trang.

---

## 1) Bức tranh lớn của dự án

## 1.1 Bài toán

SignWeather xây dựng một pipeline dữ liệu cho **Vietnamese Sign Language Weather** từ video bản tin thời tiết, với mục tiêu tạo dataset đa phương thức gồm:
- Video scene-level.
- Keypoints (pose/face/hands) theo frame.
- Transcript tiếng Việt đã tinh lọc.
- Metadata chất lượng và nhãn nội dung.

## 1.2 Động lực nghiên cứu

- Thiếu dữ liệu ngôn ngữ ký hiệu tiếng Việt chất lượng cao, đặc biệt cho miền thời tiết.
- Dữ liệu broadcast thực tế giàu thông tin nhưng nhiễu cao và khó segment tự động.
- Cần pipeline tái lập được để biến dữ liệu thô thành benchmark dùng cho SLP/Multimodal AI.

## 1.3 Đóng góp cốt lõi có thể trình bày trong khoá luận/paper

- Một pipeline end-to-end chuyển từ raw broadcast thành dataset scene-level có kiểm soát chất lượng.
- Cơ chế segmentation rule-based dựa trên hình học landmark để tách scene tự động.
- Quy trình hậu xử lý transcript bằng ASR + LLM + quality/content filtering.
- Bộ dữ liệu đa phương thức cho các tác vụ nhận dạng và truy hồi sign language.

---

## 2) Kiến trúc hệ thống (System Architecture)

## 2.1 Data flow ở mức cao

1. Raw video ingestion.
2. Audio extraction và ASR transcript có timestamp.
3. Text refinement và quality/content scoring.
4. Pose-based boundary detection trên video.
5. Scene cutting và matching scene ↔ clip metadata bằng overlap thời gian.
6. Post-processing scene, crop/scale, pose visualization và keypoint export.
7. Metadata consolidation + split train/val/test.
8. Evaluation scripts (retrieval benchmark, stats, quality reports).

## 2.2 Thành phần chính theo thư mục

- `classifier_ends/`: segmentation, scene matching/cutting, refine pipeline.
- `utils/`: ASR, GPT utils, pose extraction, ffmpeg tools.
- `docs/`: dataset docs, stats reports.
- `stats_and_eval/`: split scripts, benchmark retrieval, quality metrics.
- `data/metadata`: các bảng metadata trung tâm để tái lập pipeline.

---

## 3) Mô tả kỹ thuật pipeline tạo dataset

## 3.1 Segmentation logic (điểm nhấn thuật toán)

Segmentation dựa trên phát hiện “end signal” từ landmark hình học:
- Dùng wrist/hip landmarks từ MediaPipe Holistic.
- Điều kiện dương tính frame gồm:
  - Hai cổ tay đủ visibility.
  - Khoảng cách hai cổ tay nhỏ hơn ngưỡng.
  - Cả hai cổ tay nằm gần đường hông trung bình theo trục y.
- Các frame dương tính được gom thành event theo ngưỡng thời gian.
- Scene là các đoạn giữa event, lọc đoạn quá ngắn.

Lưu ý quan trọng cho viết học thuật:
- Đây là **rule-based geometric segmentation**, không phải motion velocity segmentation.
- Nêu rõ ưu điểm: đơn giản, diễn giải được, dễ tái lập.
- Nêu rõ hạn chế: nhạy với occlusion/pose miss và phụ thuộc domain gesture.

## 3.2 ASR và text pipeline

- ASR bằng Whisper API tạo segment transcript theo thời gian.
- Text refinement bằng LLM để làm sạch chính tả/dấu câu, chuẩn hoá diễn đạt.
- Quality/content scoring để lọc non-weather và mẫu chất lượng thấp.
- Tạo metadata clip trước khi map sang scene-level.

## 3.3 Keypoint extraction

- Dùng MediaPipe Holistic trên từng scene clip.
- Xuất landmark frame-level cho pose/face/hands (tọa độ normalized).
- Lưu JSON theo clip để phục vụ training/evaluation.

## 3.4 Scene matching và metadata hợp nhất

- Match clip vào scene theo overlap thời gian (best overlap + ngưỡng).
- Gộp text và trường metadata để tạo nhãn scene-level cuối cùng.
- Tạo các bản CSV cho huấn luyện, thống kê và đánh giá.

---

## 4) Dataset card rút gọn (dùng cho thesis/paper)

## 4.1 Snapshot hiện tại (theo metadata trong workspace)

- Scene samples: 3,139
- Unique sentences: 3,116
- Vocabulary size: 2,908
- Quality: HIGH 2,243; MEDIUM 896
- Content: WEATHER_CORE 2,871; WEATHER_SUPPORT 268
- Mean thesis score: 77.99
- Tổng thời lượng (report): ~4.42 giờ
- Độ dài trung bình: ~5.07 giây/clip

## 4.2 Splits hiện tại

- Train: 2,509
- Val: 312
- Test: 318
- Tổng: 3,139

## 4.3 Cách viết trong paper

Dùng template ngắn:
- “The final scene-level dataset contains N clips with average duration D seconds, covering C content categories and Q quality levels. We provide synchronized multimodal annotations including RGB clips, frame-level holistic keypoints, and refined Vietnamese transcripts.”

---

## 5) Hướng nghiên cứu và câu hỏi khoa học (Research Questions)

## 5.1 RQ gợi ý cho khoá luận/paper

- RQ1: Rule-based geometric boundaries có đủ ổn định để tự động segment sign weather broadcasts không?
- RQ2: LLM-based transcript refinement cải thiện chất lượng annotation ở mức nào?
- RQ3: Keypoint-only representations đạt hiệu quả ra sao cho retrieval/recognition so với text-only hoặc multimodal fusion?
- RQ4: Chất lượng segmentation ảnh hưởng thế nào đến downstream performance?

## 5.2 Giả thuyết (Hypotheses)

- H1: Boundary quality tốt hơn làm giảm nhiễu ngữ nghĩa trong scene metadata.
- H2: Transcript refinement + filtering cải thiện độ nhất quán nhãn và retrieval BLEU/WER.
- H3: Holistic keypoints cung cấp đủ tín hiệu để làm baseline mạnh cho pose alignment.

---

## 6) Thiết kế thực nghiệm chuẩn học thuật

## 6.1 Baseline đề xuất

- Text retrieval baseline (bi-encoder Vietnamese).
- Pose retrieval baseline (DTW alignment).
- Optional multimodal late fusion baseline.

## 6.2 Ablation quan trọng (nên có trong khoá luận)

- Ablation A: bỏ refinement text.
- Ablation B: thay đổi ngưỡng segmentation (dist/y/merge/min frames).
- Ablation C: chỉ pose vs pose+text metadata.
- Ablation D: có/không lọc quality/content.

## 6.3 Metrics

- Text: BLEU, WER.
- Pose alignment: DTW (normalized).
- Distribution-level: FID-approx trên pose feature.
- Dataset quality: tỷ lệ mẫu hợp lệ, duplicate ratio, missing keypoint ratio.

## 6.4 Threats to validity

- Domain bias (weather broadcast style).
- Annotation noise từ ASR và LLM post-edit.
- Rule-based segmentation thiếu khả năng tổng quát cho domain khác.
- API/model drift theo thời gian.

---

## 7) Dàn ý khoá luận hoàn chỉnh (đề xuất)

## Chương 1. Giới thiệu
- Bối cảnh và nhu cầu dữ liệu sign language tiếng Việt.
- Mục tiêu và phạm vi.
- Đóng góp chính.

## Chương 2. Cơ sở lý thuyết và công trình liên quan
- SLP, pose-based representation, ASR alignment.
- Dataset construction trong CV/NLP multimodal.
- Khoảng trống nghiên cứu mà đề tài nhắm tới.

## Chương 3. Phương pháp
- Tổng quan pipeline.
- Segmentation algorithm và tham số.
- ASR + text refinement + quality filtering.
- Keypoint extraction, metadata merging.

## Chương 4. Xây dựng dataset và triển khai hệ thống
- Nguồn dữ liệu, quy trình xử lý, lưu trữ.
- Kiểm soát lỗi và kiểm định dữ liệu.
- Dataset statistics và phân tích phân bố.

## Chương 5. Thực nghiệm và đánh giá
- Thiết lập thực nghiệm.
- Baselines, ablations, metrics.
- Kết quả và thảo luận.

## Chương 6. Kết luận và hướng phát triển
- Tóm tắt đóng góp.
- Hạn chế.
- Hướng mở rộng.

---

## 8) Dàn ý paper conference (6–8 trang)

## 8.1 Skeleton chuẩn

1. Introduction
2. Related Work
3. Dataset Construction Pipeline
4. Benchmark Setup
5. Results and Analysis
6. Limitations and Ethics
7. Conclusion

## 8.2 Viết theo “Contribution-first”

Trong Introduction, nêu ngay:
- Vấn đề chưa được giải quyết.
- Giải pháp hệ thống (pipeline + dataset).
- 2–3 đóng góp đo được.
- Preview kết quả nổi bật.

## 8.3 Figure/table bắt buộc

- Figure 1: Pipeline tổng thể.
- Figure 2: Segmentation logic minh hoạ event/scene timeline.
- Table 1: Dataset statistics.
- Table 2: Main benchmark results.
- Table 3: Ablation studies.

---

## 9) Roadmap triển khai 12 tuần (gợi ý)

## Tuần 1–2: Chuẩn hoá dữ liệu và tài liệu
- Freeze version metadata và scripts.
- Chuẩn hoá docs + reproducibility checklist.

## Tuần 3–4: Re-run pipeline subset + QC
- Chạy lại trên subset có kiểm thử thủ công.
- Audit lỗi segmentation/keypoints.

## Tuần 5–6: Benchmark baseline
- Chạy retrieval benchmark đầy đủ.
- Lưu report và seed cố định.

## Tuần 7–8: Ablation
- Segmentation thresholds.
- Text refinement on/off.
- Filtering on/off.

## Tuần 9–10: Viết khoá luận bản 1
- Hoàn thành Chương 1–4.
- Draft Chương 5 với bảng kết quả ban đầu.

## Tuần 11: Viết paper draft
- Nén nội dung thành manuscript.
- Chuẩn hoá hình/bảng.

## Tuần 12: Chốt phản biện nội bộ
- Sửa theo feedback.
- Chuẩn bị slide defense + supplemental.

---

## 10) Checklist “ready for thesis/paper”

## 10.1 Kỹ thuật
- [ ] Pipeline chạy lại được trên môi trường sạch.
- [ ] Version metadata nhất quán.
- [ ] Seed và config benchmark được cố định.
- [ ] Có script sinh thống kê tự động.

## 10.2 Học thuật
- [ ] Có tuyên bố đóng góp rõ và đo được.
- [ ] Có phần limitations + ethics.
- [ ] Có ablation chứng minh vai trò từng thành phần.
- [ ] Có so sánh baseline hợp lý.

## 10.3 Trình bày
- [ ] Figures dễ đọc, thống nhất ký hiệu.
- [ ] Tables có đơn vị, chú thích đầy đủ.
- [ ] Thuật ngữ nhất quán (clip/scene/segment).
- [ ] Tài liệu và kết quả liên kết đúng file.

---

## 11) Rủi ro phổ biến và cách giảm thiểu

- Rủi ro 1: Segmentation lỗi biên scene.
  - Giảm thiểu: kiểm tra thủ công subset + tuning threshold + rule fallback.

- Rủi ro 2: Transcript chưa sạch.
  - Giảm thiểu: thêm bước QC, đo tỷ lệ revert, thống kê lỗi chính tả.

- Rủi ro 3: Lệch phân bố train/test.
  - Giảm thiểu: stratified split theo quality/content + report imbalance.

- Rủi ro 4: Kết quả khó tái lập.
  - Giảm thiểu: khóa seed, lock version dependency, lưu run config.

---

## 12) Khung viết nhanh cho từng phần

## 12.1 One-paragraph project summary (đưa vào abstract mở rộng)

SignWeather is an end-to-end data construction framework for Vietnamese weather sign language from broadcast videos. The pipeline integrates timestamped speech transcription, language-model-based text refinement, geometric pose-based scene segmentation, and frame-level holistic keypoint extraction to produce multimodal scene annotations. The resulting dataset supports downstream sign language recognition and retrieval with reproducible metadata splits and benchmark scripts. Extensive quality filtering and post-processing are applied to improve annotation consistency and research usability.

## 12.2 Problem statement template

“Existing Vietnamese sign language resources remain limited in scale and domain specificity, especially for weather communication. This work addresses the gap by proposing an automated-yet-auditable pipeline that transforms raw weather broadcasts into scene-level multimodal training data.”

## 12.3 Contribution template

- “A reproducible pipeline for scene-level multimodal dataset construction from broadcast sign videos.”
- “A geometric landmark-based boundary detection strategy for practical segmentation.”
- “A curated Vietnamese weather sign dataset with aligned text, keypoints, and quality-aware metadata.”

---

## 13) Hướng mở rộng sau khoá luận

- Chuyển từ rule-based segmentation sang learned boundary detector.
- Huấn luyện SLP generative model từ scene_keypoints.
- Bổ sung weak/active learning cho quality auditing.
- Mở rộng domain ngoài weather để kiểm tra khả năng tổng quát.

---

## 14) Cách sử dụng tài liệu này ngay bây giờ

1. Dùng Mục 7 làm xương sống cho khoá luận.
2. Dùng Mục 8 để dựng khung paper trước khi viết chi tiết.
3. Dùng Mục 6 + 10 làm checklist thực nghiệm mỗi tuần.
4. Dùng Mục 9 làm timeline quản lý tiến độ.

Nếu cần, có thể tách tài liệu này thành:
- `THESIS_OUTLINE.md`
- `PAPER_OUTLINE.md`
- `EXPERIMENT_PLAN.md`
- `REPRODUCIBILITY_CHECKLIST.md`

---

## 15) Phiên bản và ghi chú

- Version: v1.0
- Date: 2026-02-26
- Scope: Định hướng học thuật + thực nghiệm cho SignWeather
- Status: Sẵn sàng dùng làm khung chính thức cho khoá luận/paper
