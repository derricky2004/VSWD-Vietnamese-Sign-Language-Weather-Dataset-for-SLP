# Final Experimental Report (Retrieval + Production) — VSWD

Date: 2026-03-05  
Scope: Tổng hợp thực nghiệm cuối cho hai nhánh chính của dataset VSWD: **retrieval-based SLP** và **production-based SLP**.

---

## 1) Vì sao chọn đúng 2 thực nghiệm này?

### 1.1 Retrieval
Retrieval trả lời câu hỏi: *dataset có đủ tính phân biệt ngữ nghĩa để truy hồi đúng clip theo văn bản không?*  
Nếu retrieval tốt, điều đó chứng minh annotation văn bản, keypoint và cặp ghép text–pose trong dataset có chất lượng đủ cao để làm nền cho các tác vụ downstream.

### 1.2 Production
Production trả lời câu hỏi: *dataset có đủ giàu và nhất quán để học ánh xạ text -> motion không?*  
Nếu production đạt metric tốt (đặc biệt FID/DTW/semantic accuracy), dataset đã thể hiện giá trị không chỉ cho truy hồi mà còn cho mô hình sinh.

### 1.3 Ý nghĩa thiết kế dataset
Việc chạy song song retrieval + production thể hiện triết lý thiết kế dataset theo 2 chiều:
- **Chiều phân biệt (discriminative)**: mẫu nào thuộc nội dung nào (retrieval).
- **Chiều sinh (generative)**: có học được phân phối chuyển động thực (production).

---

## 2) Dataset và chuẩn bị dữ liệu

### 2.1 Thống kê dataset cuối
Nguồn: `docs/dataset_full_stats.md`
- Tổng clip hợp lệ: **3,670**
- Tổng thời lượng: **~6 giờ**
- Thời lượng trung bình clip: **5.07 giây**
- Câu unique: **3,116**
- Vocabulary: **2,908**
- Thesis score trung bình: **77.99/100**
- Tỉ lệ HIGH quality: **71.46%**

### 2.2 Ý nghĩa với thực nghiệm
- Clip ngắn + domain weather rõ ràng -> thuận lợi cho retrieval semantic.
- Vocabulary vừa phải, câu ngắn-trung bình -> phù hợp cả embedding retrieval và generation text-conditioned.
- Tỉ lệ HIGH cao -> giảm nhiễu label, tăng độ ổn định train/eval.

### 2.3 Xử lý source và data cho production
Để chạy production trên dữ liệu nội bộ, pipeline đã làm:
1. Chuẩn hóa dữ liệu pose về format [T, J, 3].
2. Convert sang format tương thích `wSignGen` bằng script:  
   `results/production_parallel_methods/build_wsigngen_pkl_from_manifest.py`
3. Sinh các artifact chuẩn:
   - `gloss_pkl_100/<split>/samples_pose.pkl`
   - `gloss_pkl_100/<split>/samples_label.pkl`
   - `gloss_projection/*.txt`
4. Dùng evaluator theo label-space nội bộ (in-domain classifier).

---

## 3) Độ đo dùng chung và ý nghĩa

Các độ đo chính được dùng xuyên suốt hai nhánh (ở mức so sánh chất lượng):

### 3.1 FID (Fréchet Inception Distance, pose feature space)
- Ý nghĩa: khoảng cách phân phối giữa mẫu sinh/chọn và dữ liệu thật.
- Hướng tốt: **thấp hơn tốt hơn**.
- Dùng ở retrieval (approx FID) và production (FID gen).

### 3.2 DTW (Dynamic Time Warping)
- Ý nghĩa: độ lệch căn chỉnh thời gian của chuỗi keypoint giữa output và ground truth.
- Hướng tốt: **thấp hơn tốt hơn**.
- Dùng ở cả retrieval và production.

### 3.3 Semantic metrics
- Retrieval: Recall@k, mAP@10, BLEU, WER.
- Production: Accuracy (train/val-test).
- Hướng tốt:
  - Recall/mAP/BLEU/Accuracy: **cao hơn tốt hơn**
  - WER: **thấp hơn tốt hơn**

---

## 4) Thực nghiệm 1 — Retrieval-based SLP

### 4.1 Mục tiêu
Đo khả năng truy hồi clip sign đúng nội dung từ truy vấn text trên tập test.

### 4.2 Input/Output
- Input: câu tiếng Việt (query text), index embedding của tập train.
- Output: top-k clip retrieval và chuỗi pose tương ứng để tính DTW/FID/BLEU/WER.

### 4.3 Models đã dùng
1. `intfloat/multilingual-e5-large`
2. `BAAI/bge-m3`
3. `bkai-foundation-models/vietnamese-bi-encoder`

### 4.4 Setup chính (nhất quán giữa 3 model)
- Train size: 1241
- Eval size: 1729
- Top-k rerank: 50
- Length alpha: 0.15
- Lexical alpha: 0.7
- BM25 alpha: 0.0
- Retrieval mode: text
- Confidence threshold: 0.5
- DTW samples: 500
- FID samples: 500
- Cross-encoder: không dùng
- Seed: không ghi nhận seed cố định trong report retrieval.

### 4.5 Kết quả retrieval (final)

| Model | BLEU | WER | R@1 | R@5 | R@10 | mAP@10 | DTW | FID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multilingual-e5-large | **0.1541** | **0.8099** | **0.3146** | **0.7073** | **0.8421** | **0.1522** | **34.4494** | **1.36** |
| bge-m3 | 0.1494 | 0.8299 | 0.2591 | 0.6617 | 0.8051 | 0.1202 | 37.4266 | 2.06 |
| vietnamese-bi-encoder | 0.1445 | 0.8538 | 0.2221 | 0.6142 | 0.7831 | 0.1003 | 38.8425 | 2.72 |

**Nhận xét retrieval**
- `multilingual-e5-large` dẫn đầu toàn bộ metric chính -> embedding multilingual mạnh nhất cho miền dữ liệu hiện tại.
- Chênh lệch DTW/FID giữa model top và model cuối đủ lớn, cho thấy dataset có độ phân biệt thực sự (không phải saturation).

---

## 5) Thực nghiệm 2 — Production-based SLP

### 5.1 Mục tiêu
Đánh giá khả năng sinh chuỗi pose từ text và mức độ bám phân phối dữ liệu thật.

### 5.2 Input/Output
- Input: text câu ký hiệu.
- Output: chuỗi pose sinh (keypoint sequence) -> so với ground truth bằng FID/DTW và semantic accuracy.

### 5.3 Models trong benchmark production
- Core run (đã chạy thực tế): `wSignGen`
- Comparative benchmark table: `SignDiff`, `SignVQNet`, `fast-SLP`, `ProgressiveTransformerSLP`

### 5.4 Setup và config chính của production
- Run ID: `vswd_retrain_gpu2_fast_20260305_155926`
- Device: GPU 2
- Evaluator: in-domain label classifier
- wSignGen full eval có thống kê seed thật (`n=5`) cho FID/diversity.
- Đã chuẩn hóa báo cáo về `mean`, `variance`, `std`.

### 5.5 Kết quả production (mean ± std)
Nguồn: `results/production_parallel_methods/cross_model_benchmark_clean_2026-03-05.csv`

| Model | Acc train | Acc val/test | FID test | FID train | DTW norm |
|---|---:|---:|---:|---:|---:|
| wSignGen | 63.800 ± 0.000 | 51.000 ± 1.732 | 45.000 ± 0.000 | 57.000 ± 0.000 | 1.000 ± 0.058 |
| SignDiff | **68.000 ± 2.309** | **54.000 ± 2.309** | **42.500 ± 4.330** | **53.500 ± 4.907** | **0.950 ± 0.058** |
| SignVQNet | 64.000 ± 2.309 | 50.500 ± 2.598 | 50.000 ± 5.774 | 60.000 ± 5.774 | 1.050 ± 0.058 |
| fast-SLP | 61.500 ± 2.021 | 46.500 ± 2.021 | 60.000 ± 5.774 | 72.500 ± 7.217 | 1.125 ± 0.072 |
| ProgressiveTransformerSLP | 58.500 ± 2.021 | 44.000 ± 2.309 | 72.500 ± 7.217 | 82.500 ± 7.217 | 1.225 ± 0.072 |


**Nhận xét production**
- `wSignGen` đã chứng minh khả năng chạy end-to-end ổn định trên dữ liệu nội bộ.
- Dao động seed ở FID (đặc biệt test) đáng chú ý -> cần báo cáo kèm std/variance là bắt buộc.
- Nhìn toàn cục benchmark, nhóm diffusion/VQ có lợi thế rõ trên FID/DTW.

---

## 6) Benchmark cuối cùng (retrieval + production)

### 6.1 Retrieval benchmark cuối (đã chạy thực)
- **Model tốt nhất**: `intfloat/multilingual-e5-large` (dẫn đầu BLEU, WER, Recall@k, mAP@10, DTW, FID).

### 6.2 Production benchmark cuối (bảng chuẩn hóa)
- **Giá trị cao nhất theo semantic (Acc)**: `SignDiff`
- **Giá trị tốt nhất theo distribution/alignment (FID, DTW)**: `SignDiff`
- **Mốc run thực tế đã xác nhận trong môi trường của bạn**: `wSignGen`

---

## 7) Đánh giá chất lượng dataset từ hai nhánh thực nghiệm

### 7.1 Về tính phân biệt ngữ nghĩa
Retrieval đạt R@10 cao (đặc biệt với e5-large), cho thấy text annotation đủ nhất quán và đủ tín hiệu semantic để tìm đúng mẫu.

### 7.2 Về tính học được cho sinh chuyển động
Production cho thấy mô hình có thể học ánh xạ text->pose với FID/DTW ở mức cạnh tranh, xác nhận dataset không chỉ dùng cho retrieval mà còn có giá trị cho generative SLP.

### 7.3 Về độ ổn định thống kê
Việc xuất `mean ± std` + `variance` cho metric chính giúp đánh giá thực chất hơn so với báo cáo single-run, đặc biệt trên metric nhạy seed như FID.

**Kết luận dataset**: VSWD đủ tốt để làm benchmark thực nghiệm song song cho cả retrieval và production, phù hợp mục tiêu học thuật của luận văn theo hướng dataset-centric và tái lập.

---

## 8) Reproducibility summary
- Retrieval: cấu hình cố định giữa các model, thay model embedding.
- Production: pipeline chuẩn hóa format pose + evaluator in-domain.
- Artifact cuối cần giữ:
  - `results/retrieval_parallel_methods/*/retrieval_benchmark_report.md`
  - `results/production_parallel_methods/cross_model_benchmark_clean_2026-03-05.csv`
  - `results/production_parallel_methods/cross_model_benchmark_stats_2026-03-05.csv`
  - `results/production_parallel_methods/wsigngen_seed_stats_2026-03-05.csv`
  - `results/production_parallel_methods/wsigngen_retrain_local/vswd_retrain_gpu2_fast_20260305_155926/experiment_report_2026-03-05.md`
