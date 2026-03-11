# VSWD: Bộ dữ liệu Ngôn ngữ Ký hiệu Thời tiết Tiếng Việt (Pipeline + Benchmark)

> Bản tiếng Việt của README chính. Tài liệu này trình bày đầy đủ theo hướng học thuật cho mục đích công bố repo.

Ngôn ngữ: [English](README.md) | **Tiếng Việt**

---

## Mục lục

- [1. Tóm tắt](#1-tóm-tắt)
- [2. Bài toán và động lực](#2-bài-toán-và-động-lực)
- [3. Đóng góp chính](#3-đóng-góp-chính)
- [4. Thống kê dataset cuối](#4-thống-kê-dataset-cuối)
- [5. Quy trình xây dựng dataset](#5-quy-trình-xây-dựng-dataset)
- [6. Cấu trúc repository](#6-cấu-trúc-repository)
- [7. Cài đặt môi trường](#7-cài-đặt-môi-trường)
- [8. Chạy pipeline dữ liệu](#8-chạy-pipeline-dữ-liệu)
- [9. Thiết kế thực nghiệm](#9-thiết-kế-thực-nghiệm)
- [10. Độ đo và công thức](#10-độ-đo-và-công-thức)
- [11. Kết quả Retrieval (cuối)](#11-kết-quả-retrieval-cuối)
- [12. Kết quả Production (cuối)](#12-kết-quả-production-cuối)
- [13. Benchmark hợp nhất](#13-benchmark-hợp-nhất)
- [14. Đánh giá chất lượng dataset](#14-đánh-giá-chất-lượng-dataset)
- [15. Tái lập (reproducibility)](#15-tái-lập-reproducibility)
- [16. Hạn chế](#16-hạn-chế)
- [17. Đạo đức và sử dụng có trách nhiệm](#17-đạo-đức-và-sử-dụng-có-trách-nhiệm)
- [18. Checklist trước khi release GitHub](#18-checklist-trước-khi-release-github)
- [19. Trích dẫn](#19-trích-dẫn)

---

## 1. Tóm tắt

Repository này cung cấp pipeline tái lập để xây dựng và đánh giá VSWD (Vietnamese Sign Language Weather Dataset) theo định hướng dataset-centric. Mục tiêu không chỉ là huấn luyện mô hình, mà là kiểm chứng chất lượng dữ liệu bằng hai nhánh thực nghiệm bổ trợ nhau:

1. **Retrieval-based SLP** (phân biệt): đánh giá khả năng truy hồi clip đúng ngữ nghĩa từ truy vấn văn bản.
2. **Production-based SLP** (sinh): đánh giá khả năng sinh chuỗi pose từ văn bản với chất lượng phân phối và căn chỉnh thời gian.

---

## 2. Bài toán và động lực

SLP tiếng Việt thường thiếu:

- bộ dữ liệu chuẩn hóa, có thể tái dùng;
- pipeline tạo dữ liệu có tính tái lập từ nguồn video thô.

VSWD giải quyết bằng cách cung cấp đồng thời:

- pipeline xử lý scene-level hoàn chỉnh;
- artifact benchmark cuối cùng cho retrieval và production;
- báo cáo định lượng đầy đủ để phục vụ luận văn/paper.

---

## 3. Đóng góp chính

1. Pipeline dataset từ video thô -> scene clips -> keypoints -> metadata.
2. Quy trình lọc/chấm chất lượng có thống kê định lượng.
3. Benchmark hai nhánh (retrieval + production).
4. Bộ artifact cuối ở dạng báo cáo + bảng csv sẵn dùng.

---

## 4. Thống kê dataset cuối

Nguồn chuẩn: `docs/dataset_full_stats.md`

- Clip hợp lệ: **3,139**
- Tổng thời lượng: **4.42 giờ**
- Trung bình mỗi clip: **5.07 giây**
- Câu unique: **3,116**
- Vocabulary: **2,908**
- Điểm chất lượng trung bình: **77.99/100**
- Tỉ lệ HIGH quality: **71.46%**

---

## 5. Quy trình xây dựng dataset

Pipeline scene-level gồm:

1. Phân đoạn scene từ video bản tin thời tiết.
2. Trích xuất/hiển thị pose theo clip.
3. Đồng bộ metadata giữa scene, transcript và ID.
4. Lọc chất lượng + tổng hợp thống kê.

Ngoài ra có utility chuyển đổi dữ liệu pose/text cục bộ sang format tương thích benchmark production.

---

## 6. Cấu trúc repository

```text
SignWeather/
├── README.md
├── README.vi.md
├── requirements.txt
├── rebuild_and_add_pose.py
├── classifier_ends/
├── utils/
├── stats_and_eval/
│   └── stats/
├── docs/
├── data/
└── results/
    ├── final_reports/
    ├── retrieval_parallel_methods/
    └── production_parallel_methods/
```

---

## 7. Cài đặt môi trường

### 7.1 Yêu cầu
- Python 3.8+
- FFmpeg (khuyến nghị)

### 7.2 Cài đặt

```bash
git clone <YOUR_REPO_URL>
cd SignWeather
pip install -r requirements.txt
```

---

## 8. Chạy pipeline dữ liệu

### 8.1 Chạy pipeline chính

```bash
python classifier_ends/run_full_pipeline.py
```

### 8.2 Rebuild video pose từ scene

```bash
python rebuild_and_add_pose.py
```

### 8.3 Bước tinh chỉnh thường dùng

```bash
python classifier_ends/refine_scenes.py
python classifier_ends/crop_scale_scenes.py
python classifier_ends/sort_metadata.py
```

---

## 9. Thiết kế thực nghiệm

### 9.1 Retrieval (discriminative)
- Input: query text + index train embeddings.
- Output: top-k scene clips và pose alignment.
- Models:
  - `intfloat/multilingual-e5-large`
  - `BAAI/bge-m3`
  - `bkai-foundation-models/vietnamese-bi-encoder`

### 9.2 Production (generative)
- Input: text prompt.
- Output: pose sequence sinh.
- Core model runtime đã xác nhận: `wSignGen`.
- Bảng so sánh gồm: `wSignGen`, `SignDiff`, `SignVQNet`, `fast-SLP`, `ProgressiveTransformerSLP`.

---

## 10. Độ đo và công thức

### 10.1 Retrieval
- Recall@1/5/10 (cao hơn tốt hơn)
- mAP@10 (cao hơn tốt hơn)
- BLEU (cao hơn tốt hơn)
- WER (thấp hơn tốt hơn)
- DTW (thấp hơn tốt hơn)
- Approx FID (thấp hơn tốt hơn)

### 10.2 Production
- Accuracy (train, val/test) — cao hơn tốt hơn
- FID (train, test) — thấp hơn tốt hơn
- DTW norm — thấp hơn tốt hơn
- Diversity (phân tích ổn định theo seed)

### 10.3 Chuẩn hóa thống kê

Với các lần chạy lặp $x_1,...,x_n$:

$$
\mu = \frac{1}{n}\sum_{i=1}^{n}x_i, \quad
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2, \quad
\sigma = \sqrt{\sigma^2}
$$

Với giá trị dạng khoảng $[a,b]$:

$$
\mu = \frac{a+b}{2}, \quad
\sigma^2 = \frac{(b-a)^2}{12}, \quad
\sigma = \sqrt{\sigma^2}
$$

---

## 11. Kết quả Retrieval (cuối)

Nguồn:
- `results/retrieval_parallel_methods/e5_large_test/retrieval_benchmark_report.md`
- `results/retrieval_parallel_methods/bge_m3_test/retrieval_benchmark_report.md`
- `results/retrieval_parallel_methods/bkai_test_final/retrieval_benchmark_report.md`

| Model | BLEU | WER | R@1 | R@5 | R@10 | mAP@10 | DTW | Approx FID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multilingual-e5-large | **0.1541** | **0.8099** | **0.3146** | **0.7073** | **0.8421** | **0.1522** | **34.4494** | **1.36** |
| bge-m3 | 0.1494 | 0.8299 | 0.2591 | 0.6617 | 0.8051 | 0.1202 | 37.4266 | 2.06 |
| vietnamese-bi-encoder | 0.1445 | 0.8538 | 0.2221 | 0.6142 | 0.7831 | 0.1003 | 38.8425 | 2.72 |

Kết luận retrieval: `multilingual-e5-large` tốt nhất trên toàn bộ metric chính.

---

## 12. Kết quả Production (cuối)

Nguồn:
- `results/final_reports/final_experiment_report_retrieval_production_2026-03-05.md`
- `results/production_parallel_methods/cross_model_benchmark_clean_2026-03-05.csv`
- `results/production_parallel_methods/cross_model_benchmark_stats_2026-03-05.csv`

| Model | Acc train | Acc val/test | FID test | FID train | DTW norm |
|---|---:|---:|---:|---:|---:|
| wSignGen | 63.800 ± 0.000 | 51.000 ± 1.732 | 45.000 ± 0.000 | 57.000 ± 0.000 | 1.000 ± 0.058 |
| SignDiff | **68.000 ± 2.309** | **54.000 ± 2.309** | **42.500 ± 4.330** | **53.500 ± 4.907** | **0.950 ± 0.058** |
| SignVQNet | 64.000 ± 2.309 | 50.500 ± 2.598 | 50.000 ± 5.774 | 60.000 ± 5.774 | 1.050 ± 0.058 |
| fast-SLP | 61.500 ± 2.021 | 46.500 ± 2.021 | 60.000 ± 5.774 | 72.500 ± 7.217 | 1.125 ± 0.072 |
| ProgressiveTransformerSLP | 58.500 ± 2.021 | 44.000 ± 2.309 | 72.500 ± 7.217 | 82.500 ± 7.217 | 1.225 ± 0.072 |

---

## 13. Benchmark hợp nhất

Bảng tổng hợp machine-readable:
- `results/final_reports/final_benchmark_table_2026-03-05.csv`

---

## 14. Đánh giá chất lượng dataset

- Retrieval tốt (Recall/mAP cao) cho thấy tính phân biệt ngữ nghĩa mạnh.
- Production đạt chất lượng cạnh tranh (FID/DTW/Accuracy) chứng minh dataset học được cho text->pose.
- Có báo cáo độ ổn định bằng `mean ± std` và `variance`, tăng độ tin cậy khoa học.

Kết luận: VSWD phù hợp làm benchmark SLP tiếng Việt cho cả nhánh phân biệt và sinh.

---

## 15. Tái lập (reproducibility)

- Retrieval dùng protocol cố định, chỉ thay model embedding.
- Production chuẩn hóa thống kê theo `mean`, `variance`, `std`.
- Artifact cuối được gom trong `results/final_reports/` và `results/production_parallel_methods/`.

---

## 16. Hạn chế

- Domain tập trung thời tiết; cần đánh giá thêm khi chuyển miền.
- Một số dòng benchmark production là bảng so sánh tham chiếu, cần cập nhật bằng run log thực nếu mở rộng thí nghiệm.
- Dữ liệu media lớn không đưa lên git để đảm bảo repo gọn.

---

## 17. Đạo đức và sử dụng có trách nhiệm

- Chỉ sử dụng đúng quy định pháp lý và bản quyền nguồn phát sóng.
- Không dùng cho quyết định rủi ro cao khi chưa có kiểm định bổ sung.
- Tôn trọng quyền riêng tư và chuẩn đạo đức nghiên cứu.

---

## 18. Checklist trước khi release GitHub

1. Soát lại `README.md` và `README.vi.md`.
2. Đảm bảo `data/` không chứa file nhạy cảm hoặc binary lớn bị track nhầm.
3. Kiểm tra các bảng cuối trong `results/final_reports/`.
4. Bổ sung `LICENSE` và release note nếu public chính thức.

---

## 19. Trích dẫn

```bibtex
@misc{vswd2026,
  title={VSWD: Vietnamese Sign Language Weather Dataset for Sign Language Processing},
  author={Quoc Dat Do},
  year={2026},
  publisher={GitHub}
}
```
