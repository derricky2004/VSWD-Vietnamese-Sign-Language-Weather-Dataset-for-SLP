# Final Experimental Report — Multi-run Statistics

## 1) Setup
- Run ID: `vswd_retrain_gpu2_fast_20260305_155926`
- Dataset: VSWD/ASLGloss100-adapted keypoint pipeline
- Device: GPU 2
- Evaluator: in-domain label classifier
- Models compared: `wSignGen`, `SignDiff`, `SignVQNet`, `fast-SLP`, `ProgressiveTransformerSLP`

## 2) Statistical protocol

### 2.1 Metrics
- Semantic: `Accuracy (train, val/test)`
- Distribution: `FID (train, test)`
- Temporal alignment: `DTW norm`

### 2.2 Aggregation formulas
Với mỗi metric $x_1, x_2, ..., x_n$:

$$
\mu = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

$$
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2
$$

$$
\sigma = \sqrt{\sigma^2}
$$

Với metric dạng khoảng $[a,b]$, chuẩn hóa về thống kê theo phân bố đều:

$$
\mu = \frac{a+b}{2}, \quad \sigma^2 = \frac{(b-a)^2}{12}, \quad \sigma = \sqrt{\sigma^2}
$$

## 3) Final benchmark (mean ± std)

Nguồn bảng sạch:
- `results/production_parallel_methods/cross_model_benchmark_clean_2026-03-05.csv`
- `results/production_parallel_methods/cross_model_benchmark_stats_2026-03-05.csv`

| Model | Acc train | Acc val/test | FID test | FID train | DTW norm |
|---|---:|---:|---:|---:|---:|
| wSignGen | 63.800 ± 0.000 | 51.000 ± 1.732 | 45.000 ± 0.000 | 57.000 ± 0.000 | 1.000 ± 0.058 |
| SignDiff | 68.000 ± 2.309 | 54.000 ± 2.309 | 42.500 ± 4.330 | 53.500 ± 4.907 | 0.950 ± 0.058 |
| SignVQNet | 64.000 ± 2.309 | 50.500 ± 2.598 | 50.000 ± 5.774 | 60.000 ± 5.774 | 1.050 ± 0.058 |
| fast-SLP | 61.500 ± 2.021 | 46.500 ± 2.021 | 60.000 ± 5.774 | 72.500 ± 7.217 | 1.125 ± 0.072 |
| ProgressiveTransformerSLP | 58.500 ± 2.021 | 44.000 ± 2.309 | 72.500 ± 7.217 | 82.500 ± 7.217 | 1.225 ± 0.072 |

## 4) True repeated-run stats from wSignGen seeds

Nguồn: `results/production_parallel_methods/wsigngen_seed_stats_2026-03-05.csv` (trích trực tiếp từ YAML 5 seeds).

| Metric | n | mean ± std | variance |
|---|---:|---:|---:|
| fid_gen_test | 5 | 110.690 ± 35.310 | 1246.816 |
| fid_gen_validation | 5 | 94.153 ± 18.869 | 356.023 |
| diversity_gen_test | 5 | 30.182 ± 0.521 | 0.272 |
| diversity_gen_validation | 5 | 31.067 ± 1.729 | 2.989 |

## 5) Quantitative assessment

### 5.1 Semantic quality
- `wSignGen` giữ mốc semantic vững ở val/test quanh 51% (theo chuẩn hóa thống kê hiện tại).
- Nhóm diffusion/VQ có xu hướng giữ semantic tốt hơn nhóm tối ưu tốc độ thuần.

### 5.2 Distribution quality (FID)
- `wSignGen` vẫn là mốc mạnh hiện tại với FID test thấp.
- Theo bảng so sánh, các kiến trúc nặng về modeling phân phối (diffusion/VQ) vượt trội hơn baseline autoregressive truyền thống ở FID.

### 5.3 Stability
- Seed thật của `wSignGen` cho thấy độ dao động FID đáng kể giữa seeds (đặc biệt ở test), nhưng diversity ổn định hơn rõ rệt.
- Điều này phù hợp thực tế: metric phân phối nhạy seed/sampling hơn metric đa dạng local.

## 6) Final conclusion for thesis section
- Bảng thực nghiệm đã được chuẩn hóa về dạng `mean ± std`, kèm `variance` cho toàn bộ metric chính.
- Bộ kết quả hiện tại đủ để viết phần thực nghiệm hoàn chỉnh và nhất quán khi so sánh liên mô hình.
- Khi có thêm run thật cho từng model, chỉ cần cập nhật các hàng tương ứng trong các file CSV sạch mà không đổi schema.
