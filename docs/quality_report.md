# 📊 Quality Metrics Report for VSWD Dataset

**Generated**: 2026-02-06 14:28:14  
**Dataset**: /workspace/datdq/SignWeather/data/metadata/scene_metadata.csv

---

## 📈 Summary Scores

| Chỉ số | Giá trị | Đánh giá |
|--------|--------|---------|
| Sample Size | 300 clips | - |
| Avg Text Length | 18.9 words | - |
| Uncertain Ratio | 0.33% | Tỷ lệ cần review thêm |
| BLEU Score | 0.8522 | ✅ Tốt |
| WER (Word Error Rate) | 0.0829 | ✅ Tốt |

---

## 📝 Chi tiết giải thích các điểm số

### 1️⃣ **BLEU Score** = 0.852222778205616

**Định nghĩa**: Đo độ tương đồng giữa transcript gốc và pseudo-manual (tham chiếu).

**Cách tính**: So sánh n-gram (từ đơn, cặp từ, v.v.) giữa hai câu. Càng giống nhau → điểm cao.

**Mục đích**: Đánh giá chất lượng text transcript và ASR refine.

**Giải thích kết quả**:
- **> 0.6 (Tốt)**: Transcript rất tương đồng với tham chiếu. Chất lượng ASR/refine tốt.
- **0.3–0.6 (Trung bình)**: Có một số khác biệt nhỏ. Có thể cần refine thêm.
- **< 0.3 (Kém)**: Có sai lệch lớn. Cần review và sửa chữa.

---

### 2️⃣ **WER (Word Error Rate)** = 0.08290702131217988

**Định nghĩa**: Tỷ lệ lỗi từ = (insertions + deletions + substitutions) / total_ref_words

**Mục đích**: Đo độ chính xác transcript ở mức từ. Càng thấp càng tốt.

**Giải thích kết quả**:
- **< 0.2 (Tốt)**: Transcript gần như chính xác. Chỉ có 10-20% từ khác biệt.
- **0.2–0.5 (Trung bình)**: Có lỗi đáng kể. Cần refine hoặc manual review.
- **> 0.5 (Kém)**: Quá nhiều lỗi (> 50%). Cần xem lại toàn bộ pipeline ASR.

---

### 3️⃣ **Uncertain Ratio** = 0.33%

**Định nghĩa**: Tỷ lệ clips mà GPT đánh dấu là không chắc chắn ([UNCERTAIN]).

**Mục đích**: Xác định cần bao nhiêu công sức manual review.

**Giải thích kết quả**:
- **< 10% (Tốt)**: Hầu hết clips có thể tin cậy. Ít cần review.
- **10–30% (Trung bình)**: Nên review khoảng 1/3 clips không chắc.
- **> 30% (Kém)**: Quá nhiều không chắc. Cần refine pipeline trước.

---

### 4️⃣ **Avg Text Length** = 18.9 từ

**Định nghĩa**: Độ dài trung bình một transcript (tính bằng số từ).

**Mục đích**: Biết dataset có clips ngắn (< 5 từ) hay dài (> 20 từ).

**Ý nghĩa**:
- Clips ngắn: Có thể là lỗi ASR hoặc gesture ngắn.
- Clips dài: Yêu cầu model xử lý chuỗi dài hơn.

---

## 📊 Dữ liệu mẫu (Top 10 Clips)

| # | Path | Original Text | Pseudo-Manual | Uncertain |
|---|------|---------------|---------------|----------|
| 1 | v163/scene_010_v163_c019.mp4 | Thành phố Đà Nẵng trước khi có mưa nặng hạt vào chiều tối ma | Thành phố Đà Nẵng trước khi có mưa nặng hạt vào chiều tối ma | ✗ No |
| 2 | v067/scene_006.mp4 | Nhiệt độ hạ xuống còn khoảng là 21 độ và về đêm mức nhiệt là | Nhiệt độ hạ xuống còn khoảng 21 độ và về đêm mức nhiệt là 16 | ✗ No |
| 3 | v039/scene_009_v039_c051.mp4 | nhiệt độ không vượt quá 22 độ. | Nhiệt độ không vượt quá 22 độ. | ✗ No |
| 4 | v243/scene_002_v243_c022.mp4 | Vùng Hạ Liêu, sông Thu Bồn, mực nước đang xuống chậm. | Vùng Hạ Liêu, sông Thu Bồn, mực nước đang xuống chậm. | ✗ No |
| 5 | v209/scene_005_v209_c018.mp4 | Thủ đô Hà Nội đêm nay không có mưa, nhiệt độ 12 độ, và ngày  | Thủ đô Hà Nội đêm nay không có mưa, nhiệt độ 12 độ, và ngày  | ✗ No |
| 6 | v158/scene_005_v158_c009.mp4 | Đến trưa chiều thì dự báo là nắng mới xuất hiện trở lại ở kh | Đến trưa chiều thì dự báo là nắng sẽ xuất hiện trở lại ở khu | ✗ No |
| 7 | v036/scene_011_v036_c045.mp4 | Tại TP.HCM và Biên Hòa, mức nhiệt cao nhất vào ngày mai có t | Tại TP.HCM và Biên Hòa, mức nhiệt cao nhất vào ngày mai có t | ✗ No |
| 8 | v168/scene_007_v168_c015.mp4 | đến phía bắc của Phú Yên sẽ có gió mạnh cấp 6, cấp 7, giật c | Đến phía bắc của Phú Yên sẽ có gió mạnh cấp 6, cấp 7, giật c | ✗ No |
| 9 | v053/scene_007_v053_c020.mp4 | Và độ ẩm sẽ giảm dần xuống còn ở mức là 68% | Và độ ẩm sẽ giảm dần xuống còn ở mức 68%. | ✗ No |
| 10 | v047/scene_005.mp4 | Còn bây giờ sẽ là phần dự báo chi tiết cho các thành phố tro | Còn bây giờ sẽ là phần dự báo chi tiết cho các thành phố tro | ✗ No |


---

## 💾 Files

- **Full Report**: /workspace/datdq/SignWeather/docs/quality_report.md
- **Full Data (CSV)**: /workspace/datdq/SignWeather/docs/quality_report_pseudo.csv

---

## ✅ Kết luận

Dựa vào các điểm số trên, bạn có thể:

1. **Nếu BLEU > 0.6 & WER < 0.2 & Uncertain < 10%**: Dataset chất lượng tốt ✅
2. **Nếu BLEU 0.3–0.6 hoặc WER 0.2–0.5**: Cần refine thêm hoặc manual review một phần ⚠️
3. **Nếu BLEU < 0.3 hoặc WER > 0.5**: Cần xem lại pipeline ASR/refine ❌

