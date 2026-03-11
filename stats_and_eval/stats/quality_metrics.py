import pandas as pd
import numpy as np
import time
import os
import sys
import argparse
from pathlib import Path

# Try to import optional dependencies
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from jiwer import wer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Configuration paths
BASE_DIR = Path("/workspace/datdq/SignWeather")
METADATA_DIR = BASE_DIR / "data" / "metadata"

# Use canonical metadata file
CSV_PATH = str(METADATA_DIR / "scene_metadata.csv")
if not Path(CSV_PATH).exists():
    print(f"Error: No metadata CSV found at {CSV_PATH}")
    sys.exit(1)

# OpenAI API key (set in environment, do not hardcode secrets)
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
SUBSET_SIZE = 300  # Số clip để kiểm tra
OUTPUT_REPORT = str(BASE_DIR / "docs" / "quality_report.md")

def create_pseudo_manual(client, df_sample):
    """Create pseudo-manual using GPT if client available, else use original text."""
    if client is None:
        print("Note: OpenAI client not available. Using original text as pseudo-manual.")
        return pd.DataFrame({
            'path': df_sample['path'],
            'original_text': df_sample['text'],
            'pseudo_manual': df_sample['text'],
            'is_uncertain': [False] * len(df_sample)
        })
    
    results = []
    for i, row in df_sample.iterrows():
        prompt = f"""
Bạn là chuyên gia bản tin thời tiết VTV tiếng Việt, hiểu rõ ngôn ngữ ký hiệu VSL.
Transcript đã refine từ ASR + GPT: "{row['text']}"

Nhiệm vụ:
- Kiểm tra xem câu này có đúng ngữ pháp, ngữ nghĩa thời tiết, mạch lạc không.
- Nếu đúng → giữ nguyên.
- Nếu sai nhỏ (chính tả, thiếu từ) → sửa cho tự nhiên.
- Nếu không chắc → giữ nguyên + thêm "[UNCERTAIN]" ở cuối.
Chỉ trả về câu cuối cùng, không giải thích.
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            corrected = response.choices[0].message.content.strip()
            results.append({
                'path': row['path'],
                'original_text': row['text'],
                'pseudo_manual': corrected,
                'is_uncertain': "[UNCERTAIN]" in corrected
            })
        except Exception as e:
            print(f"Error at {row['path']}: {e}")
            results.append({
                'path': row['path'],
                'original_text': row['text'],
                'pseudo_manual': row['text'],
                'is_uncertain': True
            })
        time.sleep(0.5)  # Tránh rate limit
    return pd.DataFrame(results)

def calculate_metrics(df_pseudo):
    """Calculate quality metrics. If NLTK/jiwer not available, compute simple stats."""
    results = {
        'sample_size': len(df_pseudo),
        'avg_bleu': None,
        'avg_wer': None,
        'uncertain_ratio': df_pseudo['is_uncertain'].mean() * 100 if 'is_uncertain' in df_pseudo.columns else 0,
        'avg_text_length': df_pseudo['original_text'].str.split().str.len().mean() if 'original_text' in df_pseudo.columns else 0,
    }
    
    if HAS_NLTK and HAS_JIWER and 'pseudo_manual' in df_pseudo.columns and 'original_text' in df_pseudo.columns:
        smoothie = SmoothingFunction().method4
        bleu_scores = []
        wer_scores = []
        for _, row in df_pseudo.iterrows():
            ref = str(row['pseudo_manual']).replace("[UNCERTAIN]", "").strip().split()
            hyp = str(row['original_text']).split()
            try:
                bleu = sentence_bleu([ref], hyp, smoothing_function=smoothie)
                bleu_scores.append(bleu)
                wer_val = wer(' '.join(ref), ' '.join(hyp))
                wer_scores.append(wer_val)
            except Exception:
                pass
        
        if bleu_scores:
            results['avg_bleu'] = np.mean(bleu_scores)
        if wer_scores:
            results['avg_wer'] = np.mean(wer_scores)
    else:
        if not HAS_NLTK:
            print("⚠️ nltk not installed, skipping BLEU calculation")
        if not HAS_JIWER:
            print("⚠️ jiwer not installed, skipping WER calculation")
    
    # Print scores to console
    print("\n" + "="*70)
    print("🎯 QUALITY SCORES:")
    print("="*70)
    print(f"📊 Sample Size: {results['sample_size']} clips")
    print(f"📝 Avg Text Length: {results['avg_text_length']:.1f} words")
    print(f"⚠️  Uncertain Ratio: {results['uncertain_ratio']:.2f}%")
    if results['avg_bleu'] is not None:
        print(f"🔵 BLEU Score: {results['avg_bleu']:.4f}")
        if results['avg_bleu'] > 0.6:
            print(f"   ✅ Good quality (BLEU > 0.6)")
        elif results['avg_bleu'] > 0.3:
            print(f"   ⚠️  Moderate quality (BLEU 0.3-0.6)")
        else:
            print(f"   ❌ Low quality (BLEU < 0.3)")
    if results['avg_wer'] is not None:
        print(f"🔴 WER (Word Error Rate): {results['avg_wer']:.4f}")
        if results['avg_wer'] < 0.2:
            print(f"   ✅ Good quality (WER < 0.2)")
        elif results['avg_wer'] < 0.5:
            print(f"   ⚠️  Moderate quality (WER 0.2-0.5)")
        else:
            print(f"   ❌ Low quality (WER > 0.5)")
    print("="*70 + "\n")
    
    # Generate detailed markdown report
    report_md = f"""# 📊 Quality Metrics Report for VSWD Dataset

**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset**: {CSV_PATH}

---

## 📈 Summary Scores

| Chỉ số | Giá trị | Đánh giá |
|--------|--------|---------|
| Sample Size | {results['sample_size']} clips | - |
| Avg Text Length | {results['avg_text_length']:.1f} words | - |
| Uncertain Ratio | {results['uncertain_ratio']:.2f}% | Tỷ lệ cần review thêm |
"""

    if results['avg_bleu'] is not None:
        bleu_rating = "✅ Tốt" if results['avg_bleu'] > 0.6 else ("⚠️ Trung bình" if results['avg_bleu'] > 0.3 else "❌ Kém")
        report_md += f"| BLEU Score | {results['avg_bleu']:.4f} | {bleu_rating} |\n"
    
    if results['avg_wer'] is not None:
        wer_rating = "✅ Tốt" if results['avg_wer'] < 0.2 else ("⚠️ Trung bình" if results['avg_wer'] < 0.5 else "❌ Kém")
        report_md += f"| WER (Word Error Rate) | {results['avg_wer']:.4f} | {wer_rating} |\n"
    
    report_md += f"""
---

## 📝 Chi tiết giải thích các điểm số

### 1️⃣ **BLEU Score** = {results['avg_bleu'] if results['avg_bleu'] is not None else 'N/A'}

**Định nghĩa**: Đo độ tương đồng giữa transcript gốc và pseudo-manual (tham chiếu).

**Cách tính**: So sánh n-gram (từ đơn, cặp từ, v.v.) giữa hai câu. Càng giống nhau → điểm cao.

**Mục đích**: Đánh giá chất lượng text transcript và ASR refine.

**Giải thích kết quả**:
- **> 0.6 (Tốt)**: Transcript rất tương đồng với tham chiếu. Chất lượng ASR/refine tốt.
- **0.3–0.6 (Trung bình)**: Có một số khác biệt nhỏ. Có thể cần refine thêm.
- **< 0.3 (Kém)**: Có sai lệch lớn. Cần review và sửa chữa.

---

### 2️⃣ **WER (Word Error Rate)** = {results['avg_wer'] if results['avg_wer'] is not None else 'N/A'}

**Định nghĩa**: Tỷ lệ lỗi từ = (insertions + deletions + substitutions) / total_ref_words

**Mục đích**: Đo độ chính xác transcript ở mức từ. Càng thấp càng tốt.

**Giải thích kết quả**:
- **< 0.2 (Tốt)**: Transcript gần như chính xác. Chỉ có 10-20% từ khác biệt.
- **0.2–0.5 (Trung bình)**: Có lỗi đáng kể. Cần refine hoặc manual review.
- **> 0.5 (Kém)**: Quá nhiều lỗi (> 50%). Cần xem lại toàn bộ pipeline ASR.

---

### 3️⃣ **Uncertain Ratio** = {results['uncertain_ratio']:.2f}%

**Định nghĩa**: Tỷ lệ clips mà GPT đánh dấu là không chắc chắn ([UNCERTAIN]).

**Mục đích**: Xác định cần bao nhiêu công sức manual review.

**Giải thích kết quả**:
- **< 10% (Tốt)**: Hầu hết clips có thể tin cậy. Ít cần review.
- **10–30% (Trung bình)**: Nên review khoảng 1/3 clips không chắc.
- **> 30% (Kém)**: Quá nhiều không chắc. Cần refine pipeline trước.

---

### 4️⃣ **Avg Text Length** = {results['avg_text_length']:.1f} từ

**Định nghĩa**: Độ dài trung bình một transcript (tính bằng số từ).

**Mục đích**: Biết dataset có clips ngắn (< 5 từ) hay dài (> 20 từ).

**Ý nghĩa**:
- Clips ngắn: Có thể là lỗi ASR hoặc gesture ngắn.
- Clips dài: Yêu cầu model xử lý chuỗi dài hơn.

---

## 📊 Dữ liệu mẫu (Top 10 Clips)

"""
    
    if not df_pseudo.empty:
        sample_rows = df_pseudo.head(10)
        report_md += "| # | Path | Original Text | Pseudo-Manual | Uncertain |\n"
        report_md += "|---|------|---------------|---------------|----------|\n"
        for idx, (_, row) in enumerate(sample_rows.iterrows(), 1):
            orig = str(row.get('original_text', '')).strip()[:60]
            pseudo = str(row.get('pseudo_manual', '')).strip()[:60]
            uncertain = "✓ Yes" if row.get('is_uncertain', False) else "✗ No"
            path = str(row.get('path', '')).strip()[:40]
            report_md += f"| {idx} | {path} | {orig} | {pseudo} | {uncertain} |\n"
    
    report_md += f"""

---

## 💾 Files

- **Full Report**: {OUTPUT_REPORT}
- **Full Data (CSV)**: {OUTPUT_REPORT.replace('.md', '_pseudo.csv')}

---

## ✅ Kết luận

Dựa vào các điểm số trên, bạn có thể:

1. **Nếu BLEU > 0.6 & WER < 0.2 & Uncertain < 10%**: Dataset chất lượng tốt ✅
2. **Nếu BLEU 0.3–0.6 hoặc WER 0.2–0.5**: Cần refine thêm hoặc manual review một phần ⚠️
3. **Nếu BLEU < 0.3 hoặc WER > 0.5**: Cần xem lại pipeline ASR/refine ❌

"""
    
    # Print and save
    print(report_md)
    
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    csv_output = OUTPUT_REPORT.replace('.md', '_pseudo.csv')
    df_pseudo.to_csv(csv_output, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ Report saved to {OUTPUT_REPORT}")
    print(f"✓ Details saved to {csv_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate quality metrics for VSWD")
    parser.add_argument("--no-openai", action="store_true", help="Skip OpenAI-based pseudo manual (use original text)")
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE, help=f"Number of samples to evaluate (default: {SUBSET_SIZE})")
    args = parser.parse_args()
    
    print(f"📊 VSWD Quality Metrics Calculator")
    print(f"📁 Dataset: {CSV_PATH}")
    print(f"📋 Loading metadata...")
    
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['text']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ CSV missing required columns: {required_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Also check for 'path' column if available
    if 'path' not in df.columns:
        df['path'] = [f"clip_{i}" for i in range(len(df))]
    
    # Sample
    subset_size = min(args.subset_size, len(df))
    df_sample = df.sample(n=subset_size, random_state=42)
    print(f"✓ Sampled {subset_size} clips for evaluation")
    
    # Create pseudo-manual
    client = None
    if not args.no_openai and HAS_OPENAI and OPENAI_KEY and OPENAI_KEY != "sk-your-openai-api-key-here":
        try:
            client = OpenAI(api_key=OPENAI_KEY)
            print("✓ OpenAI client initialized (will use GPT for pseudo-manual)")
        except Exception as e:
            print(f"❌ Error initializing OpenAI: {e}")
            sys.exit(1)
    else:
        if args.no_openai:
            print("ℹ️ Skipping OpenAI (--no-openai flag)")
        elif not HAS_OPENAI:
            print("❌ OpenAI library not installed. Install: pip install openai")
            sys.exit(1)
        elif OPENAI_KEY == "sk-your-openai-api-key-here":
            print("❌ OPENAI_KEY not set. Update OPENAI_KEY in quality_metrics.py with your actual key")
            sys.exit(1)
    
    print(f"\n🔄 Creating pseudo-manual...")
    df_pseudo = create_pseudo_manual(client, df_sample)
    
    print(f"📈 Calculating metrics...")
    calculate_metrics(df_pseudo)
    
    print("\n✅ Done!")