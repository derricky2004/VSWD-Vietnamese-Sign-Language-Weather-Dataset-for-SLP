
# Retrieval-based SLP Benchmark on VSWD (test split)

**Generated**: 2026-03-04 13:39:33

| Metric                           | Value    | Note |
|----------------------------------|----------|------|
| Back-translation BLEU            | 0.1445 | Higher is better |
| Back-translation WER             | 0.8538 | Lower is better |
| Recall@1                         | 0.2221 | Higher is better |
| Recall@5                         | 0.6142 | Higher is better |
| Recall@10                        | 0.7831 | Higher is better |
| mAP@10                           | 0.1003 | Higher is better |
| Avg DTW (pose alignment)         | 38.8425 | Lower is better |
| Approx FID (pose distribution)   | 2.72 | Lower is better |

**Setup**:
- Train size: 1241
- Eval size: 1729
- Model: bkai-foundation-models/vietnamese-bi-encoder
- Exclude self-match: False
- Top-k rerank: 50
- Length alpha: 0.15
- Lexical alpha: 0.7
- BM25 alpha: 0.0
- Retrieval mode: text
- Cross-encoder: 
- Cross-encoder alpha: 0.0
- Cross-encoder batch size: 32
- Confidence threshold: 0.5
- DTW samples: 500
- FID samples: 500

**Notes**:
- Word-level BLEU/WER (phù hợp hơn cho Vietnamese sentence-level data).
- Text embedding: bkai-foundation-models/vietnamese-bi-encoder.
- DTW: Fixed-length alignment (100 frames) + per-landmark normalization (improved temporal alignment).
- FID computed on mean-frame features (subset).

Outputs:
- retrieval_results.csv
- retrieval_benchmark_report.md
