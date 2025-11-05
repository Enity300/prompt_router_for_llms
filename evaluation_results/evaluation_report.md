# Comprehensive Evaluation Report

**Date**: 2025-11-05 16:11:33

## 1. Overall Performance Comparison

| Model | Accuracy | Avg Latency (ms) | Throughput (q/s) |
|---|---|---|---|
| **Semantic Router** | **0.991** | 38.961 | 25.67 |
| **TF-IDF + SVM** | **0.950** | 0.330 | 3033.88 |
| **Rule-based Keywords** | **0.553** | 0.015 | 65900.03 |
| **Most Frequent Class** | **0.333** | 0.000 | 3999999.81 |
| **Random Classifier** | **0.313** | 0.000 | 3791469.26 |

## 2. Per-Category Performance (F1-Score)

| Model | coding | general_knowledge | math |
|---| --- | --- | --- |
| Semantic Router | 0.994 | 0.986 | 0.993 |
| Random Classifier | 0.322 | 0.318 | 0.299 |
| Most Frequent Class | 0.500 | 0.000 | 0.000 |
| Rule-based Keywords | 0.759 | 0.604 | 0.093 |
| TF-IDF + SVM | 0.990 | 0.929 | 0.931 |

## 3. Statistical Significance (vs Semantic Router)

| Comparison vs Baseline | t-statistic | p-value | Significant (p < 0.05) |
|---|---|---|---|
| Semantic Router vs Random Classifier | 82.071 | 0.0000 | ✅ Yes |
| Semantic Router vs Most Frequent Class | 184.374 | 0.0000 | ✅ Yes |
| Semantic Router vs Rule-based Keywords | 51.190 | 0.0000 | ✅ Yes |
| Semantic Router vs TF-IDF + SVM | 11.631 | 0.0003 | ✅ Yes |

## 4. Token Length Impact Analysis

### Semantic Router

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 5 | 100.00% |
| 6-10 | 143 | 98.60% |
| 11-20 | 221 | 96.83% |
| 21-50 | 359 | 99.44% |
| 51+ | 472 | 100.00% |

### Rule-based Keywords

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 5 | 100.00% |
| 6-10 | 143 | 99.30% |
| 11-20 | 221 | 93.21% |
| 21-50 | 359 | 26.46% |
| 51+ | 472 | 45.76% |

### TF-IDF + SVM

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 5 | 80.00% |
| 6-10 | 143 | 91.61% |
| 11-20 | 221 | 88.24% |
| 21-50 | 359 | 96.66% |
| 51+ | 472 | 98.09% |

