# Comprehensive Evaluation Report

**Date**: 2025-11-06 01:09:33

## 1. Overall Performance Comparison

| Model | Accuracy | Avg Latency (ms) | Throughput (q/s) |
|---|---|---|---|
| **Semantic Router** | **1.000** | 43.915 | 22.77 |
| **TF-IDF + Random Forest** | **0.962** | 0.357 | 2803.27 |
| **TF-IDF + Logistic Regression** | **0.955** | 0.016 | 64229.34 |
| **TF-IDF + SVM** | **0.949** | 0.488 | 2050.69 |
| **TF-IDF + CatBoost** | **0.928** | 1.950 | 512.93 |
| **TF-IDF + Naive Bayes** | **0.856** | 0.001 | 746013.49 |
| **Rule-based Keywords** | **0.681** | 0.026 | 38669.88 |
| **Random Classifier** | **0.352** | 0.000 | 5459508.64 |
| **Most Frequent Class** | **0.333** | 0.000 | 8166042.96 |

## 2. Per-Category Performance (F1-Score)

| Model | coding | general_knowledge | math |
|---| --- | --- | --- |
| Semantic Router | 1.000 | 1.000 | 1.000 |
| Random Classifier | 0.358 | 0.356 | 0.343 |
| Most Frequent Class | 0.500 | 0.000 | 0.000 |
| Rule-based Keywords | 0.790 | 0.717 | 0.465 |
| TF-IDF + SVM | 0.981 | 0.937 | 0.929 |
| TF-IDF + Logistic Regression | 0.978 | 0.948 | 0.938 |
| TF-IDF + Random Forest | 0.992 | 0.949 | 0.946 |
| TF-IDF + Naive Bayes | 0.864 | 0.776 | 0.915 |
| TF-IDF + CatBoost | 0.985 | 0.908 | 0.889 |

## 3. Statistical Significance (vs Semantic Router)

| Comparison vs Baseline | t-statistic | p-value | Significant (p < 0.05) |
|---|---|---|---|
| Semantic Router vs Random Classifier | 88.753 | 0.0000 | ✅ Yes |
| Semantic Router vs Most Frequent Class | 141.502 | 0.0000 | ✅ Yes |
| Semantic Router vs Rule-based Keywords | 38.627 | 0.0000 | ✅ Yes |
| Semantic Router vs TF-IDF + SVM | 18.326 | 0.0001 | ✅ Yes |
| Semantic Router vs TF-IDF + Logistic Regression | 23.341 | 0.0000 | ✅ Yes |
| Semantic Router vs TF-IDF + Random Forest | 11.642 | 0.0003 | ✅ Yes |
| Semantic Router vs TF-IDF + Naive Bayes | 15.019 | 0.0001 | ✅ Yes |
| Semantic Router vs TF-IDF + CatBoost | 12.780 | 0.0002 | ✅ Yes |

## 4. Token Length Impact Analysis

### Semantic Router

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 38 | 100.00% |
| 6-10 | 286 | 100.00% |
| 11-20 | 355 | 100.00% |
| 21-50 | 685 | 100.00% |
| 51+ | 1035 | 100.00% |

### Rule-based Keywords

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 38 | 100.00% |
| 6-10 | 286 | 91.26% |
| 11-20 | 355 | 82.82% |
| 21-50 | 685 | 47.59% |
| 51+ | 1035 | 69.18% |

### TF-IDF + SVM

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 38 | 97.37% |
| 6-10 | 286 | 96.50% |
| 11-20 | 355 | 89.86% |
| 21-50 | 685 | 92.99% |
| 51+ | 1035 | 97.29% |

### TF-IDF + Logistic Regression

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 38 | 97.37% |
| 6-10 | 286 | 97.20% |
| 11-20 | 355 | 94.37% |
| 21-50 | 685 | 92.85% |
| 51+ | 1035 | 97.00% |

### TF-IDF + Random Forest

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 38 | 100.00% |
| 6-10 | 286 | 96.50% |
| 11-20 | 355 | 92.96% |
| 21-50 | 685 | 94.45% |
| 51+ | 1035 | 98.26% |

### TF-IDF + CatBoost

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 38 | 100.00% |
| 6-10 | 286 | 96.15% |
| 11-20 | 355 | 92.11% |
| 21-50 | 685 | 86.72% |
| 51+ | 1035 | 95.85% |

