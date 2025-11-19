# Comprehensive Evaluation Report

**Date**: 2025-11-17 08:38:04

## 1. Overall Performance Comparison

| Model | Accuracy | Avg Latency (ms) |
|---|---|---|
| **Sentence Transformer + CatBoost Router** | **0.979** | 35.968 |
| **TF-IDF + Random Forest** | **0.963** | 2.459 |
| **TF-IDF + SVM** | **0.959** | 0.663 |
| **TF-IDF + Logistic Regression** | **0.958** | 0.278 |
| **TF-IDF + CatBoost** | **0.930** | 1.602 |
| **TF-IDF + Naive Bayes** | **0.845** | 0.313 |
| **Rule-based Keywords** | **0.770** | 0.019 |
| **Semantic Router** | **0.663** | 67.085 |
| **Most Frequent Class** | **0.408** | 0.130 |
| **Random Classifier** | **0.325** | 0.139 |

## 2. Per-Category Performance (F1-Score)

| Model | coding | general_knowledge | math |
|---| --- | --- | --- |
| Semantic Router | 0.706 | 0.692 | 0.446 |
| Sentence Transformer + CatBoost Router | 0.988 | 0.977 | 0.965 |
| Random Classifier | 0.360 | 0.342 | 0.254 |
| Most Frequent Class | 0.579 | 0.000 | 0.000 |
| Rule-based Keywords | 0.839 | 0.771 | 0.624 |
| TF-IDF + SVM | 0.986 | 0.951 | 0.919 |
| TF-IDF + Logistic Regression | 0.986 | 0.950 | 0.919 |
| TF-IDF + Random Forest | 0.993 | 0.954 | 0.920 |
| TF-IDF + Naive Bayes | 0.868 | 0.795 | 0.878 |
| TF-IDF + CatBoost | 0.987 | 0.917 | 0.834 |

## 3. Statistical Significance (vs Sentence Transformer + CatBoost Router)

| Comparison | t-statistic | p-value | Significant (p < 0.05) |
|---|---|---|---|
| Sentence Transformer + CatBoost Router vs Semantic Router | 37.615 | 0.0000 | ✅ Yes |
| Sentence Transformer + CatBoost Router vs Random Classifier | 87.860 | 0.0000 | ✅ Yes |
| Sentence Transformer + CatBoost Router vs Most Frequent Class | 86.075 | 0.0000 | ✅ Yes |
| Sentence Transformer + CatBoost Router vs Rule-based Keywords | 56.492 | 0.0000 | ✅ Yes |
| Sentence Transformer + CatBoost Router vs TF-IDF + SVM | 5.127 | 0.0069 | ✅ Yes |
| Sentence Transformer + CatBoost Router vs TF-IDF + Logistic Regression | 5.785 | 0.0044 | ✅ Yes |
| Sentence Transformer + CatBoost Router vs TF-IDF + Random Forest | 2.622 | 0.0587 | ❌ No |
| Sentence Transformer + CatBoost Router vs TF-IDF + Naive Bayes | 17.282 | 0.0001 | ✅ Yes |
| Sentence Transformer + CatBoost Router vs TF-IDF + CatBoost | 10.137 | 0.0005 | ✅ Yes |

## 4. Token Length Impact Analysis

### Semantic Router

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 93.55% |
| 6-10 | 398 | 96.73% |
| 11-20 | 462 | 92.86% |
| 21-50 | 591 | 55.16% |
| 51+ | 1387 | 52.13% |

### Sentence Transformer + CatBoost Router

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 95.16% |
| 6-10 | 398 | 97.99% |
| 11-20 | 462 | 97.19% |
| 21-50 | 591 | 95.26% |
| 51+ | 1387 | 99.28% |

### Random Classifier

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 29.03% |
| 6-10 | 398 | 29.65% |
| 11-20 | 462 | 32.68% |
| 21-50 | 591 | 33.50% |
| 51+ | 1387 | 33.02% |

### Most Frequent Class

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 0.00% |
| 6-10 | 398 | 1.01% |
| 11-20 | 462 | 7.14% |
| 21-50 | 591 | 25.38% |
| 51+ | 1387 | 71.81% |

### Rule-based Keywords

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 93.55% |
| 6-10 | 398 | 91.46% |
| 11-20 | 462 | 84.42% |
| 21-50 | 591 | 63.79% |
| 51+ | 1387 | 75.41% |

### TF-IDF + SVM

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 98.39% |
| 6-10 | 398 | 95.48% |
| 11-20 | 462 | 93.72% |
| 21-50 | 591 | 92.55% |
| 51+ | 1387 | 97.98% |

### TF-IDF + Logistic Regression

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 93.55% |
| 6-10 | 398 | 95.98% |
| 11-20 | 462 | 94.59% |
| 21-50 | 591 | 91.88% |
| 51+ | 1387 | 97.91% |

### TF-IDF + Random Forest

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 91.94% |
| 6-10 | 398 | 95.98% |
| 11-20 | 462 | 95.67% |
| 21-50 | 591 | 91.71% |
| 51+ | 1387 | 98.70% |

### TF-IDF + Naive Bayes

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 40.32% |
| 6-10 | 398 | 59.30% |
| 11-20 | 462 | 72.73% |
| 21-50 | 591 | 85.28% |
| 51+ | 1387 | 97.33% |

### TF-IDF + CatBoost

| Token Range | Sample Count | Accuracy |
|---|---|---|
| 1-5 | 62 | 93.55% |
| 6-10 | 398 | 94.22% |
| 11-20 | 462 | 93.94% |
| 21-50 | 591 | 83.59% |
| 51+ | 1387 | 96.25% |

