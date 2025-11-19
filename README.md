# 🎯 Semantic Router: Intelligent Query Classification System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance semantic routing system that intelligently classifies text queries into specialized categories (coding, math, general knowledge) using sentence embeddings and vector similarity search.

## 🌟 Key Features

- **🚀 97.9% Accuracy** - Best-in-class performance using Sentence Transformer + CatBoost
- **⚡ Fast Inference** - Sub-10ms routing with LRU caching
- **🧠 Semantic Understanding** - Goes beyond keyword matching to understand query meaning
- **📊 Comprehensive Evaluation** - Rigorous benchmarking against 9 baseline models
- **🔄 No Data Leakage** - Proper train/test splits with cross-validation
- **🎨 Rich Visualizations** - Confusion matrices, accuracy charts, token length analysis

## 📈 Performance Benchmarks

| Model | Accuracy | Avg Latency |
|-------|----------|-------------|
| **Sentence Transformer + CatBoost** | **97.9%** | 8.2ms |
| TF-IDF + Random Forest | 96.3% | 3.1ms |
| TF-IDF + SVM | 95.9% | 2.8ms |
| TF-IDF + Logistic Regression | 95.8% | 2.5ms |
| TF-IDF + CatBoost | 93.0% | 7.5ms |
| TF-IDF + Naive Bayes | 84.5% | 2.1ms |
| Rule-based Keywords | 77.0% | 0.5ms |
| **Semantic Router (Vector DB)** | **66.3%** | 9.8ms |
| Most Frequent Class | 40.8% | 0.1ms |
| Random Classifier | 32.5% | 0.1ms |

> **Note:** While the Semantic Router achieves 66.3% accuracy, the supervised Sentence Transformer + CatBoost approach (using the same embeddings) achieves 97.9%, demonstrating the power of combining semantic embeddings with supervised learning.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Sentence Transformer                           │
│           (all-MiniLM-L6-v2)                               │
│         384-dimensional embeddings                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  LRU Cache Check                            │
│            (95% similarity threshold)                       │
└────────┬────────────────────────────────────┬───────────────┘
         │ Cache Hit                          │ Cache Miss
         ▼                                    ▼
    ┌─────────┐                    ┌──────────────────────┐
    │ Return  │                    │   ChromaDB Query     │
    │ Cached  │                    │  (Cosine Distance)   │
    │Category │                    │   Top-3 Neighbors    │
    └─────────┘                    └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │  Multi-Neighbor      │
                                   │     Voting           │
                                   │  (Majority Wins)     │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │ Confidence           │
                                   │ Calibration          │
                                   │ (Threshold: 0.78)    │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │  Update Cache        │
                                   │  (LRU Eviction)      │
                                   └──────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/semantic-router.git
cd semantic-router
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Build the expertise database**
```bash
python src/build_expertise_db.py
```

This will:
- Download datasets (KodCode, GSM8K, TriviaQA, LLM-Routing)
- Generate ~12,000 embeddings for the database
- Create ~10,000 evaluation samples (with NO overlap)
- Takes ~10-15 minutes on first run

### Basic Usage

```python
from src.semantic_router import SemanticRouter

# Initialize router
router = SemanticRouter()

# Route a query
result = router.route("Write a Python function to sort a list")

print(f"Category: {result['category']}")           # 'coding'
print(f"Confidence: {result['confidence']:.2f}")   # 0.89
print(f"Explanation: {result['explanation']}")     # Human-readable reasoning
```

### Command Line Interface

```bash
# Route a single query
python main.py route "Calculate the derivative of x^2"

# Interactive mode
python main.py interactive

# Run test suite
python main.py test

# View statistics
python main.py stats
```

## 📊 Comprehensive Evaluation

Run the full evaluation suite to benchmark against all baseline models:

```bash
python src/comprehensive_evaluation.py
```

This generates:
- **Cross-validation results** (5-fold CV)
- **Statistical significance tests** (paired t-tests)
- **Confusion matrices** for all models
- **Token length analysis** (performance vs query length)
- **Publication-ready visualizations**

Results saved to `evaluation_results/`:
```
evaluation_results/
├── accuracy_comparison.png
├── latency_comparison.png
├── confusion_matrix_semantic_router.png
├── confusion_matrix_tfidf_svm.png
├── token_length_impact.png
└── evaluation_report.md
```

## 🎯 Categories

The router classifies queries into three categories:

### 1. **Coding** 🖥️
Programming, algorithms, debugging, software development
```python
"Write a binary search algorithm"
"Debug this JavaScript code"
"Explain recursion with examples"
```

### 2. **Math** 📐
Calculations, equations, mathematical concepts
```python
"Solve x^2 + 5x + 6 = 0"
"What is the derivative of sin(x)?"
"Calculate the area of a circle"
```

### 3. **General Knowledge** 🌍
Science, history, general information
```python
"What is photosynthesis?"
"Who wrote 1984?"
"Explain climate change"
```

## 🔧 Configuration

Edit `config.py` or set environment variables:

```python
# Model Configuration
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.78  # Routing confidence threshold

# Database Configuration
CHROMADB_PATH = "./data/db"
COLLECTION_NAME = "expertise-manifolds"

# Dataset Sizes
CODING_DATASET_SIZE = 6000
MATH_DATASET_SIZE = 3000
GENERAL_DATASET_SIZE = 3000
EVALUATION_SET_SIZE = 2000

# Cache Configuration
CACHE_SIZE = 100
CACHE_SIMILARITY_THRESHOLD = 0.95

# Performance Tuning
TOP_K_NEIGHBORS = 3
EMBEDDING_BATCH_SIZE = 32
```

## 📁 Project Structure

```
semantic-router/
├── src/
│   ├── semantic_router.py           # Core routing engine
│   ├── build_expertise_db.py        # Database builder
│   ├── comprehensive_evaluation.py  # Evaluation framework
│   ├── specialist_clients.py        # LLM client integrations
│   └── utils/
│       └── model_loader.py          # Singleton model loader
├── config.py                        # Configuration settings
├── main.py                          # CLI interface
├── requirements.txt                 # Python dependencies
├── evaluation_dataset.json          # Test data (generated)
└── data/
    └── db/                          # ChromaDB storage
```

## 🧪 How It Works

### 1. **Embedding Generation**
Queries are converted to 384-dimensional vectors using `all-MiniLM-L6-v2`:
```python
embedding = model.encode("Write a quicksort function")
# → [0.23, -0.45, 0.12, ..., 0.67]  (384 dimensions)
```

### 2. **Vector Normalization**
Embeddings are normalized to unit length for consistent similarity:
```python
norm = np.linalg.norm(embedding)
embedding = embedding / norm  # Now ||embedding|| = 1.0
```

### 3. **Similarity Search**
ChromaDB finds the 3 nearest neighbors using cosine distance:
```python
results = collection.query(
    query_embeddings=[embedding],
    n_results=3,
    metric="cosine"
)
```

### 4. **Multi-Neighbor Voting**
The router uses majority voting from top-3 neighbors:
```
Neighbors:
1. "Implement merge sort" (coding) - similarity: 0.89
2. "Write recursive function" (coding) - similarity: 0.85
3. "Debug algorithm" (coding) - similarity: 0.82

Vote: coding (3/3) → High confidence
```

### 5. **Confidence Calibration**
```python
if similarity > 0.95:  # Very high confidence
if similarity > 0.85:  # High confidence
if similarity > 0.78:  # Medium confidence (threshold)
else:                  # Low confidence → fallback to general_knowledge
```

## 🎓 Key Innovations

### 1. **No Data Leakage**
- Database and evaluation sets are split BEFORE any processing
- Ensures honest performance metrics
- Prevents the router from being tested on seen data

### 2. **LRU Caching**
- Caches recent queries with 95% similarity threshold
- Proper LRU eviction using `OrderedDict`
- ~100x speedup for cache hits

### 3. **Singleton Model Loading**
- Single shared model instance across all components
- Reduces memory usage by 3x
- Faster startup time

### 4. **Embedding Normalization**
- Consistent normalization in both database and queries
- Ensures accurate cosine similarity calculations
- Critical for routing accuracy

## 📊 Evaluation Methodology

### Cross-Validation (5-Fold)
- Splits training data into 5 parts
- Each part validated once
- Ensures model generalization

### Statistical Significance
- Paired t-tests compare router vs baselines
- p-value < 0.05 indicates significant difference
- Proves improvements aren't due to chance

### Held-Out Test Set (40%)
- Completely unseen data for final evaluation
- Prevents overfitting
- Provides unbiased accuracy estimates

### Token Length Analysis
- Tests performance across query lengths
- Identifies weaknesses (short vs long queries)
- Buckets: 1-5, 6-10, 11-20, 21-50, 51+ tokens

## 🔬 Baseline Models

We compare against 9 baseline approaches:

**Dummy Baselines:**
- Random Classifier (33% accuracy)
- Most Frequent Class (41% accuracy)

**Rule-Based:**
- Keyword Matching (77% accuracy)

**Traditional ML (TF-IDF + Classifier):**
- Naive Bayes (84.5% accuracy)
- CatBoost (93.0% accuracy)
- Logistic Regression (95.8% accuracy)
- SVM (95.9% accuracy)
- Random Forest (96.3% accuracy)

**Deep Learning:**
- **Sentence Transformer + CatBoost (97.9% accuracy)** ⭐ Best

## 🚧 Limitations & Future Work

### Current Limitations
1. **Supervised approach outperforms unsupervised** - The vector DB router (66.3%) is beaten by supervised learning (97.9%)
2. **Fixed categories** - Only supports 3 categories (coding, math, general)
3. **English only** - Model trained on English text
4. **Cold start** - First query is slow (~500ms) due to model loading

### Future Improvements
- [ ] Add supervised fine-tuning layer
- [ ] Support dynamic category addition
- [ ] Multi-language support
- [ ] Hybrid approach (vector DB + classifier)
- [ ] Real-time learning from user feedback
- [ ] GPU acceleration for batch processing

## 🙏 Acknowledgments

- **Sentence Transformers** - For the excellent embedding models
- **ChromaDB** - For the fast vector database
- **CatBoost** - For the high-performance gradient boosting
- **Hugging Face** - For dataset hosting and model hub
