# 🎯 Semantic Router for Specialized LLMs

A high-performance semantic routing system that intelligently directs user queries to specialized AI models based on content classification. Built with ChromaDB vector database and powered by sentence transformers for accurate query routing.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Details](#dataset-details)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)

---

## 🎓 Overview

This project implements an intelligent routing system that classifies user queries into three categories (**Coding**, **Math**, **General Knowledge**) and routes them to specialized LLM models for optimal performance. The system uses semantic embeddings and vector similarity search to achieve high accuracy routing.

### Why Semantic Routing?

- **Better Accuracy**: Specialized models outperform general-purpose models in their domain
- **Cost Efficiency**: Route complex queries to powerful models, simple ones to lightweight models
- **Scalability**: Easily add new categories and models
- **Speed**: Fast vector similarity search with ChromaDB

---

## ✨ Features

- 🔍 **Semantic Classification**: Uses `all-MiniLM-L6-v2` sentence transformer for query embedding
- 📊 **Variable Sample Database**: Balanced dataset across coding, math, and general knowledge that can be increased depending on user
- 🎯 **5 Dataset Integration**: Combines KodCode, GSM8K, TriviaQA, and LLM-routing datasets
- 🧠 **Memory Efficient**: Sequential processing prevents memory overload
- 📈 **Comprehensive Evaluation**: Accuracy, latency, confusion matrices, and visualization
- 🔄 **Local & Cloud Models**: Supports both Ollama local models and cloud APIs

---

## 🏗️ Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Semantic Router (ChromaDB)     │
│  - Embed query                  │
│  - Find K-nearest neighbors     │
│  - Classify category            │
└────────┬────────────────────────┘
         │
    ┌────┴────┬─────────────┐
    ▼         ▼             ▼
┌────────┐ ┌────────┐  ┌──────────┐
│ Coding │ │  Math  │  │ General  │
│ Model  │ │ Model  │  │  Model   │
└────────┘ └────────┘  └──────────┘
```

### Core Components

1. **Expertise Database Builder** (`src/build_expertise_db.py`)
   - Loads and processes 5 datasets
   - Creates 384-dimensional embeddings
   - Stores in ChromaDB vector database

2. **Semantic Router** (`src/semantic_router.py`)
   - Performs K-NN search (K=3)
   - Classification via majority voting
   - Cosine similarity threshold: 0.78

3. **Specialist Clients** (`src/specialist_clients.py`)
   - Coding: `deepseek-coder:1.3b`
   - Math: `qwen2-math:1.5b`
   - General: `llama3.2:1b`

4. **Evaluation System** (`src/comprehensive_evaluation.py`)
   - Tests on variable unseen samples (user can also define the number of test samples, similar to variable sample database)
   - Generates performance metrics
   - Creates visualization charts

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Ollama (for local models) - [Install Ollama](https://ollama.ai/)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Pull Ollama Models (Optional - for local inference)

```bash
ollama pull deepseek-coder:1.3b
ollama pull qwen2-math:1.5b
ollama pull llama3.2:1b
ollama pull phi3:mini
```

### Step 3: Configure Environment (Optional)

Create a `.env` file for custom configuration:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings
# Set USE_LOCAL_MODELS=true for Ollama
# Or add API keys for cloud models
```

---

## 📖 Usage

### 1️⃣ Build the Expertise Database

This creates the vector database with variable embeddings:

```bash
python src/build_expertise_db.py
```

**Expected Output:**
- ChromaDB at `./data/db/`
- Evaluation dataset at `./evaluation_dataset.json`

### 2️⃣ Run Evaluation

Test the router's performance:

```bash
python src/comprehensive_evaluation.py
```

### 3️⃣ Interactive Demo (Streamlit)

Launch the web interface:

```bash
streamlit run streamlit_app.py
```

Open browser at `http://localhost:8501`

**Features:**
- Real-time query routing
- Category confidence scores
- Model response generation
- Visual similarity heatmaps

## 📊 Dataset Details

### Total: 18,000 Training + 6,000 Evaluation Samples

| Category | Datasets | Training Samples | Eval Samples | Distribution |
|----------|----------|------------------|--------------|--------------|
| **Coding** | KodCode/KodCode-V1 | 6,000 | 2,000 | 33.3% |
| **Math** | GSM8K + LLM-routing | 6,000 (3K+3K) | 2,000 | 33.3% |
| **General** | TriviaQA + LLM-routing | 6,000 (3K+3K) | 2,000 | 33.3% |

### Dataset Breakdown

#### 1. **Coding Dataset - KodCode** (6,000 samples)
- **Source**: [KodCode/KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1)
- **Content**: Programming questions across multiple languages
- **Examples**: "Write a Python function to implement binary search"

#### 2. **Math Dataset - GSM8K** (3,000 samples)
- **Source**: [gsm8k](https://huggingface.co/datasets/gsm8k)
- **Content**: Grade school math word problems
- **Examples**: "Janet has 16 marbles. She lost 7. How many does she have now?"

#### 3. **Math Dataset - LLM Routing** (3,000 samples)
- **Source**: [jeanvydes/llm-routing-text-classification](https://huggingface.co/datasets/jeanvydes/llm-routing-text-classification)
- **Filter**: `task == 'math'`
- **Content**: Mathematical reasoning problems

#### 4. **General Dataset - TriviaQA** (3,000 samples)
- **Source**: [trivia_qa](https://huggingface.co/datasets/trivia_qa)
- **Content**: Factual questions across various topics
- **Examples**: "What is the capital of France?"

#### 5. **General Dataset - LLM Routing (Balanced)** (3,000 samples)
- **Source**: [jeanvydes/llm-routing-text-classification](https://huggingface.co/datasets/jeanvydes/llm-routing-text-classification)
- **Filter**: 6 classes with **equal distribution (500 each)**:
  - `conversation` (500)
  - `science` (500)
  - `toxic_harmful` (500)
  - `logical_reasoning` (500)
  - `sex` (500)
  - `creative_writing` (500)

### Evaluation Dataset Creation

**Key Feature**: Evaluation samples are **subsets of the training database**

- After all 18,000 samples are stored in ChromaDB
- Query ChromaDB by category
- Randomly select 2,000 samples from each category (6,000 total)
- **Advantage**: All eval samples guaranteed to exist in the vector database
- Ensures realistic routing performance measurement

## 📁 Project Structure

```
prompt_router_for_llms/
├── src/
│   ├── build_expertise_db.py        # Database builder (18K embeddings)
│   ├── semantic_router.py           # K-NN routing logic
│   ├── specialist_clients.py        # LLM API wrappers
│   ├── comprehensive_evaluation.py  # Performance testing
│   ├── embedding_visualization.py   # t-SNE/PCA plots
│   ├── gpu_monitor.py               # Resource tracking
│   └── test_reproducibility.py      # Seed verification
├── config/                           # Configuration files
├── data/
│   └── db/                          # ChromaDB vector store
├── evaluation_results/              # Performance metrics & charts
├── embedding_visualizations/        # Interactive plots
├── config.py                        # Main configuration
├── requirements.txt                 # Python dependencies
├── streamlit_app.py                # Web UI
├── main.py                         # CLI interface
└── README.md                       # This file
```

---

## 🔧 Advanced Usage

### Custom Dataset Integration

Add your own dataset in `src/build_expertise_db.py`:

```python
def load_custom_dataset(self) -> List[Dict[str, Any]]:
    dataset_list = list(load_dataset("your-dataset-name", split="train"))
    db_samples = []
    
    for item in tqdm(dataset_list[:YOUR_SIZE], desc="Processing custom samples"):
        prompt = item.get('text_field')
        if prompt and prompt.strip():
            db_samples.append({
                "prompt": prompt.strip(),
                "category": "your_category",
                "source": "custom-dataset"
            })
    
    return db_samples
```

Then add to `datasets_to_process` list in `build_database()`.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **HuggingFace** for dataset hosting and sentence-transformers library
- **ChromaDB** for efficient vector database implementation
- **Ollama** for local LLM inference capabilities
- **Streamlit** for interactive web interface

---

## 📞 Contact

**Project Maintainer**: Enity300

**Repository**: [github.com/Enity300/prompt_router_for_llms](https://github.com/Enity300/prompt_router_for_llms)

**Issues**: [GitHub Issues](https://github.com/Enity300/prompt_router_for_llms/issues)

---

