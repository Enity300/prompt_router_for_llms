import os
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for SS-GER Router System"""

    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    WIZARDMATH_API_KEY: Optional[str] = os.getenv("WIZARDMATH_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.78"))
    SENTENCE_TRANSFORMER_MODEL: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", "./data/db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "expertise-manifolds")

    CODING_DATASET_SIZE: int = int(os.getenv("CODING_DATASET_SIZE", "14000"))
    MATH_DATASET_SIZE: int = int(os.getenv("MATH_DATASET_SIZE", "7000"))
    GENERAL_DATASET_SIZE: int = int(os.getenv("GENERAL_DATASET_SIZE", "7000"))
    EVALUATION_SET_SIZE: int = int(os.getenv("EVALUATION_SET_SIZE", "3000"))

    TOP_K_NEIGHBORS: int = int(os.getenv("TOP_K_NEIGHBORS", "7"))
    CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "100"))
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))

    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    NUMPY_SEED: int = int(os.getenv("NUMPY_SEED", "42"))
    TORCH_SEED: int = int(os.getenv("TORCH_SEED", "42"))
    ENABLE_REPRODUCIBILITY: bool = os.getenv("ENABLE_REPRODUCIBILITY", "true").lower() == "true"

    FALLBACK_CODING_PROMPTS: List[str] = [
        "Write a Python function to implement binary search",
        "Create a class for a binary tree",
        "Implement a merge sort algorithm",
        "Write a function to find the longest palindromic substring",
        "Create a graph class with DFS and BFS",
        "Implement a LRU cache",
        "Write a function to detect cycles in a linked list",
        "Create a priority queue",
        "Implement the quicksort algorithm",
        "Write a function to validate a binary search tree"
    ]

    FALLBACK_MATH_PROMPTS: List[str] = [
        "Solve for x: 2x + 5 = 15",
        "What is the derivative of x^3?",
        "Calculate the area of a circle with radius 7",
        "Find the prime factorization of 84",
        "Solve the quadratic equation x^2 - 5x + 6 = 0",
        "What is the integral of 3x^2?",
        "Probability of rolling two dice and getting 7",
        "Slope of the line passing through (2,3) and (5,9)",
        "What is 15% of 240?",
        "Solve the system: 2x + y = 7, x - y = 2"
    ]

    FALLBACK_GENERAL_PROMPTS: List[str] = [
        "What is the capital of France?",
        "Explain the theory of relativity",
        "What are the main causes of climate change?",
        "Describe photosynthesis",
        "History of the Internet?",
        "Explain how vaccines work",
        "Benefits of renewable energy?",
        "Describe the structure of an atom",
        "What is artificial intelligence?",
        "Explain the water cycle",
        "Significance of the Renaissance?",
        "How do economies work?",
        "What is quantum mechanics?",
        "Describe the human digestive system",
        "What is machine learning?",
        "Explain evolution",
        "What are human rights?",
        "Describe the solar system",
        "What is democracy?",
        "Explain how the brain works"
    ]


    USE_LOCAL_MODELS: bool = os.getenv("USE_LOCAL_MODELS", "true").lower() == "true"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LOCAL_CODING_MODEL: str = os.getenv("LOCAL_CODING_MODEL", "deepseek-coder:1.3b")
    LOCAL_MATH_MODEL: str = os.getenv("LOCAL_MATH_MODEL", "qwen2-math:1.5b")
    LOCAL_GENERAL_MODEL: str = os.getenv("LOCAL_GENERAL_MODEL", "llama3.2:1b")
    LOCAL_FALLBACK_MODEL: str = os.getenv("LOCAL_FALLBACK_MODEL", "phi3:mini")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "100"))

    @classmethod
    def get_example_env_content(cls) -> str:
        """Returns example .env file content for setup"""
        return """# API Keys for Specialist LLMs
DEEPSEEK_API_KEY=your_deepseek_api_key_here
WIZARDMATH_API_KEY=your_wizardmath_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

SIMILARITY_THRESHOLD=0.78
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
CHROMADB_PATH=./data/db
COLLECTION_NAME=expertise-manifolds

CODING_DATASET_SIZE=2000
MATH_DATASET_SIZE=2000
GENERAL_DATASET_SIZE=1000

TOP_K_NEIGHBORS=3
CACHE_SIMILARITY_THRESHOLD=0.95

RANDOM_SEED=42
NUMPY_SEED=42
TORCH_SEED=42
ENABLE_REPRODUCIBILITY=true

DEEPSEEK_API_BASE=https://api.deepseek.com/v1
WIZARDMATH_API_BASE=https://api.wizardmath.com/v1
OPENAI_API_BASE=https://api.openai.com/v1

USE_LOCAL_MODELS=true
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_CODING_MODEL=deepseek-coder:1.3b
LOCAL_MATH_MODEL=qwen2-math:1.5b
LOCAL_GENERAL_MODEL=llama3.2:1b
LOCAL_FALLBACK_MODEL=phi3:mini
OLLAMA_TIMEOUT=30"""

    @classmethod
    def validate_config(cls) -> bool:
        """Validates that required configuration is present"""
        if cls.USE_LOCAL_MODELS:
            print("✅ Local Ollama models enabled - no API keys required!")
            return True

        missing_keys = []

        if not cls.DEEPSEEK_API_KEY:
            missing_keys.append("DEEPSEEK_API_KEY")
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")

        if missing_keys:
            print(f"Warning: Missing required environment variables: {', '.join(missing_keys)}")
            print("Please set these in your .env file or set USE_LOCAL_MODELS=true for Ollama.")
            return False

        return True

    @classmethod
    def initialize_seeds(cls):
        """Initialize all random seeds for reproducibility"""
        if not cls.ENABLE_REPRODUCIBILITY:
            print("⚠️ Reproducibility disabled - results may vary between runs")
            return

        try:
            import random
            random.seed(cls.RANDOM_SEED)

            import numpy as np
            np.random.seed(cls.NUMPY_SEED)

            try:
                import torch
                torch.manual_seed(cls.TORCH_SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(cls.TORCH_SEED)
                    torch.cuda.manual_seed_all(cls.TORCH_SEED)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except ImportError:
                pass

            try:
                import tensorflow as tf
                tf.random.set_seed(cls.TORCH_SEED)
            except ImportError:
                pass

            print(f"✅ Reproducibility enabled - using seeds: random={cls.RANDOM_SEED}, numpy={cls.NUMPY_SEED}, torch={cls.TORCH_SEED}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize seeds: {e}")

config = Config()
