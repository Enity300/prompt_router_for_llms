import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Configuration class for SS-GER Router System"""
    
    # API Keys
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    WIZARDMATH_API_KEY: Optional[str] = os.getenv("WIZARDMATH_API_KEY") 
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Core Configuration - Enhanced for 5K Embeddings
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.78"))
    SENTENCE_TRANSFORMER_MODEL: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", "./data/db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "expertise-manifolds")
    
    # Dataset Configuration - Conservative Scaling (5,000 total embeddings)
    CODING_DATASET_SIZE: int = int(os.getenv("CODING_DATASET_SIZE", "6000"))
    MATH_DATASET_SIZE: int = int(os.getenv("MATH_DATASET_SIZE", "3000"))
    GENERAL_DATASET_SIZE: int = int(os.getenv("GENERAL_DATASET_SIZE", "3000"))
    
    # Routing Configuration - Enhanced for Larger Dataset
    TOP_K_NEIGHBORS: int = int(os.getenv("TOP_K_NEIGHBORS", "3"))
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
    
    # Reproducibility Configuration
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    NUMPY_SEED: int = int(os.getenv("NUMPY_SEED", "42"))
    TORCH_SEED: int = int(os.getenv("TORCH_SEED", "42"))
    ENABLE_REPRODUCIBILITY: bool = os.getenv("ENABLE_REPRODUCIBILITY", "true").lower() == "true"
    
    
    # Local Ollama Models Configuration
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

# Configuration Parameters - Enhanced for 5K Embeddings
SIMILARITY_THRESHOLD=0.78
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
CHROMADB_PATH=./data/db
COLLECTION_NAME=expertise-manifolds

# Dataset Configuration - Conservative Scaling (5,000 total)
CODING_DATASET_SIZE=2000
MATH_DATASET_SIZE=2000
GENERAL_DATASET_SIZE=1000

# Routing Configuration - Enhanced Performance
TOP_K_NEIGHBORS=3
CACHE_SIMILARITY_THRESHOLD=0.95

# Reproducibility Configuration
RANDOM_SEED=42
NUMPY_SEED=42
TORCH_SEED=42
ENABLE_REPRODUCIBILITY=true

# API Endpoints (customize as needed)
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
WIZARDMATH_API_BASE=https://api.wizardmath.com/v1
OPENAI_API_BASE=https://api.openai.com/v1

# Local Ollama Models (for impressive demo!)
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
            # Standard library random
            import random
            random.seed(cls.RANDOM_SEED)
            
            # NumPy random
            import numpy as np
            np.random.seed(cls.NUMPY_SEED)
            
            # PyTorch random (if available)
            try:
                import torch
                torch.manual_seed(cls.TORCH_SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(cls.TORCH_SEED)
                    torch.cuda.manual_seed_all(cls.TORCH_SEED)
                    # Additional CUDA reproducibility settings
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except ImportError:
                pass  # PyTorch not available, skip
            
            # TensorFlow random (if available)
            try:
                import tensorflow as tf
                tf.random.set_seed(cls.TORCH_SEED)
            except ImportError:
                pass  # TensorFlow not available, skip
            
            print(f"✅ Reproducibility enabled - using seeds: random={cls.RANDOM_SEED}, numpy={cls.NUMPY_SEED}, torch={cls.TORCH_SEED}")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize seeds: {e}")

# Global config instance
config = Config()
