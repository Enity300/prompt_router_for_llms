"""
Singleton model loader for SentenceTransformer to avoid duplicate loading.
This module ensures only one instance of the model is loaded in memory.
"""

import sys
import os
from sentence_transformers import SentenceTransformer

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config

# Singleton instance
_model_instance = None


def get_sentence_transformer() -> SentenceTransformer:
    """
    Get or create the singleton SentenceTransformer model instance.
    
    Returns:
        SentenceTransformer: The shared model instance
    """
    global _model_instance
    
    if _model_instance is None:
        print(f"Loading SentenceTransformer model (singleton): {config.SENTENCE_TRANSFORMER_MODEL}")
        _model_instance = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
        print("✅ Model loaded successfully (shared instance)")
    
    return _model_instance


def reset_model():
    """Reset the singleton instance (useful for testing)"""
    global _model_instance
    _model_instance = None
