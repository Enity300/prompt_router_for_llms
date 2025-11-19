import os
import sys
import time
import numpy as np
import chromadb
from chromadb.config import Settings
from catboost import CatBoostClassifier
import pickle
from typing import Dict, Any, List
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from utils.model_loader import get_sentence_transformer


class SentenceTransformerCatBoostRouter:
    """
    Production router that uses Sentence Transformer + CatBoost

    Architecture:
    1. Encode query with Sentence Transformer → 384-dim embedding
    2. Pass embedding to trained CatBoost classifier → category
    3. Return category + confidence
    """

    def __init__(self, model_path="models/sentence_catboost_router.pkl"):
        self.model_path = model_path
        self.classifier = None
        self.sentence_model = None
        self.is_trained = False
        self._cache = {}

        self._load_model()

        if self.sentence_model is None:
            print("Loading Sentence Transformer model...")
            self.sentence_model = get_sentence_transformer()

    def _load_model(self):
        """Load pre-trained CatBoost classifier if available"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                self.is_trained = True
                print(f"✅ Loaded Sentence Transformer + CatBoost model from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                self.is_trained = False
        else:
            print(f"⚠️ Model not found at {self.model_path}")
            print("   Train it first: python src/sentence_transformer_catboost_router.py")
            self.is_trained = False

    def train_from_chromadb(self) -> bool:
        """
        Extract training data from ChromaDB and train CatBoost classifier

        Returns:
            bool: True if training succeeded, False otherwise
        """
        print("\n" + "="*60)
        print("EXTRACTING TRAINING DATA FROM CHROMADB")
        print("="*60)

        db_path = config.CHROMADB_PATH
        if not os.path.exists(db_path):
            print(f"❌ ChromaDB not found at {db_path}")
            print("   Build it first: python src/build_expertise_db.py")
            return False

        try:
            client = chromadb.PersistentClient(path=db_path, settings=Settings())
            collection = client.get_collection(name=config.COLLECTION_NAME)
        except Exception as e:
            print(f"❌ Failed to load ChromaDB: {e}")
            return False

        print("Extracting embeddings and labels...")
        all_data = collection.get(include=['embeddings', 'metadatas'])

        if all_data['embeddings'] is None or len(all_data['embeddings']) == 0:
            print("❌ ChromaDB is empty!")
            return False

        X_train = np.array(all_data['embeddings'])
        y_train = [m['category'] for m in all_data['metadatas']]

        print(f"✅ Extracted {len(X_train)} training samples")
        print(f"   Embedding dimension: {X_train.shape[1]}")
        print(f"   Categories: {set(y_train)}")

        print("\n" + "="*60)
        print("TRAINING CATBOOST CLASSIFIER")
        print("="*60)

        self.classifier = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=50,
            allow_writing_files=False
        )

        self.classifier.fit(X_train, y_train)
        print("\n✅ Training complete!")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)

        print(f"✅ Model saved to {self.model_path}")
        self.is_trained = True
        return True

    def route(self, prompt: str) -> Dict[str, Any]:
        """
        Route a query to the appropriate category

        Args:
            prompt: User query

        Returns:
            dict: {
                'category': predicted category,
                'confidence': prediction confidence (0-1),
                'reasoning': explanation,
                'routing_time': time in seconds,
                'prompt': original prompt
            }
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Run training first.")

        if prompt in self._cache:
            return self._cache[prompt]

        start_time = time.time()

        embedding = self.sentence_model.encode([prompt], convert_to_numpy=True, show_progress_bar=False)[0]
        embedding = embedding.reshape(1, -1)

        prediction = self.classifier.predict(embedding)
        predicted_category = prediction[0]
        
        if hasattr(predicted_category, 'item'):
            predicted_category = predicted_category.item()
        
        predicted_category = str(predicted_category).strip()
        
        valid_categories = ['coding', 'math', 'general_knowledge']
        if predicted_category not in valid_categories:
            print(f"⚠️ Warning: Unexpected category '{predicted_category}' from classifier")
            print(f"   Valid categories: {valid_categories}")
            print(f"   Raw prediction: {prediction}")
            print(f"   Type: {type(prediction[0])}")
        
        probabilities = self.classifier.predict_proba(embedding)[0]
        confidence = float(np.max(probabilities))

        routing_time = time.time() - start_time

        result = {
            "category": predicted_category,
            "confidence": confidence,
            "reasoning": f"Sentence Transformer + CatBoost (prob: {confidence:.3f})",
            "routing_time": routing_time,
            "prompt": prompt
        }

        self._cache[prompt] = result
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics"""
        cache_hits = len(self._cache)
        return {
            "is_trained": self.is_trained,
            "cache_size": cache_hits,
            "model_path": self.model_path,
            "categories": list(self.classifier.classes_) if self.is_trained else []
        }


if __name__ == "__main__":
    """
    Train Sentence Transformer + CatBoost router from ChromaDB data

    Usage:
        python src/sentence_transformer_catboost_router.py

    This will:
    1. Extract embeddings and labels from ChromaDB
    2. Train CatBoost classifier on embeddings
    3. Save model to models/sentence_catboost_router.pkl

    For evaluation, run: python src/comprehensive_evaluation.py
    """
    print("\n" + "="*60)
    print("SENTENCE TRANSFORMER + CATBOOST ROUTER - TRAINING")
    print("="*60)

    router = SentenceTransformerCatBoostRouter()

    if router.train_from_chromadb():
        print("\n✅ Training complete!")
        print("   Model saved to: models/sentence_catboost_router.pkl")
        print("\nNext step: Run evaluation with:")
        print("   python src/comprehensive_evaluation.py")

        print("\n" + "="*60)
        print("QUICK TEST (Sample Predictions)")
        print("="*60)

        test_queries = [
            "How to implement binary search in Python?",
            "What is the derivative of sin(x)?",
            "Who invented the telephone?",
        ]

        for query in test_queries:
            result = router.route(query)
            print(f"\nQuery: {query}")
            print(f"  → Category: {result['category']} (confidence: {result['confidence']:.1%})")
    else:
        print("\n❌ Training failed!")
        print("   Make sure ChromaDB is built first:")
        print("   python src/build_expertise_db.py")

