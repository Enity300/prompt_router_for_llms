import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import time
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class SemanticRouterError(Exception):
    """Custom exception for semantic router errors"""
    pass


class SemanticRouter:
    
    def __init__(self, model_name: Optional[str] = None, db_path: Optional[str] = None):
       
        self.model_name = model_name or config.SENTENCE_TRANSFORMER_MODEL
        self.db_path = db_path or config.CHROMADB_PATH
        self.model = None
        self.client = None
        self.collection = None
        self.cache = {}  
        
        self._initialize_components()
    
    def _initialize_components(self):
        try:
            # Initialize seeds for reproducibility
            config.initialize_seeds()
            
            print(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Verify database exists
            if not os.path.exists(self.db_path):
                raise SemanticRouterError(
                    f"Database not found at {self.db_path}. "
                    "Please run 'python build_expertise_db.py' first to build the expertise database."
                )
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings()
            )
            
            # Get the collection, 
            try:
                self.collection = self.client.get_collection(name=config.COLLECTION_NAME)
                print(f"Connected to collection: {config.COLLECTION_NAME}")
                
                # Verify collection has data
                count = self.collection.count()
                if count == 0:
                    raise SemanticRouterError(
                        "Expertise database is empty. Please run 'python build_expertise_db.py' to populate it."
                    )
                print(f"Loaded expertise database with {count} embeddings")
                
            except Exception as e:
                raise SemanticRouterError(
                    f"Could not access collection '{config.COLLECTION_NAME}': {e}. "
                    "Please run 'python build_expertise_db.py' first."
                )
                
        except Exception as e:
            raise SemanticRouterError(f"Failed to initialize router components: {e}")
    
    def _embed_prompt(self, prompt: str) -> np.ndarray:

        try:
            embedding = self.model.encode([prompt], convert_to_numpy=True)
            return embedding[0]  # Return single embedding vector
        except Exception as e:
            raise SemanticRouterError(f"Failed to embed prompt: {e}")
    
    def _check_cache(self, prompt: str, embedding: np.ndarray) -> Optional[str]:
       
        for cached_prompt, cached_data in self.cache.items():
            cached_embedding = cached_data["embedding"]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding], [cached_embedding])[0][0]
            
            if similarity >= config.CACHE_SIMILARITY_THRESHOLD:
                print(f"🎯 Cache hit! Similarity: {similarity:.4f} with: '{cached_prompt[:50]}...'")
                return cached_data["category"]
        
        return None
    
    def _update_cache(self, prompt: str, embedding: np.ndarray, category: str):
       
        # Simple cache management: keep only the most recent 100 entries
        if len(self.cache) >= 100:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[prompt] = {
            "embedding": embedding,
            "category": category,
            "timestamp": time.time()
        }
    
    def _query_expertise_manifolds(self, embedding: np.ndarray) -> Tuple[str, float, Dict[str, Any]]: 
       
        try:
            # Query ChromaDB for nearest neighbors
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=config.TOP_K_NEIGHBORS,
                include=["metadatas", "documents", "distances"]
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                raise SemanticRouterError("No similar examples found in expertise database")
            
            # Get the closest match
            closest_metadata = results['metadatas'][0][0]
            closest_distance = results['distances'][0][0]  # ChromaDB returns L2 distance
            closest_document = results['documents'][0][0]
            
            # Convert L2 distance to cosine similarity approximation
            # For normalized vectors: cosine_similarity ≈ 1 - (L2_distance^2 / 2)
            # This is an approximation, but works well for routing decisions
            similarity_score = max(0, 1 - (closest_distance ** 2 / 2))
            
            category = closest_metadata['category']
            
            routing_info = {
                "closest_example": closest_document,
                "similarity_score": similarity_score,
                "l2_distance": closest_distance,
                "source": closest_metadata.get('source', 'unknown'),
                "metadata": closest_metadata
            }
            
            return category, similarity_score, routing_info
            
        except Exception as e:
            raise SemanticRouterError(f"Failed to query expertise manifolds: {e}")
    
    def route(self, prompt: str) -> Dict[str, Any]:
       
        start_time = time.time()
        
        try:
            # Input validation
            if not prompt or not prompt.strip():
                raise SemanticRouterError("Empty prompt provided")
            
            prompt = prompt.strip()
            
            # Step 1: Embed the prompt
            embedding = self._embed_prompt(prompt)
            
            # Step 2: Check semantic cache
            cached_category = self._check_cache(prompt, embedding)
            if cached_category:
                return {
                    "category": cached_category,
                    "confidence": 1.0,  # High confidence for cached results
                    "reasoning": "Retrieved from semantic cache",
                    "routing_time": time.time() - start_time,
                    "from_cache": True,
                    "prompt": prompt
                }
            
            # Step 3: Query expertise manifolds
            category, similarity_score, routing_info = self._query_expertise_manifolds(embedding)
            
            # Step 4: Apply similarity threshold for out-of-distribution detection
            if similarity_score < config.SIMILARITY_THRESHOLD:
                category = "general_knowledge"
                reasoning = f"Out-of-distribution query (similarity: {similarity_score:.4f} < threshold: {config.SIMILARITY_THRESHOLD})"
                confidence = 1 - similarity_score  # Lower similarity = higher confidence it's OOD
            else:
                reasoning = f"Matched {category} expertise (similarity: {similarity_score:.4f})"
                confidence = similarity_score
            
            # Step 5: Update cache
            self._update_cache(prompt, embedding, category)
            
            routing_result = {
                "category": category,
                "confidence": confidence,
                "reasoning": reasoning,
                "routing_time": time.time() - start_time,
                "from_cache": False,
                "prompt": prompt,
                "routing_info": routing_info
            }
            
            return routing_result
            
        except SemanticRouterError:
            raise
        except Exception as e:
            raise SemanticRouterError(f"Unexpected error during routing: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
     
        try:
            collection_count = self.collection.count() if self.collection else 0
            
            # Get category distribution
            category_stats = {}
            if self.collection:
                for category in ["coding", "math", "general_knowledge"]:
                    try:
                        results = self.collection.query(
                            query_texts=["sample"],
                            where={"category": category},
                            n_results=1
                        )
                        # This is a rough way to count 
                        category_stats[category] = "available"
                    except:
                        category_stats[category] = "not available"
            
            return {
                "model_name": self.model_name,
                "db_path": self.db_path,
                "collection_name": config.COLLECTION_NAME,
                "total_embeddings": collection_count,
                "similarity_threshold": config.SIMILARITY_THRESHOLD,
                "cache_size": len(self.cache),
                "cache_threshold": config.CACHE_SIMILARITY_THRESHOLD,
                "category_stats": category_stats,
                "top_k_neighbors": config.TOP_K_NEIGHBORS
            }
            
        except Exception as e:
            return {"error": f"Failed to get statistics: {e}"}
    
    def test_routing(self, test_prompts: List[str]) -> List[Dict[str, Any]]:
      
        results = []
        
        print(f"Testing router with {len(test_prompts)} prompts...")
        for i, prompt in enumerate(test_prompts, 1):
            try:
                print(f"\n[{i}/{len(test_prompts)}] Testing: '{prompt[:60]}...'")
                result = self.route(prompt)
                print(f"  → Routed to: {result['category']} (confidence: {result['confidence']:.3f})")
                results.append(result)
                
            except Exception as e:
                print(f"  → Error: {e}")
                results.append({
                    "prompt": prompt,
                    "category": "error",
                    "error": str(e)
                })
        
        return results
