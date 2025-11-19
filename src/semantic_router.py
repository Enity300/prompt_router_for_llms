import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import chromadb
from chromadb.config import Settings
import time
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from utils.model_loader import get_sentence_transformer

logger = logging.getLogger(__name__)


class SemanticRouterError(Exception):
    """Custom exception for semantic router errors"""
    pass


class SemanticRouter:
    EXPECTED_EMBEDDING_DIM = 384

    def __init__(self, model_name: Optional[str] = None, db_path: Optional[str] = None):

        self.model_name = model_name or config.SENTENCE_TRANSFORMER_MODEL
        self.db_path = db_path or config.CHROMADB_PATH
        self.model = None
        self.client = None
        self.collection = None
        self.cache = OrderedDict()

        self.cache_hits = 0
        self.cache_misses = 0

        self._initialize_components()

    def _initialize_components(self):
        try:
            config.initialize_seeds()

            print(f"Loading sentence transformer model: {self.model_name}")
            self.model = get_sentence_transformer()

            if not os.path.exists(self.db_path):
                raise SemanticRouterError(
                    f"Database not found at {self.db_path}. "
                    "Please run 'python build_expertise_db.py' first to build the expertise database."
                )

            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings()
            )

            try:
                self.collection = self.client.get_collection(name=config.COLLECTION_NAME)
                print(f"Connected to collection: {config.COLLECTION_NAME}")

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
        """Embed prompt and normalize to unit norm (consistent with database)"""
        try:
            embedding = self.model.encode([prompt], convert_to_numpy=True)
            embedding = embedding[0]

            if embedding.shape[0] != self.EXPECTED_EMBEDDING_DIM:
                raise SemanticRouterError(
                    f"Embedding dimension mismatch: expected {self.EXPECTED_EMBEDDING_DIM}, "
                    f"got {embedding.shape[0]}. Model may be misconfigured."
                )

            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm

            return embedding
        except Exception as e:
            raise SemanticRouterError(f"Failed to embed prompt: {e}")

    def _check_cache(self, prompt: str, embedding: np.ndarray) -> Optional[str]:
        """Check if a similar prompt exists in cache (with LRU update)"""
        for cached_prompt, cached_data in self.cache.items():
            cached_embedding = cached_data["embedding"]

            similarity = cosine_similarity([embedding], [cached_embedding])[0][0]

            if similarity >= config.CACHE_SIMILARITY_THRESHOLD:
                logger.info(f"Cache hit! Similarity: {similarity:.4f} with: '{cached_prompt[:50]}...'")
                self.cache.move_to_end(cached_prompt)
                self.cache_hits += 1
                return cached_data["category"]

        self.cache_misses += 1
        return None

    def _update_cache(self, prompt: str, embedding: np.ndarray, category: str):
        """Update cache with LRU eviction policy using OrderedDict"""
        if len(self.cache) >= config.CACHE_SIZE:
            self.cache.popitem(last=False)

        self.cache[prompt] = {
            "embedding": embedding,
            "category": category,
            "timestamp": time.time()
        }

    def _calibrate_confidence(self, similarity: float, category: str) -> Tuple[float, str]:
        """Calibrate confidence score and provide interpretation (Recommendation #2)"""
        if similarity > 0.95:
            return similarity, f"Very high confidence - strong match to {category} examples"
        elif similarity > 0.85:
            return similarity, f"High confidence - good match to {category} examples"
        elif similarity > config.SIMILARITY_THRESHOLD:
            return similarity, f"Medium confidence - moderate match to {category} examples"
        else:
            return 1 - similarity, "Low confidence - no strong match found, using fallback"

    def _query_expertise_manifolds(self, embedding: np.ndarray) -> Tuple[str, float, str, Dict[str, Any]]:
        """Query database with multi-neighbor voting (Recommendation #3)"""
        try:
            initial_k = min(config.TOP_K_NEIGHBORS * 2, 15)

            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=initial_k,
                include=["metadatas", "documents", "distances"]
            )

            if not results['metadatas'] or not results['metadatas'][0]:
                raise SemanticRouterError("No similar examples found in expertise database")

            votes = {}
            weighted_votes = {}
            all_neighbors = []

            for i, (metadata, distance, document) in enumerate(zip(
                results['metadatas'][0],
                results['distances'][0],
                results['documents'][0]
            )):
                category = metadata['category']
                similarity = 1 - distance

                votes[category] = votes.get(category, 0) + 1

                weight = similarity ** 2
                weighted_votes[category] = weighted_votes.get(category, 0.0) + weight

                all_neighbors.append({
                    "rank": i + 1,
                    "category": category,
                    "similarity": similarity,
                    "weight": weight,
                    "document": document[:100] + "..." if len(document) > 100 else document,
                    "source": metadata.get('source', 'unknown')
                })

            first_similarity = 1 - results['distances'][0][0]

            if first_similarity > 0.9:
                active_k = min(3, len(all_neighbors))
            elif first_similarity > 0.75:
                active_k = min(config.TOP_K_NEIGHBORS, len(all_neighbors))
            else:
                active_k = min(10, len(all_neighbors))

            votes_final = {}
            weighted_votes_final = {}
            for i in range(active_k):
                cat = all_neighbors[i]['category']
                votes_final[cat] = votes_final.get(cat, 0) + 1
                weighted_votes_final[cat] = weighted_votes_final.get(cat, 0.0) + all_neighbors[i]['weight']

            category = max(weighted_votes_final, key=weighted_votes_final.get)
            vote_confidence = votes_final[category] / active_k

            total_weight = sum(weighted_votes_final.values())
            weighted_confidence = weighted_votes_final[category] / total_weight if total_weight > 0 else 0.0

            closest_metadata = results['metadatas'][0][0]
            closest_distance = results['distances'][0][0]
            closest_document = results['documents'][0][0]
            similarity_score = 1 - closest_distance

            calibrated_confidence, confidence_level = self._calibrate_confidence(similarity_score, category)

            routing_info = {
                "closest_example": closest_document,
                "similarity_score": similarity_score,
                "cosine_distance": closest_distance,
                "source": closest_metadata.get('source', 'unknown'),
                "metadata": closest_metadata,
                "vote_confidence": vote_confidence,
                "weighted_confidence": weighted_confidence,
                "votes": votes_final,
                "weighted_votes": weighted_votes_final,
                "active_k": active_k,
                "all_neighbors": all_neighbors[:active_k],
                "confidence_level": confidence_level
            }

            return category, similarity_score, confidence_level, routing_info

        except Exception as e:
            raise SemanticRouterError(f"Failed to query expertise manifolds: {e}")

    def _generate_explanation(self, category: str, similarity: float, confidence_level: str) -> str:
        """Generate human-readable explanation for routing decision (Recommendation #5)"""
        if similarity > 0.9:
            return f"Very similar to {category} examples in database ({confidence_level})"
        elif similarity > config.SIMILARITY_THRESHOLD:
            return f"Moderately similar to {category} examples ({confidence_level})"
        else:
            return f"No strong match found (similarity: {similarity:.3f}), using general knowledge fallback"

    def route(self, prompt: str) -> Dict[str, Any]:
        """Route a prompt to the appropriate specialist category"""
        start_time = time.time()

        try:
            if not prompt or not prompt.strip():
                raise SemanticRouterError("Empty prompt provided")

            prompt = prompt.strip()

            embedding = self._embed_prompt(prompt)

            cached_category = self._check_cache(prompt, embedding)
            if cached_category:
                return {
                    "category": cached_category,
                    "confidence": 1.0,
                    "confidence_level": "cached",
                    "reasoning": "Retrieved from semantic cache",
                    "explanation": "Exact or very similar query found in cache",
                    "routing_time": time.time() - start_time,
                    "from_cache": True,
                    "prompt": prompt
                }

            category, similarity_score, confidence_level, routing_info = self._query_expertise_manifolds(embedding)

            if similarity_score < config.SIMILARITY_THRESHOLD:
                category = "general_knowledge"
                confidence = 1 - similarity_score
                confidence_level = "low"
            else:
                confidence = similarity_score

            explanation = self._generate_explanation(category, similarity_score, confidence_level)
            reasoning = f"Matched {category} expertise (similarity: {similarity_score:.4f})"

            self._update_cache(prompt, embedding, category)

            routing_result = {
                "category": category,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "reasoning": reasoning,
                "explanation": explanation,
                "routing_time": time.time() - start_time,
                "from_cache": False,
                "prompt": prompt,
                "routing_info": routing_info
            }

            logger.info(f"Routed to {category} (confidence: {confidence:.3f}, level: {confidence_level})")

            return routing_result

        except SemanticRouterError:
            raise
        except Exception as e:
            raise SemanticRouterError(f"Unexpected error during routing: {e}")

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (Recommendation #1)"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive router statistics"""
        try:
            collection_count = self.collection.count() if self.collection else 0

            category_stats = {}
            if self.collection:
                for category in ["coding", "math", "general_knowledge"]:
                    try:
                        results = self.collection.query(
                            query_texts=["sample"],
                            where={"category": category},
                            n_results=1
                        )
                        category_stats[category] = "available"
                    except Exception:
                        category_stats[category] = "not available"

            cache_hit_rate = self.get_cache_hit_rate()

            return {
                "model_name": self.model_name,
                "db_path": self.db_path,
                "collection_name": config.COLLECTION_NAME,
                "total_embeddings": collection_count,
                "similarity_threshold": config.SIMILARITY_THRESHOLD,
                "cache_size": len(self.cache),
                "cache_max_size": config.CACHE_SIZE,
                "cache_threshold": config.CACHE_SIMILARITY_THRESHOLD,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "category_stats": category_stats,
                "top_k_neighbors": config.TOP_K_NEIGHBORS,
                "embedding_dimension": self.EXPECTED_EMBEDDING_DIM
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
