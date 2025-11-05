"""
Reproducibility Testing Module for SS-GER

This module tests whether the database building process produces
consistent, deterministic results across multiple runs.
"""

import os
import shutil
import tempfile
import time
import hashlib
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

from .build_expertise_db import ExpertiseDBBuilder
from .semantic_router import SemanticRouter
from config import config


class ReproducibilityTester:
    """Test reproducibility of SS-GER database building process"""
    
    def __init__(self):
        self.temp_dirs = []
        self.test_results = {}
    
    def create_test_database(self, build_id: str) -> str:
        """Create a test database in a temporary directory"""
        # Create unique temporary directory
        timestamp = str(int(time.time() * 1000))
        pid = str(os.getpid())
        temp_dir = tempfile.mkdtemp(prefix=f"ss_ger_test_{build_id}_{timestamp}_{pid}_")
        self.temp_dirs.append(temp_dir)
        
        # Set up paths
        db_path = os.path.join(temp_dir, "test_db")
        collection_name = f"test_expertise_{build_id}_{timestamp}"
        
        # Temporarily modify config for this test
        original_db_path = config.CHROMADB_PATH
        original_collection = config.COLLECTION_NAME
        original_coding_size = config.CODING_DATASET_SIZE
        original_math_size = config.MATH_DATASET_SIZE
        original_general_size = config.GENERAL_DATASET_SIZE
        
        try:
            # Override config for test
            config.CHROMADB_PATH = db_path
            config.COLLECTION_NAME = collection_name
            config.CODING_DATASET_SIZE = 20  # Small for testing
            config.MATH_DATASET_SIZE = 20
            config.GENERAL_DATASET_SIZE = 10
            
            # Build database with limited samples for speed
            builder = ExpertiseDBBuilder()
            builder.initialize_components()
            builder.build_database()
            
        finally:
            # Restore original config
            config.CHROMADB_PATH = original_db_path
            config.COLLECTION_NAME = original_collection
            config.CODING_DATASET_SIZE = original_coding_size
            config.MATH_DATASET_SIZE = original_math_size
            config.GENERAL_DATASET_SIZE = original_general_size
        
        # Allow time for cleanup
        time.sleep(0.5)
        
        return db_path
    
    def _cleanup_chromadb_client(self, client):
        """Properly cleanup ChromaDB client"""
        try:
            if hasattr(client, '_client') and client._client:
                client._client.close()
            if hasattr(client, 'close'):
                client.close()
        except Exception:
            pass  # Ignore cleanup errors
    
    def get_database_signature(self, db_path: str) -> str:
        """Generate a signature for database contents"""
        try:
            # Create router to access database
            original_path = config.CHROMADB_PATH
            config.CHROMADB_PATH = db_path
            
            router = SemanticRouter()
            embeddings = router.get_all_embeddings()
            
            # Generate hash of embeddings
            if embeddings is not None and len(embeddings) > 0:
                # Convert to consistent format and hash
                embeddings_bytes = embeddings.tobytes()
                signature = hashlib.md5(embeddings_bytes).hexdigest()
            else:
                signature = "empty_database"
            
            # Cleanup
            self._cleanup_chromadb_client(router.client)
            config.CHROMADB_PATH = original_path
            
            return signature
            
        except Exception as e:
            config.CHROMADB_PATH = original_path
            return f"error_{str(e)[:50]}"
    
    def compare_databases(self, db_path1: str, db_path2: str) -> Dict[str, Any]:
        """Compare two databases for identical content"""
        try:
            sig1 = self.get_database_signature(db_path1)
            sig2 = self.get_database_signature(db_path2)
            
            identical = sig1 == sig2 and sig1 != "empty_database"
            
            # Calculate similarity if both have valid embeddings
            similarity = 1.0 if identical else 0.0
            
            return {
                "databases_identical": identical,
                "signature1": sig1,
                "signature2": sig2,
                "embedding_similarity": similarity
            }
            
        except Exception as e:
            return {
                "databases_identical": False,
                "signature1": "error",
                "signature2": "error", 
                "embedding_similarity": 0.0,
                "error": str(e)
            }
    
    def run_reproducibility_test(self, num_builds: int = 2) -> Dict[str, Any]:
        """Run complete reproducibility test"""
        print(f"🧪 Building {num_builds} test databases for reproducibility testing...")
        
        # Build multiple databases
        db_paths = []
        for i in range(num_builds):
            print(f"  Building database {i+1}/{num_builds}...")
            try:
                db_path = self.create_test_database(f"build_{i}")
                db_paths.append(db_path)
                time.sleep(1)  # Allow cleanup between builds
            except Exception as e:
                print(f"  ❌ Failed to build database {i+1}: {e}")
                return {
                    "reproducibility_achieved": False,
                    "error": f"Failed to build database {i+1}: {e}",
                    "comparisons": []
                }
        
        # Compare all pairs
        comparisons = []
        all_identical = True
        
        for i in range(len(db_paths)):
            for j in range(i + 1, len(db_paths)):
                print(f"  Comparing database {i+1} vs {j+1}...")
                comparison = self.compare_databases(db_paths[i], db_paths[j])
                comparison["pair"] = f"Build {i+1} vs Build {j+1}"
                comparisons.append(comparison)
                
                if not comparison["databases_identical"]:
                    all_identical = False
        
        return {
            "reproducibility_achieved": all_identical,
            "comparisons": comparisons,
            "num_builds": num_builds,
            "databases_built": len(db_paths)
        }
    
    def cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup {temp_dir}: {e}")
        self.temp_dirs.clear()