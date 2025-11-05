import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import random
import json
import traceback
import gc  # For garbage collection to free memory

# Assume config is in the parent directory
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import config
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)

# <<< MODIFICATION: Added constant for evaluation set size
EVALUATION_SET_SIZE = 2000

class ExpertiseDBBuilder:

    def __init__(self):
        self.model = None
        self.client = None
        self.collection = None

    def initialize_components(self):
        print(f"Loading sentence transformer model: {config.SENTENCE_TRANSFORMER_MODEL}")
        try:
            self.model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
        except Exception as e:
            print(f"❌ Failed to load SentenceTransformer model: {e}")
            raise

        os.makedirs(config.CHROMADB_PATH, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=config.CHROMADB_PATH,
                settings=Settings(allow_reset=True)
            )
        except Exception as e:
            print(f"❌ Failed to initialize ChromaDB client at path {config.CHROMADB_PATH}: {e}")
            raise

        try:
            print(f"Checking for existing collection: {config.COLLECTION_NAME}")
            self.client.delete_collection(name=config.COLLECTION_NAME)
            print("Cleared existing collection for fresh build.")
        except Exception:
            print(f"No existing collection '{config.COLLECTION_NAME}' found or error clearing. Proceeding to create.")
            pass

        try:
            print(f"Creating new collection: {config.COLLECTION_NAME}")
            self.collection = self.client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Successfully created collection '{config.COLLECTION_NAME}'.")
        except Exception as e:
            print(f"❌ Failed to create ChromaDB collection '{config.COLLECTION_NAME}': {e}")
            raise

    # <<< MODIFICATION: Function now only returns db_samples (no eval samples yet)
    def load_coding_dataset(self) -> List[Dict[str, Any]]:
        print("Loading coding dataset (KodCode/KodCode-V1)...")
        dataset_name = "KodCode/KodCode-V1"
        prompt_field_name = 'question'

        try:
            dataset_list = list(load_dataset(dataset_name, split="train", streaming=False))
            if not dataset_list:
                raise ValueError(f"Loaded {dataset_name} dataset split is empty.")

            db_size = config.CODING_DATASET_SIZE
            if len(dataset_list) < db_size:
                print(f"⚠️ Warning: Dataset has {len(dataset_list)} records, but {db_size} are required for DB. Adjusting sizes.")
                db_size = min(db_size, len(dataset_list))
            
            # --- MODIFICATION: Random sampling from dataset for DB ---
            print(f"Randomly sampling {db_size} samples for the database...")
            all_indices = list(range(len(dataset_list)))
            random.shuffle(all_indices)
            db_indices = all_indices[:db_size]
            db_records = [dataset_list[i] for i in db_indices]
            # ---
            
            db_samples = []

            for item in tqdm(db_records, desc="Processing coding DB samples"):
                prompt = item.get(prompt_field_name)
                if prompt and prompt.strip():
                    db_samples.append({
                        "prompt": prompt.strip(),
                        "category": "coding",
                        "source": "KodCode-V1"
                    })

            print(f"Successfully processed {len(db_samples)} DB samples.")
            return db_samples

        except Exception as e:
            print(f"Error loading or processing {dataset_name} dataset: {e}")
            print("Using fallback coding prompts...")
            fallback_prompts = [ "Write a Python function to implement binary search", "Create a class for a binary tree", "Implement a merge sort algorithm", "Write a function to find the longest palindromic substring", "Create a graph class with DFS and BFS", "Implement a LRU cache", "Write a function to detect cycles in a linked list", "Create a priority queue", "Implement the quicksort algorithm", "Write a function to validate a binary search tree"]
            actual_fallback_size = min(len(fallback_prompts), config.CODING_DATASET_SIZE)
            db_samples = [{"prompt": prompt, "category": "coding", "source": "synthetic"} for prompt in fallback_prompts[:actual_fallback_size]]
            return db_samples

    # <<< MODIFICATION: Function now only returns db_samples (no eval samples yet)
    def load_math_dataset(self) -> List[Dict[str, Any]]:
        """Load and prepare math dataset"""
        print("Loading math dataset...")
        try:
            dataset_list = list(load_dataset("gsm8k", "main", split="train", streaming=False))
            if not dataset_list:
                raise ValueError("Loaded math (gsm8k) dataset is empty.")
            
            db_size = config.MATH_DATASET_SIZE
            if len(dataset_list) < db_size:
                print(f"⚠️ Warning: Dataset has {len(dataset_list)} records, but {db_size} are required for DB. Adjusting sizes.")
                db_size = min(db_size, len(dataset_list))

            # --- MODIFICATION: Random sampling from dataset for DB ---
            print(f"Randomly sampling {db_size} samples for the database...")
            all_indices = list(range(len(dataset_list)))
            random.shuffle(all_indices)
            db_indices = all_indices[:db_size]
            db_records = [dataset_list[i] for i in db_indices]
            # ---

            db_samples = []
            
            for item in tqdm(db_records, desc="Processing math DB samples"):
                prompt = item.get('question')
                if prompt and prompt.strip():
                    db_samples.append({"prompt": prompt.strip(), "category": "math", "source": "gsm8k"})

            print(f"Successfully processed {len(db_samples)} DB samples.")
            return db_samples

        except Exception as e:
            print(f"Error loading or processing math dataset: {e}")
            print("Using fallback math prompts...")
            fallback_prompts = ["Solve for x: 2x + 5 = 15", "What is the derivative of x^3?", "Calculate the area of a circle with radius 7", "Find the prime factorization of 84", "Solve the quadratic equation x^2 - 5x + 6 = 0", "What is the integral of 3x^2?", "Probability of rolling two dice and getting 7", "Slope of the line passing through (2,3) and (5,9)", "What is 15% of 240?", "Solve the system: 2x + y = 7, x - y = 2"]
            actual_fallback_size = min(len(fallback_prompts), config.MATH_DATASET_SIZE)
            db_samples = [{"prompt": prompt, "category": "math", "source": "synthetic"} for prompt in fallback_prompts[:actual_fallback_size]]
            return db_samples

    # <<< MODIFICATION: Function now only returns db_samples (no eval samples yet)
    def load_general_dataset(self) -> List[Dict[str, Any]]:
        """Load and prepare general knowledge dataset using TriviaQA"""
        print("Loading general knowledge dataset (TriviaQA)...")
        try:
            dataset_list = list(load_dataset("trivia_qa", "rc.nocontext", split="validation", streaming=False))
            if not dataset_list:
                raise ValueError("Loaded TriviaQA dataset split is empty.")

            db_size = config.GENERAL_DATASET_SIZE
            if len(dataset_list) < db_size:
                print(f"⚠️ Warning: Dataset has {len(dataset_list)} records, but {db_size} are required. Adjusting sizes.")
                db_size = min(db_size, len(dataset_list))
            
            # --- MODIFICATION: Random sampling from dataset for DB ---
            print(f"Randomly sampling {db_size} samples for the database...")
            all_indices = list(range(len(dataset_list)))
            random.shuffle(all_indices)
            db_indices = all_indices[:db_size]
            db_records = [dataset_list[i] for i in db_indices]
            # ---

            db_samples = []

            for item in tqdm(db_records, desc="Processing general DB samples"):
                prompt = item.get('question')
                if prompt and prompt.strip():
                    db_samples.append({"prompt": prompt.strip(), "category": "general_knowledge", "source": "trivia_qa"})

            print(f"Successfully processed {len(db_samples)} DB samples.")
            return db_samples

        except Exception as e:
            print(f"Error loading or processing TriviaQA dataset: {e}")
            print("Using fallback general knowledge prompts...")
            fallback_prompts = ["What is the capital of France?", "Explain the theory of relativity", "What are the main causes of climate change?", "Describe photosynthesis", "History of the Internet?", "Explain how vaccines work", "Benefits of renewable energy?", "Describe the structure of an atom", "What is artificial intelligence?", "Explain the water cycle", "Significance of the Renaissance?", "How do economies work?", "What is quantum mechanics?", "Describe the human digestive system", "What is machine learning?", "Explain evolution", "What are human rights?", "Describe the solar system", "What is democracy?", "Explain how the brain works"]
            actual_fallback_size = min(len(fallback_prompts), config.GENERAL_DATASET_SIZE)
            db_samples = [{"prompt": prompt, "category": "general_knowledge", "source": "synthetic"} for prompt in fallback_prompts[:actual_fallback_size]]
            return db_samples

    # NEW: Load LLM Routing dataset for GENERAL classes
    def load_llm_routing_general_dataset(self) -> List[Dict[str, Any]]:
        """Load and prepare general knowledge from LLM Routing dataset with balanced sampling per class"""
        print("Loading LLM Routing dataset for GENERAL classes...")
        
        # Classes to include in general category
        general_classes = ['conversation', 'science', 'toxic_harmful', 'logical_reasoning', 'sex', 'creative_writing']
        
        try:
            dataset_list = list(load_dataset("jeanvydes/llm-routing-text-classification", split="train", streaming=False))
            if not dataset_list:
                raise ValueError("Loaded LLM Routing dataset is empty.")
            
            # Calculate samples per class for equal distribution
            db_size = config.GENERAL_DATASET_SIZE
            samples_per_class = db_size // len(general_classes)
            print(f"Target: {samples_per_class} samples per class (total {db_size})")
            
            # Group samples by class
            print(f"Filtering and grouping samples by class: {general_classes}")
            class_samples = {cls: [] for cls in general_classes}
            
            for item in dataset_list:
                task_class = item.get('task')
                if task_class in general_classes:
                    class_samples[task_class].append(item)
            
            # Print distribution
            print("Class distribution in dataset:")
            for cls in general_classes:
                print(f"  - {cls}: {len(class_samples[cls])} samples")
            
            # Sample equally from each class
            db_samples = []
            
            for cls in general_classes:
                available = len(class_samples[cls])
                if available == 0:
                    print(f"⚠️ Warning: No samples found for class '{cls}'. Skipping.")
                    continue
                
                # Take samples_per_class or all available (whichever is smaller)
                actual_count = min(samples_per_class, available)
                
                # Random sampling from this class
                selected_indices = random.sample(range(available), actual_count)
                
                for idx in tqdm(selected_indices, desc=f"Processing {cls} samples"):
                    item = class_samples[cls][idx]
                    prompt = item.get('prompt')
                    if prompt and prompt.strip():
                        db_samples.append({
                            "prompt": prompt.strip(), 
                            "category": "general_knowledge", 
                            "source": f"llm-routing-{cls}"
                        })
                
                print(f"✅ Sampled {actual_count} samples from class '{cls}'")
            
            print(f"Successfully processed {len(db_samples)} DB samples from LLM Routing dataset (balanced).")
            return db_samples
            
        except Exception as e:
            print(f"Error loading or processing LLM Routing dataset for general classes: {e}")
            print("Skipping LLM Routing general dataset.")
            return []

    # NEW: Load LLM Routing dataset for MATH class
    def load_llm_routing_math_dataset(self) -> List[Dict[str, Any]]:
        """Load and prepare math samples from LLM Routing dataset"""
        print("Loading LLM Routing dataset for MATH class...")
        
        try:
            dataset_list = list(load_dataset("jeanvydes/llm-routing-text-classification", split="train", streaming=False))
            if not dataset_list:
                raise ValueError("Loaded LLM Routing dataset is empty.")
            
            # Filter for math class only
            print("Filtering for 'math' class...")
            filtered_list = [item for item in dataset_list if item.get('task') == 'math']
            print(f"Found {len(filtered_list)} math samples")
            
            if not filtered_list:
                print("⚠️ No math samples found. Skipping.")
                return []
            
            db_size = config.MATH_DATASET_SIZE
            if len(filtered_list) < db_size:
                print(f"⚠️ Warning: Filtered dataset has {len(filtered_list)} records, but {db_size} are required. Adjusting sizes.")
                db_size = min(db_size, len(filtered_list))
            
            # Random sampling from dataset for DB
            print(f"Randomly sampling {db_size} samples for the database...")
            all_indices = list(range(len(filtered_list)))
            random.shuffle(all_indices)
            db_indices = all_indices[:db_size]
            db_records = [filtered_list[i] for i in db_indices]
            
            db_samples = []
            
            for item in tqdm(db_records, desc="Processing LLM Routing math DB samples"):
                prompt = item.get('prompt')
                if prompt and prompt.strip():
                    db_samples.append({"prompt": prompt.strip(), "category": "math", "source": "llm-routing-dataset"})
            
            print(f"Successfully processed {len(db_samples)} DB samples from LLM Routing math dataset.")
            return db_samples
            
        except Exception as e:
            print(f"Error loading or processing LLM Routing dataset for math class: {e}")
            print("Skipping LLM Routing math dataset.")
            return []

    def embed_and_store_samples(self, samples: List[Dict[str, Any]], batch_size: int = 32):
        """Embed samples and store them in ChromaDB"""
        if not samples:
            print("No samples to embed and store.")
            return
        print(f"Embedding and storing {len(samples)} samples...")
        for i in tqdm(range(0, len(samples), batch_size), desc="Storing embeddings"):
            batch = samples[i:i + batch_size]
            prompts = [sample["prompt"] for sample in batch]
            try:
                embeddings = self.model.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
            except Exception as e:
                print(f"\nError encoding batch starting at index {i}: {e}")
                continue
            ids = [f"{sample.get('category', 'unknown')}_{i + j}_{random.randint(10000, 99999)}" for j, sample in enumerate(batch)]
            metadatas = [{"category": sample.get("category", "unknown"), "source": sample.get("source", "unknown"), "prompt": sample.get("prompt", "")[:500] + ("..." if len(sample.get("prompt", "")) > 500 else "")} for sample in batch]
            documents = prompts
            try:
                self.collection.add(embeddings=embeddings.tolist(), metadatas=metadatas, documents=documents, ids=ids)
            except Exception as e:
                print(f"\nError adding batch to ChromaDB starting at index {i}: {e}")
                continue

    def build_database(self):
        """Main method to build the complete expertise database and evaluation file - MEMORY EFFICIENT VERSION"""
        print("=== Semantic Router Expertise Database Builder ===")
        print(f"Target database path: {config.CHROMADB_PATH}")
        print(f"Collection name: {config.COLLECTION_NAME}")
        print(f"Configured sizes: Coding={config.CODING_DATASET_SIZE}, Math={config.MATH_DATASET_SIZE}, General={config.GENERAL_DATASET_SIZE}")
        print()

        try:
            if hasattr(config, 'initialize_seeds') and callable(config.initialize_seeds):
                 config.initialize_seeds()
                 print("✅ Reproducibility enabled via config.initialize_seeds()")
            else:
                 print("Warning: config.initialize_seeds() not found or not callable. Using default random seeds.")
                 random.seed(42)

            self.initialize_components()

            # Process datasets ONE AT A TIME (memory efficient!)
            total_db_stored = 0
            
            # Define dataset processing order
            datasets_to_process = [
                ('coding', self.load_coding_dataset),
                ('math_gsm8k', self.load_math_dataset),
                ('math_llm_routing', self.load_llm_routing_math_dataset),
                ('general_trivia', self.load_general_dataset),
                ('general_llm_routing', self.load_llm_routing_general_dataset)
            ]
            
            print("\n" + "="*60)
            print("PROCESSING DATASETS SEQUENTIALLY (Memory Efficient)")
            print("="*60)
            
            for dataset_name, load_func in datasets_to_process:
                print(f"\n[{dataset_name.upper()}] Loading dataset...")
                
                # Load dataset (only this one in memory)
                db_samples = load_func()
                
                print(f"[{dataset_name.upper()}] Loaded {len(db_samples)} DB samples")
                
                # Store DB samples in ChromaDB immediately
                if db_samples:
                    print(f"[{dataset_name.upper()}] Embedding and storing in ChromaDB...")
                    self.embed_and_store_samples(db_samples)
                    total_db_stored += len(db_samples)
                    print(f"[{dataset_name.upper()}] ✅ Stored {len(db_samples)} samples in ChromaDB")
                
                # Free memory explicitly
                print(f"[{dataset_name.upper()}] Clearing from memory...")
                del db_samples
                gc.collect()  # Force garbage collection
                print(f"[{dataset_name.upper()}] ✅ Memory cleared\n")

            # Verify ChromaDB storage
            try:
                count = self.collection.count()
                print(f"\n{'='*60}")
                print(f"✅ ChromaDB Build Complete!")
                print(f"Total embeddings stored: {count}")
                print(f"Database location: {os.path.abspath(config.CHROMADB_PATH)}")
                print(f"{'='*60}\n")
            except Exception as count_e:
                print(f"Warning: Could not get final count from collection: {count_e}")

            # Combine all evaluation samples from temp files
            print("\n" + "="*60)
            print("CREATING EVALUATION DATASET FROM CHROMADB")
            print("="*60)
            
            all_eval_samples = []
            
            # For each category, query ChromaDB and randomly pick eval samples
            categories_eval_count = {
                "coding": EVALUATION_SET_SIZE,  # 2000 from 6000 coding samples
                "math": EVALUATION_SET_SIZE,    # 2000 from 6000 math samples  
                "general_knowledge": EVALUATION_SET_SIZE  # 2000 from 6000 general samples
            }
            
            for category, eval_count in categories_eval_count.items():
                print(f"\n[{category.upper()}] Querying ChromaDB for samples...")
                try:
                    # Query all samples for this category
                    results = self.collection.get(
                        where={"category": category},
                        include=["documents", "metadatas"]
                    )
                    
                    if not results or not results['documents']:
                        print(f"⚠️ No samples found for category '{category}'. Skipping.")
                        continue
                    
                    category_samples = results['documents']
                    print(f"Found {len(category_samples)} total samples for '{category}'")
                    
                    # Randomly select eval_count samples
                    actual_eval_count = min(eval_count, len(category_samples))
                    print(f"Randomly sampling {actual_eval_count} evaluation samples...")
                    
                    # Create indices and shuffle
                    eval_indices = random.sample(range(len(category_samples)), actual_eval_count)
                    
                    for idx in eval_indices:
                        all_eval_samples.append({
                            "query": category_samples[idx],
                            "category": category
                        })
                    
                    print(f"✅ Added {actual_eval_count} eval samples for '{category}'")
                    
                except Exception as e:
                    print(f"❌ Error querying ChromaDB for category '{category}': {e}")
            
            if not all_eval_samples:
                print("⚠️ No evaluation samples were generated. Skipping evaluation file creation.")
            else:
                print(f"\nTotal evaluation samples collected: {len(all_eval_samples)}")
                print("Shuffling evaluation samples (mixing all categories)...")
                random.shuffle(all_eval_samples)
                
                eval_filepath = "evaluation_dataset.json"
                try:
                    with open(eval_filepath, 'w', encoding='utf-8') as f:
                        json.dump(all_eval_samples, f, indent=2)
                    print(f"✅ Successfully created '{eval_filepath}' with {len(all_eval_samples)} mixed samples.")
                    print(f"   Categories mixed: coding, math, general_knowledge")
                except Exception as e:
                    print(f"❌ Failed to write evaluation file: {e}")
            
            print("\n" + "="*60)
            print("✅ DATABASE BUILD COMPLETE!")
            print("="*60)

        except Exception as e:
            print(f"\n❌ An error occurred during the database build process: {e}")
            raise

def main():
    """Main entry point"""
    try:
        builder = ExpertiseDBBuilder()
        builder.build_database()
        print("\nScript finished successfully.")
    except KeyboardInterrupt:
        print("\n\n⚠️ Build process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Final error during database build: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()