import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import chromadb
from chromadb.config import Settings
import random
import json
import traceback
import gc
import re

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import config
    from utils.model_loader import get_sentence_transformer
except ImportError as e:
    print(f"Error importing config or utils: {e}")
    sys.exit(1)

class ExpertiseDBBuilder:
 
    def __init__(self):
        self.model = None
        self.client = None
        self.collection = None

        self.math_coding_ambiguous_keywords = [
            'algorithm', 'complexity', 'time complexity', 'space complexity',
            'big o', 'runtime', 'optimize', 'efficient', 'implement',
            'code', 'program', 'function', 'array', 'matrix multiplication',
            'recursion', 'iteration', 'loop', 'binary search', 'sorting',
            'fibonacci', 'dynamic programming', 'graph', 'tree traversal'
        ]

        self.pure_coding_keywords = [
            'syntax', 'debug', 'error', 'exception', 'class definition',
            'import', 'library', 'api', 'rest', 'database', 'sql query',
            'git', 'version control', 'docker', 'deployment', 'test',
            'front-end', 'backend', 'web', 'async', 'promise', 'callback'
        ]

        self.pure_math_keywords = [
            'theorem', 'proof', 'lemma', 'axiom', 'derivative', 'integral',
            'calculus', 'algebra', 'geometry', 'trigonometry', 'probability',
            'statistics', 'distribution', 'hypothesis', 'vector space',
            'eigenvalue', 'matrix determinant', 'polynomial', 'equation',
            'solve for x', 'simplify', 'factor', 'expand', 'evaluate'
        ]

    def _is_ambiguous_math_sample(self, prompt: str) -> bool:
        """Filter out math samples that are too computational/algorithmic"""
        prompt_lower = prompt.lower()

        coding_mentions = sum(1 for kw in self.pure_coding_keywords if kw in prompt_lower)
        if coding_mentions >= 2:
            return True

        ambiguous_mentions = sum(1 for kw in self.math_coding_ambiguous_keywords if kw in prompt_lower)
        if ambiguous_mentions >= 2:
            return True

        return False

    def _is_ambiguous_coding_sample(self, prompt: str) -> bool:
        """Filter out coding samples that are too mathematical/algorithmic"""
        prompt_lower = prompt.lower()

        math_problem_patterns = [
            r'\bcalculate\s+(the\s+)?(sum|product|integral|derivative)',
            r'\bsolve\s+(for|the)?\s+[a-z]\s+(equation|inequality)?',
            r'\bfind\s+(the\s+)?(derivative|integral|limit|area|volume)',
            r'\bprove\s+(that|the\s+)?.*theorem',
            r'\bsimplify\s+.*expression',
            r'\bevaluate\s+.*\d+.*[\+\-\*/\^]',
            r'\bfactor\s+(the\s+)?polynomial',
            r'what\s+is\s+\d+.*[\+\-\*/].*\d+'
        ]

        for pattern in math_problem_patterns:
            if re.search(pattern, prompt_lower):
                return True

        has_algorithm_talk = any(kw in prompt_lower for kw in ['algorithm', 'complexity', 'big o', 'time complexity'])
        has_coding_context = any(kw in prompt_lower for kw in ['implement', 'code', 'write a', 'function', 'program'])

        if has_algorithm_talk and not has_coding_context:
            return True

        return False

    def _generate_boundary_reinforcement_samples(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate hard negative samples that explicitly define category boundaries.
        Returns (db_samples, eval_samples) - samples split for training and evaluation.

        These are carefully crafted to push Math and Coding clusters apart in embedding space.
        """
        print("\n" + "="*60)
        print("GENERATING BOUNDARY REINFORCEMENT SAMPLES")
        print("="*60)

        pure_coding_samples = [
            "How do I fix a CORS error in my Express.js API?",
            "What's the difference between async/await and promises in JavaScript?",
            "How to set up environment variables in a React application?",
            "How do I create a REST API endpoint using Flask?",
            "What are React hooks and how do I use useState?",
            "How to handle form validation in React?",
            "Explain the difference between GET and POST requests",
            "What is the difference between REST and GraphQL APIs?",
            "How to handle AJAX requests in jQuery?",
            "How do I set up HTTPS on my Node.js server?",
            "What is Server-Side Rendering in Next.js?",
            "How to handle cookies in Express.js?",
            "What are WebSockets and how do they work?",
            "How to implement file uploads in Django?",
            "What is the Virtual DOM in React?",

            "Explain the difference between SQL JOIN types with examples",
            "What's the difference between SQL and NoSQL databases?",
            "How do I optimize a slow SQL query?",
            "What is database indexing and when should I use it?",
            "How to prevent SQL injection attacks?",
            "What is the difference between INNER JOIN and LEFT JOIN?",
            "How to backup a PostgreSQL database?",
            "What are database transactions and ACID properties?",
            "How to design a database schema for an e-commerce site?",
            "What is database normalization?",

            "Debug this Python syntax error: invalid syntax at line 42",
            "How to debug memory leaks in Node.js applications?",
            "How to write unit tests for a Python Flask API?",
            "What is Test-Driven Development (TDD)?",
            "How to use breakpoints in VS Code debugger?",
            "What is the difference between unit tests and integration tests?",
            "How to mock API calls in Jest?",
            "How to debug a production error in JavaScript?",
            "What are assertion errors in Python?",
            "How to profile Python code performance?",

            "How to deploy a Docker container to AWS?",
            "How to configure Git to ignore node_modules?",
            "How to set up a CI/CD pipeline with GitHub Actions?",
            "What is the difference between Docker and Virtual Machines?",
            "How to set up load balancing with Nginx?",
            "How to deploy a React app to Netlify?",
            "What are environment variables and how do I use them?",
            "How to rollback a deployment on Heroku?",
            "What is Kubernetes and when should I use it?",
            "How to set up automatic backups on AWS?",

            "Explain the MVC pattern in web development",
            "Explain dependency injection in Spring Boot",
            "What is the difference between abstract classes and interfaces in Java?",
            "What are design patterns in software engineering?",
            "Explain the Singleton pattern with an example",
            "What is the Observer pattern?",
            "What are SOLID principles?",
            "What is the difference between composition and inheritance?",
            "Explain the Factory pattern in Java",
            "What is dependency inversion?",

            "How do I authenticate users with JWT tokens?",
            "What are CSS Grid and Flexbox and when to use each?",
            "How do I import a CSV file in Python using pandas?",
            "What does the 'this' keyword mean in JavaScript?",
            "How to create a virtual environment in Python?",
            "Explain the difference between '==' and '===' in JavaScript",
            "How do I read a file line by line in Python?",
            "What is the purpose of the 'super' keyword in Java?",
            "How to handle exceptions in Python using try-except?",
            "What are lambda functions in Python?",
            "How do I create a class in Python?",
            "What is the difference between a list and a tuple in Python?",
            "What are Python decorators?",
            "How to use list comprehensions in Python?",
            "What is the difference between var, let, and const in JavaScript?",
            "What are closures in JavaScript?",
            "How to use async/await in Python?",
            "What are generators in Python?",
            "How to handle null values in JavaScript?",
            "What is the spread operator in JavaScript?",

            "How to revert a commit in Git?",
            "What is the difference between git merge and git rebase?",
            "How to resolve merge conflicts in Git?",
            "What is a pull request?",
            "How to create a new branch in Git?",
            "What is .gitignore and how do I use it?",
            "How to undo git add before commit?",
            "What is Git stash?",
            "How to delete a remote branch?",
            "What is the difference between git fetch and git pull?",

            "How to hash passwords securely in Node.js?",
            "What is CSRF and how do I prevent it?",
            "How to implement OAuth authentication?",
            "What is XSS and how do I prevent it?",
            "How to securely store API keys?",
            "What is rate limiting and how do I implement it?",
            "How to implement Two-Factor Authentication?",
            "What are JSON Web Tokens (JWT)?",
            "How to validate user input to prevent attacks?",
            "What is HTTPS and why is it important?"
        ]

        pure_math_samples = [
            "Prove that the square root of 2 is irrational",
            "What is the derivative of sin(x) with respect to x?",
            "Calculate the integral of x³ from 0 to 5",
            "Find the limit of (x² - 4) / (x - 2) as x approaches 2",
            "What is the Taylor series expansion of e^x around x=0?",
            "Find the Fourier transform of the function f(t) = e^(-t²)",
            "What is the derivative of ln(x)?",
            "Calculate the integral of 1/x dx",
            "What is the second derivative test for finding maxima and minima?",
            "Find the derivative of tan(x)",
            "What is the fundamental theorem of calculus?",
            "Calculate the integral of e^x dx",
            "What is L'Hôpital's rule?",
            "Find the partial derivatives of f(x,y) = x²y + y³",
            "What is the chain rule for derivatives?",
            "Calculate the surface integral of a sphere",
            "What is the divergence theorem?",
            "Find the gradient of f(x,y,z) = xyz",
            "What is Green's theorem?",
            "Calculate the line integral along a curve",

            "Solve the quadratic equation: 3x² + 7x - 2 = 0",
            "If f(x) = 2x + 3, what is f⁻¹(x)?",
            "Simplify the expression: (x² - 9) / (x - 3)",
            "Solve the system of equations: 2x + y = 7, x - y = 1",
            "What is the quadratic formula?",
            "Factor the polynomial: x³ - 8",
            "Solve for x: log(x) + log(3) = log(12)",
            "What is the binomial theorem?",
            "Expand (x + y)³ using the binomial theorem",
            "What is a polynomial?",
            "What are the roots of x² - 5x + 6 = 0?",
            "Simplify: (a³b²)/(ab)",
            "What is the difference of squares formula?",
            "Solve the inequality: 2x + 3 > 7",
            "What are complex numbers?",
            "What is i² equal to?",
            "Convert to polar form: 3 + 4i",
            "What is the remainder theorem?",
            "Factor completely: x⁴ - 16",
            "Solve the absolute value equation: |x - 3| = 5",

            "Find the area of a circle with radius 7cm",
            "Calculate the volume of a sphere with radius 4 meters",
            "What is the slope of the line passing through (2,3) and (5,9)?",
            "What is the Pythagorean theorem and how do you prove it?",
            "What is the perimeter of a rectangle with length 12cm and width 5cm?",
            "Find the area of a triangle with base 10 and height 6",
            "What is the circumference of a circle with diameter 14?",
            "Calculate the surface area of a cube with side length 5",
            "What is the volume of a cylinder with radius 3 and height 10?",
            "Find the hypotenuse of a right triangle with legs 3 and 4",
            "What is the area of a trapezoid?",
            "Calculate the volume of a cone",
            "What is the distance formula between two points?",
            "Find the midpoint between (2,3) and (8,7)",
            "What is the equation of a circle?",
            "What are similar triangles?",
            "What is the Pythagorean identity in trigonometry?",
            "Calculate the area of a sector of a circle",
            "What is the volume of a rectangular prism?",
            "Find the perimeter of a regular hexagon with side 5",

            "What is the cosine of 60 degrees?",
            "What is the sine of 90 degrees?",
            "What is the tangent of 45 degrees?",
            "What is the law of sines?",
            "What is the law of cosines?",
            "Convert 180 degrees to radians",
            "What is sin²(x) + cos²(x) equal to?",
            "Find the amplitude of y = 3sin(x)",
            "What is the period of tan(x)?",
            "What is arcsin(1/2)?",
            "What is the unit circle?",
            "Evaluate cos(π/3)",
            "What are the reciprocal trigonometric functions?",
            "What is the double angle formula for sine?",
            "Simplify: sin(2x)",

            "What is the probability of rolling two sixes with two dice?",
            "Calculate the standard deviation of the dataset: [2,4,6,8,10]",
            "What is the mean of 3, 7, 2, 9, 5, 11?",
            "What is the median of the numbers: 3, 7, 2, 9, 5, 11?",
            "What is the mode of a dataset?",
            "Calculate the variance of [1, 2, 3, 4, 5]",
            "What is a normal distribution?",
            "What is the Central Limit Theorem?",
            "Calculate P(A and B) if A and B are independent",
            "What is conditional probability?",
            "What is Bayes' theorem?",
            "Calculate the expected value of a dice roll",
            "What is a confidence interval?",
            "What is hypothesis testing?",
            "What is a p-value?",

            "What is the determinant of the matrix [[2,3],[1,4]]?",
            "Find all eigenvalues of the matrix [[5,2],[2,5]]",
            "What is the angle between vectors (1,2,3) and (4,5,6)?",
            "What is the dot product of two vectors?",
            "What is the cross product?",
            "Calculate the magnitude of vector (3,4)",
            "What is a matrix inverse?",
            "What is matrix multiplication?",
            "What is the identity matrix?",
            "What is the trace of a matrix?",
            "What is the rank of a matrix?",
            "What are eigenvectors?",
            "What is the null space of a matrix?",
            "What is linear independence?",
            "What is a basis of a vector space?"
        ]

        math_boundary_samples = [
            "How many ways can you arrange 5 books on a shelf?",
            "What is 15% of 240?",
            "If a train travels at 60 mph for 2.5 hours, how far does it go?",
            "Factor the polynomial: x³ - 8",
            "What is the median of the numbers: 3, 7, 2, 9, 5, 11?",
            "Convert 3/8 to a decimal",
            "What is the perimeter of a rectangle with length 12cm and width 5cm?",
            "Solve for x: log(x) + log(3) = log(12)",
            "What is the cosine of 60 degrees?",
            "If y = x² - 4x + 7, what is the vertex of this parabola?"
        ]

        coding_boundary_samples = [
            "How do I import a CSV file in Python using pandas?",
            "What does the 'this' keyword mean in JavaScript?",
            "How to create a virtual environment in Python?",
            "Explain the difference between '==' and '===' in JavaScript",
            "How do I read a file line by line in Python?",
            "What is the purpose of the 'super' keyword in Java?",
            "How to handle exceptions in Python using try-except?",
            "What are lambda functions in Python?",
            "How do I create a class in Python?",
            "What is the difference between a list and a tuple in Python?"
        ]

        all_coding = pure_coding_samples + coding_boundary_samples
        all_math = pure_math_samples + math_boundary_samples

        random.shuffle(all_coding)
        random.shuffle(all_math)

        coding_split = int(len(all_coding) * 0.8)
        math_split = int(len(all_math) * 0.8)

        db_samples = []
        eval_samples = []

        for prompt in all_coding[:coding_split]:
            db_samples.append({"prompt": prompt, "category": "coding", "source": "boundary_reinforcement"})
        for prompt in all_coding[coding_split:]:
            eval_samples.append({"query": prompt, "category": "coding"})

        for prompt in all_math[:math_split]:
            db_samples.append({"prompt": prompt, "category": "math", "source": "boundary_reinforcement"})
        for prompt in all_math[math_split:]:
            eval_samples.append({"query": prompt, "category": "math"})

        print(f"✅ Generated {len(db_samples)} DB samples and {len(eval_samples)} EVAL samples for boundary reinforcement")
        print(f"   - Pure Coding: {len(pure_coding_samples)} samples")
        print(f"   - Pure Math: {len(pure_math_samples)} samples")
        print(f"   - Boundary-defining: {len(math_boundary_samples) + len(coding_boundary_samples)} samples")
        print("="*60 + "\n")

        return db_samples, eval_samples

    def initialize_components(self):
        print(f"Loading sentence transformer model: {config.SENTENCE_TRANSFORMER_MODEL}")
        try:
            self.model = get_sentence_transformer()
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
            print(f"Creating new collection: {config.COLLECTION_NAME} with cosine distance metric")
            self.collection = self.client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ Successfully created collection '{config.COLLECTION_NAME}' with cosine distance.")
        except Exception as e:
            print(f"❌ Failed to create ChromaDB collection '{config.COLLECTION_NAME}': {e}")
            raise

    def load_coding_dataset(self, num_eval_samples: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        print("Loading coding dataset (KodCode/KodCode-V1)...")
        dataset_name = "KodCode/KodCode-V1"
        prompt_field_name = 'question'

        try:
            dataset_list = list(load_dataset(dataset_name, split="train", streaming=False))
            if not dataset_list:
                raise ValueError(f"Loaded {dataset_name} dataset split is empty.")

            db_size = config.CODING_DATASET_SIZE
            eval_size = num_eval_samples
            total_needed = db_size + eval_size

            if len(dataset_list) < total_needed:
                print(f"⚠️ Warning: Dataset has {len(dataset_list)} records, but {total_needed} are required. Adjusting sizes.")
                ratio = len(dataset_list) / total_needed
                db_size = int(db_size * ratio)
                eval_size = int(eval_size * ratio)

            print(f"Splitting dataset: {db_size} for DB, {eval_size} for evaluation (NO OVERLAP)")
            all_indices = list(range(len(dataset_list)))
            random.shuffle(all_indices)
            db_indices = all_indices[:db_size]
            eval_indices = all_indices[db_size:db_size + eval_size]
            db_records = [dataset_list[i] for i in db_indices]
            eval_records = [dataset_list[i] for i in eval_indices]

            db_samples = []
            eval_samples = []

            filtered_ambiguous_db = 0
            filtered_ambiguous_eval = 0

            for item in tqdm(db_records, desc="Processing coding DB samples"):
                prompt = item.get(prompt_field_name)
                if prompt and prompt.strip():

                    if self._is_ambiguous_coding_sample(prompt):
                        filtered_ambiguous_db += 1
                        continue
                    db_samples.append({
                        "prompt": prompt.strip(),
                        "category": "coding",
                        "source": "KodCode-V1"
                    })

            for item in tqdm(eval_records, desc="Processing coding EVAL samples"):
                prompt = item.get(prompt_field_name)
                if prompt and prompt.strip():
                    if self._is_ambiguous_coding_sample(prompt):
                        filtered_ambiguous_eval += 1
                        continue
                    eval_samples.append({
                        "query": prompt.strip(),
                        "category": "coding"
                    })

            print(f"🔍 Filtered {filtered_ambiguous_db} ambiguous DB samples and {filtered_ambiguous_eval} ambiguous EVAL samples from coding dataset")

            print(f"Successfully processed {len(db_samples)} DB samples and {len(eval_samples)} EVAL samples.")
            return db_samples, eval_samples

        except Exception as e:
            print(f"Error loading or processing {dataset_name} dataset: {e}")
            print("Using fallback coding prompts from config...")
            actual_fallback_size = min(len(config.FALLBACK_CODING_PROMPTS), config.CODING_DATASET_SIZE)
            db_samples = [{"prompt": prompt, "category": "coding", "source": "synthetic"}
                         for prompt in config.FALLBACK_CODING_PROMPTS[:actual_fallback_size]]
            eval_samples = []
            return db_samples, eval_samples

    def load_math_dataset(self, num_eval_samples: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load and prepare math dataset - returns (db_samples, eval_samples)"""
        print("Loading math dataset...")
        try:
            dataset_list = list(load_dataset("gsm8k", "main", split="train", streaming=False))
            if not dataset_list:
                raise ValueError("Loaded math (gsm8k) dataset is empty.")

            db_size = config.MATH_DATASET_SIZE
            eval_size = num_eval_samples
            total_needed = db_size + eval_size

            if len(dataset_list) < total_needed:
                print(f"⚠️ Warning: Dataset has {len(dataset_list)} records, but {total_needed} are required. Adjusting sizes.")
                db_size = min(db_size, len(dataset_list))
                eval_size = min(eval_size, len(dataset_list) - db_size)

            print(f"Splitting {total_needed} samples: {db_size} for DB, {eval_size} for EVAL (no overlap)...")
            all_indices = list(range(len(dataset_list)))
            random.shuffle(all_indices)
            db_indices = all_indices[:db_size]
            eval_indices = all_indices[db_size:db_size + eval_size]

            db_records = [dataset_list[i] for i in db_indices]
            eval_records = [dataset_list[i] for i in eval_indices]

            db_samples = []
            eval_samples = []
            filtered_ambiguous_db = 0
            filtered_ambiguous_eval = 0

            for item in tqdm(db_records, desc="Processing math DB samples"):
                prompt = item.get('question')
                if prompt and prompt.strip():
                    if self._is_ambiguous_math_sample(prompt):
                        filtered_ambiguous_db += 1
                        continue
                    db_samples.append({"prompt": prompt.strip(), "category": "math", "source": "gsm8k"})

            for item in tqdm(eval_records, desc="Processing math EVAL samples"):
                prompt = item.get('question')
                if prompt and prompt.strip():
                    if self._is_ambiguous_math_sample(prompt):
                        filtered_ambiguous_eval += 1
                        continue
                    eval_samples.append({
                        "query": prompt.strip(),
                        "category": "math"
                    })

            print(f"🔍 Filtered {filtered_ambiguous_db} ambiguous DB samples and {filtered_ambiguous_eval} ambiguous EVAL samples from math dataset")

            print(f"Successfully processed {len(db_samples)} DB samples and {len(eval_samples)} EVAL samples.")
            return db_samples, eval_samples

        except Exception as e:
            print(f"Error loading or processing math dataset: {e}")
            print("Using fallback math prompts from config...")
            actual_fallback_size = min(len(config.FALLBACK_MATH_PROMPTS), config.MATH_DATASET_SIZE)
            db_samples = [{"prompt": prompt, "category": "math", "source": "synthetic"}
                         for prompt in config.FALLBACK_MATH_PROMPTS[:actual_fallback_size]]
            eval_samples = []
            return db_samples, eval_samples

    def load_general_dataset(self, num_eval_samples: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load and prepare general knowledge dataset using TriviaQA - returns (db_samples, eval_samples)"""
        print("Loading general knowledge dataset (TriviaQA)...")
        try:
            dataset_list = list(load_dataset("trivia_qa", "rc.nocontext", split="validation", streaming=False))
            if not dataset_list:
                raise ValueError("Loaded TriviaQA dataset split is empty.")

            db_size = config.GENERAL_DATASET_SIZE
            eval_size = num_eval_samples
            total_needed = db_size + eval_size

            if len(dataset_list) < total_needed:
                print(f"⚠️ Warning: Dataset has {len(dataset_list)} records, but {total_needed} are required. Adjusting sizes.")
                db_size = min(db_size, len(dataset_list))
                eval_size = min(eval_size, len(dataset_list) - db_size)

            print(f"Splitting {total_needed} samples: {db_size} for DB, {eval_size} for EVAL (no overlap)...")
            all_indices = list(range(len(dataset_list)))
            random.shuffle(all_indices)
            db_indices = all_indices[:db_size]
            eval_indices = all_indices[db_size:db_size + eval_size]

            db_records = [dataset_list[i] for i in db_indices]
            eval_records = [dataset_list[i] for i in eval_indices]

            db_samples = []
            eval_samples = []

            for item in tqdm(db_records, desc="Processing general DB samples"):
                prompt = item.get('question')
                if prompt and prompt.strip():
                    db_samples.append({"prompt": prompt.strip(), "category": "general_knowledge", "source": "trivia_qa"})

            for item in tqdm(eval_records, desc="Processing general EVAL samples"):
                prompt = item.get('question')
                if prompt and prompt.strip():
                    eval_samples.append({
                        "query": prompt.strip(),
                        "category": "general_knowledge"
                    })

            print(f"Successfully processed {len(db_samples)} DB samples and {len(eval_samples)} EVAL samples.")
            return db_samples, eval_samples

        except Exception as e:
            print(f"Error loading or processing TriviaQA dataset: {e}")
            print("Using fallback general knowledge prompts from config...")
            actual_fallback_size = min(len(config.FALLBACK_GENERAL_PROMPTS), config.GENERAL_DATASET_SIZE)
            db_samples = [{"prompt": prompt, "category": "general_knowledge", "source": "synthetic"}
                         for prompt in config.FALLBACK_GENERAL_PROMPTS[:actual_fallback_size]]
            eval_samples = []
            return db_samples, eval_samples

    def load_llm_routing_general_dataset(self, num_eval_samples: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load and prepare general knowledge from LLM Routing dataset with balanced sampling per class - returns (db_samples, eval_samples)"""
        print("Loading LLM Routing dataset for GENERAL classes...")

        general_classes = ['conversation', 'science', 'toxic_harmful', 'logical_reasoning', 'sex', 'creative_writing']

        try:
            dataset_list = list(load_dataset("jeanvydes/llm-routing-text-classification", split="train", streaming=False))
            if not dataset_list:
                raise ValueError("Loaded LLM Routing dataset is empty.")

            db_size = config.GENERAL_DATASET_SIZE
            eval_size = num_eval_samples
            samples_per_class_db = db_size // len(general_classes)
            samples_per_class_eval = eval_size // len(general_classes)
            print(f"Target: {samples_per_class_db} DB samples per class (total {db_size}), {samples_per_class_eval} EVAL samples per class (total {eval_size})")

            print(f"Filtering and grouping samples by class: {general_classes}")
            class_samples = {cls: [] for cls in general_classes}

            for item in dataset_list:
                task_class = item.get('task')
                if task_class in general_classes:
                    class_samples[task_class].append(item)

            print("Class distribution in dataset:")
            for cls in general_classes:
                print(f"  - {cls}: {len(class_samples[cls])} samples")

            db_samples = []
            eval_samples = []

            for cls in general_classes:
                available = len(class_samples[cls])
                if available == 0:
                    print(f"⚠️ Warning: No samples found for class '{cls}'. Skipping.")
                    continue

                actual_db_count = min(samples_per_class_db, available)
                actual_eval_count = min(samples_per_class_eval, available - actual_db_count)

                shuffled_indices = random.sample(range(available), actual_db_count + actual_eval_count)
                db_indices = shuffled_indices[:actual_db_count]
                eval_indices = shuffled_indices[actual_db_count:actual_db_count + actual_eval_count]

                for idx in tqdm(db_indices, desc=f"Processing {cls} DB samples"):
                    item = class_samples[cls][idx]
                    prompt = item.get('prompt')
                    if prompt and prompt.strip():
                        db_samples.append({
                            "prompt": prompt.strip(),
                            "category": "general_knowledge",
                            "source": f"llm-routing-{cls}"
                        })

                for idx in tqdm(eval_indices, desc=f"Processing {cls} EVAL samples"):
                    item = class_samples[cls][idx]
                    prompt = item.get('prompt')
                    if prompt and prompt.strip():
                        eval_samples.append({
                            "query": prompt.strip(),
                            "category": "general_knowledge"
                        })

                print(f"✅ Sampled {actual_db_count} DB and {actual_eval_count} EVAL samples from class '{cls}'")

            print(f"Successfully processed {len(db_samples)} DB samples and {len(eval_samples)} EVAL samples from LLM Routing dataset (balanced).")
            return db_samples, eval_samples

        except Exception as e:
            print(f"Error loading or processing LLM Routing dataset for general classes: {e}")
            print("Skipping LLM Routing general dataset.")
            return [], []

    def load_llm_routing_math_dataset(self, num_eval_samples: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load and prepare math samples from LLM Routing dataset - returns (db_samples, eval_samples)"""
        print("Loading LLM Routing dataset for MATH class...")

        try:
            dataset_list = list(load_dataset("jeanvydes/llm-routing-text-classification", split="train", streaming=False))
            if not dataset_list:
                raise ValueError("Loaded LLM Routing dataset is empty.")

            print("Filtering for 'math' class...")
            filtered_list = [item for item in dataset_list if item.get('task') == 'math']
            print(f"Found {len(filtered_list)} math samples")

            if not filtered_list:
                print("⚠️ No math samples found. Skipping.")
                return [], []

            db_size = config.MATH_DATASET_SIZE
            eval_size = num_eval_samples
            total_needed = db_size + eval_size

            if len(filtered_list) < total_needed:
                print(f"⚠️ Warning: Filtered dataset has {len(filtered_list)} records, but {total_needed} are required. Adjusting sizes.")
                db_size = min(db_size, len(filtered_list))
                eval_size = min(eval_size, len(filtered_list) - db_size)

            print(f"Splitting {total_needed} samples: {db_size} for DB, {eval_size} for EVAL (no overlap)...")
            all_indices = list(range(len(filtered_list)))
            random.shuffle(all_indices)
            db_indices = all_indices[:db_size]
            eval_indices = all_indices[db_size:db_size + eval_size]

            db_records = [filtered_list[i] for i in db_indices]
            eval_records = [filtered_list[i] for i in eval_indices]

            db_samples = []
            eval_samples = []
            filtered_ambiguous_db = 0
            filtered_ambiguous_eval = 0

            for item in tqdm(db_records, desc="Processing LLM Routing math DB samples"):
                prompt = item.get('prompt')
                if prompt and prompt.strip():
                    if self._is_ambiguous_math_sample(prompt):
                        filtered_ambiguous_db += 1
                        continue
                    db_samples.append({"prompt": prompt.strip(), "category": "math", "source": "llm-routing-dataset"})

            for item in tqdm(eval_records, desc="Processing LLM Routing math EVAL samples"):
                prompt = item.get('prompt')
                if prompt and prompt.strip():
                    if self._is_ambiguous_math_sample(prompt):
                        filtered_ambiguous_eval += 1
                        continue
                    eval_samples.append({
                        "query": prompt.strip(),
                        "category": "math"
                    })

            print(f"🔍 Filtered {filtered_ambiguous_db} ambiguous DB samples and {filtered_ambiguous_eval} ambiguous EVAL samples from LLM Routing math dataset")

            print(f"Successfully processed {len(db_samples)} DB samples and {len(eval_samples)} EVAL samples from LLM Routing math dataset.")
            return db_samples, eval_samples

        except Exception as e:
            print(f"Error loading or processing LLM Routing dataset for math class: {e}")
            print("Skipping LLM Routing math dataset.")
            return [], []

    def embed_and_store_samples(self, samples: List[Dict[str, Any]], batch_size: int = None):
        """Embed samples and store them in ChromaDB"""
        if not samples:
            print("No samples to embed and store.")
            return

        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE

        print(f"Embedding and storing {len(samples)} samples (batch_size={batch_size})...")
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            prompts = [sample["prompt"] for sample in batch]
            try:
                embeddings = self.model.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, 1e-8, None)
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

            total_db_stored = 0
            all_eval_samples = []

            coding_eval_samples = config.EVALUATION_SET_SIZE
            math_eval_samples_per_source = config.EVALUATION_SET_SIZE // 2
            general_eval_samples_per_source = config.EVALUATION_SET_SIZE // 2

            print(f"\n{'='*60}")
            print("EVALUATION DATASET BALANCING")
            print(f"{'='*60}")
            print(f"Target per category: {config.EVALUATION_SET_SIZE} samples")
            print(f"  - Coding: {coding_eval_samples} samples (1 source)")
            print(f"  - Math: {math_eval_samples_per_source} × 2 sources = {math_eval_samples_per_source * 2} samples")
            print(f"  - General: {general_eval_samples_per_source} × 2 sources = {general_eval_samples_per_source * 2} samples")
            print(f"{'='*60}\n")

            datasets_to_process = [
                ('boundary_reinforcement', lambda n: self._generate_boundary_reinforcement_samples(), 0),
                ('coding', self.load_coding_dataset, coding_eval_samples),
                ('math_gsm8k', self.load_math_dataset, math_eval_samples_per_source),
                ('math_llm_routing', self.load_llm_routing_math_dataset, math_eval_samples_per_source),
                ('general_trivia', self.load_general_dataset, general_eval_samples_per_source),
                ('general_llm_routing', self.load_llm_routing_general_dataset, general_eval_samples_per_source)
            ]

            print("\n" + "="*60)
            print("PROCESSING DATASETS SEQUENTIALLY (Memory Efficient)")
            print("="*60)

            for dataset_name, load_func, num_eval_samples in datasets_to_process:
                print(f"\n[{dataset_name.upper()}] Loading dataset...")
                print(f"[{dataset_name.upper()}] Requesting {num_eval_samples} evaluation samples")

                db_samples, eval_samples = load_func(num_eval_samples)

                print(f"[{dataset_name.upper()}] Loaded {len(db_samples)} DB samples and {len(eval_samples)} EVAL samples")

                if db_samples:
                    print(f"[{dataset_name.upper()}] Embedding and storing DB samples in ChromaDB...")
                    self.embed_and_store_samples(db_samples)
                    total_db_stored += len(db_samples)
                    print(f"[{dataset_name.upper()}] ✅ Stored {len(db_samples)} DB samples in ChromaDB")

                if eval_samples:
                    all_eval_samples.extend(eval_samples)
                    print(f"[{dataset_name.upper()}] ✅ Collected {len(eval_samples)} EVAL samples (NOT in ChromaDB)")

                print(f"[{dataset_name.upper()}] Clearing from memory...")
                del db_samples, eval_samples
                gc.collect()
                print(f"[{dataset_name.upper()}] ✅ Memory cleared\n")

            try:
                count = self.collection.count()
                print(f"\n{'='*60}")
                print(f"✅ ChromaDB Build Complete!")
                print(f"Total embeddings stored: {count}")
                print(f"Database location: {os.path.abspath(config.CHROMADB_PATH)}")
                print(f"{'='*60}\n")
            except Exception as count_e:
                print(f"Warning: Could not get final count from collection: {count_e}")

            print("\n" + "="*60)
            print("CREATING EVALUATION DATASET (from collected samples, NO DATA LEAKAGE)")
            print("="*60)

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
