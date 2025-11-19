import sys
import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import logging
from tqdm import tqdm

from semantic_router import SemanticRouter
from sentence_transformer_catboost_router import SentenceTransformerCatBoostRouter
from rich.console import Console
from config import config
from utils.model_loader import get_sentence_transformer

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

class ComprehensiveEvaluator:

    def __init__(self):
        self.router = None
        self.results = {}
        self.sentence_model = None

        evaluation_data = self._load_evaluation_dataset()
        if not evaluation_data:
            console.print("❌ No data loaded. Exiting.")
            sys.exit(1)

        queries = np.array([item["query"] for item in evaluation_data])
        ground_truth = np.array([item["category"] for item in evaluation_data])

        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
            queries, ground_truth, test_size=0.40, random_state=42, stratify=ground_truth
        )
        console.print(f"✅ Data loaded: {len(self.X_train_val)} training/validation samples (60%), {len(self.X_test)} test samples (40%).")

    def _load_evaluation_dataset(self) -> List[Dict[str, Any]]:
        filepath = "evaluation_dataset.json"
        console.print(f"Attempting to load evaluation data from '{filepath}'...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            console.print(f"✅ Successfully loaded {len(data)} records from '{filepath}'.")
            return data
        except FileNotFoundError:
            console.print(f"❌ Error: The file '{filepath}' was not found. Please create it first.")
            return []
        except Exception as e:
            console.print(f"❌ An unexpected error occurred while loading '{filepath}': {e}")
            return []

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        console.print("\n[bold green]Comprehensive Evaluation Suite[/bold green]")

        results = {}
        self._initialize_components()

        console.print("\n[bold cyan]Step 1: Performing cross-validation and baseline comparison...[/bold cyan]")
        cv_results, baseline_cv_scores = self._cross_validation_with_baselines(self.X_train_val, self.y_train_val)
        results['cross_validation'] = cv_results

        console.print("\n[bold cyan]Step 2: Computing statistical significance...[/bold cyan]")
        reference_accuracies = results['cross_validation'].get('st_catboost_fold_accuracies')
        if not reference_accuracies or len(reference_accuracies) == 0:
            console.print("⚠️ Sentence Transformer + CatBoost Router not available, using Semantic Router for statistical tests")
            reference_accuracies = results['cross_validation']['fold_accuracies']
            reference_model_name = "Semantic Router"
        else:
            reference_model_name = "Sentence Transformer + CatBoost Router"
        
        results['statistical_tests'] = self._statistical_significance_testing(
            reference_accuracies, baseline_cv_scores, reference_model_name
        )

        console.print("\n[bold cyan]Step 3: Running final evaluation & benchmarking on unseen test set...[/bold cyan]")
        all_final_performances = self._evaluate_final_performance_for_all(self.X_test, self.y_test)
        results['final_performance'] = all_final_performances

        console.print("\n[bold cyan]Step 4: Analyzing performance impact of query token length for all models...[/bold cyan]")
        results['token_length_analysis'] = self._analyze_token_length_for_all_models(
            results['final_performance']
        )

        console.print("\n[bold cyan]📊 Generating visualizations...[/bold cyan]")
        self._generate_visualizations(results)

        console.print("\n[bold cyan]📝 Generating final report...[/bold cyan]")
        self._generate_report(results)

        console.print("\n[bold cyan]💾 Saving results for future visualization...[/bold cyan]")
        self._save_results(results)

        return results

    def _initialize_components(self):
        try:
            self.router = SemanticRouter()
            console.print(f"Loading Sentence Transformer model for baseline: {config.SENTENCE_TRANSFORMER_MODEL}")
            self.sentence_model = get_sentence_transformer()
        except Exception as e:
            console.print(f"❌ Failed to initialize components: {e}")
            raise

    def _evaluate_final_performance_for_all(self, X_test, y_test) -> Dict[str, Any]:
        if len(X_test) != len(y_test):
            raise ValueError(f"Length mismatch: X_test has {len(X_test)} samples, y_test has {len(y_test)} labels")
        if len(X_test) == 0:
            raise ValueError("Empty test set provided")

        all_performances = {}

        console.print("Benchmarking Semantic Router...")
        sr_predictions, sr_latencies = [], []
        for query in tqdm(X_test, desc="Evaluating Semantic Router", leave=False):
            start_time = time.perf_counter()
            try:
                result = self.router.route(query)
                predicted_category = result['category'] if result else None
            except Exception as e:
                logger.warning(f"Router failed on query: '{query[:50]}...' Error: {e}")
                predicted_category = None
            latency = time.perf_counter() - start_time
            sr_predictions.append(predicted_category)
            sr_latencies.append(latency)
        all_performances['Semantic Router'] = self._calculate_performance_metrics(
            "Semantic Router", y_test, sr_predictions, sr_latencies, X_test
        )

        try:
            console.print("Benchmarking Sentence Transformer + CatBoost Router (Pre-trained)...")
            st_catboost_router = SentenceTransformerCatBoostRouter()
            if st_catboost_router.is_trained:
                st_predictions, st_latencies = [], []
                for query in tqdm(X_test, desc="Evaluating Sentence Transformer + CatBoost Router", leave=False):
                    start_time = time.perf_counter()
                    result = st_catboost_router.route(query)
                    predicted_category = result['category']
                    latency = time.perf_counter() - start_time
                    st_predictions.append(predicted_category)
                    st_latencies.append(latency)
                all_performances['Sentence Transformer + CatBoost Router'] = self._calculate_performance_metrics(
                    "Sentence Transformer + CatBoost Router", y_test, st_predictions, st_latencies, X_test
                )
            else:
                console.print("⚠️ Sentence Transformer + CatBoost Router not trained yet. Skipping.")
                console.print("   Train it first: python src/sentence_transformer_catboost_router.py")
        except Exception as e:
            console.print(f"⚠️ Failed to load Sentence Transformer + CatBoost Router: {e}")
            logger.warning(f"Sentence Transformer + CatBoost Router evaluation skipped: {e}")

        baselines = self._get_baseline_models(self.X_train_val, self.y_train_val)

        for name, model in baselines.items():
            console.print(f"Benchmarking {model.name}...")

            if name.startswith("tfidf_"):
                model.fit(self.X_train_val, self.y_train_val)
            else:
                model.fit(self.X_train_val.reshape(-1, 1), self.y_train_val)

            inference_latencies = []
            if name.startswith("tfidf_"):
                for test_query in X_test:
                    start_time = time.perf_counter()
                    pred = model.predict([test_query])
                    latency = time.perf_counter() - start_time
                    inference_latencies.append(latency)
                preds = model.predict(X_test)
            else:
                for test_query in X_test:
                    start_time = time.perf_counter()
                    pred = model.predict(np.array([test_query]).reshape(-1, 1))
                    latency = time.perf_counter() - start_time
                    inference_latencies.append(latency)
                preds = model.predict(X_test.reshape(-1, 1))

            all_performances[model.name] = self._calculate_performance_metrics(
                model.name, y_test, preds, inference_latencies, X_test
            )
        return all_performances

    def _calculate_performance_metrics(self, model_name, y_true, y_pred, latencies, queries=None) -> Dict:
        y_pred_flat = np.array(y_pred).flatten().tolist()
        y_true_flat = np.array(y_true).flatten().tolist()

        report = classification_report(y_true_flat, y_pred_flat, output_dict=True, zero_division=0)

        routing_decisions = None
        if queries is not None:
            routing_decisions = [{
                "query": q,
                "true_category": gt,
                "predicted_category": p,
                "correct": gt == p
            } for q, gt, p in zip(queries, y_true_flat, y_pred_flat)]

        return {
            "accuracy": accuracy_score(y_true_flat, y_pred_flat),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_true_flat, y_pred_flat).tolist(),
            "labels": sorted(list(set(y_true_flat) | set(y_pred_flat))),
            "avg_latency_ms": np.mean(latencies) * 1000,
            "routing_decisions": routing_decisions
        }

    def _get_baseline_models(self, X_data, y_data) -> Dict[str, Any]:
        class RuleBasedPredictor:
            name = "Rule-based Keywords"
            def fit(self, X, y):
                return self
            def predict(self, queries):
                queries = queries.flatten()
                return np.array([self._predict_single(q) for q in queries])
            def _predict_single(self, query):
                query_lower = query.lower()
                coding_keywords = [
                    'code', 'function', 'algorithm', 'program', 'script', 'debug', 'compile',
                    'class', 'method', 'variable', 'array', 'loop', 'syntax', 'api', 'library',
                    'framework', 'database', 'query', 'sql', 'python', 'java', 'javascript',
                    'implement', 'execute', 'runtime', 'exception', 'bug', 'test', 'git',
                    'repository', 'commit', 'branch', 'merge', 'deploy', 'docker', 'kubernetes',
                    'recursion', 'iteration', 'binary', 'search', 'sort', 'hash', 'tree', 'graph',
                    'stack', 'queue', 'linked list', 'pointer', 'memory', 'optimization',
                    'complexity', 'big o', 'refactor', 'inheritance', 'polymorphism', 'encapsulation'
                ]
                math_keywords = [
                    'calculate', 'solve', 'equation', 'math', 'derivative', 'integral', 'calculus',
                    'algebra', 'geometry', 'trigonometry', 'probability', 'statistics', 'theorem',
                    'proof', 'formula', 'polynomial', 'exponential', 'logarithm', 'matrix',
                    'vector', 'scalar', 'coefficient', 'variable', 'constant', 'factor',
                    'multiply', 'divide', 'sum', 'difference', 'quotient', 'remainder',
                    'prime', 'composite', 'fraction', 'decimal', 'percentage', 'ratio',
                    'proportion', 'inequality', 'limit', 'series', 'sequence', 'convergence',
                    'divergence', 'eigenvalue', 'determinant', 'slope', 'tangent', 'cosine',
                    'sine', 'angle', 'radius', 'diameter', 'circumference', 'area', 'volume',
                    'perimeter', 'pythagorean', 'quadratic', 'cubic', 'linear'
                ]
                if any(word in query_lower for word in coding_keywords): return 'coding'
                elif any(word in query_lower for word in math_keywords): return 'math'
                else: return 'general_knowledge'

        dummy_random = DummyClassifier(strategy="uniform", random_state=42)
        dummy_random.fit(X_data.reshape(-1, 1), y_data)
        dummy_random.name = "Random Classifier"
        dummy_freq = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy_freq.fit(X_data.reshape(-1, 1), y_data)
        dummy_freq.name = "Most Frequent Class"

        svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', SVC(kernel='rbf', random_state=42))
        ])
        svm_pipeline.name = "TF-IDF + SVM"

        logistic_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        logistic_pipeline.name = "TF-IDF + Logistic Regression"

        rf_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        rf_pipeline.name = "TF-IDF + Random Forest"

        nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        nb_pipeline.name = "TF-IDF + Naive Bayes"

        catboost_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=False,
            allow_writing_files=False
            ))
        ])
        catboost_pipeline.name = "TF-IDF + CatBoost"

        return {
            "random": dummy_random,
            "most_frequent": dummy_freq,
            "rule_based": RuleBasedPredictor(),
            "tfidf_svm": svm_pipeline,
            "tfidf_logistic": logistic_pipeline,
            "tfidf_rf": rf_pipeline,
            "tfidf_nb": nb_pipeline,
            "tfidf_catboost": catboost_pipeline
        }

    def _cross_validation_with_baselines(self, X, y) -> Tuple[Dict, Dict]:
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        ss_ger_fold_accuracies = []
        st_catboost_fold_accuracies = []
        baseline_fold_scores = {
            "Semantic Router": [],
            "Random Classifier": [],
            "Most Frequent Class": [],
            "Rule-based Keywords": [],
            "TF-IDF + SVM": [],
            "TF-IDF + Logistic Regression": [],
            "TF-IDF + Random Forest": [],
            "TF-IDF + Naive Bayes": [],
            "TF-IDF + CatBoost": []
        }

        try:
            st_catboost_router = SentenceTransformerCatBoostRouter()
            has_catboost_router = st_catboost_router.is_trained
        except Exception as e:
            console.print(f"⚠️ Sentence Transformer + CatBoost Router not available for CV: {e}")
            has_catboost_router = False

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            console.print(f"Running Cross-Validation Fold {fold+1}/{k_folds}...")
            X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

            val_preds_sr = []
            for q in X_val:
                try:
                    result = self.router.route(q)
                    predicted_category = result['category'] if result else None
                except Exception as e:
                    logger.warning(f"Router failed during CV fold {fold+1}: {e}")
                    predicted_category = None
                val_preds_sr.append(predicted_category)
            ss_ger_fold_accuracies.append(accuracy_score(y_val, val_preds_sr))
            baseline_fold_scores["Semantic Router"].append(accuracy_score(y_val, val_preds_sr))

            if has_catboost_router:
                val_preds_stcb = []
                for q in X_val:
                    try:
                        result = st_catboost_router.route(q)
                        predicted_category = result['category']
                    except Exception as e:
                        logger.warning(f"CatBoost Router failed during CV fold {fold+1}: {e}")
                        predicted_category = None
                    val_preds_stcb.append(predicted_category)
                st_catboost_fold_accuracies.append(accuracy_score(y_val, val_preds_stcb))

            baselines = self._get_baseline_models(X_train, y_train)

            for name, model in baselines.items():
                if name.startswith("tfidf_"):
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                else:
                    model.fit(X_train.reshape(-1, 1), y_train)
                    preds = model.predict(X_val.reshape(-1, 1))
                baseline_fold_scores[model.name].append(accuracy_score(y_val, preds))

        return {
            "fold_accuracies": ss_ger_fold_accuracies,
            "st_catboost_fold_accuracies": st_catboost_fold_accuracies
        }, baseline_fold_scores

    def _statistical_significance_testing(self, reference_accuracies: List[float], baseline_cv_scores: Dict[str, List[float]], reference_model_name: str) -> Dict[str, Any]:
        significance_tests = {"reference_model": reference_model_name}
        for baseline_name, baseline_scores in baseline_cv_scores.items():
            if baseline_name == reference_model_name:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                t_stat, p_value = stats.ttest_rel(reference_accuracies, baseline_scores)
            significance_tests[baseline_name] = {"t_statistic": t_stat, "p_value": p_value, "significant": p_value < 0.05}
        return significance_tests

    def _perform_token_length_analysis(self, model_name: str, routing_decisions: List[Dict]) -> Dict:
        if not routing_decisions: return {}
        df = pd.DataFrame(routing_decisions)
        df['token_count'] = df['query'].apply(lambda x: len(str(x).split()))
        buckets, results = [(1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, 1000, "51+")], []
        for min_tokens, max_tokens, label in buckets:
            bucket_df = df[(df['token_count'] >= min_tokens) & (df['token_count'] <= max_tokens)]
            if len(bucket_df) == 0: continue
            accuracy = bucket_df['correct'].mean()
            results.append({'token_range': label, 'sample_count': len(bucket_df), 'accuracy': accuracy})
        return {"results": results, "model_name": model_name}

    def _analyze_token_length_for_all_models(self, all_performances: Dict[str, Any]) -> Dict[str, Any]:
        all_token_analyses = {}

        for model_name, performance_data in all_performances.items():
            if performance_data.get('routing_decisions'):
                console.print(f"Analyzing token length impact for {model_name}...")
                analysis = self._perform_token_length_analysis(model_name, performance_data['routing_decisions'])
                all_token_analyses[model_name] = analysis
            else:
                logger.debug(f"Skipping token analysis for {model_name} (no routing decisions)")

        return all_token_analyses

    def _generate_visualizations(self, results: Dict[str, Any]):
        viz_dir = "evaluation_results"
        os.makedirs(viz_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        performances = results['final_performance']
        model_names = list(performances.keys())
        accuracies = [p['accuracy'] for p in performances.values()]
        latencies = [p['avg_latency_ms'] for p in performances.values()]

        color_map = {
            'Semantic Router': '#8B4789',
            'Random Classifier': '#5B9BD5',
            'Most Frequent Class': '#70AD47',
            'Rule-based Keywords': '#FFC000',
            'TF-IDF + SVM': '#C55A11',
            'TF-IDF + Logistic Regression': '#E74C3C',
            'TF-IDF + Random Forest': '#16A085',
            'TF-IDF + Naive Bayes': '#9B59B6',
            'TF-IDF + CatBoost': '#E67E22',
            'Sentence Transformer + CatBoost Router': '#2ECC71'
        }

        fig1, ax1 = plt.subplots(figsize=(10, 6))

        sorted_data = sorted(zip(model_names, accuracies), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_acc = zip(*sorted_data)

        sorted_colors = [color_map.get(name, '#808080') for name in sorted_names]

        bars1 = ax1.barh(sorted_names, sorted_acc, color=sorted_colors,
                         edgecolor='black', linewidth=1.5, alpha=0.85)
        ax1.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Model', fontsize=13, fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontsize=15, fontweight='bold', pad=15)
        ax1.set_xlim(0, 1.05)
        ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.set_axisbelow(True)

        for bar, acc in zip(bars1, sorted_acc):
            ax1.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc*100:.1f}%',
                    va='center', ha='left', fontsize=11, fontweight='bold')

        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

        plt.tight_layout()
        plt.savefig(f"{viz_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(10, 6))

        sorted_data_lat = sorted(zip(model_names, latencies), key=lambda x: x[1])
        sorted_names_lat, sorted_lat_only = zip(*sorted_data_lat)
        sorted_colors_lat = [color_map.get(name, '#808080') for name in sorted_names_lat]

        bars2 = ax2.barh(sorted_names_lat, sorted_lat_only, color=sorted_colors_lat,
                         edgecolor='black', linewidth=1.5, alpha=0.85)
        ax2.set_xlabel('Average Latency (ms)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Model', fontsize=13, fontweight='bold')
        ax2.set_title('Model Latency Comparison', fontsize=15, fontweight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.set_axisbelow(True)

        for bar, lat in zip(bars2, sorted_lat_only):
            ax2.text(lat + (max(sorted_lat_only) * 0.01), bar.get_y() + bar.get_height()/2,
                    f'{lat:.2f} ms',
                    va='center', ha='left', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{viz_dir}/latency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        for name, data in performances.items():
            if data["labels"]:
                fig, ax = plt.subplots(figsize=(8, 7))

                sns.heatmap(np.array(data["confusion_matrix"]),
                           annot=True,
                           fmt='d',
                           cmap='Blues',
                           ax=ax,
                           xticklabels=data["labels"],
                           yticklabels=data["labels"],
                           cbar_kws={'label': 'Count'},
                           linewidths=0.5,
                           linecolor='gray')

                ax.set_title(f'Confusion Matrix: {name}', fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
                ax.set_ylabel('True Category', fontsize=12, fontweight='bold')

                plt.tight_layout()

                safe_name = name.replace(' ', '_').replace('+', 'plus').lower()
                plt.savefig(f"{viz_dir}/confusion_matrix_{safe_name}.png", dpi=300, bbox_inches='tight')
                plt.close()

        token_analyses = results.get('token_length_analysis', {})
        if token_analyses:
            fig, ax = plt.subplots(figsize=(28, 14))

            color_map = {
                'Semantic Router': '#8B4789',
                'Rule-based Keywords': '#FFC000',
                'TF-IDF + SVM': '#C55A11',
                'TF-IDF + Logistic Regression': '#E74C3C',
                'TF-IDF + Random Forest': '#16A085',
                'TF-IDF + Naive Bayes': '#9B59B6',
                'TF-IDF + CatBoost': '#E67E22',
                'Sentence Transformer + CatBoost Router': '#2ECC71',
                'Random Classifier': '#5B9BD5',
                'Most Frequent Class': '#70AD47'
            }

            token_ranges = None
            model_data = {}

            for model_name, analysis in token_analyses.items():
                if analysis.get('results'):
                    df = pd.DataFrame(analysis['results'])
                    if token_ranges is None:
                        token_ranges = df['token_range'].tolist()
                    model_data[model_name] = df['accuracy'].tolist()

            if token_ranges and model_data:
                x = np.arange(len(token_ranges))
                width = 0.08
                num_models = len(model_data)

                start_offset = -(num_models - 1) * width / 2

                all_bar_data = []

                for idx, (model_name, accuracies) in enumerate(model_data.items()):
                    offset = start_offset + idx * width
                    color = color_map.get(model_name, '#808080')

                    bars = ax.bar(x + offset, accuracies, width,
                                 label=model_name,
                                 color=color,
                                 edgecolor='black',
                                 linewidth=0.6,
                                 alpha=0.85)

                    for bar_idx, (bar, acc) in enumerate(zip(bars, accuracies)):
                        height = bar.get_height()
                        if height > 0.03:
                            all_bar_data.append({
                                'x': bar.get_x() + bar.get_width()/2.,
                                'y': height,
                                'acc': acc,
                                'bar_idx': bar_idx,
                                'model_idx': idx
                            })

                for i, data in enumerate(all_bar_data):
                    overlaps = False
                    for j, other in enumerate(all_bar_data):
                        if i != j and data['bar_idx'] == other['bar_idx']:
                            x_dist = abs(data['x'] - other['x'])
                            y_dist = abs(data['y'] - other['y'])
                            if x_dist < width * 1.5 and y_dist < 0.08:
                                overlaps = True
                                break

                    if overlaps and data['y'] > 0.85:
                        rotation = 45
                        fontsize = 10
                        y_offset = 0.015
                    elif overlaps:
                        rotation = 90
                        fontsize = 9
                        y_offset = 0.02
                    else:
                        rotation = 0
                        fontsize = 11
                        y_offset = 0.015

                    ax.text(data['x'], data['y'] + y_offset,
                           f'{data["acc"]:.0%}',
                           ha='center', va='bottom',
                           fontsize=fontsize, fontweight='bold',
                           rotation=rotation)

                ax.set_xlabel("Number of Tokens", fontsize=16, fontweight='bold')
                ax.set_ylabel("Accuracy", fontsize=16, fontweight='bold')
                ax.set_title('Model Performance Across Token Length Ranges (Grouped Comparison)',
                            fontsize=18, fontweight='bold', pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels(token_ranges, fontsize=14)

                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                ax.tick_params(axis='y', labelsize=13)

                all_accuracies = [acc for accs in model_data.values() for acc in accs]
                if all_accuracies:
                    min_acc = min(all_accuracies)
                    if min_acc > 0.4:
                        ax.set_ylim(0, 1.10)
                    else:
                        ax.set_ylim(0, 1.10)

                ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                ax.set_axisbelow(True)

                ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                         fontsize=12, framealpha=0.95,
                         edgecolor='gray', fancybox=False)

                plt.tight_layout()
                plt.savefig(f"{viz_dir}/token_length_impact.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                console.print("⚠️ No token length data available for visualization")

        console.print(f"✅ Visualizations saved to '{viz_dir}/' directory.")

    def _generate_report(self, results: Dict[str, Any]):
        report_file = "evaluation_results/evaluation_report.md"
        performances = results['final_performance']
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Evaluation Report\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 1. Overall Performance Comparison\n\n")
            f.write("| Model | Accuracy | Avg Latency (ms) |\n|---|---|---|\n")
            for name, data in sorted(performances.items(), key=lambda item: item[1]['accuracy'], reverse=True):
                f.write(f"| **{name}** | **{data['accuracy']:.3f}** | {data['avg_latency_ms']:.3f} |\n")
            f.write("\n")
            if performances and list(performances.values())[0]['labels']:
                f.write("## 2. Per-Category Performance (F1-Score)\n\n")
                categories = sorted(list(performances.values())[0]['labels'])
                header = "| Model | " + " | ".join([cat if cat is not None else "None" for cat in categories]) + " |\n"
                f.write(header)
                f.write("|---| " + " | ".join(["---"] * len(categories)) + " |\n")
                for name, data in performances.items():
                    row = f"| {name} |"
                    for cat in categories:
                        f1_score = data['classification_report'].get(str(cat), {}).get('f1-score', 0.0)
                        row += f" {f1_score:.3f} |"
                    f.write(row + "\n")
                f.write("\n")
            if results.get('statistical_tests'):
                reference_model = results['statistical_tests'].get('reference_model', 'Semantic Router')
                f.write(f"## 3. Statistical Significance (vs {reference_model})\n\n")
                f.write("| Comparison | t-statistic | p-value | Significant (p < 0.05) |\n|---|---|---|---|\n")
                for name, data in results['statistical_tests'].items():
                    if name == 'reference_model':
                        continue
                    f.write(f"| {reference_model} vs {name} | {data['t_statistic']:.3f} | {data['p_value']:.4f} | {'✅ Yes' if data['significant'] else '❌ No'} |\n")
                f.write("\n")

            token_analyses = results.get('token_length_analysis', {})
            if token_analyses:
                f.write("## 4. Token Length Impact Analysis\n\n")
                for model_name, analysis in token_analyses.items():
                    if analysis and analysis.get('results'):
                        f.write(f"### {analysis['model_name']}\n\n")
                        f.write("| Token Range | Sample Count | Accuracy |\n|---|---|---|\n")
                        for res in analysis['results']:
                            f.write(f"| {res['token_range']} | {res['sample_count']} | {res['accuracy']:.2%} |\n")
                        f.write("\n")
        console.print(f"✅ Comprehensive report saved to '{report_file}'")

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON file for later visualization"""
        results_file = "evaluation_results/evaluation_results.json"
        
        serializable_results = {
            'final_performance': {},
            'token_length_analysis': results.get('token_length_analysis', {}),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for model_name, perf_data in results['final_performance'].items():
            serializable_results['final_performance'][model_name] = {
                'accuracy': perf_data['accuracy'],
                'avg_latency_ms': perf_data['avg_latency_ms'],
                'labels': perf_data['labels'],
                'confusion_matrix': perf_data['confusion_matrix'],
                'classification_report': perf_data['classification_report']
            }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2)
            console.print(f"✅ Results saved to '{results_file}'")
        except Exception as e:
            console.print(f"⚠️ Failed to save results: {e}")

def main():
    evaluator = ComprehensiveEvaluator()
    try:
        results = evaluator.run_comprehensive_evaluation()
        console.print("\n[bold green]Comprehensive evaluation complete![/bold green]")
        console.print("📁 Results saved to 'evaluation_results/' directory.")
        return results
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
