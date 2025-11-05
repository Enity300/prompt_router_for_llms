#!/usr/bin/env python3
"""
- Statistical significance testing
- Cross-validation evaluation
- Multiple baseline comparisons & speed benchmarking
- Performance visualizations (Accuracy, Confusion Matrices, Token Length Impact)
- Comparative error and per-category analysis
"""

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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

# --- Real Imports ---
from semantic_router import SemanticRouter
from rich.console import Console

warnings.filterwarnings('ignore')

# Initialize the real console for rich text output
console = Console()

class ComprehensiveEvaluator:

    def __init__(self):
        self.router = None
        self.results = {}
        
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
        results['statistical_tests'] = self._statistical_significance_testing(
            results['cross_validation']['fold_accuracies'], baseline_cv_scores
        )

        console.print("\n[bold cyan]Step 3: Running final evaluation & benchmarking on unseen test set...[/bold cyan]")
        all_final_performances = self._evaluate_final_performance_for_all(self.X_test, self.y_test)
        results['final_performance'] = all_final_performances

        console.print("\n[bold cyan]Step 4: Analyzing performance impact of query token length for all models...[/bold cyan]")
        # MODIFIED: Call the new function that analyzes multiple models
        results['token_length_analysis'] = self._analyze_token_length_for_all_models(
            results['final_performance']
        )
        
        console.print("\n[bold cyan]📊 Generating visualizations...[/bold cyan]")
        self._generate_visualizations(results)
        
        console.print("\n[bold cyan]📝 Generating final report...[/bold cyan]")
        self._generate_report(results)
        
        return results

    def _initialize_components(self):
        try:
            self.router = SemanticRouter()
        except Exception as e:
            console.print(f"❌ Failed to initialize SemanticRouter: {e}")
            raise
    
    def _evaluate_final_performance_for_all(self, X_test, y_test) -> Dict[str, Any]:
        all_performances = {}
        
        # 1. Evaluate Semantic Router
        console.print("Benchmarking Semantic Router...")
        sr_predictions, sr_latencies = [], []
        for query in X_test:
            start_time = time.perf_counter()
            try:
                result = self.router.route(query)
                predicted_category = result['category'] if result else None
            except Exception:
                predicted_category = None
            latency = time.perf_counter() - start_time
            sr_predictions.append(predicted_category)
            sr_latencies.append(latency)
        all_performances['Semantic Router'] = self._calculate_performance_metrics(
            "Semantic Router", y_test, sr_predictions, sr_latencies, X_test
        )
        
        # 2. Evaluate Baselines
        baselines = self._get_baseline_models(self.X_train_val, self.y_train_val)
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_train_val_vec = vectorizer.fit_transform(self.X_train_val)
        X_test_vec = vectorizer.transform(X_test)

        for name, model in baselines.items():
            console.print(f"Benchmarking {model.name}...")
            start_time = time.perf_counter()
            if name == "tfidf_svm":
                model.fit(X_train_val_vec, self.y_train_val)
                preds = model.predict(X_test_vec)
            else:
                preds = model.predict(X_test.reshape(-1, 1))
            latency = time.perf_counter() - start_time
            avg_latency = (latency / len(X_test)) if len(X_test) > 0 else 0
            all_performances[model.name] = self._calculate_performance_metrics(
                model.name, y_test, preds, [avg_latency] * len(X_test), X_test
            )
        return all_performances
        
    def _calculate_performance_metrics(self, model_name, y_true, y_pred, latencies, queries=None) -> Dict:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        routing_decisions = None
        if queries is not None:
             routing_decisions = [{
                "query": q, "true_category": gt, "predicted_category": p,
                "correct": gt == p
            } for q, gt, p in zip(queries, y_true, y_pred)]
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "labels": sorted(list(set(y_true) | set(y_pred))),
            "avg_latency_ms": np.mean(latencies) * 1000,
            "throughput_qps": 1 / np.mean(latencies) if np.mean(latencies) > 0 else float('inf'),
            "routing_decisions": routing_decisions
        }

    def _get_baseline_models(self, X_data, y_data) -> Dict[str, Any]:
        class RuleBasedPredictor:
            name = "Rule-based Keywords"
            def predict(self, queries):
                queries = queries.flatten()
                return np.array([self._predict_single(q) for q in queries])
            def _predict_single(self, query):
                query_lower = query.lower()
                if any(word in query_lower for word in ['code', 'function', 'algorithm', 'program', 'script']): return 'coding'
                elif any(word in query_lower for word in ['calculate', 'solve', 'equation', 'math', 'derivative']): return 'math'
                else: return 'general_knowledge'
        dummy_random = DummyClassifier(strategy="uniform", random_state=42)
        dummy_random.fit(X_data.reshape(-1, 1), y_data)
        dummy_random.name = "Random Classifier"
        dummy_freq = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy_freq.fit(X_data.reshape(-1, 1), y_data)
        dummy_freq.name = "Most Frequent Class"
        svm_classifier = SVC(kernel='rbf', random_state=42)
        svm_classifier.name = "TF-IDF + SVM"
        return {"random": dummy_random, "most_frequent": dummy_freq, "rule_based": RuleBasedPredictor(), "tfidf_svm": svm_classifier}

    def _cross_validation_with_baselines(self, X, y) -> Tuple[Dict, Dict]:
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        ss_ger_fold_accuracies, baseline_fold_scores = [], { "Random Classifier": [], "Most Frequent Class": [], "Rule-based Keywords": [], "TF-IDF + SVM": [] }
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            console.print(f"Running Cross-Validation Fold {fold+1}/{k_folds}...")
            X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
            val_preds_sr = []
            for q in X_val:
                try:
                    result = self.router.route(q)
                    predicted_category = result['category'] if result else None
                except Exception:
                    predicted_category = None
                val_preds_sr.append(predicted_category)
            ss_ger_fold_accuracies.append(accuracy_score(y_val, val_preds_sr))
            baselines = self._get_baseline_models(X_train, y_train)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_val_vec = vectorizer.transform(X_val)
            for name, model in baselines.items():
                if name == "tfidf_svm":
                    model.fit(X_train_vec, y_train)
                    preds = model.predict(X_val_vec)
                else:
                    preds = model.predict(X_val.reshape(-1, 1))
                baseline_fold_scores[model.name].append(accuracy_score(y_val, preds))
        return {"fold_accuracies": ss_ger_fold_accuracies}, baseline_fold_scores

    def _statistical_significance_testing(self, ss_ger_accuracies: List[float], baseline_cv_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        significance_tests = {}
        for baseline_name, baseline_scores in baseline_cv_scores.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                t_stat, p_value = stats.ttest_rel(ss_ger_accuracies, baseline_scores)
            significance_tests[baseline_name] = {"t_statistic": t_stat, "p_value": p_value, "significant": p_value < 0.05}
        return significance_tests

    # NEW: Helper function for token analysis logic
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

    # MODIFIED: This function now orchestrates the analysis for multiple models
    def _analyze_token_length_for_all_models(self, all_performances: Dict[str, Any]) -> Dict[str, Any]:
        all_token_analyses = {}
        # Define which models are relevant for this analysis
        relevant_models = ["Semantic Router", "TF-IDF + SVM", "Rule-based Keywords"]
        for model_name, performance_data in all_performances.items():
            if model_name in relevant_models:
                console.print(f"Analyzing token length impact for {model_name}...")
                analysis = self._perform_token_length_analysis(model_name, performance_data['routing_decisions'])
                all_token_analyses[model_name] = analysis
        return all_token_analyses

    def _generate_visualizations(self, results: Dict[str, Any]):
        viz_dir = "evaluation_results"
        os.makedirs(viz_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        performances = results['final_performance']
        model_names = list(performances.keys())
        accuracies = [p['accuracy'] for p in performances.values()]
        latencies = [p['avg_latency_ms'] for p in performances.values()]

        # Professional color palette (consistent with academic style)
        color_map = {
            'Semantic Router': '#8B4789',      # Purple
            'Random Classifier': '#5B9BD5',    # Blue
            'Most Frequent Class': '#70AD47',  # Green
            'Rule-based Keywords': '#FFC000',  # Gold
            'TF-IDF + SVM': '#C55A11'         # Orange
        }
        
        # Figure 1a: Accuracy Comparison (Separate PNG)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Sort models by accuracy for better visualization
        sorted_data = sorted(zip(model_names, accuracies), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_acc = zip(*sorted_data)
        
        # Get colors in sorted order
        sorted_colors = [color_map.get(name, '#808080') for name in sorted_names]
        
        # Accuracy Chart
        bars1 = ax1.barh(sorted_names, sorted_acc, color=sorted_colors, 
                         edgecolor='black', linewidth=1.5, alpha=0.85)
        ax1.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Model', fontsize=13, fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontsize=15, fontweight='bold', pad=15)
        ax1.set_xlim(0, 1.05)
        ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.set_axisbelow(True)
        
        # Add percentage labels on bars
        for bar, acc in zip(bars1, sorted_acc):
            ax1.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{acc*100:.1f}%',
                    va='center', ha='left', fontsize=11, fontweight='bold')
        
        # Format x-axis as percentage
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 1b: Latency Comparison (Separate PNG)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Sort by latency (lower is better, so sort ascending)
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
        
        # Add latency value labels on bars
        for bar, lat in zip(bars2, sorted_lat_only):
            ax2.text(lat + (max(sorted_lat_only) * 0.01), bar.get_y() + bar.get_height()/2, 
                    f'{lat:.2f} ms',
                    va='center', ha='left', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/latency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: Confusion Matrices - SEPARATE PNG FOR EACH MODEL
        for name, data in performances.items():
            if data["labels"]:
                fig, ax = plt.subplots(figsize=(8, 7))
                
                # Create heatmap
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
                
                # Save with model name in filename
                safe_name = name.replace(' ', '_').replace('+', 'plus').lower()
                plt.savefig(f"{viz_dir}/confusion_matrix_{safe_name}.png", dpi=300, bbox_inches='tight')
                plt.close()

        # Figure 3: Token Length Impact - MULTI-LINE CHART (Academic Style)
        token_analyses = results.get('token_length_analysis', {})
        if token_analyses:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Professional color palette matching scatter plot
            color_map = {
                'Semantic Router': '#8B4789',
                'Rule-based Keywords': '#FFC000',
                'TF-IDF + SVM': '#C55A11',
                'Random Classifier': '#5B9BD5',
                'Most Frequent Class': '#70AD47'
            }
            
            # Distinct markers and line styles
            marker_map = {
                'Semantic Router': 'o',
                'Rule-based Keywords': 's',
                'TF-IDF + SVM': '^',
                'Random Classifier': 'D',
                'Most Frequent Class': 'v'
            }
            
            linestyle_map = {
                'Semantic Router': '-',
                'Rule-based Keywords': '--',
                'TF-IDF + SVM': '-',
                'Random Classifier': ':',
                'Most Frequent Class': '-.'
            }
            
            # Plot each model
            for model_name, analysis in token_analyses.items():
                if analysis.get('results'):
                    df = pd.DataFrame(analysis['results'])
                    token_ranges = df['token_range'].tolist()
                    accuracies_per_range = df['accuracy'].tolist()
                    
                    color = color_map.get(model_name, '#808080')
                    marker = marker_map.get(model_name, 'o')
                    linestyle = linestyle_map.get(model_name, '-')
                    
                    # Plot line
                    ax.plot(token_ranges, accuracies_per_range, 
                           marker=marker, 
                           markersize=8,
                           linewidth=2,
                           linestyle=linestyle,
                           label=model_name,
                           color=color,
                           markeredgecolor='black',
                           markeredgewidth=1,
                           alpha=0.9)
            
            # Formatting
            ax.set_xlabel("Query Token Length Range", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title('Model Performance Across Token Length Ranges', 
                        fontsize=13, pad=12)
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            
            # Professional grid
            ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
            ax.set_axisbelow(True)
            
            # Set appropriate y-limits
            all_accuracies = []
            for analysis in token_analyses.values():
                if analysis.get('results'):
                    all_accuracies.extend([r['accuracy'] for r in analysis['results']])
            
            if all_accuracies:
                min_acc = min(all_accuracies)
                max_acc = max(all_accuracies)
                range_acc = max_acc - min_acc
                
                # Set limits based on data range
                if range_acc > 0.3:  # Large variation
                    ax.set_ylim(0, 1.05)
                else:  # Small variation - zoom in
                    y_padding = max(0.05, range_acc * 0.2)
                    ax.set_ylim(max(0, min_acc - y_padding), min(1.05, max_acc + y_padding))
            
            # Legend
            ax.legend(loc='best', fontsize=10, framealpha=0.95, 
                     edgecolor='gray', fancybox=False)
            
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/token_length_impact.png", dpi=300, bbox_inches='tight')
            plt.close()

        console.print(f"✅ Visualizations saved to '{viz_dir}/' directory.")

    def _generate_report(self, results: Dict[str, Any]):
        report_file = "evaluation_results/evaluation_report.md"
        performances = results['final_performance']
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Evaluation Report\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 1. Overall Performance Comparison\n\n")
            f.write("| Model | Accuracy | Avg Latency (ms) | Throughput (q/s) |\n|---|---|---|---|\n")
            for name, data in sorted(performances.items(), key=lambda item: item[1]['accuracy'], reverse=True):
                f.write(f"| **{name}** | **{data['accuracy']:.3f}** | {data['avg_latency_ms']:.3f} | {data['throughput_qps']:.2f} |\n")
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
                f.write("## 3. Statistical Significance (vs Semantic Router)\n\n")
                f.write("| Comparison vs Baseline | t-statistic | p-value | Significant (p < 0.05) |\n|---|---|---|---|\n")
                for name, data in results['statistical_tests'].items():
                    f.write(f"| Semantic Router vs {name} | {data['t_statistic']:.3f} | {data['p_value']:.4f} | {'✅ Yes' if data['significant'] else '❌ No'} |\n")
                f.write("\n")
            
            # MODIFIED: Generate a report section for each token length analysis
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