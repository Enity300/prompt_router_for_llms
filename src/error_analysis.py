
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class ErrorAnalyzer:
    """Analyze classification errors and failure patterns"""

    def __init__(self, evaluation_results_path: str = "evaluation_results"):
        self.results_path = Path(evaluation_results_path)
        self.output_dir = self.results_path / "error_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_data = None
        self.results = {}

    def load_evaluation_results(self, results_file: str = "evaluation_dataset.json") -> bool:
        """Load evaluation dataset"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.evaluation_data = json.load(f)
            console.print(f"✅ Loaded {len(self.evaluation_data)} evaluation samples")
            return True
        except FileNotFoundError:
            console.print(f"❌ Evaluation file '{results_file}' not found")
            return False
        except Exception as e:
            console.print(f"❌ Error loading evaluation data: {e}")
            return False

    def analyze_model_errors(self, routing_decisions: List[Dict], model_name: str) -> Dict[str, Any]:
        """Perform comprehensive error analysis for a single model"""

        correct = [d for d in routing_decisions if d['correct']]
        incorrect = [d for d in routing_decisions if not d['correct']]

        if not routing_decisions:
            return {"error": "No routing decisions provided"}

        total = len(routing_decisions)
        error_count = len(incorrect)
        error_rate = error_count / total if total > 0 else 0

        analysis = {
            "model_name": model_name,
            "total_samples": total,
            "correct_predictions": len(correct),
            "incorrect_predictions": error_count,
            "error_rate": error_rate,
            "accuracy": 1 - error_rate
        }

        if error_count == 0:
            console.print(f"[green]✨ {model_name}: Perfect classification (0 errors)![/green]")
            analysis["failure_patterns"] = "No failures to analyze"
            return analysis

        confusion_pairs = defaultdict(int)
        for err in incorrect:
            pair = (err['true_category'], err['predicted_category'])
            confusion_pairs[pair] += 1

        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        analysis["confusion_pairs"] = {
            f"{true_cat} → {pred_cat}": count
            for (true_cat, pred_cat), count in sorted_pairs
        }

        error_by_category = defaultdict(list)
        for err in incorrect:
            error_by_category[err['true_category']].append(err)

        category_error_rates = {}
        for category in set(d['true_category'] for d in routing_decisions):
            cat_total = len([d for d in routing_decisions if d['true_category'] == category])
            cat_errors = len(error_by_category[category])
            category_error_rates[category] = {
                "total": cat_total,
                "errors": cat_errors,
                "error_rate": cat_errors / cat_total if cat_total > 0 else 0
            }

        analysis["error_by_category"] = category_error_rates

        analysis["example_failures"] = []
        for i, err in enumerate(incorrect[:10]):
            analysis["example_failures"].append({
                "query": err['query'][:150] + ("..." if len(err['query']) > 150 else ""),
                "true_category": err['true_category'],
                "predicted_category": err['predicted_category']
            })

        error_lengths = [len(err['query'].split()) for err in incorrect]
        if error_lengths:
            analysis["error_query_stats"] = {
                "mean_length": np.mean(error_lengths),
                "median_length": np.median(error_lengths),
                "min_length": min(error_lengths),
                "max_length": max(error_lengths)
            }

        return analysis

    def compare_model_errors(self, model_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare error patterns across multiple models"""

        comparison = {
            "models_compared": len(model_analyses),
            "error_rate_comparison": {},
            "common_failures": None,
            "unique_failures": {}
        }

        for model_name, analysis in model_analyses.items():
            if "error_rate" in analysis:
                comparison["error_rate_comparison"][model_name] = {
                    "error_rate": analysis["error_rate"],
                    "error_count": analysis.get("incorrect_predictions", 0)
                }

        model_failures = {}
        for model_name, analysis in model_analyses.items():
            if "example_failures" in analysis and analysis["example_failures"]:
                model_failures[model_name] = set(
                    f["query"] for f in analysis["example_failures"]
                )

        if len(model_failures) >= 2:
            all_failed_queries = []
            for queries in model_failures.values():
                all_failed_queries.extend(queries)

            query_fail_count = Counter(all_failed_queries)
            common = [(q, c) for q, c in query_fail_count.items() if c > 1]

            if common:
                comparison["common_failures"] = {
                    "count": len(common),
                    "examples": sorted(common, key=lambda x: x[1], reverse=True)[:5]
                }

        return comparison

    def visualize_error_patterns(self, model_analyses: Dict[str, Dict]):
        """Create visualizations for error patterns"""

        models_with_errors = {
            name: analysis.get("error_rate", 0)
            for name, analysis in model_analyses.items()
            if "error_rate" in analysis
        }

        if models_with_errors:
            fig, ax = plt.subplots(figsize=(12, 6))

            sorted_models = sorted(models_with_errors.items(), key=lambda x: x[1], reverse=True)
            model_names = [m[0] for m in sorted_models]
            error_rates = [m[1] for m in sorted_models]

            colors = ['#e74c3c' if er > 0.1 else '#f39c12' if er > 0.05 else '#2ecc71'
                     for er in error_rates]

            bars = ax.barh(model_names, error_rates, color=colors,
                          edgecolor='black', linewidth=1.5, alpha=0.85)

            ax.set_xlabel('Error Rate', fontsize=13, fontweight='bold')
            ax.set_ylabel('Model', fontsize=13, fontweight='bold')
            ax.set_title('Model Error Rate Comparison', fontsize=15, fontweight='bold', pad=15)
            ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)

            for bar, er in zip(bars, error_rates):
                ax.text(er + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{er*100:.2f}%',
                       va='center', ha='left', fontsize=11, fontweight='bold')

            plt.tight_layout()
            plt.savefig(self.output_dir / "error_rate_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

            console.print(f"✅ Saved error rate comparison to {self.output_dir}")

        models_with_category_errors = {
            name: analysis.get("error_by_category", {})
            for name, analysis in model_analyses.items()
            if "error_by_category" in analysis and analysis.get("incorrect_predictions", 0) > 0
        }

        if models_with_category_errors:
            categories = set()
            for cat_errors in models_with_category_errors.values():
                categories.update(cat_errors.keys())
            categories = sorted(categories)

            fig, ax = plt.subplots(figsize=(14, 7))

            x = np.arange(len(categories))
            width = 0.15
            num_models = len(models_with_category_errors)
            start_offset = -(num_models - 1) * width / 2

            for idx, (model_name, cat_errors) in enumerate(models_with_category_errors.items()):
                error_rates = [cat_errors.get(cat, {}).get('error_rate', 0) for cat in categories]
                offset = start_offset + idx * width

                ax.bar(x + offset, error_rates, width,
                      label=model_name, alpha=0.85, edgecolor='black', linewidth=0.8)

            ax.set_xlabel('Category', fontsize=13, fontweight='bold')
            ax.set_ylabel('Error Rate', fontsize=13, fontweight='bold')
            ax.set_title('Error Rate by Category and Model', fontsize=15, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

            plt.tight_layout()
            plt.savefig(self.output_dir / "error_by_category.png", dpi=300, bbox_inches='tight')
            plt.close()

            console.print(f"✅ Saved category error analysis to {self.output_dir}")

    def generate_error_report(self, model_analyses: Dict[str, Dict], comparison: Dict):
        """Generate comprehensive error analysis report"""

        report_file = self.output_dir / "error_analysis_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Error Analysis Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Overview\n\n")
            f.write(f"- **Models Analyzed:** {len(model_analyses)}\n")
            f.write(f"- **Total Comparisons:** {comparison.get('models_compared', 0)}\n\n")

            f.write("## Error Rate Summary\n\n")
            f.write("| Model | Total Samples | Errors | Error Rate | Accuracy |\n")
            f.write("|-------|---------------|--------|------------|----------|\n")

            sorted_models = sorted(
                model_analyses.items(),
                key=lambda x: x[1].get('error_rate', 0)
            )

            for model_name, analysis in sorted_models:
                if 'error_rate' in analysis:
                    f.write(f"| {model_name} | {analysis['total_samples']} | "
                           f"{analysis['incorrect_predictions']} | "
                           f"{analysis['error_rate']:.4f} | "
                           f"{analysis['accuracy']:.4f} |\n")

            f.write("\n")

            f.write("## Detailed Model Analysis\n\n")

            for model_name, analysis in sorted_models:
                if analysis.get('incorrect_predictions', 0) > 0:
                    f.write(f"### {model_name}\n\n")
                    f.write(f"**Error Statistics:**\n")
                    f.write(f"- Total Errors: {analysis['incorrect_predictions']}\n")
                    f.write(f"- Error Rate: {analysis['error_rate']:.2%}\n\n")

                    if "confusion_pairs" in analysis:
                        f.write(f"**Most Common Confusion Patterns:**\n\n")
                        for pair, count in list(analysis['confusion_pairs'].items())[:5]:
                            f.write(f"- {pair}: {count} occurrences\n")
                        f.write("\n")

                    if "error_by_category" in analysis:
                        f.write(f"**Error Distribution by Category:**\n\n")
                        f.write("| Category | Total | Errors | Error Rate |\n")
                        f.write("|----------|-------|--------|------------|\n")
                        for cat, stats in analysis["error_by_category"].items():
                            f.write(f"| {cat.replace('_', ' ').title()} | {stats['total']} | "
                                   f"{stats['errors']} | {stats['error_rate']:.2%} |\n")
                        f.write("\n")

                    if "example_failures" in analysis and analysis["example_failures"]:
                        f.write(f"**Example Failure Cases:**\n\n")
                        for i, failure in enumerate(analysis["example_failures"][:5], 1):
                            f.write(f"{i}. **Query:** \"{failure['query']}\"\n")
                            f.write(f"   - **True Category:** {failure['true_category']}\n")
                            f.write(f"   - **Predicted:** {failure['predicted_category']}\n\n")
                else:
                    f.write(f"### {model_name}\n\n")
                    f.write(f"✅ **Perfect Classification** - No errors detected!\n\n")

            if comparison.get("common_failures"):
                f.write("## Cross-Model Analysis\n\n")
                f.write(f"**Common Difficult Queries:**\n\n")
                f.write(f"Found {comparison['common_failures']['count']} queries that "
                       f"multiple models failed on.\n\n")

                if comparison['common_failures']['examples']:
                    f.write("Top examples:\n\n")
                    for query, fail_count in comparison['common_failures']['examples']:
                        f.write(f"- \"{query[:100]}...\" (failed by {fail_count} models)\n")
                    f.write("\n")

            f.write("## Key Findings\n\n")

            best_model = sorted_models[0]
            f.write(f"1. **Best Performer:** {best_model[0]} "
                   f"(Error Rate: {best_model[1].get('error_rate', 0):.4f})\n")

            worst_model = sorted_models[-1]
            f.write(f"2. **Needs Improvement:** {worst_model[0]} "
                   f"(Error Rate: {worst_model[1].get('error_rate', 0):.4f})\n")

            error_rates = [a.get('error_rate', 0) for _, a in sorted_models if 'error_rate' in a]
            if error_rates:
                f.write(f"3. **Error Rate Range:** {min(error_rates):.4f} to {max(error_rates):.4f}\n")

        console.print(f"✅ Generated error analysis report: {report_file}")
        return str(report_file)

    def run_complete_analysis(self, comprehensive_results: Dict = None) -> Dict[str, Any]:
        """Run complete error analysis pipeline"""

        console.print("\n[bold cyan]🔍 Starting Comprehensive Error Analysis[/bold cyan]\n")

        if comprehensive_results is None:
            console.print("No results provided. Load evaluation data manually or run comprehensive_evaluation first.")
            return {}

        if 'final_performance' not in comprehensive_results:
            console.print("❌ No final_performance data in results")
            return {}

        model_analyses = {}

        for model_name, performance_data in comprehensive_results['final_performance'].items():
            if 'routing_decisions' in performance_data and performance_data['routing_decisions']:
                console.print(f"Analyzing errors for: [cyan]{model_name}[/cyan]")

                analysis = self.analyze_model_errors(
                    performance_data['routing_decisions'],
                    model_name
                )
                model_analyses[model_name] = analysis

        console.print("\nComparing error patterns across models...")
        comparison = self.compare_model_errors(model_analyses)

        console.print("\nGenerating error visualizations...")
        self.visualize_error_patterns(model_analyses)

        console.print("\nGenerating comprehensive report...")
        report_path = self.generate_error_report(model_analyses, comparison)

        console.print(f"\n[bold green]✅ Error Analysis Complete![/bold green]")
        console.print(f"📁 Results saved to: {self.output_dir}")
        console.print(f"📄 Report: {report_path}")

        return {
            "model_analyses": model_analyses,
            "comparison": comparison,
            "report_path": report_path
        }


def main():
    """Main entry point for standalone execution"""

    console.print("[bold green]🔍 Error Analysis for SS-GER[/bold green]\n")

    try:
        from src.comprehensive_evaluation import ComprehensiveEvaluator

        console.print("Running comprehensive evaluation to get fresh data...")
        evaluator = ComprehensiveEvaluator()
        results = evaluator.run_comprehensive_evaluation()

        if results:
            console.print("\n" + "="*60)
            analyzer = ErrorAnalyzer()
            error_results = analyzer.run_complete_analysis(results)

            console.print("\n[bold green]🎉 Analysis pipeline completed successfully![/bold green]")
            return error_results
        else:
            console.print("❌ Failed to get evaluation results")
            return None

    except Exception as e:
        console.print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

