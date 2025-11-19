"""
Regenerate Token Length Impact Visualization
=============================================
This script regenerates ONLY the token length impact chart
without rerunning the entire evaluation.

Usage:
    python regenerate_token_viz.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """Load saved evaluation results"""
    results_file = "evaluation_results/evaluation_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("   Run 'python src/comprehensive_evaluation.py' first to generate results.")
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"✅ Loaded results from {results_file}")
        print(f"   Timestamp: {results.get('timestamp', 'Unknown')}")
        return results
    except Exception as e:
        print(f"❌ Failed to load results: {e}")
        return None

def regenerate_token_length_viz(results):
    """Regenerate token length impact visualization"""
    token_analyses = results.get('token_length_analysis', {})
    
    if not token_analyses:
        print("❌ No token length analysis data found in results.")
        return
    
    print(f"\n📊 Regenerating token length visualization...")
    print(f"   Models: {list(token_analyses.keys())}")
    
    viz_dir = "evaluation_results"
    os.makedirs(viz_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
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
        output_file = f"{viz_dir}/token_length_impact.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Token length visualization saved to: {output_file}")
    else:
        print("⚠️ No token length data available for visualization")

def main():
    print("="*60)
    print("TOKEN LENGTH VISUALIZATION REGENERATOR")
    print("="*60)
    
    results = load_results()
    if results is None:
        return
    
    regenerate_token_length_viz(results)
    
    print("\n" + "="*60)
    print("✅ DONE! Token length visualization regenerated.")
    print("="*60)

if __name__ == "__main__":
    main()

