"""
Features:
- t-SNE and PCA dimensionality reduction
- Expertise manifold boundary visualization
- Query routing path analysis
- Interactive 3D visualizations
- Publication-quality plots
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from semantic_router import SemanticRouter
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import config
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import chromadb
except ImportError as e:
    print(f"Missing dependencies. Install with: pip install plotly scikit-learn")
    print(f"Error: {e}")
    sys.exit(1)

console = Console()

class EmbeddingVisualizer:

    def __init__(self):
        self.router = None
        self.embeddings_data = []
        self.categories = []
        self.queries = []

    def initialize_components(self):
        try:
            self.router = SemanticRouter()
            console.print("✅ SemanticRouter initialized")

            self._load_embeddings_from_db()
            console.print(f"✅ Loaded {len(self.embeddings_data)} embeddings")

        except Exception as e:
            console.print(f"❌ Failed to initialize components: {e}")
            raise

    def _load_embeddings_from_db(self):
        try:
            results = self.router.collection.get(
                include=['embeddings', 'metadatas', 'documents']
            )

            if results['embeddings'] is not None and len(results['embeddings']) > 0:
                self.embeddings_data = np.array(results['embeddings'])
                self.categories = [meta['category'] for meta in results['metadatas']]
                self.queries = results['documents']

                console.print(f"Loaded embeddings shape: {self.embeddings_data.shape}")
                console.print(f"Categories distribution: {pd.Series(self.categories).value_counts().to_dict()}")
            else:
                console.print("⚠️ No embeddings found in database")

        except Exception as e:
            console.print(f"❌ Failed to load embeddings: {e}")
            raise

    def create_comprehensive_visualizations(self):

        console.print("\n[bold green]🎨 Generating Comprehensive Embedding Visualizations[/bold green]")

        viz_dir = "embedding_visualizations"
        os.makedirs(viz_dir, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task1 = progress.add_task("Creating 2D t-SNE visualization...", total=100)
            self._create_tsne_2d_plot(viz_dir)
            progress.update(task1, advance=100)

            task2 = progress.add_task("Creating 3D PCA visualization...", total=100)
            self._create_pca_3d_plot(viz_dir)
            progress.update(task2, advance=100)

            task3 = progress.add_task("Analyzing expertise manifold boundaries...", total=100)
            self._create_manifold_boundary_plot(viz_dir)
            progress.update(task3, advance=100)

            task4 = progress.add_task("Creating similarity heatmap...", total=100)
            self._create_similarity_heatmap(viz_dir)
            progress.update(task4, advance=100)

            task5 = progress.add_task("Visualizing query routing paths...", total=100)
            self._create_routing_visualization(viz_dir)
            progress.update(task5, advance=100)

            task6 = progress.add_task("Performing cluster analysis...", total=100)
            self._create_cluster_analysis(viz_dir)
            progress.update(task6, advance=100)

        console.print(f"\n✅ All visualizations saved to '{viz_dir}/' directory")
        return viz_dir

    def _create_tsne_2d_plot(self, output_dir: str):
        """Create 2D t-SNE visualization of embedding space"""

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = tsne.fit_transform(self.embeddings_data)

        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'category': self.categories,
            'query': [q[:50] + '...' if len(q) > 50 else q for q in self.queries]
        })

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = {'coding': '#2E86AB', 'math': '#A23B72', 'general_knowledge': '#F18F01'}

        for category in df['category'].unique():
            mask = df['category'] == category
            ax.scatter(df[mask]['x'], df[mask]['y'],
                      c=colors.get(category, '#000000'),
                      label=category.replace('_', ' ').title(),
                      alpha=0.7, s=60, edgecolors='black', linewidths=0.5)

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('SS-GER Expertise Manifolds in 2D Semantic Space\n(t-SNE Projection)',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Expertise Domain', title_fontsize=11, fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/tsne_2d_manifolds.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig_interactive = px.scatter(
            df, x='x', y='y', color='category',
            hover_data=['query'],
            title='Interactive 2D t-SNE: SS-GER Expertise Manifolds',
            labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2', 'category': 'Expertise Domain'},
            color_discrete_map=colors
        )
        fig_interactive.update_traces(marker_size=8)
        fig_interactive.write_html(f"{output_dir}/tsne_2d_interactive.html")

    def _create_pca_3d_plot(self, output_dir: str):
        """Create 3D PCA visualization"""

        pca = PCA(n_components=3, random_state=42)
        embeddings_3d = pca.fit_transform(self.embeddings_data)

        df = pd.DataFrame({
            'x': embeddings_3d[:, 0],
            'y': embeddings_3d[:, 1],
            'z': embeddings_3d[:, 2],
            'category': self.categories,
            'query': [q[:50] + '...' if len(q) > 50 else q for q in self.queries]
        })

        colors = {'coding': '#2E86AB', 'math': '#A23B72', 'general_knowledge': '#F18F01'}

        fig = go.Figure()

        for category in df['category'].unique():
            mask = df['category'] == category
            fig.add_trace(go.Scatter3d(
                x=df[mask]['x'],
                y=df[mask]['y'],
                z=df[mask]['z'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=colors.get(category, '#000000'),
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                name=category.replace('_', ' ').title(),
                text=df[mask]['query'],
                hovertemplate='<b>%{text}</b><br>' +
                             'PC1: %{x:.2f}<br>' +
                             'PC2: %{y:.2f}<br>' +
                             'PC3: %{z:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title='3D PCA: Expertise Manifolds in Semantic Space',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)',
            ),
            legend=dict(title='Expertise Domain'),
            width=900,
            height=700
        )

        fig.write_html(f"{output_dir}/pca_3d_manifolds.html")

        fig.write_image(f"{output_dir}/pca_3d_manifolds.png", width=900, height=700)

    def _create_manifold_boundary_plot(self, output_dir: str):

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = tsne.fit_transform(self.embeddings_data)

        h = 0.1
        x_min, x_max = embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1
        y_min, y_max = embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        from sklearn.ensemble import RandomForestClassifier

        X = embeddings_2d
        y = np.array(self.categories)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(mesh_points)

        unique_labels = np.unique(y)
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        Z_numeric = np.array([label_to_num[label] for label in Z])
        Z_numeric = Z_numeric.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(14, 10))

        contour = ax.contourf(xx, yy, Z_numeric, alpha=0.3, levels=len(unique_labels)-1)

        colors = {'coding': '#2E86AB', 'math': '#A23B72', 'general_knowledge': '#F18F01'}

        for i, category in enumerate(unique_labels):
            mask = y == category
            ax.scatter(X[mask, 0], X[mask, 1],
                      c=colors.get(category, '#000000'),
                      label=category.replace('_', ' ').title(),
                      alpha=0.8, s=80, edgecolors='black', linewidths=1)

        ax.contour(xx, yy, Z_numeric, colors='black', linewidths=1, alpha=0.6)

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(r'\Expertise Manifold Decision Boundaries\n(Random Forest Classification on t-SNE Space)',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Expertise Domain', title_fontsize=11, fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/manifold_boundaries.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_similarity_heatmap(self, output_dir: str):

        category_embeddings = {}
        for category in np.unique(self.categories):
            mask = np.array(self.categories) == category
            category_embeddings[category] = np.mean(self.embeddings_data[mask], axis=0)

        categories = list(category_embeddings.keys())
        similarity_matrix = np.zeros((len(categories), len(categories)))

        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                emb1 = category_embeddings[cat1]
                emb2 = category_embeddings[cat2]
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarity_matrix[i, j] = similarity

        fig, ax = plt.subplots(figsize=(8, 6))

        category_names = [cat.replace('_', ' ').title() for cat in categories]

        sns.heatmap(similarity_matrix,
                   xticklabels=category_names,
                   yticklabels=category_names,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0.5,
                   square=True,
                   ax=ax)

        ax.set_title('Inter-Category Semantic Similarity Matrix\n(Cosine Similarity of Average Embeddings)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/similarity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_routing_visualization(self, output_dir: str):

        test_queries = [
            {"query": "Write a Python sorting algorithm", "expected": "coding"},
            {"query": "Calculate integral of x^2", "expected": "math"},
            {"query": "What is artificial intelligence?", "expected": "general_knowledge"},
            {"query": "Debug memory leak in JavaScript", "expected": "coding"},
            {"query": "Solve quadratic equation", "expected": "math"}
        ]

        query_embeddings = []
        routing_results = []

        for item in test_queries:
            embedding = self.router._embed_prompt(item["query"])
            query_embeddings.append(embedding)

            result = self.router.route(item["query"])
            routing_results.append(result)

        query_embeddings = np.array(query_embeddings)

        all_embeddings = np.vstack([self.embeddings_data, query_embeddings])
        all_categories = self.categories + ['query'] * len(test_queries)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        all_embeddings_2d = tsne.fit_transform(all_embeddings)

        manifold_embeddings_2d = all_embeddings_2d[:-len(test_queries)]
        query_embeddings_2d = all_embeddings_2d[-len(test_queries):]

        fig, ax = plt.subplots(figsize=(14, 10))

        colors = {'coding': '#2E86AB', 'math': '#A23B72', 'general_knowledge': '#F18F01'}

        for category in np.unique(self.categories):
            mask = np.array(self.categories) == category
            ax.scatter(manifold_embeddings_2d[mask, 0], manifold_embeddings_2d[mask, 1],
                      c=colors.get(category, '#000000'),
                      alpha=0.4, s=30,
                      label=f'{category.replace("_", " ").title()} Manifold')

        for i, (query_item, result) in enumerate(zip(test_queries, routing_results)):
            x, y = query_embeddings_2d[i]
            predicted_category = result["category"]
            confidence = result["confidence"]

            color = colors.get(predicted_category, '#000000')

            size = 50 + confidence * 200

            ax.scatter(x, y, c=color, s=size, alpha=0.9,
                      edgecolors='black', linewidths=2,
                      marker='*')

            ax.annotate(f"Q{i+1}: {query_item['query'][:30]}...\n→ {predicted_category} ({confidence:.2f})",
                       xy=(x, y), xytext=(10, 10),
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       fontsize=8, ha='left')

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Query Routing Visualization\n(Test Queries in Expertise Manifold Space)',
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/routing_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_cluster_analysis(self, output_dir: str):
        """Perform and visualize cluster analysis"""

        k_range = range(2, 8)
        silhouette_scores = []
        inertias = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings_data)

            silhouette_avg = silhouette_score(self.embeddings_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Silhouette Score', fontsize=12)
        ax1.set_title('Cluster Quality: Silhouette Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        best_k = k_range[np.argmax(silhouette_scores)]
        ax1.axvline(x=best_k, color='red', linestyle='--',
                   label=f'Optimal k={best_k} (Score={max(silhouette_scores):.3f})')
        ax1.legend()

        ax2.plot(k_range, inertias, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Inertia (WCSS)', fontsize=12)
        ax2.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/cluster_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        optimal_k = best_k
        kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans_optimal.fit_predict(self.embeddings_data)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(self.embeddings_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        colors_category = {'coding': '#2E86AB', 'math': '#A23B72', 'general_knowledge': '#F18F01'}
        for category in np.unique(self.categories):
            mask = np.array(self.categories) == category
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=colors_category.get(category, '#000000'),
                       label=category.replace('_', ' ').title(),
                       alpha=0.7, s=60, edgecolors='black', linewidths=0.5)

        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax1.set_title('Ground Truth Categories', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        colors_cluster = plt.cm.Set3(np.linspace(0, 1, optimal_k))
        for i in range(optimal_k):
            mask = cluster_labels == i
            ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors_cluster[i]], label=f'Cluster {i+1}',
                       alpha=0.7, s=60, edgecolors='black', linewidths=0.5)

        ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax2.set_title(f'K-Means Clustering (k={optimal_k})', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_vs_clusters.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():

    console.print("[bold green] Embedding Space Visualization[/bold green]")

    visualizer = EmbeddingVisualizer()

    try:
        visualizer.initialize_components()

        output_dir = visualizer.create_comprehensive_visualizations()

        console.print(f"\n[bold green]Visualization suite complete![/bold green]")
        console.print(f"📁 All visualizations saved to '{output_dir}/' directory")
        console.print("📊 Publication-quality plots generated")
        console.print("🌐 Interactive 3D visualizations created")

        return output_dir

    except Exception as e:
        console.print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
