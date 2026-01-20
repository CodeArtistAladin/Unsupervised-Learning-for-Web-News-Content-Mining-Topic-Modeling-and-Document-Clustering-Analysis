"""VISUALIZATION SCRIPT

This script generates all required visualizations:
- Word clouds for each topic
- Topic distribution bar chart
- Cluster distribution bar chart
- 2D PCA scatter plot of clusters
- Topic vs Cluster heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from gensim.models.ldamodel import LdaModel
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys

# ============= CONFIGURATION =============
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')

# Input files
CLUSTERS_FILE = os.path.join(data_dir, 'news_with_clusters.csv')
TOPICS_FILE = os.path.join(data_dir, 'document_topics.csv')
LDA_MODEL_PATH = os.path.join(results_dir, 'lda_model')

# ============= HYPERPARAMETERS =============
NUM_TOPICS = 8
N_CLUSTERS = 8
DPI = 100
FIGSIZE_NORMAL = (12, 6)
FIGSIZE_LARGE = (14, 10)

def generate_wordclouds():
    """Generate word clouds for each topic."""
    print("Generating word clouds...")
    lda = LdaModel.load(LDA_MODEL_PATH)
    
    for i in range(NUM_TOPICS):
        topic_words = lda.show_topic(i, 50)
        words = " ".join([word for word, _ in topic_words])
        
        if words.strip():
            wc = WordCloud(width=800, height=600, background_color='white').generate(words)
            output_path = os.path.join(results_dir, f'topic{i}_wordcloud.png')
            wc.to_file(output_path)
            print(f"  ✓ Topic {i} word cloud saved")

def generate_topic_distribution():
    """Generate bar chart of topic distribution."""
    print("Generating topic distribution chart...")
    df = pd.read_csv(TOPICS_FILE)
    
    plt.figure(figsize=FIGSIZE_NORMAL)
    topic_counts = df['dominant_topic'].value_counts().sort_index()
    plt.bar(topic_counts.index, topic_counts.values, color='steelblue', alpha=0.8)
    plt.xlabel('Topic', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Documents', fontsize=12, fontweight='bold')
    plt.title('Topic Distribution (LDA)', fontsize=14, fontweight='bold')
    plt.xticks(range(NUM_TOPICS))
    plt.grid(axis='y', alpha=0.3)
    
    output_path = os.path.join(results_dir, 'topic_distribution.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Topic distribution chart saved")

def generate_cluster_distribution():
    """Generate bar chart of cluster distribution."""
    print("Generating cluster distribution chart...")
    df = pd.read_csv(CLUSTERS_FILE)
    
    plt.figure(figsize=FIGSIZE_NORMAL)
    cluster_counts = df['cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values, color='coral', alpha=0.8)
    plt.xlabel('Cluster', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Documents', fontsize=12, fontweight='bold')
    plt.title('Cluster Distribution (K-Means)', fontsize=14, fontweight='bold')
    plt.xticks(range(N_CLUSTERS))
    plt.grid(axis='y', alpha=0.3)
    
    output_path = os.path.join(results_dir, 'cluster_distribution.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Cluster distribution chart saved")

def generate_pca_plot():
    """Generate 2D PCA scatter plot of document clusters."""
    print("Generating PCA scatter plot...")
    df = pd.read_csv(CLUSTERS_FILE)
    
    # Vectorize text
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())
    
    # Create scatter plot
    plt.figure(figsize=FIGSIZE_LARGE)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], 
                         cmap='tab10', alpha=0.6, s=50, edgecolors='black', linewidth=0.3)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    plt.title('Document Clusters - PCA Visualization (K-Means)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, label='Cluster')
    plt.grid(alpha=0.3)
    
    output_path = os.path.join(results_dir, 'clusters_pca.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ PCA scatter plot saved")

def generate_topic_cluster_heatmap():
    """Generate heatmap of topic vs cluster distribution."""
    print("Generating topic vs cluster heatmap...")
    
    # Load data
    clusters_df = pd.read_csv(CLUSTERS_FILE)
    topics_df = pd.read_csv(TOPICS_FILE)
    
    # Merge dataframes
    df = clusters_df[['cluster']].copy()
    df['topic'] = topics_df['dominant_topic']
    
    # Create contingency table
    contingency = pd.crosstab(df['topic'], df['cluster'])
    
    # Create heatmap
    plt.figure(figsize=FIGSIZE_NORMAL)
    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
    plt.xlabel('Cluster', fontsize=12, fontweight='bold')
    plt.ylabel('Topic', fontsize=12, fontweight='bold')
    plt.title('Topic vs Cluster Distribution', fontsize=14, fontweight='bold')
    
    output_path = os.path.join(results_dir, 'topic_cluster_heatmap.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Topic vs Cluster heatmap saved")

def merge_and_save_final_results():
    """Merge topic and cluster assignments into final output."""
    print("Creating final comprehensive results file...")
    
    clusters_df = pd.read_csv(CLUSTERS_FILE)
    topics_df = pd.read_csv(TOPICS_FILE)
    
    # Merge
    final_df = clusters_df[['News Title', 'Category', 'clean_text', 'cluster']].copy()
    final_df['dominant_topic'] = topics_df['dominant_topic']
    
    # Save
    output_path = os.path.join(data_dir, 'final_results.csv')
    final_df.to_csv(output_path, index=False)
    print(f"  ✓ Final results saved: {output_path}")
    
    # Statistics
    print(f"\nFinal Dataset Statistics:")
    print(f"  - Total documents: {len(final_df)}")
    print(f"  - Number of clusters: {final_df['cluster'].nunique()}")
    print(f"  - Number of topics: {final_df['dominant_topic'].nunique()}")
    print(f"  - Categories: {', '.join(final_df['Category'].unique())}")

def main():
    """Main visualization pipeline."""
    try:
        print("\n" + "="*60)
        print("STEP 4: VISUALIZATION")
        print("="*60)
        
        # Verify all input files exist
        for filepath, name in [
            (CLUSTERS_FILE, "Clusters file"),
            (TOPICS_FILE, "Topics file"),
            (LDA_MODEL_PATH, "LDA model"),
        ]:
            if not os.path.exists(filepath):
                print(f"ERROR: {name} not found: {filepath}")
                sys.exit(1)
        
        print(f"\nGenerating all visualizations...\n")
        
        # Generate all outputs
        generate_wordclouds()
        generate_topic_distribution()
        generate_cluster_distribution()
        generate_pca_plot()
        generate_topic_cluster_heatmap()
        merge_and_save_final_results()
        
        print(f"\n✓ All visualizations complete!")
        print(f"✓ Results saved to: {results_dir}\n")
        
    except Exception as e:
        print(f"ERROR during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
