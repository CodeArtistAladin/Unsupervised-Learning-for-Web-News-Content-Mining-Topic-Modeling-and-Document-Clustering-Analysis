"""CLUSTERING SCRIPT (K-Means)

This script applies K-Means clustering to documents:
- Vectorizes text using TF-IDF
- Performs K-Means clustering
- Saves cluster assignments
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
import sys

# ============= CONFIGURATION =============
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

INPUT_FILE = os.path.join(data_dir, 'cleaned_news.csv')
OUTPUT_FILE = os.path.join(data_dir, 'news_with_clusters.csv')

# ============= HYPERPARAMETERS =============
N_CLUSTERS = 8              # Number of clusters
MAX_FEATURES = 5000         # Maximum TF-IDF features
RANDOM_STATE = 42           # For reproducibility

def main():
    """Main clustering pipeline."""
    try:
        print("\n" + "="*60)
        print("STEP 3: K-MEANS CLUSTERING")
        print("="*60)
        
        # Load cleaned data
        print(f"\nLoading cleaned data from: {INPUT_FILE}")
        if not os.path.exists(INPUT_FILE):
            print(f"ERROR: Input file not found: {INPUT_FILE}")
            sys.exit(1)
        
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} documents")
        
        # Clean data
        print("\nValidating and cleaning data...")
        df = df.dropna(subset=['clean_text'])
        df = df[df['clean_text'].str.strip() != '']
        print(f"Valid documents: {len(df)}")
        
        # Vectorize text
        print(f"\nVectorizing text with TF-IDF...")
        print(f"  - Max features: {MAX_FEATURES}")
        tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
        X = tfidf.fit_transform(df['clean_text'])
        print(f"  - Feature matrix shape: {X.shape}")
        
        # Apply K-Means
        print(f"\nApplying K-Means clustering...")
        print(f"  - Number of clusters: {N_CLUSTERS}")
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        df['cluster'] = kmeans.fit_predict(X)
        print(f"  - Inertia: {kmeans.inertia_:.4f}")
        
        # Save results
        print(f"\nSaving clustered data to: {OUTPUT_FILE}")
        df.to_csv(OUTPUT_FILE, index=False)
        
        # Statistics
        print(f"\nCluster Distribution:")
        cluster_dist = df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  Cluster {cluster_id}: {count:6d} documents ({percentage:5.1f}%)")
        
        print(f"\n✓ Clustering complete!\n")
        
    except Exception as e:
        print(f"ERROR during clustering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
