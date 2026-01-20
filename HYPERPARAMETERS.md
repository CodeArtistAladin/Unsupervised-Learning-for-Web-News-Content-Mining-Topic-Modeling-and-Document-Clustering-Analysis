# WEB MINING PROJECT - HYPERPARAMETER CONFIGURATION
# ===================================================
# This file documents all hyperparameters used in the project.
# To modify behavior, edit the values in the individual script files.

## PREPROCESSING CONFIGURATION
## File: scripts/preprocess.py

TEXT_COLUMN = 'News Title'          # Column name in dataset containing text
DATA_SEP = ';'                      # CSV delimiter
OUTPUT_FILE = 'cleaned_news.csv'    # Output filename

Preprocessing Notes:
- Removes all non-alphabetic characters
- Converts text to lowercase
- Removes English stopwords
- Filters out empty documents


## TOPIC MODELING CONFIGURATION
## File: scripts/topic_model.py

NUM_TOPICS = 8                      # Number of topics to extract (try: 5, 10, 15, 20)
PASSES = 15                         # Number of passes through corpus (try: 10, 20, 30)
WORDS_PER_TOPIC = 10               # Number of top words per topic
RANDOM_STATE = 42                   # For reproducibility

Topic Modeling Notes:
- Uses Latent Dirichlet Allocation (LDA)
- Produces document-topic assignments
- Saves trained model for later use
- Each topic is a distribution over words


## CLUSTERING CONFIGURATION  
## File: scripts/cluster.py

N_CLUSTERS = 8                      # Number of clusters (try: 5, 10, 12, 15)
MAX_FEATURES = 5000                 # Maximum TF-IDF features
RANDOM_STATE = 42                   # For reproducibility
N_INIT = 10                         # Number of times algorithm runs with different seeds

Clustering Notes:
- Uses K-Means algorithm
- TF-IDF vectorization applied first
- Produces cluster assignments for documents
- Inertia value indicates cluster quality


## VISUALIZATION CONFIGURATION
## File: scripts/visualize.py

NUM_TOPICS = 8                      # Must match topic_model.py
N_CLUSTERS = 8                      # Must match cluster.py
DPI = 100                           # Resolution of saved plots
FIGSIZE_NORMAL = (12, 6)           # Figure size for standard charts
FIGSIZE_LARGE = (14, 10)           # Figure size for large charts

Visualization Outputs:
- 8 word cloud images (topics)
- 1 topic distribution chart
- 1 cluster distribution chart
- 1 PCA scatter plot (2D)
- 1 topic-cluster heatmap


## RECOMMENDED HYPERPARAMETER VALUES FOR EXPERIMENTATION

For different dataset sizes:

Small dataset (< 10k documents):
  NUM_TOPICS = 5-8
  N_CLUSTERS = 5-8
  PASSES = 10-15

Medium dataset (10k-100k documents):
  NUM_TOPICS = 8-15
  N_CLUSTERS = 8-12
  PASSES = 15-20

Large dataset (> 100k documents):
  NUM_TOPICS = 10-20
  N_CLUSTERS = 10-15
  PASSES = 15-30


## PARAMETER ADJUSTMENT GUIDELINES

To find optimal NUM_TOPICS:
1. Try values: 5, 8, 10, 15, 20
2. Look at word clouds - topics should be distinct
3. Check topic distribution - avoid heavily unbalanced topics
4. Choose value with best topic coherence

To find optimal N_CLUSTERS:
1. Try values: 5, 8, 10, 12, 15
2. Check cluster distribution - avoid extremely unbalanced clusters
3. Look at PCA plot - clusters should be visually separable
4. Lower inertia indicates better clustering (but check for overfitting)

To improve LDA quality:
- Increase PASSES (15 → 20-30) for better topic quality
- Increase NUM_TOPICS if dataset is large
- Ensure good preprocessing (sufficient text per document)

To improve K-Means quality:
- Increase N_INIT (10 → 20) for more stable results
- Adjust MAX_FEATURES (5000 → 3000-10000)
- Ensure balanced cluster sizes in distribution


## CURRENT PROJECT SETTINGS

Current configuration analyzed these settings:
- Dataset: AG News (~65k documents)
- NUM_TOPICS: 8 (well-balanced distribution)
- N_CLUSTERS: 8 (matches topic count)
- PASSES: 15 (good quality topics)
- TF-IDF features: 5000 (comprehensive coverage)

Results achieved:
✓ Clear, distinct topics in word clouds
✓ Balanced topic distribution
✓ Interpretable clusters
✓ Good visualization quality
✓ Reproducible results


## HOW TO MODIFY HYPERPARAMETERS

1. Edit the value in the script (top section marked HYPERPARAMETERS)
2. Delete intermediate files (optional):
   - document_topics.csv (if changing NUM_TOPICS)
   - news_with_clusters.csv (if changing N_CLUSTERS)
3. Rerun affected script(s):
   python scripts/topic_model.py        # if changed LDA hyperparameters
   python scripts/cluster.py            # if changed clustering hyperparameters
   python scripts/visualize.py          # to update visualizations
4. Compare results with previous run


## PERFORMANCE NOTES

Current execution times (~65k documents):
- Preprocessing: ~10-15 seconds
- Topic modeling: ~1-2 minutes
- Clustering: ~10-30 seconds
- Visualization: ~30-60 seconds
- Total: ~3-5 minutes

Performance varies by:
- Dataset size (larger = longer)
- NUM_TOPICS (higher = longer for LDA)
- N_CLUSTERS (higher = longer for K-Means)
- PASSES value (higher = longer for LDA)
- Computer specs (CPU, RAM)


═══════════════════════════════════════════════════════════════

For questions or issues, see README.md in project root.

Last Updated: January 15, 2026
