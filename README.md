# Web Mining Project: News Article Analysis

## Project Overview

This project performs unsupervised learning on a news articles dataset using **Topic Modeling (LDA)** and **Document Clustering (K-Means)**. The goal is to discover hidden topics and group similar news articles without labeled data.

## What This Project Does

1. **Data Preprocessing**: Cleans and preprocesses raw text data (tokenization, stopword removal)
2. **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to extract topics
3. **Document Clustering**: Groups documents using K-Means on TF-IDF vectors
4. **Visualization**: Generates comprehensive visualizations and analysis

## Dataset

- **Name**: AG News Dataset
- **Source**: Kaggle
- **Size**: ~120,000 news articles
- **Format**: CSV with columns: `No`, `News Title`, `Category`
- **Location**: `data/news_dataset.csv`

## Algorithms Used

### 1. Topic Modeling (LDA)
- **What it does**: Discovers hidden topics and represents documents as mixtures of topics
- **Parameters**:
  - Number of topics: **8**
  - Passes through corpus: **15**
  - Output: Topic-word distributions and dominant topic per document

### 2. K-Means Clustering
- **What it does**: Groups similar documents into clusters based on TF-IDF features
- **Parameters**:
  - Number of clusters: **8**
  - Max TF-IDF features: **5000**
  - Output: Cluster assignments for each document

### 3. Dimensionality Reduction (PCA)
- **What it does**: Reduces high-dimensional data to 2D for visualization
- **Output**: PCA scatter plot of document clusters

## Folder Structure

```
Final Project/
│
├── data/                          # Data files
│   ├── news_dataset.csv           # Original dataset (input)
│   ├── cleaned_news.csv           # Preprocessed text
│   ├── document_topics.csv        # Documents with dominant topics
│   ├── news_with_clusters.csv     # Documents with cluster assignments
│   └── final_results.csv          # Final combined results
│
├── scripts/                       # Python code
│   ├── preprocess.py              # Data preprocessing
│   ├── topic_model.py             # LDA topic modeling
│   ├── cluster.py                 # K-Means clustering
│   └── visualize.py               # Generate visualizations
│
├── results/                       # Output visualizations
│   ├── topic0_wordcloud.png       # Word cloud for each topic
│   ├── topic1_wordcloud.png
│   ├── ... (topics 2-7)
│   ├── topic_distribution.png     # Topic distribution chart
│   ├── cluster_distribution.png   # Cluster distribution chart
│   ├── clusters_pca.png           # PCA scatter plot
│   ├── topic_cluster_heatmap.png  # Topic vs Cluster heatmap
│   └── lda_model/                 # Saved LDA model files
│
├── notebooks/                     # Jupyter notebooks (optional)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset

1. Download the **AG News Dataset** from [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
2. Extract the CSV file
3. Place it in the `data/` folder as `news_dataset.csv`

The expected format:
```
No;News Title;Category
1;Google+ rolls out 'Stories';Technology
2;Dov Charney's Redeeming Quality;Business
...
```

## How to Run the Project

### Run All Steps in Order

```bash
# Navigate to project directory
cd "Final Project"

# Step 1: Preprocess data
python scripts/preprocess.py

# Step 2: Extract topics with LDA
python scripts/topic_model.py

# Step 3: Cluster documents with K-Means
python scripts/cluster.py

# Step 4: Generate visualizations
python scripts/visualize.py
```

### Run Individual Steps

You can run each script independently after the previous step is complete:

```bash
python scripts/preprocess.py    # Cleans raw text
python scripts/topic_model.py   # Extracts topics from cleaned text
python scripts/cluster.py       # Clusters cleaned documents
python scripts/visualize.py     # Generates all visualizations
```

## Output Files

### Data Files (in `data/` folder)

| File | Contents | Purpose |
|------|----------|---------|
| `cleaned_news.csv` | Original text + cleaned text | Preprocessed data for modeling |
| `document_topics.csv` | Documents + dominant topic | Topic assignments |
| `news_with_clusters.csv` | Documents + cluster ID | Cluster assignments |
| `final_results.csv` | All information combined | Complete analysis results |

### Visualization Files (in `results/` folder)

| File | Contents |
|------|----------|
| `topic{0-7}_wordcloud.png` | Word cloud for each discovered topic |
| `topic_distribution.png` | Bar chart showing topic distribution |
| `cluster_distribution.png` | Bar chart showing cluster distribution |
| `clusters_pca.png` | 2D scatter plot of documents colored by cluster |
| `topic_cluster_heatmap.png` | Heatmap showing topic-cluster relationships |

## Customizing Hyperparameters

To experiment with different settings, edit the hyperparameter values in each script:

### Topic Modeling (`scripts/topic_model.py`)
```python
NUM_TOPICS = 8          # Change number of topics (try: 5, 10, 15)
PASSES = 15             # Change LDA passes (try: 10, 20, 30)
WORDS_PER_TOPIC = 10    # Words displayed per topic
```

### Clustering (`scripts/cluster.py`)
```python
N_CLUSTERS = 8          # Change number of clusters (try: 5, 10, 12)
MAX_FEATURES = 5000     # Maximum TF-IDF features
```

## Expected Results

After successful execution, you should see:

1. **Console Output**:
   - Preprocessing statistics (token counts, document count)
   - Extracted topics with top words
   - Topic distribution summary
   - Cluster distribution summary

2. **Data Files**:
   - 4-5 CSV files in `data/` with analysis results

3. **Visualizations**:
   - 13+ PNG files in `results/` including word clouds, charts, and plots

## Code Quality Features

✓ Clear comments explaining each step  
✓ Error handling for missing files  
✓ Reproducible results (fixed random seeds)  
✓ Modular design (independent scripts)  
✓ Progress messages during execution  
✓ Statistics printed to console  

## Performance Notes

- **Dataset Size**: ~120,000 documents
- **Preprocessing Time**: ~10-15 seconds
- **Topic Modeling Time**: ~1-2 minutes (depends on corpus size)
- **Clustering Time**: ~10-30 seconds
- **Visualization Time**: ~30-60 seconds
- **Total Runtime**: ~3-5 minutes (first run)

## Troubleshooting

### Issue: "FileNotFoundError: news_dataset.csv not found"
**Solution**: Ensure the dataset is downloaded and placed in `data/news_dataset.csv`

### Issue: "ModuleNotFoundError: No module named 'gensim'"
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: "NLTK stopwords not found"
**Solution**: The script auto-downloads them, but if needed manually run:
```python
import nltk
nltk.download('stopwords')
```

### Issue: "Low topic quality or unclear clusters"
**Solution**: Experiment with hyperparameters:
- Try different `NUM_TOPICS` values (5, 10, 15, 20)
- Try different `N_CLUSTERS` values (5, 10, 12)
- Adjust `PASSES` in LDA (10, 20, 30)

## Next Steps After Project Completion

1. **Analyze Results**: Review generated visualizations and statistical summaries
2. **Interpret Topics**: Examine word clouds to understand discovered topics
3. **Analyze Clusters**: Use PCA plot to visualize document relationships
4. **Write Report**: Use insights for academic/technical reporting
5. **Fine-tune**: Experiment with different hyperparameters for better results

## References

- Blei, D.M., Ng, A.Y., & Jordan, M.I. (2003). "Latent Dirichlet Allocation" - Journal of Machine Learning Research
- Scikit-learn K-Means: https://scikit-learn.org/stable/modules/clustering.html#k-means
- Gensim LDA: https://radimrehurek.com/gensim/models/ldamodel.html

## Notes for Users

- This project is designed for a **Web Mining course** - it prioritizes clarity over production optimization
- All scripts are **independent** and can be rerun individually
- Results are **deterministic** (same input produces same output due to fixed random seeds)
- The project **requires no GUI** - runs entirely from command line

---

**Last Updated**: January 2026  
**Status**: Complete and Ready for Analysis
