"""TOPIC MODELING SCRIPT (LDA - Latent Dirichlet Allocation)

This script applies LDA to extract topics from cleaned text:
- Builds dictionary and corpus from cleaned documents
- Trains LDA model
- Extracts dominant topic for each document
- Saves results and model
"""

import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import os
import sys

# ============= CONFIGURATION =============
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')

INPUT_FILE = os.path.join(data_dir, 'cleaned_news.csv')
MODEL_OUTPUT = os.path.join(results_dir, 'lda_model')
DOC_TOPICS_OUTPUT = os.path.join(data_dir, 'document_topics.csv')

# ============= HYPERPARAMETERS =============
NUM_TOPICS = 8          # Number of topics to extract
PASSES = 15             # Number of passes through the corpus
WORDS_PER_TOPIC = 10    # Words to display per topic

def get_dominant_topic(lda, doc_topics):
    """Extract dominant topic for a document."""
    if doc_topics:
        return max(doc_topics, key=lambda x: x[1])[0]
    return -1

def main():
    """Main topic modeling pipeline."""
    try:
        print("\n" + "="*60)
        print("STEP 2: TOPIC MODELING (LDA)")
        print("="*60)
        
        # Load cleaned data
        print(f"\nLoading cleaned data from: {INPUT_FILE}")
        if not os.path.exists(INPUT_FILE):
            print(f"ERROR: Input file not found: {INPUT_FILE}")
            sys.exit(1)
        
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} documents")
        
        # Clean data
        print("Validating and cleaning data...")
        df = df.dropna(subset=['clean_text'])
        df = df[df['clean_text'].str.strip() != '']
        print(f"Valid documents: {len(df)}")
        
        # Prepare texts
        print("\nPreparing texts for topic modeling...")
        texts = [doc.split() for doc in df['clean_text']]
        
        # Build dictionary and corpus
        print("Building dictionary and corpus...")
        dict_ = corpora.Dictionary(texts)
        print(f"Dictionary size: {len(dict_)} unique tokens")
        
        corpus = [dict_.doc2bow(text) for text in texts]
        
        # Train LDA model
        print(f"\nTraining LDA model...")
        print(f"  - Number of topics: {NUM_TOPICS}")
        print(f"  - Number of passes: {PASSES}")
        print(f"  - Number of documents: {len(corpus)}")
        
        lda = LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dict_, 
                       passes=PASSES, random_state=42, per_word_topics=True)
        
        # Display topics
        print(f"\nExtracted Topics:")
        print("-" * 60)
        topics = lda.print_topics(num_words=WORDS_PER_TOPIC)
        for topic_id, words in topics:
            print(f"Topic {topic_id}: {words}")
        
        # Get dominant topic for each document
        print("\nExtracting dominant topics for each document...")
        dominant_topics = []
        for i, doc_topics in enumerate([lda.get_document_topics(doc) for doc in corpus]):
            dominant_topics.append(get_dominant_topic(lda, doc_topics))
        
        df['dominant_topic'] = dominant_topics
        
        # Save document-topic assignments
        print(f"\nSaving document-topic assignments to: {DOC_TOPICS_OUTPUT}")
        df.to_csv(DOC_TOPICS_OUTPUT, index=False)
        
        # Save model
        print(f"Saving LDA model to: {MODEL_OUTPUT}")
        lda.save(MODEL_OUTPUT)
        
        # Statistics
        print(f"\nTopic Distribution:")
        print(df['dominant_topic'].value_counts().sort_index())
        
        print(f"\n✓ Topic modeling complete!\n")
        
    except Exception as e:
        print(f"ERROR during topic modeling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
