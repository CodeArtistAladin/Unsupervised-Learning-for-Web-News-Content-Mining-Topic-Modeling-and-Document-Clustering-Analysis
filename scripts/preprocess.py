"""DATA PREPROCESSING SCRIPT

This script loads raw text data and applies cleaning:
- Remove punctuation and lowercase
- Remove stopwords
- Save cleaned text to CSV
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import os
import sys

# ============= CONFIGURATION =============
# Input/Output paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

INPUT_FILE = os.path.join(data_dir, 'news_dataset.csv')
OUTPUT_FILE = os.path.join(data_dir, 'cleaned_news.csv')
DATA_SEP = ';'  # CSV separator
TEXT_COLUMN = 'News Title'  # Column name containing text

# Download stopwords
try:
    nltk.download('stopwords', quiet=True)
except:
    pass  # Use built-in stopwords if download fails
stop_words = set(stopwords.words('english'))

# ============= FUNCTIONS =============
def clean_text(text):
    """Clean text by removing punctuation, converting to lowercase, and removing stopwords."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove all non-alphabetic characters
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())
    
    # Remove stopwords
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 0]
    
    return " ".join(tokens)

def main():
    """Main preprocessing pipeline."""
    try:
        print("\n" + "="*60)
        print("STEP 1: TEXT PREPROCESSING")
        print("="*60)
        
        # Load data
        print(f"\nLoading data from: {INPUT_FILE}")
        if not os.path.exists(INPUT_FILE):
            print(f"ERROR: Input file not found: {INPUT_FILE}")
            sys.exit(1)
        
        df = pd.read_csv(INPUT_FILE, sep=DATA_SEP)
        print(f"Loaded {len(df)} documents")
        
        if TEXT_COLUMN not in df.columns:
            print(f"ERROR: Column '{TEXT_COLUMN}' not found in dataset")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        
        # Clean text
        print(f"\nCleaning text in column '{TEXT_COLUMN}'...")
        df['clean_text'] = df[TEXT_COLUMN].apply(clean_text)
        
        # Remove empty documents
        df = df[df['clean_text'].str.strip() != '']
        print(f"Removed empty documents. Remaining: {len(df)} documents")
        
        # Save output
        print(f"\nSaving cleaned data to: {OUTPUT_FILE}")
        df.to_csv(OUTPUT_FILE, index=False)
        
        # Statistics
        print(f"\nPreprocessing Statistics:")
        print(f"  - Total documents: {len(df)}")
        print(f"  - Avg tokens per document: {df['clean_text'].str.split().str.len().mean():.1f}")
        print(f"  - Min tokens: {df['clean_text'].str.split().str.len().min()}")
        print(f"  - Max tokens: {df['clean_text'].str.split().str.len().max()}")
        
        print(f"\n✓ Preprocessing complete!\n")
        
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
