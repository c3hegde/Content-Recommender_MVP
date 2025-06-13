# Purpose: Reusable module to load and clean your processed dataset.
# scripts/data_loader.py

import os
import pandas as pd
import re

def clean_text(text):
    """Simple text normalization: lowercasing, strip, remove extra spaces, etc."""
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII
    return text

def load_clean_data(data_path="data/processed/combined_documents.csv", min_len=30):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data file not found at {data_path}")

    print(f" Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    print(" Raw data loaded. Cleaning...")

    df["title"] = df["title"].fillna("").apply(clean_text)
    df["content"] = df["content"].fillna("").apply(clean_text)
    df["category"] = df["category"].fillna("unknown")

    df = df[df["content"].str.len() > min_len].reset_index(drop=True)

    print(f" Cleaned {len(df)} documents.")
    return df
