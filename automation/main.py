#Python utility to run the entire pipeline:
# Dynamically add the parent directory (project root) to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_clean_data
from src.search_tfidf import TFIDFSearchEngine

def run_pipeline():
    df = load_clean_data()
    engine = TFIDFSearchEngine(df)
    print("✅ TF-IDF engine initialized.")

if __name__ == "__main__":
    run_pipeline()