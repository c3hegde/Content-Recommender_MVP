#Create a file tests/test_search.py to run sample queries and check accuracy manually.

# Dynamically add the parent directory (project root) to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_clean_data
from src.search_tfidf import TFIDFSearchEngine


def test_queries():
    df = load_clean_data()
    engine = TFIDFSearchEngine(df)
    test_cases = [
        "employee benefits",
        "project management",
        "health insurance policy",
        "vacation leave process",
        "python programming"
    ]
    print(" TF-IDF Matrix shape:", engine.tfidf_matrix.shape)

    for query in test_cases:
        print(f"\n Query: {query}")
        results = engine.search(query, top_n=3)
        for i, row in results.iterrows():
            print(f"{i+1}. {row['title']} (Score: {row['score']:.3f})")

if __name__ == "__main__":
    test_queries()
