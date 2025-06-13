# Quick test file for testing TF-IDF search engine

# Dynamically add the parent directory (project root) to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search_tfidf import TFIDFSearchEngine
from src.data_loader import load_clean_data

# Load data
df = load_clean_data()

# Create and index the search engine
search_engine = TFIDFSearchEngine()
search_engine.index(df)

# Run a search
query = "employee benefits policy"
results = search_engine.search(query, top_n=5)

#print(search_engine.vectorizer.vocabulary_.keys())

#print(results.head())

if results.empty:
    print(" No relevant documents found.")
else:
    for i, row in results.iterrows():
        print(f"\n {row['title']} [{row['category']}] (score: {row['score']:.3f})\n{row['content'][:300]}...")


       #Code recommended if you have to debug:
'''
      \def search(self, query: str, top_n=5):
    query_vector = self.vectorizer.transform([query])
    scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

    print(f"Min score: {scores.min():.4f}, Max score: {scores.max():.4f}")

    top_indices = scores.argsort()[-top_n:][::-1]
    results = self.documents.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    return results
'''