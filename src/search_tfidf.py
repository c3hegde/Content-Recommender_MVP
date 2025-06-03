#TF-IDF search engine

import os
import sys
# Dynamically add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_clean_data

class TFIDFSearchEngine:
   #  commenting this block coz I have an alternative below that does initilization of the search engine 
   #def __init__(self, min_df=2, max_df=0.8):
   #     self.vectorizer = TfidfVectorizer(
    #        stop_words='english',
    #        lowercase=True,
   #         max_df=max_df,
    #        min_df=min_df
   #     )
     #   self.tfidf_matrix = None
    #    self.documents = None

    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.tfidf_matrix = self.vectorizer.fit_transform(documents['content'])

    def index(self, df: pd.DataFrame, text_field="content"):
        self.documents = df.copy()
        combined_text = df["title"] + " " + df[text_field]
        self.tfidf_matrix = self.vectorizer.fit_transform(combined_text)
        print(f"✅ Indexed {self.tfidf_matrix.shape[0]} documents with {self.tfidf_matrix.shape[1]} terms.")

    def search(self, query: str, top_n=5):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        results = self.documents.iloc[top_indices].copy()
        results["score"] = scores[top_indices]
        return results[["title", "category", "source", "score", "content"]]

