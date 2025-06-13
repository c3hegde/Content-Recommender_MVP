# src/semantic_search.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import sys
# Dynamically add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class SemanticSearch:
    def __init__(self, data_path="data/processed/combined_documents.csv", model_name="all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)
        self.df = pd.read_csv(data_path)
        self.texts = self.df['content'].fillna("").tolist()
        self.index = None
        self.embeddings = None

    def build_index(self, save_path="data/faiss_index"):
        print(" Generating embeddings...")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True, convert_to_numpy=True)

        print(" Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_path, "semantic.index"))
        with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.df, f)
        print(" Index built and saved.")

    def load_index(self, path="data/faiss_index"):
        print(" Loading index...")
        self.index = faiss.read_index(os.path.join(path, "semantic.index"))
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            self.df = pickle.load(f)
            self.texts = self.df['content'].fillna("").tolist()
        print(" Index loaded.")

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        scores, indices = self.index.search(query_vec, top_k)
        results = self.df.iloc[indices[0]]
        return results[['title', 'content']]

