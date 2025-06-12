# playground_semantic.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.semantic_search import SemanticSearch

if __name__ == "__main__":
    retriever = SemanticSearch()
    
    # Build index if needed
    retriever.build_index()  # Run once; then comment out

    # Load for querying
    retriever.load_index()

    query = "employee benefits policy"
    results = retriever.search(query, top_k=5)

    for i, row in results.iterrows():
        print(f"\n🔹 Title: {row['title']}")
        print(f"📄 Content snippet: {row['content'][:300]}...")
