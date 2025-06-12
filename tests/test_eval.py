#to evaluate TF-IDF search engine using Precision@k and Recall@k with hand-labeled relevance judgments.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_clean_data
from src.search_tfidf import TFIDFSearchEngine
from tests.searchIndex import search_index_data

# Sample evaluation queries with ground truth relevance (these are fake IDs, replace with real ones)
eval_queries = [
    "employee benefits policy",
    "project management templates",
    "health insurance coverage",
    "annual performance review process"
]


def build_test_cases(queries, min_len=30):
    test_cases = []
    for query in queries:
        try:
            filtered_df = search_index_data(query=query, min_len=min_len)
            relevant_ids = filtered_df.index.tolist()
            test_cases.append({
                "query": query,
                "relevant_ids": relevant_ids
            })
        except Exception as e:
            print(f"⚠️ Failed for query: {query} → {e}")
    return test_cases

def evaluate_query(engine, query, relevant_ids, top_k=5):
    results = engine.search(query, top_n=top_k)
    retrieved_ids = results.index.tolist()

    true_positives = len(set(retrieved_ids) & set(relevant_ids))
    precision = true_positives / top_k
    if len(relevant_ids) == 0:
     print(f"⚠️ No relevant documents found for query: '{query}'")
     recall = 0.0
    else:
     print(f"✅ Retrieved top {top_k} results.")
     print(f"📌 Ground truth relevant_ids: {relevant_ids}")
     recall = true_positives / len(relevant_ids)
   
    return {
        "query": query,
        "precision@k": precision,
        "recall@k": recall,
        "true_positives": true_positives,
        "retrieved_ids": retrieved_ids,
        "relevant_ids": relevant_ids,
    }

def evaluate_all(engine, test_cases, k=5):
    print(f"🔍 Running evaluation on {len(test_cases)} queries (Top-{k})...")
    all_results = []

    for case in test_cases:
        result = evaluate_query(engine, case["query"], case["relevant_ids"], top_k=k)
        all_results.append(result)
        print(f"\nQuery: '{result['query']}'")
        print(f"Precision@{k}: {result['precision@k']:.2f}")
        print(f"Recall@{k}: {result['recall@k']:.2f}")
        print(f"True Positives: {result['true_positives']}")
        print(f"Retrieved IDs: {result['retrieved_ids']}")
        print(f"Relevant IDs: {result['relevant_ids']}")

    avg_precision = sum(r["precision@k"] for r in all_results) / len(all_results)
    avg_recall = sum(r["recall@k"] for r in all_results) / len(all_results)

    print("\n📊 Summary:")
    print(f"Average Precision@{k}: {avg_precision:.2f}")
    print(f"Average Recall@{k}: {avg_recall:.2f}")

    return all_results

if __name__ == "__main__":
    df = load_clean_data()
    engine = TFIDFSearchEngine(df)

    queries = [
        "employee benefits policy",
        "project management templates",
        "health insurance coverage",
        "annual performance review process"
    ]

    test_cases = build_test_cases(queries)
    evaluate_all(engine, test_cases, k=5)
