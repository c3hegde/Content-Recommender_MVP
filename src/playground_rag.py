# playground_rag.py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline()
    query = "What does the employee benefits policy include?"
    
    answer = rag.generate_answer(query)
    print("\n Generated Answer:")
    print(answer)
