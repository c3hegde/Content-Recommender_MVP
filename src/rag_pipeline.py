# src/rag_pipeline.py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.semantic_search import SemanticSearch

class RAGPipeline:
    def __init__(self, retriever_model="all-MiniLM-L6-v2", generator_model="google/flan-t5-small"):
        print(" Loading Retriever...")
        self.retriever = SemanticSearch(model_name=retriever_model)
        self.retriever.load_index()
        
        print(" Loading Generator...")
        self.tokenizer = T5Tokenizer.from_pretrained(generator_model)
        self.generator = T5ForConditionalGeneration.from_pretrained(generator_model)

    def generate_answer(self, query, top_k=3, max_length=180):
        # Step 1: Retrieve top-k documents
        retrieved_docs = self.retriever.search(query, top_k=top_k)
        context = "\n".join(retrieved_docs["content"].tolist())

        # Step 2: Prepare prompt
        prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"

        # Step 3: Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.generator.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

        # Step 4: Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
