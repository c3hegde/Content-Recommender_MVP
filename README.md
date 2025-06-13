# Content Recommender - MVP
A machine learning application that recommends relevant internal documents based on their search queries, department, role, or behavioral patterns. The app accepts user input such as search text or metadata (e.g., role, department) and retrieves content from a company-specific knowledge base—including a corpus of documents, FAQs, and other internal resources. For each query, the system generates a ranked list of the Top-N most relevant documents, ensuring employees quickly access the information they need.

## Features
Developed and evaluated two distinct models to create a unified recommender system capable of handling both unstructured and structured content:
- A TF-IDF-based search engine that utilizes sparse vector representations, offering faster computation times. This works well with structured or repetitive vocabulary and performs words search.
- A semantic similarity search model using pre-trained language models (e.g., BERT, SBERT, MiniLM) to convert queries and documents into dense vector embeddings. This captures semantic meaning.
- Interactive UI (Streamlit)
- Evaluation using Precision and Recall to test whether the algorithms miss relevant docs.
- Datasets taken from Wikipedia_sample that includes unstructured documents (text articles) and amazon books sold list that     includes structured metadata 


## Use Case
This MVP demonstrates how companies can retrieve company specific information that is ranked and recommended based on search parameters. 

## Models Used

| Algorithm           | Purpose            |
|---------------------|--------------------|
| TfidfVectorizer and Cosine similarity  | From scikit-learn library for converting a collection of raw text documents into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features and perform cosine vector search|
| RAG pipeline       | Built using SentenceTransformers to create embeddings, retriever built using model="all-MiniLM-L6-v2", and LLM model to answer questions built using model="google/flan-t5-small" |


## Project Structure

diabetes-predictor-ai-mvp/
- ├── data/               # Raw and processed datasets
- ├── notebooks/          # Exploratory Data Analysis. Load and preprocess documents and metadata
- ├── automation/         # data update and retraining pipeline
- ├── src/             #TF-IDF, embdeddings, Ranking algorithms
- ├── tests/          # Evaluation, Precision@K, Recall

## Potential Future Enhancements
-	User personalization based on past queries or role
-	Use metadata (tags, departments) in ranking model
-	Add feedback capture (likes/dislikes or implicit click data


