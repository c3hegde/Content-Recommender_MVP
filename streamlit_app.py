
import streamlit as st
# Dynamically add the parent directory (project root) to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search_tfidf import TFIDFSearchEngine
from src.data_loader import load_clean_data

# Load data and initialize search engine
@st.cache_resource
def load_engine():
    df = load_clean_data()
    engine = TFIDFSearchEngine(df)
    return engine

engine = load_engine()

# UI
st.title("📚 Document Recommendation Engine")
query = st.text_input("🔍 Enter your search query:")
top_n = st.slider("Top N results", min_value=1, max_value=10, value=5)

if query:
    results = engine.search(query, top_n=top_n)

    if results.empty:
        st.warning("No results found.")
    else:
        for _, row in results.iterrows():
            st.markdown(f"### {row['title']} ({row['score']:.3f})")
            st.markdown(f"**Category:** {row['category']}")
            st.markdown(f"{row['content'][:400]}...")
            st.markdown("---")
