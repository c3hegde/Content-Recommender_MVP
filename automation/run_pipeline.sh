#!/bin/bash

echo "🚀 Running data cleaning and indexing..."
python src/data_loader.py

echo "✅ Launching Streamlit app..."
streamlit run streamlit_app.py
