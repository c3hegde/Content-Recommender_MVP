#run this in command prompt at this level - G:\My Drive\Github\Content Recommender_MVP

import os
import pandas as pd
import uuid

# Set input/output paths
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load Wikipedia sample
wiki_path = os.path.join(RAW_DIR, "wikipedia_sample.csv")
print(f"🔍 Loading Wikipedia data from {wiki_path}")
df_wiki = pd.read_csv(wiki_path)

# Normalize Wikipedia fields
df_wiki_normalized = pd.DataFrame({
    "id": df_wiki["id"].fillna("").astype(str),
    "title": df_wiki["title"],
    "content": df_wiki["text"],
    "category": df_wiki["department"],
    "source": "wikipedia",
    "popularity": None
})

# Load Amazon books dataset
books_path = os.path.join(RAW_DIR, "bestsellers_with_categories.csv")
print(f"📚 Loading Amazon Books data from {books_path}")
df_books = pd.read_csv(books_path)

# Normalize Amazon book fields
df_books_normalized = pd.DataFrame({
    "id": [str(uuid.uuid4()) for _ in range(len(df_books))],  # generate unique IDs
    "title": df_books["Name"],
    "content": "By " + df_books["Author"] + ". Genre: " + df_books["Genre"] +
               ". Published: " + df_books["Year"].astype(str),
    "category": df_books["Genre"],
    "source": "amazon_books",
    "popularity": df_books["User Rating"]
})

# Combine both sources
df_combined = pd.concat([df_wiki_normalized, df_books_normalized], ignore_index=True)

# Save to processed file
output_path = os.path.join(PROCESSED_DIR, "combined_documents.csv")
df_combined.to_csv(output_path, index=False)
print(f"✅ Combined dataset saved to {output_path}")
