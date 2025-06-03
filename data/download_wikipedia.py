#run this in command prompt at this level - G:\My Drive\Github\Content Recommender_MVP

import os
import sys
import pandas as pd
import random
from datasets import load_dataset

# Set paths
RAW_DATA_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RAW_DATA_DIR, "wikipedia_sample.csv")

# Step 1: Load 1% of the Wikipedia dataset
print("🔄 Loading Wikipedia dataset from Hugging Face...")
dataset = load_dataset("wikipedia", "20220301.en", split="train[:3%]", trust_remote_code=True)

# Step 2: Convert to pandas DataFrame
df = pd.DataFrame(dataset)
print(f"✅ Loaded {len(df)} records.")

# Step 3: Add simulated metadata for personalization
departments = ["Engineering", "HR", "Finance", "Legal", "Support", "Marketing"]
df["department"] = [random.choice(departments) for _ in range(len(df))]

# Step 4: Select and save essential columns
df_subset = df[["id", "title", "text", "department"]]
df_subset.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved dataset to: {OUTPUT_PATH}")
