#dentify and collect the correct document indices for your relevant_ids

import os
import pandas as pd
import re

import os
import pandas as pd

def search_index_data(data_path="data/processed/combined_documents.csv",
                      query="employee benefits",
                      export=False,
                      export_path="data/processed/relevant_docs.csv",
                      min_len=30):

    if not os.path.exists(data_path):
        raise FileNotFoundError(f" Data file not found at {data_path}")
    
    if not query:
        raise ValueError(" Query string must be provided.")
    
    query = query.strip()

    print(f" Searching for: '{query}' in {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    # Basic filtering
    filtered_df = df[df["content"].str.contains(query, case=False, na=False)]

    # Filter out very short content
    filtered_df = filtered_df[filtered_df["content"].str.len() > min_len]

    print(f" Found {len(filtered_df)} matching documents.")
    print(f" Document indices: {filtered_df.index.tolist()[:10]} ...")
    print(filtered_df[["title", "content"]].head(5))

    if export:
        filtered_df.to_csv(export_path, index=True)
        print(f" Exported to: {export_path}")

    return filtered_df.copy()
    
if __name__ == "__main__":
    df = search_index_data(query="performance review", export=True)
    print(df.index.tolist())

