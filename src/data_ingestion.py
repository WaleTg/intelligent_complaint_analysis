# src/data_ingestion.py

import pandas as pd
import os
import re

# Supported product categories
TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later (BNPL)",
    "Savings account",
    "Money transfer, virtual currency"
]

def clean_text(text):
    """Lowercase, strip, and remove excessive whitespace."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def load_and_clean_data(input_path: str, output_path: str):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print("Filtering rows with complaint narratives...")
    df = df[df['Consumer complaint narrative'].notnull()]

    print("Filtering to relevant financial products...")
    df = df[df['Product'].isin(TARGET_PRODUCTS)]

    print("Cleaning text fields...")
    df['clean_narrative'] = df['Consumer complaint narrative'].apply(clean_text)
    df = df.rename(columns={
        "Consumer complaint narrative": "narrative",
        "Date received": "date"
    })

    print(f"Saving cleaned data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Done! {len(df)} complaints saved.")

if __name__ == "__main__":
    input_csv = "data/raw/complaints.csv"        # <- Put original file here
    output_csv = "data/processed/cleaned_complaints.csv"
    load_and_clean_data(input_csv, output_csv)
