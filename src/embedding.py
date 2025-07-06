# src/embedding.py

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

def generate_embeddings(df, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    print("Generating embeddings and storing in ChromaDB...")

    client = chromadb.Client(Settings(
        persist_directory="models/chroma_db",  # Directory for persistence
        chroma_db_impl="duckdb+parquet"
    ))

    # Reset collection if rerunning
    if "complaints" in [c.name for c in client.list_collections()]:
        client.delete_collection("complaints")

    collection = client.create_collection(name="complaints")

    # Add data in chunks (to avoid memory blowup)
    batch_size = 500
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        collection.add(
            documents=batch["clean_narrative"].tolist(),
            metadatas=batch[["Product", "Company", "date"]].to_dict(orient="records"),
            ids=[str(j) for j in batch.index]
        )
        print(f"✅ Added {i+len(batch)} / {len(df)} complaints")

    client.persist()
    print("✅ ChromaDB embeddings saved.")

def main():
    df = pd.read_csv("data/processed/cleaned_complaints.csv")
    generate_embeddings(df)

if __name__ == "__main__":
    main()
