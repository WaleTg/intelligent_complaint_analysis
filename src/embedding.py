import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

def generate_embeddings(df, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    print("Generating embeddings and storing in ChromaDB...")

    # Initialize the persistent client
    client = chromadb.PersistentClient(path="models/chroma_db")

    # Delete old collection if it exists
    if "complaints" in [c.name for c in client.list_collections()]:
        client.delete_collection("complaints")

    collection = client.create_collection(name="complaints")

    batch_size = 500
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]

        # Generate sentence embeddings
        embeddings = model.encode(batch["clean_narrative"].tolist(), show_progress_bar=False)

        collection.add(
            documents=batch["clean_narrative"].tolist(),
            embeddings=embeddings.tolist(),
            metadatas=batch[["Product", "Company", "date"]].to_dict(orient="records"),
            ids=[str(j) for j in batch.index]
        )
        print(f"✅ Added {i + len(batch)} / {len(df)} complaints")

    print("✅ All embeddings stored in ChromaDB.")

def main():
    df = pd.read_csv("data/processed/cleaned_complaints.csv")
    generate_embeddings(df)

if __name__ == "__main__":
    main()
