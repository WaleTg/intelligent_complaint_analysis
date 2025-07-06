import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def load_chroma_collection(persist_dir="models/chroma_db", collection_name="complaints"):
    client = chromadb.Client(Settings(
        persist_directory=persist_dir,
        chroma_db_impl="duckdb+parquet"
    ))
    return client.get_collection(collection_name)

def semantic_search(query, top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = load_chroma_collection()

    print(f"\n Searching for: \"{query}\"")

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    for i in range(top_k):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        print(f"\n--- Result {i+1} ---")
        print(f" {doc}")
        print(f"Metadata: {meta}")

if __name__ == "__main__":
    user_query = input(" Ask a question about complaints (e.g., 'Why are users upset with BNPL?'):\n> ")
    semantic_search(user_query)
