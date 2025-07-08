import os
import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import multiprocessing
from chromadb import Client
from chromadb.config import Settings

# Fix multiprocessing issue on macOS
multiprocessing.set_start_method("spawn", force=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB client
from chromadb import HttpClient

chroma_client = HttpClient(host="localhost", port=8000)


# Load collection
collection = chroma_client.get_collection(name="complaints")

# Load text-to-text generation model
generator = pipeline("text2text-generation", model="google/flan-t5-base", device=0 if device == "cuda" else -1)

# Prompt template
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

def retrieve_context(question, k=5):
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=k)
    documents = results["documents"][0]
    return "\n".join(documents)

def ask_question(question):
    context = retrieve_context(question)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return response, context

def main():
    questions = [
        "Was my account opened fraudulently?",
        "What company issued the disputed credit card?",
        "Is there any information about my complaint being resolved?",
        "Who opened the account in my name?",
        "How much was the fraudulent charge?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer, ctx = ask_question(q)
        print(f" Answer: {answer}")
        print(f"Retrieved Context:\n{ctx[:300]}...\n")

if __name__ == "__main__":
    main()
