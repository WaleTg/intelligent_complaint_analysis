
from src.rag_pipeline import answer_question

# Sample test questions
sample_questions = [
    "Why are users unhappy with credit cards?",
    "What issues do people report about personal loans?",
    "What are the most common complaints about Buy Now, Pay Later?",
    "Are there fraud-related issues in money transfers?",
    "Why are users closing their savings accounts?"
]

print("Running RAG Demo...\n")

for idx, question in enumerate(sample_questions, 1):
    print(f"=== Question {idx}: {question} ===")
    answer, context, metadata = answer_question(question)
    
    print("\nGenerated Answer:")
    print(answer.strip())

    print("\nTop 2 Retrieved Complaint Excerpts:")
    for i in range(min(2, len(context))):
        print(f"- {context[i]} (Product: {metadata[i].get('Product')}, Company: {metadata[i].get('Company')})")

    print("\n" + "="*60 + "\n")
