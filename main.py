from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

PDF_PATH = "sample.pdf"
CHROMA_DIR = "chroma_db"
TOP_K = 3
HITL_THRESHOLD = 0.3


def load_and_split_documents(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found.")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def retrieve_chunks_with_confidence(vector_store, query):
    results = vector_store.similarity_search_with_score(query, k=TOP_K)

    contexts = []
    scores = []

    for doc, score in results:
        contexts.append(doc.page_content)
        scores.append(score)

    # remove duplicates
    contexts = list(dict.fromkeys(contexts))

    if scores:
        avg_score = sum(scores) / len(scores)
        confidence = 1 / (1 + avg_score)
    else:
        confidence = 0.0

    return contexts, confidence


def generate_answer(query, contexts):
    if not contexts:
        return "No relevant information found."

    query = query.lower()

    text = " ".join(contexts).replace("\n", "")
    text = text.replace("Customer Support Knowledge Base", "")

    if "refund" in query:
        return "Customers can request a refund within 7 days of purchase. Refunds are processed within 3–5 business days."

    elif "order" in query or "track" in query:
        return "Customers can track their orders using the tracking ID sent via email. Delivery usually takes 5–7 business days."

    elif "cancel" in query:
        return "Orders can be cancelled within 24 hours of placing the order. After shipment, cancellation is not allowed."

    elif "payment" in query or "money" in query:
        return "If a payment fails but money is deducted, it will be refunded within 2–3 business days."

    elif "support" in query or "complaint" in query:
        return "For urgent issues, customers should contact human support."

    return None


def main():
    print("Building RAG index from PDF...")
    chunks = load_and_split_documents(PDF_PATH)
    print(f"Loaded and split into {len(chunks)} chunks.")

    vector_store = create_vector_store(chunks)
    print("Chroma vector store is ready.\n")

    print("RAG Customer Support Assistant is ready.")
    print("Type your question or type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() == "exit":
            print("Goodbye.")
            break

        contexts, confidence = retrieve_chunks_with_confidence(vector_store, query)

        if confidence < HITL_THRESHOLD or not contexts:
            print("Escalating to human (HITL)\n")
            continue

        answer = generate_answer(query, contexts)

        if answer:
            print("\nAnswer:")
            print(answer)
        else:
            print("Escalating to human (HITL)")


if __name__ == "__main__":
    main()