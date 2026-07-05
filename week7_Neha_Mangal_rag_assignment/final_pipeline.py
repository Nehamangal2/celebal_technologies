from data_loader import get_data
from chunker import build_text_chunks
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def run_pipeline():
    print("Step 1: Loading raw data...")
    dataset = get_data()

    print("Step 2: Splitting into chunks...")
    text_chunks = build_text_chunks(dataset)

    print(f"Step 3: Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Step 4: Embedding {len(text_chunks)} chunks (may take a bit)...")
    vectors = embedder.embed_documents(text_chunks)

    print(f"\nDone! Created {len(vectors)} embedding vectors.")
    print(f"First vector preview (5 dims): {vectors[0][:5]}")

    return vectors


if __name__ == "__main__":
    run_pipeline()
