import chromadb
from data_loader import get_data
from chunker import build_text_chunks
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "./my_vector_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEST_QUERY = "What is the Catholic character of the school?"


def main():
    # --- Step 1: Load and chunk data ---
    print("Loading and chunking data...")
    dataset = get_data()
    text_chunks = build_text_chunks(dataset)

    # --- Step 2: Load embedding model ---
    print("Loading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # --- Step 3: Set up ChromaDB collection ---
    print("Storing vectors in ChromaDB...")
    db_client = chromadb.PersistentClient(path=DB_PATH)
    doc_collection = db_client.create_collection(name=COLLECTION_NAME, get_or_create=True)

    # --- Step 4: Embed and store chunks ---
    # generate a unique id per chunk
    chunk_ids = [str(i) for i in range(len(text_chunks))]
    chunk_vectors = embedder.embed_documents(text_chunks)

    doc_collection.add(
        documents=text_chunks,
        embeddings=chunk_vectors,
        ids=chunk_ids
    )
    print("Success! Data stored in ChromaDB.")

    # --- Step 5: Quick search test ---
    query_vector = embedder.embed_query(TEST_QUERY)
    results = doc_collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )

    print(f"\nQuery: {TEST_QUERY}")
    print(f"Top result: {results['documents'][0][0]}")


if __name__ == "__main__":
    main()
