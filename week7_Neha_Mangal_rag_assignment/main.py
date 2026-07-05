import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

DB_PATH = "./my_vector_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3"
TOP_K = 2

# --- Setup ---
db_client = chromadb.PersistentClient(path=DB_PATH)
doc_collection = db_client.get_collection(name=COLLECTION_NAME)
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
llm = OllamaLLM(model=LLM_MODEL_NAME)  # local model, runs via Ollama


def get_answer(question):
    # --- Retrieval step ---
    q_embedding = embedder.embed_query(question)
    results = doc_collection.query(query_embeddings=[q_embedding], n_results=TOP_K)
    retrieved_context = "\n".join(results["documents"][0])

    # --- Build prompt for generation ---
    prompt = f"""
    Answer the question based ONLY on the following context:
    {retrieved_context}

    Question: {question}
    Answer:
    """

    # send prompt to the LLM
    answer = llm.invoke(prompt)
    return answer


if __name__ == "__main__":
    user_question = input("Ask a question: ")
    print("Thinking...")
    print("\nAI Answer:", get_answer(user_question))
