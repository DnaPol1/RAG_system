from loader import PDFLoader
from vectorStore import VectorStore
from embeddingIndexer import EmbeddingIndexer
from RetrievalComponentForRAG import RetrievalComponent
from ragPipeline import RAGPipeline
from gigaChat import GigaChatClient
from sentence_transformers import SentenceTransformer

with open("API.txt", "r", encoding="utf-8") as f:
    API_KEY = f.readline().strip()

if __name__ == "__main__":
    # ====== Загрузка модели эмбеддингов ======
    embedding_model = SentenceTransformer("sentence-transformers/LaBSE")

    # ====== Загрузка векторной БД ======
    vector_store = VectorStore.load("rag_vector_db_v3")

    retrieval = RetrievalComponent(vector_store, embedding_model)

    gigachat = GigaChatClient(
        credentials=API_KEY
    )

    rag = RAGPipeline(retrieval, gigachat)

    query = "Какие документы нужно предоставить лизинговым компаниям в рамках программы Лизинговые проекты?"

    print("\n=== БЕЗ RAG ===")
    print(rag.answer_without_rag(query))

    print("\n=== С RAG ===")
    result = rag.answer_with_rag(query)

    print("Ответ:")
    print(result["answer"])

    print("\nСходство:", round(result["similarity"], 4))
    print("Источник:", result["source"])
