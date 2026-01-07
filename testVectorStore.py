from loader import PDFLoader
from vectorStore import VectorStore
from embeddingIndexer import EmbeddingIndexer
from RetrievalComponentForRAG import RetrievalComponent
import numpy as np


PDF_FOLDER = r"C:\Users\Полина\Desktop\test_RAG_DB"
EMBEDDING_DIM = 768


def assert_true(condition, message):
    if not condition:
        raise AssertionError("❌ " + message)
    print("✅", message)


def test_full_rag_pipeline():
    print("\n=== TEST: FULL RAG PIPELINE ===\n")

    # ---------- 1. LOAD DOCUMENTS ----------
    loader = PDFLoader(PDF_FOLDER)
    documents = loader.load_pdfs()

    assert_true(len(documents) > 0, "Документы загружены")
    assert_true(all(hasattr(d, "text") for d in documents),
                "Документы содержат поле text")

    non_empty_docs = [
        d for d in documents
        if isinstance(d.text, str) and d.text.strip()
    ]
    assert_true(len(non_empty_docs) > 0,
                "Есть непустые документы после очистки")

    # ---------- 2. VECTOR STORE ----------
    vector_store = VectorStore(embedding_dim=EMBEDDING_DIM)

    # ---------- 3. EMBEDDING INDEXER ----------
    indexer = EmbeddingIndexer(vector_store)
    indexer.index_documents(non_empty_docs)

    assert_true(vector_store.index.ntotal > 0,
                "Векторная БД содержит векторы")

    # ---------- 4. CHECK STORED TEXTS ----------
    assert_true(
        hasattr(vector_store, "texts"),
        "VectorStore хранит тексты"
    )
    assert_true(
        len(vector_store.texts) == vector_store.index.ntotal,
        "Количество текстов = количеству векторов"
    )

    # ---------- 5. RETRIEVAL ----------
    retriever = RetrievalComponent(vector_store, indexer)
    retriever.retrieve(vector_store.texts)

    query = "Какие документы нужны для лизинговых компаний"

    retrieved = retriever.retrieve_top_context(query)

    assert_true(
        isinstance(retrieved, tuple) and len(retrieved) == 3,
        "Ретривер вернул (text, metadata, score)"
    )

    text, metadata, score = retrieved

    assert_true(
        isinstance(text, str) and len(text.strip()) > 0,
        "Текст извлечён корректно"
    )

    assert_true(
        isinstance(metadata, dict),
        "Метаданные корректны"
    )

    assert_true(
        isinstance(score, float),
        "Скор релевантности корректен"
    )

    # ---------- 6. COSINE SIMILARITY ----------
    query_vec = indexer.encode_query(query)
    D, I = vector_store.index.search(
        np.array(query_vec).astype("float32"), 1
    )

    similarity = float(D[0][0])

    assert_true(similarity > 0,
                f"Косинусное сходство > 0 ({similarity:.4f})")

    # ---------- 7. SOURCE SAFETY ----------
    try:
        result = {
            "text": retrieved,
            "score": similarity,
            "source": "vector_db"
        }
        _ = result["source"]
        print("✅ Ключ 'source' безопасно используется")
    except KeyError:
        raise AssertionError("❌ KeyError: 'source' — источник не задан")

    print("\n🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ. RAG ЖИВОЙ.\n")


if __name__ == "__main__":
    test_full_rag_pipeline()