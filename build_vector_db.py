from loader import PDFLoader
from vectorStore import VectorStore
from embeddingIndexer import EmbeddingIndexer

PDF_FOLDER = r"C:\Users\Полина\Desktop\test_RAG_DB"

loader = PDFLoader(PDF_FOLDER)
documents = loader.load_pdfs()

vector_store = VectorStore(embedding_dim=768)
indexer = EmbeddingIndexer(vector_store)

indexer.index_documents(documents)

vector_store.save("rag_vector_db_v3")

print("✅ Векторная БД сохранена")
