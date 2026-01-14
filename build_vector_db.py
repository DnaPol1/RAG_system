from vector_db_builder import build_vector_db

PDF_FOLDER = r"C:\Users\Полина\Desktop\test_RAG_DB"
DB_PATH = "C:/IT/ragData/rag_vector_db_v6/struct"

build_vector_db(
    pdf_folder=PDF_FOLDER,
    db_path=DB_PATH,
    chunker_type="struct",
    chunker_config={
        "chunk_size": 50
    }
)

print("✅ Векторная БД сохранена")
