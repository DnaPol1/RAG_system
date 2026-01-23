from vector_db_builder import build_vector_db

PDF_FOLDER = r"C:\Users\Полина\Desktop\test_RAG_DB"
DB_PATH = "C:/IT/ragData/rag_vector_db_v7"

build_vector_db(
    pdf_folder=PDF_FOLDER,
    db_path=DB_PATH,
    chunker_type="recursive",
    chunker_config={
        "chunk_size": 256,
        "overlap": 64
    }
)

print("✅ Векторная БД сохранена")
