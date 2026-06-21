from vector_db_builder import build_vector_db

PDF_FOLDER = r"C:\Users\Полина\Desktop\test_RAG_DB"
DB_PATH = r"C:\IT\ragData\hnsw\0617\lc_200_50_bge-m3"

build_vector_db(
    pdf_folder=PDF_FOLDER,
    db_path=DB_PATH,
    chunker_type="langchain",
    chunker_config={
        "chunk_size": 200,
        "overlap": 50
    }
)

print("✅ Векторная БД сохранена")