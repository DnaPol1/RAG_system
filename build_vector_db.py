from vector_db_builder import build_vector_db

PDF_FOLDER = r"C:\Users\Полина\Desktop\test_RAG_DB"
DB_PATH = "C:/IT/ragData/rag_vector_db_v6"

build_vector_db(
    pdf_folder=PDF_FOLDER,
    db_path=DB_PATH,
    chunker_type="semantic",
    chunker_config={
        "min_chars": 300,
        "max_chars": 1200,
        "overlap": 150
    }
)

print("✅ Векторная БД сохранена")
