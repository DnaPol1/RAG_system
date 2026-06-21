import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from core.loader import PDFLoader
from core.vectorStore import VectorStore
from sentence_transformers import SentenceTransformer
import time

from configs.config import LOCAL_BAAI_PATH
from core.chunking.factory import create_chunker

def build_vector_db(
    pdf_folder: str,
    db_path: str,
    chunker_type: str,
    chunker_config: dict,
    embedding_dim: int = 1024
):
    start = time.time()

    print("Загрузка документов")

    loader = PDFLoader(pdf_folder)
    documents = loader.load_pdfs()

    print("Чанкирование")

    chunker = create_chunker(chunker_type, **chunker_config)

    all_chunks = []
    for doc in documents:
        chunks = chunker.split(
            text=doc.text,
            metadata=doc.metadata
        )
        all_chunks.extend(chunks)

    texts = [c["text"] for c in all_chunks]

    print("Генерация эмбеддингов")

    model = SentenceTransformer(LOCAL_BAAI_PATH)
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Сохранение новой БД")

    vector_store = VectorStore(embedding_dim=embedding_dim)
    vector_store.add(embeddings, all_chunks)
    vector_store.save(db_path)
    print(f"Время генерации векторной БД:{(time.time() - start):.4f} сек")

    vector_store.summary()