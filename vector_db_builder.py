from loader import PDFLoader
from vectorStore import VectorStore
from chunking import create_chunker
from sentence_transformers import SentenceTransformer


def build_vector_db(
    pdf_folder: str,
    db_path: str,
    chunker_type: str,
    chunker_config: dict,
    embedding_model_name: str = "sentence-transformers/LaBSE",
    embedding_dim: int = 768
) -> VectorStore:

    loader = PDFLoader(pdf_folder)
    documents = loader.load_pdfs()

    chunker = create_chunker(chunker_type, **chunker_config)

    all_chunks = []
    for doc in documents:
        chunks = chunker.split(
            text=doc.text,
            metadata={"source": doc.source}
        )
        all_chunks.extend(chunks)

    texts = [c["text"] for c in all_chunks]

    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    vector_store = VectorStore(embedding_dim=embedding_dim)
    vector_store.add(embeddings, all_chunks)
    vector_store.save(db_path)

    return vector_store, model
