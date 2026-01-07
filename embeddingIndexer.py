from sentence_transformers import SentenceTransformer
from typing import List

from vectorStore import VectorStore

class EmbeddingIndexer:
    """
    Класс для:
    - чанкинга документов
    - генерации эмбеддингов
    - сохранения в VectorStore
    """

    def __init__(self, vector_store: VectorStore, chunk_size=1000, overlap=200):
        self.model = SentenceTransformer("sentence-transformers/LaBSE")
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.overlap = overlap


    def index_documents(self, documents: List[str], batch_size: int = 16):
        """
        Принимает список Document и индексирует их
        """

        all_chunks = []
        all_metadatas = []
        all_documents = []

        for doc in documents:
            if not hasattr(doc, "text"):
                raise TypeError("Ожидается объект Document с полем text")

            chunks = self.chunk_text(doc.text)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_documents.append(doc.text) # источник
                all_metadatas.append({
                    "source": doc.source,
                    "chunk_id": i
                })

        if not all_chunks:
            raise ValueError("Нет текста для индексации (все документы пустые?)")

        embeddings = self.model.encode(
            all_chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.vector_store.add(
            embeddings=embeddings,
            documents=all_documents,
            texts=all_chunks,
            metadatas=all_metadatas
        )

    def encode_query(self, query: str):
        """
        Эмбеддинг пользовательского запроса
        """
        return self.model.encode(
            [query],
            normalize_embeddings=True
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Чанкование текста с overlap
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks