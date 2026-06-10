from typing import List, Dict

class SimpleVectorRetriever:
    """
    Только плотный поиск в векторном хранилище.
    Каждый результат — отдельный chunk с vector_id, текстом и метаданными.
    """
    def __init__(self, vector_store, embedding_model,  top_k: int = 10):
        self.vector_store = vector_store
        self.top_k = top_k
        self.embedding_model = embedding_model

    def retrieve_top_chunks(self, query: str) -> List[Dict]:
        """
        Возвращает top_k ближайших chunk'ов по cosine similarity.
        """
        embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )

        results = self.vector_store.search(embedding, top_k=self.top_k)

        # Каждый результат включает vector_id, текст и метаданные
        output = []
        for doc in results:
            output.append({
                "vector_id": doc["vector_id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": doc["score"]
            })

        return output