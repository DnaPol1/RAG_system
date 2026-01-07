import numpy as np


class RetrievalComponent:
    """
    Retrieval для RAG:
    - работает с VectorStore (FAISS)
    - использует embedding model
    - возвращает score + текст + metadata
    """

    def __init__(self, vector_store, embedding_model, top_k=1, min_similarity=0.2):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str):
        """
        Возвращает:
        - text (str)
        - metadata (dict)
        - similarity_score (float)
        """

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=self.top_k
        )

        if not results or not results[0]["text"].strip():
            return None, None, 0.0

        best = results[0]

        return (
            best["text"],
            best["metadata"],
            best["score"]
        )

    def retrieve_top_context(self, query: str):
        """
        Возвращает:
        - text (str)
        - metadata (dict)
        - similarity_score (float)
        """
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        results = self.vector_store.search(query_embedding, top_k=1)

        if not results:
            return None, None, 0.0

        best = results[0]

        if best["score"] < self.min_similarity:
            return None, None, best["score"]

        if not best["text"].strip():
            return None, None, best["score"]

        return (
            best["text"],
            best["metadata"],
            best["score"]
        )