from typing import List, Dict
from sentence_transformers import SentenceTransformer
from vectorStore import VectorStore
from reranker import CrossEncoderReranker

class RetrievalComponent:
    """
    Retrieval для RAG:
    - работает с VectorStore (FAISS)
    - использует embedding model
    - возвращает score + текст + metadata
    """

    def __init__(self,
                 vector_store: VectorStore,
                 embedding_model: SentenceTransformer,
                 top_k: int = 10,
                 top_k_candidates: int = 20,
                 use_reranker: bool = False,
                 reranker: CrossEncoderReranker = None):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        self.top_k = top_k
        self.top_k_candidates = top_k_candidates

        self.use_reranker = use_reranker
        self.reranker = reranker

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

    def retrieve_top_context(self, query: str) -> List[Dict]:
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

        # 1. Bi-encoder retrieval
        candidates = self.vector_store.search(
            query_embedding,
            top_k=self.top_k_candidates
        )

        # 2. Optional reranking
        if self.use_reranker and self.reranker is not None:
            return self.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=self.top_k
            )

        # fallback: просто берём top-k
        return candidates[:self.top_k]