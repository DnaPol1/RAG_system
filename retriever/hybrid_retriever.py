from typing import List, Dict
from collections import defaultdict
import numpy as np

from retriever.base import BaseRetriever
from retriever.lexical import lexical_score


class HybridRetriever(BaseRetriever):
    """
    Универсальный retriever для RAG:
    - vector retrieval
    - lexical scoring
    - hybrid scoring
    - multi-query aggregation (phase 5.1)
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        top_k: int = 10,
        top_k_candidates: int = 30,
        alpha: float = 1.0,
        beta: float = 0.0,
        multi_query: bool = False,
        expansion_weight: float = 0.4,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        self.top_k = top_k
        self.top_k_candidates = top_k_candidates

        self.alpha = alpha
        self.beta = beta

        self.multi_query = multi_query
        self.expansion_weight = expansion_weight

    # =========================
    # Public API
    # =========================

    def retrieve_top_context(
        self,
        query: str,
        expanded_queries: List[str] | None = None
    ) -> List[Dict]:
        """
        Основной retrieval pipeline
        """

        aggregated = defaultdict(lambda: {
            "text": None,
            "metadata": None,
            "vector_score": 0.0,
            "lexical_score": 0.0,
            "score": 0.0,
        })

        # основной запрос
        self._retrieve_and_merge(
            query=query,
            weight=1.0,
            storage=aggregated
        )

        # расширенные запросы
        if self.multi_query and expanded_queries:
            for q in expanded_queries:
                self._retrieve_and_merge(
                    query=q,
                    weight=self.expansion_weight,
                    storage=aggregated
                )

        results = list(aggregated.values())

        self._apply_hybrid_scoring(results)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.top_k]

    # =========================
    # Internal methods
    # =========================

    def _retrieve_and_merge(
        self,
        query: str,
        weight: float,
        storage: dict
    ) -> None:
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        docs = self.vector_store.search(
            query_embedding,
            top_k=self.top_k_candidates
        )

        for doc in docs:
            # ЕДИНСТВЕННЫЙ идентификатор документа
            key = doc["metadata"]["source_path"]

            vector_score = float(doc.get("score", 0.0))
            lex_score = lexical_score(query, doc.get("text", ""))

            if storage[key]["text"] is None:
                storage[key]["text"] = doc["text"]
                storage[key]["metadata"] = doc["metadata"]

            storage[key]["vector_score"] += vector_score * weight
            storage[key]["lexical_score"] += lex_score * weight

    def _apply_hybrid_scoring(self, docs: List[Dict]) -> None:
        """
        score = alpha * norm(vector) + beta * norm(lexical)
        """

        v_scores = np.array(
            [d["vector_score"] for d in docs],
            dtype=np.float32
        )
        l_scores = np.array(
            [d["lexical_score"] for d in docs],
            dtype=np.float32
        )

        if v_scores.max() > 0:
            v_scores /= v_scores.max()

        if l_scores.max() > 0:
            l_scores /= l_scores.max()

        for d, v, l in zip(docs, v_scores, l_scores):
            d["score"] = self.alpha * v + self.beta * l
