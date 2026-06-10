from sentence_transformers import CrossEncoder
from typing import List, Dict


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None
    ):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        documents: [{ "text": ..., "metadata": ... }]
        """

        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents = sorted(
            documents,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return documents[:top_k]
