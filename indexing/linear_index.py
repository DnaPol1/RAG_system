import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from indexing.base import BaseIndex

class LinearIndex(BaseIndex):
    def build(self, chunk_texts, embeddings, chunk_metadata):
        self.chunk_texts = chunk_texts
        self.chunk_metadata = chunk_metadata
        self.embeddings = np.array(embeddings)

    def query(self, query_embedding, top_k=5, query_text=None):
        if query_embedding is None:
            raise ValueError("LinearIndex requires query_embedding")

        sims = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            self.embeddings
        )[0]

        ranked = sorted(
            enumerate(sims),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [{
            "text": self.chunk_texts[i],
            "metadata": self.chunk_metadata[i],
            "score": float(score)
        } for i, score in ranked]
