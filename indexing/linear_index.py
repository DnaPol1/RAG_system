# indexing/linear_index.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseIndex

class LinearIndex(BaseIndex):
    def build(self, texts, embeddings, metadata):
        self.texts = texts
        self.metadata = metadata
        self.embeddings = np.array(embeddings)

    def query(self, query_embedding, top_k=5):
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
            "text": self.texts[i],
            "metadata": self.metadata[i],
            "score": float(score)
        } for i, score in ranked]
