# indexing/vector_index.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseIndex

class VectorIndex(BaseIndex):
    def build(self, texts, embeddings, metadata):
        self.texts = texts
        self.embeddings = np.array(embeddings)
        self.metadata = metadata

    def query(self, query_embedding, top_k=5):
        query_vec = np.array(query_embedding).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.embeddings)[0]

        top_indices = sims.argsort()[::-1][:top_k]

        results = []
        for i in top_indices:
            results.append({
                "text": self.texts[i],
                "metadata": self.metadata[i],
                "score": float(sims[i])
            })
        return results
