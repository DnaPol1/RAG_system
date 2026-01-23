import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from indexing.base import BaseIndex

class VectorIndex(BaseIndex):
    def build(self, chunk_texts, embeddings, chunk_metadata):
        self.chunk_texts = chunk_texts
        self.embeddings = np.array(embeddings)
        self.chunk_metadata = chunk_metadata

    def query(self, query_embedding, top_k=5, query_text=None):
        if query_embedding is None:
            raise ValueError("VectorIndex requires query_embedding")

        query_vec = np.array(query_embedding).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.embeddings)[0]

        top_indices = sims.argsort()[::-1][:top_k]

        results = []
        for i in top_indices:
            results.append({
                "text": self.chunk_texts[i],
                "metadata": self.chunk_metadata[i],
                "score": float(sims[i])
            })
        return results
