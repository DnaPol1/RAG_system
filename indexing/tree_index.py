import numpy as np
from sklearn.neighbors import BallTree
from indexing.base import BaseIndex

class TreeIndex(BaseIndex):
    def build(self, chunk_texts, embeddings, chunk_metadata):
        self.chunk_texts = chunk_texts
        self.chunk_metadata = chunk_metadata
        self.embeddings = np.array(embeddings)
        self.tree = BallTree(self.embeddings, metric="euclidean")

    def query(self, query_embedding, top_k=5, query_text=None):
        if query_embedding is None:
            raise ValueError("TreeIndex requires query_embedding")

        dist, idx = self.tree.query(
            np.array(query_embedding).reshape(1, -1),
            k=top_k
        )

        results = []
        for d, i in zip(dist[0], idx[0]):
            results.append({
                "text": self.chunk_texts[i],
                "metadata": self.chunk_metadata[i],
                "score": 1 - float(d)
            })
        return results
