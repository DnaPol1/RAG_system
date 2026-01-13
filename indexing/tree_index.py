# indexing/tree_index.py
import numpy as np
from sklearn.neighbors import BallTree
from .base import BaseIndex

class TreeIndex(BaseIndex):
    def build(self, texts, embeddings, metadata):
        self.texts = texts
        self.metadata = metadata
        self.embeddings = np.array(embeddings)
        self.tree = BallTree(self.embeddings, metric="euclidean")

    def query(self, query_embedding, top_k=5):
        dist, idx = self.tree.query(
            np.array(query_embedding).reshape(1, -1),
            k=top_k
        )

        results = []
        for d, i in zip(dist[0], idx[0]):
            results.append({
                "text": self.texts[i],
                "metadata": self.metadata[i],
                "score": 1 - float(d)
            })
        return results
