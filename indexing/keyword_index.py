# indexing/keyword_index.py
import re
from collections import defaultdict
from .base import BaseIndex

class KeywordIndex(BaseIndex):
    def build(self, texts, embeddings, metadata):
        self.texts = texts
        self.metadata = metadata
        self.index = defaultdict(set)

        for i, text in enumerate(texts):
            tokens = set(re.findall(r"[а-яА-Яa-zA-Z]{3,}", text.lower()))
            for token in tokens:
                self.index[token].add(i)

    def query(self, query_embedding, top_k=5, query_text: str = ""):
        tokens = set(re.findall(r"[а-яА-Яa-zA-Z]{3,}", query_text.lower()))
        scores = defaultdict(int)

        for token in tokens:
            for idx in self.index.get(token, []):
                scores[idx] += 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [{
            "text": self.texts[i],
            "metadata": self.metadata[i],
            "score": score
        } for i, score in ranked]
