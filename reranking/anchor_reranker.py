from typing import List, Dict
import numpy as np


class AnchorReranker:
    """
    Reranks retrieved chunks using short anchor text instead of full chunk
    """

    def __init__(self, cross_encoder, top_k: int = 10):
        self.model = cross_encoder
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        candidates: list of dicts with keys:
        - text
        - metadata (must contain 'anchor')
        - score (optional)
        """

        pairs = []
        valid_items = []

        for item in candidates:
            anchor = item["metadata"].get("anchor")
            if not anchor:
                continue
            pairs.append((query, anchor))
            valid_items.append(item)

        if not pairs:
            return candidates[:self.top_k]

        scores = self.model.predict(pairs)

        scored = list(zip(scores, valid_items))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in scored[:self.top_k]]
