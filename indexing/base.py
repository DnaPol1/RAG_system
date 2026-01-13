# indexing/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseIndex(ABC):
    def __init__(self):
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    @abstractmethod
    def build(self, texts: List[str], embeddings: List[list], metadata: List[dict]):
        pass

    @abstractmethod
    def query(self, query_embedding: list, top_k: int = 5) -> List[Dict]:
        """
        Возвращает список:
        {
            "text": str,
            "metadata": dict,
            "score": float
        }
        """
        pass
