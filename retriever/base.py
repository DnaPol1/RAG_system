from abc import ABC, abstractmethod
from typing import List, Dict


class BaseRetriever(ABC):
    """
    Базовый интерфейс retriever'а для RAG
    """

    @abstractmethod
    def retrieve_top_context(
        self,
        query: str,
        expanded_queries: List[str] | None = None
    ) -> List[Dict]:
        """
        Возвращает список документов:
        {
            "text": str,
            "metadata": dict,
            "score": float,
            ...
        }
        """
        pass
