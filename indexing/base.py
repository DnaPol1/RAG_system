from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseIndex(ABC):
    @abstractmethod
    def build(
            self,
            texts: List[str],
            embeddings: Optional[List[list]],
            metadata: List[Dict[str, Any]]
    ):
        """
        :param texts: - чанки
        :param embeddings: - эмбеддниги (если применимо)
        :param metadata: - метаданные чанков
        :return:
        """
        pass

    @abstractmethod
    def query(
            self,
            query_embedding: Optional[list],
            top_k: int = 5,
            query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список:
        {
            "text": str,
            "metadata": dict,
            "score": float
        }
        """
        pass
