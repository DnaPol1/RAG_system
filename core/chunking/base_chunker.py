from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunker(ABC):
    """
    Базовый интерфейс для чанкования текста.
    """

    @abstractmethod
    def split(
        self,
        text: str,
        metadata: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Разбивает текст на чанки.

        Возвращает список словарей:
        {
            "text": str,
            "metadata": dict
        }
        """
        raise NotImplementedError