from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chunking.base_chunker import BaseChunker


class RecursiveTextChunker(BaseChunker):
    """
    Умное чанкование на основе RecursiveCharacterTextSplitter.

    Подходит для русского языка.
    Не использует LLM.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        :param chunk_size: максимальный размер чанка (в символах)
        :param chunk_overlap: перекрытие между чанками
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # абзацы
                "\n",    # строки
                ". ",    # предложения
                " ",     # слова
                ""
            ]
        )

    def split(
        self,
        text: str,
        metadata: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Разбивает текст и аккуратно размножает metadata.
        """

        if not text or not text.strip():
            return []

        metadata = metadata or {}

        chunks = self.splitter.split_text(text)

        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                "text": chunk.strip(),
                "metadata": {
                    **metadata,
                    "chunk_id": i,
                    "chunk_size": len(chunk)
                }
            })

        return results
