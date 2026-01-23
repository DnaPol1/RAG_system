from typing import List, Dict, Any
from chunking.base_chunker import BaseChunker
from chunking.rules import split_legal_text

class SemanticChunker(BaseChunker):
    """
    Семантическое чанкование юридических текстов
    на основе структурных паттернов
    """
    def __init__(
        self,
        min_chars: int = 300,
        max_chars: int = 1200,
        overlap: int = 150
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap = overlap

    def split(self, text: str, metadata: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        raw_chunks = split_legal_text(text)
        chunks = []

        current = ""
        chunk_id = 0

        for part in raw_chunks:
            if len(current) + len(part) <= self.max_chars:
                current += "\n" + part
            else:
                if len(current) >= self.min_chars:
                    chunks.append(self._make_chunk(current, metadata, chunk_id))
                    chunk_id += 1

                current = current[-self.overlap:] + "\n" + part

        if len(current) >= self.min_chars:
            chunks.append(self._make_chunk(current, metadata, chunk_id))

        return chunks

    def _make_chunk(self, text: str, metadata: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        text = text.strip()

        meta = metadata.copy()
        meta.update({
            "chunk_id": chunk_id,
            "chunk_size": len(text),
            "chunking_strategy": "semantic"
        })

        return {
            "text": text.strip(),
            "metadata": meta
        }