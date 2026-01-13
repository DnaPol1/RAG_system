from typing import List, Dict
from .rules import split_legal_text

class SemanticChunker:
    def __init__(
        self,
        min_chars: int = 300,
        max_chars: int = 1200,
        overlap: int = 150
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap = overlap

    def split(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Возвращает список чанков:
        {
            "text": str,
            "metadata": dict
        }
        """
        raw_chunks = split_legal_text(text)
        chunks = []

        current = ""
        chunk_id = 0

        for part in raw_chunks:
            if len(current) + len(part) <= self.max_chars:
                current += "\n" + part
            else:
                if len(current) >= self.min_chars:
                    chunks.append(self._make_chunk(
                        current, metadata, chunk_id
                    ))
                    chunk_id += 1

                # overlap
                current = current[-self.overlap:] + "\n" + part

        if len(current) >= self.min_chars:
            chunks.append(self._make_chunk(
                current, metadata, chunk_id
            ))

        return chunks

    def _make_chunk(self, text: str, metadata: Dict, chunk_id: int) -> Dict:
        meta = metadata.copy()
        meta["chunk_id"] = chunk_id
        meta["chunk_size"] = len(text)

        return {
            "text": text.strip(),
            "metadata": meta
        }
