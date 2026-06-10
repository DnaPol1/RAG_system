from typing import List, Dict, Any
from chunking.base_chunker import BaseChunker


class FixedChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 100
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap


    def split(
        self,
        text: str,
        metadata: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:

        if metadata is None:
            metadata = {}

        tokens = text.split()

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(tokens):

            end = start + self.chunk_size

            chunk_tokens = tokens[start:end]

            chunk_text = " ".join(chunk_tokens)

            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = chunk_id

            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

            chunk_id += 1
            start += self.chunk_size - self.overlap

        return chunks