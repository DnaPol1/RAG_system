import re
from typing import List, Dict, Any
from chunking.base_chunker import BaseChunker


SECTION_PATTERN = re.compile(
    r"""
    (?P<title>
        (Раздел|Глава|Статья|Пункт)\s+\d+.* |
        \d+(\.\d+)*\.\s+.* |
        [IVXLCDM]+\.\s+.*
    )
    """,
    re.VERBOSE | re.IGNORECASE
)


class StructuralChunker(BaseChunker):
    """
    Структурное чанкование по пунктам / статьям / разделам.
    """

    def split(
        self,
        text: str,
        metadata: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:

        if metadata is None or "document_id" not in metadata:
            raise ValueError("metadata must contain document_id")

        lines = text.splitlines()

        chunks = []
        current_title = None
        current_lines = []
        chunk_id = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = SECTION_PATTERN.match(line)

            if match:
                # сохраняем предыдущий чанк
                if current_title and current_lines:
                    chunks.append(
                        self._make_chunk(
                            current_title,
                            current_lines,
                            metadata,
                            chunk_id
                        )
                    )
                    chunk_id += 1

                current_title = match.group("title")
                current_lines = []
            else:
                if current_title:
                    current_lines.append(line)

        # последний чанк
        if current_title and current_lines:
            chunks.append(
                self._make_chunk(
                    current_title,
                    current_lines,
                    metadata,
                    chunk_id
                )
            )

        return chunks

    def _make_chunk(
        self,
        title: str,
        lines: List[str],
        metadata: Dict[str, Any],
        chunk_id: int
    ) -> Dict[str, Any]:

        text = title + "\n" + "\n".join(lines)

        meta = metadata.copy()
        meta.update({
            "chunk_id": chunk_id,
            "anchor": title.strip(),
            "chunk_size": len(text),
            "structure_type": "section"
        })

        return {
            "text": text,
            "metadata": meta
        }
