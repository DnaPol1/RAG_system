import uuid
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]

    @staticmethod
    def create(
            text: str,
            source_name: str | None = None,
            source_path: str | None = None,
            doc_type: str | None = None,
            document_id: str | None = None
    ) -> "Document":
        return Document(
            text=text,
            metadata={
                "document_id": document_id or str(uuid.uuid4()),
                "source_name": source_name,
                "source_path": source_path,
                "doc_type": doc_type
            }
        )