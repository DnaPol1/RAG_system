from collections import defaultdict
from typing import List,Dict

def aggregate_chunks_to_docments(retrieved_chunks: List[Dict], doc_key: str = "document_id"
                                 ) -> Dict[str, Dict]:
    """
    Группирует чанки по документам
    Возвращает:
    {
        doc_id: {
            "doc_score": float,
            "chunks" : [chunk, ...]
        }
    }
    :param retrieved_chunks:
    :param doc_key:
    :return:
    """

    docs = defaultdict(lambda : {
        "doc_score": 0.0,
        "chunks": []
    })

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        doc_id = metadata.get(doc_key)

        if not doc_id:
            continue

        docs[doc_id]["chunks"].append(chunk)
        docs[doc_id]["doc_score"] = max(docs[doc_id]["doc_score"], chunk.get("score", 0.0))

    return docs