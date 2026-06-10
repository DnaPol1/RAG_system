def doc_recall_at_k(results, gold_doc_id, k):
    top_k = results[:k]
    return int(any(d["doc_id"] == gold_doc_id for d in top_k))
