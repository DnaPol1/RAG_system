# indexing/metrics.py
import time

def measure_query(index, query_embedding, top_k=5, **kwargs):
    start = time.time()
    results = index.query(query_embedding, top_k=top_k, **kwargs)
    elapsed = time.time() - start

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    return {
        "time_sec": elapsed,
        "avg_score": avg_score,
        "results": results
    }
