import os
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from datetime import datetime

from config import SEC_EMB_MODEL_NAME, HF_TOKEN, TEST_DATA_PATH, VEC_STORE_PATH, LOCAL_BAAI_PATH
from vectorStore import VectorStore
from retriever.SimpleVectorRetriever import SimpleVectorRetriever
from evaluation.retrieval_evaluation import RetrievalEvaluation

def save_results(
        results: dict,
        output_dir: str,
        model_name: str,
        retriever_name: str,
        total_questions: int,
        k_values: list
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    save_data = {
        "model": model_name,
        "retriever": retriever_name,
        "total_questions": total_questions,
        "k_values": k_values,
        "metrics": results
    }

    output_path = (
        output_dir /
        f"retrieval_eval_{timestamp}.json"
    )

    with open(
            output_path,
            "w",
            encoding="utf-8"
    ) as f:

        json.dump(
            save_data,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"[OK] Results saved: {output_path}"
    )

def save_retrieval_results_txt(
        query: str,
        results: list,
        output_dir: str = "retrieval_debug"
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        output_dir /
        f"retrieval_{timestamp}.txt"
    )

    with open(
            output_path,
            "w",
            encoding="utf-8"
    ) as f:

        f.write(f"QUERY:\n{query}\n\n")
        f.write("=" * 100 + "\n\n")

        for i, r in enumerate(results):

            f.write(f"RESULT #{i + 1}\n")
            f.write(f"vector_id: {r['vector_id']}\n")
            f.write(f"score: {r['score']:.6f}\n")

            if r.get("metadata"):
                f.write(f"metadata: {r['metadata']}\n")

            f.write("\nTEXT:\n")
            f.write(r["text"])

            f.write("\n\n")
            f.write("=" * 100)
            f.write("\n\n")

    print(f"[OK] Retrieval results saved: {output_path}")


if __name__ == "__main__":

    k_values = [1, 3, 5, 10, 15]
    top_k = max(k_values)
    vector_store = VectorStore.load(path=r"C:\IT\ragData\hnsw\0602\lc_200_50_bge-m3")
    os.environ[HF_TOKEN] = HF_TOKEN
    embedding_model = SentenceTransformer(LOCAL_BAAI_PATH)
    retriever = SimpleVectorRetriever(vector_store, embedding_model=embedding_model, top_k=top_k)
    evaluator = RetrievalEvaluation(
        k_values=k_values,
        json_path=r"C:\Users\Полина\PycharmProjects\RAG_system\dataset_0602_200_50.json",
        retriever=retriever
    )
    results = evaluator.calculate_hit_rate()
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    save_results(
        results=results,
        output_dir="evaluation_results",
        model_name=SEC_EMB_MODEL_NAME,
        retriever_name="FAISS_HNSW",
        total_questions=len(evaluator.test_data),
        k_values=k_values
    )