import json
import numpy as np
from typing import List, Dict

from RetrievalComponentForRAG import RetrievalComponent


class RetrievalEvaluation:
    def __init__(self, retriever: RetrievalComponent = None):
        self.retriever = retriever

    def load_data_from_json(self, json_path: str) -> List[Dict]:
        """
        Загружает тестовые данные из JSON файла.

        Ожидаемый формат:
        [
            {
                "query": str,
                "relevant_sources": [str, str, ...]
            }
        ]
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_data = []
        for item in data:
            test_data.append({
                "query": item["query"],
                "relevant_sources": set(item["relevant_sources"])
            })

        return test_data

    def calculate_hit_rate_simple(
        self,
        test_data: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Вычисляет Hit Rate@k.

        Hit@k = 1, если хотя бы один релевантный документ
        попал в top-k результатов.
        """

        results = {f"hit_rate@{k}": [] for k in k_values}

        max_k = max(k_values)
        self.retriever.top_k = max_k

        for item in test_data:
            query = item["query"]
            relevant_sources = item["relevant_sources"]

            retrieved_items = self.retriever.retrieve_top_context(query)

            retrieved_sources = [
                r["metadata"].get("source")
                for r in retrieved_items
                if r.get("metadata")
            ]

            for k in k_values:
                top_k_sources = retrieved_sources[:k]

                hit = any(
                    src in relevant_sources
                    for src in top_k_sources
                )

                results[f"hit_rate@{k}"].append(1 if hit else 0)

        return {
            metric: float(np.mean(values))
            for metric, values in results.items()
        }

    def calculate_document_recall(
            self,
            test_data: List[Dict],
            k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Document Recall@k:
        доля релевантных документов, найденных в top-k
        """

        results = {f"doc_recall@{k}": [] for k in k_values}

        for item in test_data:
            query = item["query"]
            relevant_sources = set(item["relevant_sources"])

            if not relevant_sources:
                continue  # на всякий случай

            max_k = max(k_values)
            self.retriever.top_k = max_k

            retrieved_items = self.retriever.retrieve_top_context(query)

            # source документов, найденных ретривером
            retrieved_sources = [
                r["metadata"].get("source")
                for r in retrieved_items
                if r.get("metadata")
            ]

            for k in k_values:
                top_k_sources = set(retrieved_sources[:k])

                found = relevant_sources & top_k_sources
                recall = len(found) / len(relevant_sources)

                results[f"doc_recall@{k}"].append(recall)

        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in results.items()
        }
