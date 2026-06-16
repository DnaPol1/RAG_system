import json
from typing import List, Dict

import numpy as np

from core.retriever.SimpleVectorRetriever import SimpleVectorRetriever


class RetrievalEvaluation:
    def __init__(self, k_values, json_path: str, retriever: SimpleVectorRetriever = None):
        self.retriever = retriever
        self.k_values = k_values
        self.test_data = self.load_data_from_json(json_path)

    def load_data_from_json(self, json_path: str) -> List[Dict]:
        """
        Загружает тестовые данные из JSON файла.

        Ожидаемый формат:
        [
            {
                "question": str,
                "answer": str,
                "pos_vec_ids": [int, int, ...]
            }
        ]
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_data = []
        for item in data:
            test_data.append({
                "question": item["question"],
                "pos_vec_ids": set(item["pos_vec_ids"])
            })

        return test_data

    def calculate_hit_rate(self):
        """
        test_data: [
            {"question": str, "pos_vec_ids": set(int, ...)}
        ]
        retriever: объект, у которого есть метод retrieve_top_chunks(query)
        """

        results = {f"hit_rate@{k}": [] for k in self.k_values}

        for item in self.test_data:
            query = item["question"]
            relevant_ids = set(item["pos_vec_ids"])

            retrieved_items = self.retriever.retrieve_top_chunks(query)
            retrieved_ids = [r["vector_id"] for r in retrieved_items if r["vector_id"] is not None]

            for k in self.k_values:
                top_k_ids = retrieved_ids[:k]
                hit = any(_id in relevant_ids for _id in top_k_ids)
                results[f"hit_rate@{k}"].append(1 if hit else 0)

        # средние значения
        hit_rate_scores = {metric: float(np.mean(vals)) for metric, vals in results.items()}
        return hit_rate_scores