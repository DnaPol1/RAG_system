import numpy as np
import json
from typing import List, Dict, Any

from RetrievalComponentForRAG import RetrievalComponent

class RetrievalEvaluation:
    def __init__(self, retriever : RetrievalComponent):
        self.retriever = retriever

    def load_data_from_json(self, json_path: str) -> List[Dict]:
        """
                Загружает тестовые данные из JSON файла

                Args:
                    json_path: путь к JSON файлу

                Returns:
                    Список словарей в формате, подходящем для calculate_hit_rate_simple
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Преобразуем в нужный формат
        test_data = []
        for item in data:
            test_data.append({
                'query': item['query'],
                "relevant_sources": item['relevant_sources']
            })

        return test_data

    def calculate_hit_rate_simple(self,
                                  test_data: List[Dict],
                                  k_values: List[int] = [1, 3, 5, 10],
                                  use_semantic_match : bool = False
                                  ) -> Dict[str, float]:
        """
        Вычисляет Hit Rate@k для разных k

        Args:
            retriever: объект ритривера с методом retrieve()
            test_data: список словарей с query и relevant_doc
            k_values: какие значения k оценивать
            use_semantic_match: если True, сравниваем текст документов,
                              если False, сравниваем ID документов
        """
        results = {f'hit_rate@{k}' : [] for k in k_values}
        for item in test_data:
            query = item['query']
            relevant_doc = item['relevant_sources']

            max_k = max(k_values)
            self.retriever.top_k = max_k
            retrieved_items = self.retriever.retrieve_top_context(query)

            for k in k_values:
                # Берем только топ-k результатов
                top_k_results = retrieved_items[:k]

                # Проверяем, есть ли релевантный документ в топ-k
                found = False
                for retrieved in top_k_results:
                    if use_semantic_match:
                        # Семантическое сравнение текстов
                        # Нужно реализовать отдельно
                        pass
                    else:
                        if retrieved['metadata'].get('source') == relevant_doc:
                            found = True
                            break

                results[f'hit_rate@{k}'].append(1 if found else 0)

            # Усредняем по всем запросам
        return {k: np.mean(v) for k, v in results.items()}