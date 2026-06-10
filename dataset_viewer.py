import pandas as pd
from typing import List, Dict, Any


class DatasetViewer:
    """
    Просмотр чанков, эмбеддингов и результатов retriever в виде таблицы.
    """

    @staticmethod
    def from_chunks(chunks: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        chunks = [
            {
                "text": "...",
                "metadata": {...}
            }
        ]
        """

        rows = []
        for c in chunks:
            row = {
                "text": c["text"],
                **c.get("metadata", {})
            }
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def from_retrieval_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        results = [
            {
                "score": float,
                "text": str,
                "metadata": dict
            }
        ]
        """

        rows = []
        for r in results:
            rows.append({
                "score": round(r["score"], 4),
                "text_preview": r["text"][:200].replace("\n", " "),
                **r.get("metadata", {})
            })

        df = pd.DataFrame(rows)
        return df.sort_values(by="score", ascending=False)

    @staticmethod
    def pretty_print(df: pd.DataFrame, max_rows: int = 10):
        """
        Красивый вывод в консоль
        """
        from tabulate import tabulate

        print(
            tabulate(
                df.head(max_rows),
                headers="keys",
                tablefmt="fancy_grid",
                showindex=False
            )
        )
