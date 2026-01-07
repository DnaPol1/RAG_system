import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any

class VectorStore:
    """
    Векторное хранилище на FAISS для RAG:
    - хранит эмбеддинги
    - хранит тексты
    - хранит метаданные (source и др.)
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

        # Cosine similarity = Inner Product + нормализация
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Хранение данных параллельно индексам FAISS
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.documents = []

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]], documents):
        """
        Добавление эмбеддингов + текстов + метаданных
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be numpy array")

        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise  ValueError(
                f"Expected embeddings shape (N, {self.embedding_dim}),"
                f"got {embeddings.shape}"
            )

        if len(texts) != embeddings.shape[0]:
            raise ValueError("texts count must match embeddings count")

        if len(metadatas) != embeddings.shape[0]:
            raise ValueError("metadatas count musy match embeddings count")

        #нормализация для cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 1):
        """
        Поиск ближайших документов
        Возвращает список словарей:
        {score, text, metadata}
        """

        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[00]):
            if idx == -1:
                continue

            results.append({
                "score": float(score),
                "text": self.texts[idx],
                "metadata": self.metadatas[idx]
            })

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "texts.pkl"), "wb") as f:
            pickle.dump(self.texts, f)

        with open(os.path.join(path, "metadatas.pkl"), "wb") as f:
            pickle.dump(self.metadatas, f)

        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(
                {"embedding_dim": self.embedding_dim},
                f
            )

    @classmethod
    def load(cls, path: str):
        index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "texts.pkl"), "rb") as f:
            texts = pickle.load(f)

        with open(os.path.join(path, "metadatas.pkl"), "rb") as f:
            metadatas = pickle.load(f)

        with open(os.path.join(path, "config.pkl"), "rb") as f:
            config = pickle.load(f)

        store = cls(config["embedding_dim"])
        store.index = index
        store.texts = texts
        store.metadatas = metadatas

        return store

    def summary(self, show_example: bool = True):
        print("\n=== VECTOR STORE SUMMARY ===")
        print(f"Embedding dim: {self.embedding_dim}")
        print(f"Total vectors: {self.index.ntotal}")
        print(f"Index type: {type(self.index).__name__}")

        print(f"Stored texts: {len(self.texts) if self.texts else 'NONE'}")
        print(f"Stored metadatas: {len(self.metadatas) if self.metadatas else 'NONE'}")

        if show_example and self.texts:
            print("\n--- Example document ---")
            preview = self.texts[0][:500]
            print(preview + ("..." if len(self.texts[0]) > 500 else ""))
            print("Metadata:", self.metadatas[0])
            print("------------------------")

        print("========================\n")