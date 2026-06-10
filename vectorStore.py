import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any

from config import EMB_DIM


class VectorStore:
  """
  Векторное хранилище на FAISS для RAG с использованием IndexIDMap.
  - Каждый вектор получает стабильный уникальный ID.
  - хранит эмбеддинги, тексты и метаданные.
  - Поддержка поиска по cosine similarity.
  """

  def __init__(self, embedding_dim=None):
    self.embedding_dim = EMB_DIM if embedding_dim is None else embedding_dim
    # создаём HNSW индекс
    hnsw_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
    hnsw_index.hnsw.efConstruction = 100
    hnsw_index.hnsw.efSearch = 64
    # оборачиваем его в IDMap для стабильных индетификаторов
    self.index = faiss.IndexIDMap(hnsw_index)
    # словари для стабильной связи ID -> текст / метаданные
    self.id_to_text: Dict[int, str] = {}
    self.id_to_metadata: Dict[int, Dict[str, Any]] = {}
    self.id_to_vector: Dict[int, np.ndarray] = {}
    # счётчик для генерации ID
    self.next_id = 1

  def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
    """
    Добавление эмбеддингов + текстов + метаданных с генерацией уникальных ID
    """
    if not isinstance(embeddings, np.ndarray):
      raise TypeError("Embeddings must be numpy array")
    if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
      raise ValueError(
        f"Expected embeddings shape (N, {self.embedding_dim}),"
        f"got {embeddings.shape}"
      )
    if len(chunks) != embeddings.shape[0]:
      raise ValueError(
        f"Chunks count ({len(chunks)}) "
        f"must match embeddings count ({embeddings.shape[0]})"
      )
    # нормализация для cosine similarity
    faiss.normalize_L2(embeddings)
    # генерация уникальных int64 iD
    new_ids = np.array([self.next_id + i for i in range(len(chunks))],
                       dtype=np.int64)
    self.next_id += len(chunks)
    # добавление в FAISS
    self.index.add_with_ids(embeddings, new_ids)
    # сохранение текста и метаданных по ID
    for i, chunk in enumerate(chunks):
      self.id_to_text[new_ids[i]] = chunk["text"]
      self.id_to_metadata[new_ids[i]] = chunk["metadata"]
      self.id_to_vector[new_ids[i]] = embeddings[i]

  def search(self, query_embedding: np.ndarray, top_k: int = 10):
    """
    Поиск ближайших документов
    Возвращает список словарей: {score, text, metadata}
    """

    if self.index.ntotal == 0:
      return []
    if query_embedding.ndim == 1:
      query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    scores, indices = self.index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
      if idx == -1:
        continue
      results.append({
        "vector_id": int(idx),
        "score": float(score),
        "text": self.id_to_text[idx],
        "metadata": self.id_to_metadata[idx]
      })
    return results

  def save(self, path: str):
    os.makedirs(path, exist_ok=True)
    faiss.write_index(self.index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "id_to_text.pkl"), "wb") as f:
      pickle.dump(self.id_to_text, f)
    with open(os.path.join(path, "id_to_metadata.pkl"), "wb") as f:
      pickle.dump(self.id_to_metadata, f)
    with open(os.path.join(path, "config.pkl"), "wb") as f:
      pickle.dump(
        {
          "embedding_dim": self.embedding_dim,
          "index_type": "HNSW+IDMap",
        },
        f
      )
    with open(os.path.join(path, "id_to_vector.pkl"), "wb") as f:
      pickle.dump(self.id_to_vector, f)

  @classmethod
  def load(cls, path: str):
    index_path = os.path.join(path, "index.faiss")
    if not os.path.exists(index_path):
      raise FileNotFoundError(
        f"FAISS index not found: {index_path}\n"
        f"Files in directory: {os.listdir(path) if os.path.exists(path) else "DIR NOT FOUND"}"
      )
    index = faiss.read_index(index_path)
    store = cls()
    store.index = index
    with open(os.path.join(path, "id_to_text.pkl"), "rb") as f:
      store.id_to_text = pickle.load(f)
    with open(os.path.join(path, "id_to_metadata.pkl"), "rb") as f:
      store.id_to_metadata = pickle.load(f)
    with open(os.path.join(path, "config.pkl"), "rb") as f:
      config = pickle.load(f)
      store.embedding_dim = config["embedding_dim"]
    with open(os.path.join(path, "id_to_vector.pkl"), "rb") as f:
      store.id_to_vector = pickle.load(f)
    # устанавливаем next_id на максимум уже существующих ID + 1
    store.next_id = max(store.id_to_text.keys(), default=0) + 1

    return store

  def summary(self, show_example: bool = True):
    print("\n=== VECTOR STORE SUMMARY ===")
    print(f"Embedding dim: {self.embedding_dim}")
    print(f"Total vectors: {self.index.ntotal}")
    print(f"Index type: {type(self.index).__name__}")
    try:
      inner_index = faiss.downcast_index(self.index.index)
      hnsw = inner_index.hnsw
      print(f"HNSW parameters:")
      print(f"  - efSearch (current): {hnsw.efSearch}")
      print(f"  - efConstruction: {hnsw.efConstruction}")
      print(f"  - Max levels: {hnsw.max_level}")
    except AttributeError:
      print("HNSW parameters unavailable")
    print(
      f"Stored texts: {len(self.id_to_text) if self.id_to_text else 'NONE'}")
    print(
      f"Stored metadatas: {len(self.id_to_metadata) if self.id_to_metadata else 'NONE'}")
    if show_example and self.id_to_text:
      first_id = next(iter(self.id_to_text))
      print("\n--- Example document ---")
      preview = self.id_to_text[first_id][:500]
      print(preview + ("..." if len(self.id_to_text[first_id]) > 500 else ""))
      print("Metadata:", self.id_to_metadata[first_id])
      print("------------------------")
    print("========================\n")

  def brute_force_search(
    self,
    query_embedding: np.ndarray,
    top_k: int = 10
  ):
    """
    Полный перебор всех векторов.
    Используется для дебага качества retrieval.
    """

    if len(self.id_to_vector) == 0:
      return []

    if query_embedding.ndim == 1:
      query_embedding = query_embedding.reshape(1, -1)

    faiss.normalize_L2(query_embedding)

    query_vec = query_embedding[0]

    results = []

    for vector_id, vector in self.id_to_vector.items():
      # cosine similarity
      score = np.dot(query_vec, vector)

      results.append({
        "vector_id": int(vector_id),
        "score": float(score),
        "text": self.id_to_text[vector_id],
        "metadata": self.id_to_metadata[vector_id]
      })

    results.sort(
      key=lambda x: x["score"],
      reverse=True
    )

    return results[:top_k]
