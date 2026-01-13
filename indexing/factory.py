# indexing/factory.py
from .vector_index import VectorIndex
from .tree_index import TreeIndex
from .linear_index import LinearIndex
from .keyword_index import KeywordIndex

def create_index(index_type: str):
    index_type = index_type.lower()

    if index_type == "vector":
        return VectorIndex()
    elif index_type == "tree":
        return TreeIndex()
    elif index_type == "linear":
        return LinearIndex()
    elif index_type == "keyword":
        return KeywordIndex()
    else:
        raise ValueError(f"Неизвестный тип индекса: {index_type}")
