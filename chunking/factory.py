from .recursive_chunker import RecursiveTextChunker
from .semantic_chunker import SemanticChunker

def create_chunker(name: str, **kwargs):
    if name == "recursive":
        return RecursiveTextChunker(**kwargs)

    if name == "semantic":
        return SemanticChunker(**kwargs)

    raise ValueError(f"Unknown chunker: {name}")
