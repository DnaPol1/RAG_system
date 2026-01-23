from chunking.recursive_chunker import RecursiveTextChunker
from chunking.semantic_chunker import SemanticChunker
from chunking.structural_chunker import StructuralChunker

def create_chunker(name: str, **kwargs):
    name = name.lower()

    if name == "recursive":
        return RecursiveTextChunker(**kwargs)

    if name == "semantic":
        return SemanticChunker(**kwargs)

    if name in ("structural","struct"):
        return StructuralChunker()

    raise ValueError(f"Unknown chunker: {name}")
