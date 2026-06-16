from chunking.recursive_lang_chain import RecursiveLangChainChunker

def create_chunker(name: str, **kwargs):
    name = name.lower()

    if name == "langchain":
        return RecursiveLangChainChunker(**kwargs)

    raise ValueError(f"Unknown chunker: {name}")