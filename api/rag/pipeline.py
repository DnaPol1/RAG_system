from core.llm.gigaChat import GigaChatClient
from core.retriever.SimpleVectorRetriever import SimpleVectorRetriever as Retriever

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GIGACHAT_API_KEY")

class RAGPipline:

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.retriever = Retriever(vector_store=self.vector_store,
                                   embedding_model=self.embedding_model,
                                   top_k=10)

    def build_augmented_prompt(self, query, context):
        return f"""
ДАННЫЕ ИЗ БАЗЫ ЗНАНИЙ:
{context}
ЗАДАЧА:
Ответь на вопрос пользователя, используя только данные выше.
ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}
ОТВЕТ:
"""

    def run(self, query: str):
        print("STEP 1: retrieve")
        result = self.retriever.retrieve_top_chunks(query)
        print("STEP 2: context build")
        context = "\n\n".join(
            [f"[Фрагмент {i + 1}]\n{res["text"]}" for i, res in enumerate(result)]
        )
        print("STEP 3: calling LLM")
        prompt = self.build_augmented_prompt(query, context)
        llm = GigaChatClient(credentials=API_KEY)
        answer = llm.generate(prompt=prompt, rag=True)
        print("STEP 4: got answer")
        return {
            "answer": answer
        }