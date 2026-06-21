from core.llm.gigaChat import GigaChatClient
from core.retriever.SimpleVectorRetriever import SimpleVectorRetriever as Retriever
from core.memory import SummaryMemory, RedisClient

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
        self.redis = RedisClient()

    def build_augmented_prompt(self, query, context, history_text):
        return f"""
ИСТОРИЯ ДИАЛОГА:
{history_text}
ДАННЫЕ ИЗ БАЗЫ ЗНАНИЙ:
{context}
ЗАДАЧА:
Ответь на вопрос пользователя, используя только данные выше.
ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}
ОТВЕТ:
"""

    def run(self, query: str, session_id: str = "default"):
        memory = SummaryMemory(redis_client=self.redis, session_id=session_id)
        result = self.retriever.retrieve_top_chunks(query)
        chunks = []
        for i, res in enumerate(result):
            text = res["text"]
            chunks.append(f"[Фрагмент {i + 1}]\n{text}")
        context = "\n\n".join(chunks)
        history_text = memory.build_context()
        prompt = self.build_augmented_prompt(query, context, history_text)
        llm = GigaChatClient(credentials=API_KEY)
        answer = llm.generate(prompt=prompt, rag=True)
        memory.add("user", query)
        memory.add("assistant", answer)
        return {
            "answer": answer
        }