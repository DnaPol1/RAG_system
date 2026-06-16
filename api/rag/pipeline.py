from core.llm.gigaChat import GigaChatClient
from core.retriever.SimpleVectorRetriever import SimpleVectorRetriever as Retriever


class RAGPipline:

  def __init__(self, vector_store, embedding_model):
    self.vector_store = vector_store
    self.embedding_model = embedding_model
    self.retriever = Retriever(vector_store=self.vector_store,
                               embedding_model=self.embedding_model,
                               top_k=10)

  def build_augmented_prompt(self, query, context):
    return f"""КОНТЕКСТ:
                   {context}
                   ВОПРОС:
                   {query}
                """

  def run(self, query: str):
    result = self.retriever.retrieve_top_chunks(query)
    context = "\n\n".join([res["text"] for res in result])
    prompt = self.build_augmented_prompt(query, context)
    with open(
      r"/llm/API.txt", "r",
      encoding="utf-8") as f:
      API_KEY = f.readline().strip()
    llm = GigaChatClient(credentials=API_KEY)
    answer = llm.generate(prompt=prompt, rag=True)
    return {
      "answer": answer
    }