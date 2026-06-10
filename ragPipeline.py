class RAGPipeline:
    def __init__(self, retrieval_component, gigachat_client):
        self.retrieval = retrieval_component
        self.gigachat = gigachat_client

    def build_augmented_prompt(self, query, context):
        return f"""
    КОНТЕКСТ:
    {context}

    ВОПРОС:
    {query}

    ИНСТРУКЦИЯ:
    Ответь на вопрос, используя приведённый контекст.
    Если точного ответа нет, попробуй дать максимально релевантный ответ на основе доступной информации.
    Не пиши, что информации нет, если можно сделать вывод.
    """

    def answer_with_rag(self, query):
        results = self.retrieval.retrieve_top_context(query)

        if not results:
            return {
                "answer": "Релевантные документы не найдены",
                "similarity": 0.0
            }

        top_k = 3
        selected_docs = results[:top_k]

        context = "\n\n".join([doc["text"] for doc in selected_docs])

        metadata = [doc["metadata"] for doc in selected_docs]
        score = selected_docs[0]["score"]

        prompt = self.build_augmented_prompt(query, context)

        response = self.gigachat.generate(prompt, rag=True)

        return {
            "answer": response,
            "similarity": score,
            "source": metadata
        }

    def answer_without_rag(self, query):
        response = self.gigachat.generate(query, rag=False)
        return response
