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

    ДАЙ ЧЁТКИЙ И КРАТКИЙ ОТВЕТ, ИСПОЛЬЗУЯ ТОЛЬКО КОНТЕКСТ.
    """

    def answer_with_rag(self, query):
        context, metadata, score = self.retrieval.retrieve_top_context(query)

        print("CONTEXT:\n", context)

        if context is None:
            return {
                "answer": "Релевантные документы не найдены.",
                "similarity": 0.0
            }

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
