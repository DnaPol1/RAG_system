from typing import List, Dict, Any, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from core.chunking.base_chunker import BaseChunker
from configs.config import LOCAL_SBERT_PATH


class RecursiveLangChainChunker(BaseChunker):
    """
    Чанкование на основе RecursiveCharacterTextSplitter с интеграцией токенизатора Hugging Face
    для контроля длины чанков в токенах.

    Особенности:
        - Использует from_huggingface_tokenizer для точного соблюдения лимитов токенов
        - Подходит для подготовки текстов под LLM с фиксированным контекстным окном
        - Сохраняет совместимость с LangChain и Hugging Face экосистемой
    """

    def __init__(
            self,
            model_name: str = "ai-forever/sbert_large_nlu_ru",
            chunk_size: int = 768,
            overlap: int = 128,
            separators: Optional[List[str]] = None,
            add_start_index: bool = True
    ):
        """
        :param model_name: идентификатор модели в Hugging Face Hub для загрузки токенизатора
        :param chunk_size: максимальный размер чанка в ТОКЕНАХ
        :param overlap: перекрытие между чанками в ТОКЕНАХ
        :param separators: список разделителей в порядке приоритета (если None — используется
        стандартный)
        :param add_start_index: добавлять ли в метаданные индекс начала чанка в исходном тексте
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = overlap
        self.add_start_index = add_start_index
        #Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_SBERT_PATH)
        # Устанавливаем разделители (можно переопределить через параметр)
        if separators is None:
            self.separators = [
                "\n\n",  # абзацы (наивысший приоритет)
                "\n",  # строки
                ". ",  # предложения (с точкой и пробелом)
                " ",  # слова
                ""  # в крайнем случае — по символам
            ]
        else:
            self.separators = separators
        # Создаем сплиттер, который работает с токенами
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=self.separators,
            add_start_index=add_start_index
        )

    def split(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Разбивает текст на чанки с контролем длины в токенах.

        :param text: исходный текст для разбиения
        :param metadata: метаданные, общие для всех чанков (должны содержать document_id)
        :return: список словарей с ключами "text", "metadata", и возможно "start_index"
        """
        if not text or not text.strip():
            return []

        if metadata is None or "document_id" not in metadata:
            raise ValueError("metadata must contain 'document_id'")

        # Метод create_documents возвращает список Document-объектов LangChain
        # Каждый Document содержит page_content (текст) и metadata
        documents = self.splitter.create_documents([text], [metadata])

        results = []
        for i, doc in enumerate(documents):
            # Извлекаем метаданные, которые LangChain добавил автоматически
            doc_metadata = doc.metadata.copy() if doc.metadata else {}

            # Добавляем наш chunk_id
            doc_metadata["chunk_size_chars"] = len(doc.page_content)
            doc_metadata["chunk_size_tokens"] = len(self.tokenizer.encode(doc.page_content))

            # Если сплиттер вычислил start_index, он уже будет в doc_metadata
            # (при add_start_index=True)

            results.append({
                "text": doc.page_content,
                "metadata": {
                    **doc_metadata
                }
            })

        return results