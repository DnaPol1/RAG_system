import os
from typing import List
from PyPDF2 import PdfReader

from document import Document

class PDFLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load_one_pdf(self, file_path):
        if file_path.lower().endswith(".pdf"):
            text = self._extract_text_from_pdf(file_path)
            cleaned_text = self.remove_empty_lines(text)

            if cleaned_text.strip():
                document = Document.create(
                    text=cleaned_text,
                    source_name=os.path.basename(file_path),
                    source_path=file_path,
                    doc_type="pdf"
                )

            return document

    def load_pdfs(self) -> List[Document]:
        documents = []

        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.folder_path, filename)

                text = self._extract_text_from_pdf(file_path)
                cleaned_text = self.remove_empty_lines(text)

                if cleaned_text.strip():
                    documents.append(
                        Document.create(
                            text=cleaned_text,
                            source_name=os.path.basename(file_path),
                            source_path=file_path,
                            doc_type="pdf"
                        )
                    )

        return documents

    @staticmethod
    def _extract_text_from_pdf(file_path: str) -> str:
        """
        Извлекает текст из одного PDF-файла
        """
        reader = PdfReader(file_path)
        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

        #out = "Файл:".join(file_path)
        return "\n".join(pages_text)

    @staticmethod
    def remove_empty_lines(text: str) -> str:
        '''
        Удаляет пустые строки из текста
        '''
        lines = text.splitlines(0)

        cleaned_lines = [
            line.strip()
            for line in lines
            if line.strip()
        ]

        return "\n".join(cleaned_lines)