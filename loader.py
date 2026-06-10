import os
from typing import List
import re
import pdfplumber
from collections import Counter

from document import Document


class PDFLoader:
  def __init__(self, folder_path: str = None):
    self.folder_path = folder_path

  def _process_pdf_file(self, file_path: str):
    text = self._extract_text_from_pdf(file_path)
    cleaned_text = self.clean_legal_text(text)

    if cleaned_text.strip():
      return Document.create(
        text=cleaned_text,
        source_name=os.path.basename(file_path),
        source_path=file_path,
        doc_type="pdf"
      )
    return None

  def load_one_pdf(self, file_path):
    if file_path.lower().endswith(".pdf"):
      return self._process_pdf_file(file_path)

  def load_pdfs(self) -> List[Document]:
    documents = []

    for filename in os.listdir(self.folder_path):
      if filename.lower().endswith(".pdf"):
        file_path = os.path.join(self.folder_path, filename)
        doc = self._process_pdf_file(file_path)
        if doc:
          documents.append(doc)
    return documents

  @staticmethod
  def _extract_text_from_pdf(file_path: str) -> str:
    """
    Извлекает текст из одного PDF-файла
    """
    with pdfplumber.open(file_path) as pdf:
      full_text = []
      for page in pdf.pages:
        text = page.extract_text()
        if text:
          full_text.append(text)
    return "\n\n".join(full_text)

  @staticmethod
  def remove_empty_lines(text: str) -> str:
    """
    Удаляет пустые строки из текста
    """
    lines = text.splitlines(0)

    cleaned_lines = [
      line.strip()
      for line in lines
      if line.strip()
    ]

    return "\n".join(cleaned_lines)

  @staticmethod
  def remove_extra_whitespace(text: str) -> str:
    """Удаляет лишние пробелы: множественные пробелы -> один пробел"""
    # Замена множественных пробелов на один
    text = re.sub(r' +', ' ', text)
    # Удаление пробелов перед знаками препинания
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()

  @staticmethod
  def remove_headers_footers(text: str) -> str:
    """Удаляет типичные колонтитулы юридических документов"""
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
      line = line.strip()
      # Пропускаем строки, которые выглядят как номера страниц
      if re.match(r'^\s*\d+\s*$', line):
        continue
      # Пропускаем строки с разделителями страниц
      if re.match(r'^[-=_]{3,}$', line):
        continue
      # Пропускаем строки с датами в формате "Страница X из Y"
      if re.match(r'^страница\s*\d+\s*из\s*\d+\s*$', line.lower()):
        continue
      cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

  @staticmethod
  def normalize_unicode_and_symbols(text: str) -> str:
    """Нормализует Unicode символы и заменяет спецсимволы"""
    # Нормализация Unicode (например, ё -> е, если нужно)
    import unicodedata
    text = unicodedata.normalize('NFC', text)

    # Замена разных тире на обычное
    text = text.replace('—', '-').replace('–', '-')

    # Замена разных кавычек на стандартные
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")

    # Удаление непечатных символов (кроме основных)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text

  @staticmethod
  def fix_pdf_line_breaks(text: str) -> str:
    """
    Склеивает строки, разорванные PDF-версткой
    :param text:
    :return:
    """
    # убираем переносы строк внутри предложений
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

  @staticmethod
  def remove_repeated_lines(text: str, threshold: float = 0.9) -> str:
    """
    Удаляет строки, которые повторяются слишком часто (headers/footers)
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    freq = Counter(lines)
    total = len(lines)

    cleaned = [
      line for line in lines
      if freq[line] / total < threshold
    ]

    return "\n".join(cleaned)

  @staticmethod
  def remove_noise_lines(text: str) -> str:
    """
    Убирает строки без смысловой нагрузки
    """
    cleaned = []

    for line in text.split("\n"):
      line = line.strip()

      if not line:
        continue

      # слишком короткие строки
      if len(line) < 3:
        continue

      # только цифры (страницы)
      if re.fullmatch(r"\d+", line):
        continue

      # слишком много символов пунктуации
      if len(re.findall(r"[^\w\s]", line)) > len(line) * 0.5:
        continue

      cleaned.append(line)

    return "\n".join(cleaned)

  @staticmethod
  def clean_legal_text(text: str) -> str:
    # 1. нормализация
    text = PDFLoader.normalize_unicode_and_symbols(text)

    # 2. фиксим PDF-разрывы
    text = PDFLoader.fix_pdf_line_breaks(text)

    # 3. убираем колонтитулы
    text = PDFLoader.remove_headers_footers(text)

    # 4. убираем повторяющиеся строки
    # text = PDFLoader.remove_repeated_lines(text) - убирает все строки

    # 5. убираем шум
    text = PDFLoader.remove_noise_lines(text)

    # 6. whitespace cleanup
    text = PDFLoader.remove_extra_whitespace(text)

    # 7. пустые строки
    text = PDFLoader.remove_empty_lines(text)

    return text
