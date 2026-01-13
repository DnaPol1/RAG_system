import re

# Разделители для юридических текстов
LEGAL_SPLIT_PATTERNS = [
    r"\n\s*\d+\.\s+",          # 1. 2. 3.
    r"\n\s*\d+\.\d+\s+",       # 1.1 1.2
    r"\n\s*[А-ЯЁ][^.]{3,}:\n", # ЗАГОЛОВКИ:
    r"\n\s*[-–•]\s+",          # маркированные списки
]

def split_legal_text(text: str):
    """
    Делит юридический текст по смысловым блокам
    """
    pattern = "(" + "|".join(LEGAL_SPLIT_PATTERNS) + ")"
    parts = re.split(pattern, text)

    chunks = []
    buffer = ""

    for part in parts:
        if re.match(pattern, part):
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = part
        else:
            buffer += part

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks
