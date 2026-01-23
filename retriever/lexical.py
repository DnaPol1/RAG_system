import re
from typing import Set


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[а-яА-Яa-zA-Z]{2,}", text.lower()))


def lexical_score(query: str, text: str) -> float:
    """
    Простейший overlap score:
    |Q ∩ D|
    """
    if not query or not text:
        return 0.0

    q_tokens = tokenize(query)
    d_tokens = tokenize(text)

    return float(len(q_tokens & d_tokens))