from __future__ import annotations

import unicodedata

TAMIL_RANGE = (0x0B80, 0x0BFF)
DEVANAGARI_RANGE = (0x0900, 0x097F)


def detect_language(text: str) -> str:
    for char in text:
        code = ord(char)
        if TAMIL_RANGE[0] <= code <= TAMIL_RANGE[1]:
            return "ta"
        if DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]:
            return "hi"
    return "en"


def normalize(text: str) -> str:
    lowered = text.lower()
    cleaned_chars: list[str] = []
    for char in lowered:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(char)
    collapsed = " ".join("".join(cleaned_chars).split())
    return collapsed.strip()
