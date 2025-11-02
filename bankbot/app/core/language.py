"""Utilities for language detection and normalization."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

DEVANAGARI_RANGE: Tuple[int, int] = (0x0900, 0x097F)
TAMIL_RANGE: Tuple[int, int] = (0x0B80, 0x0BFF)

LANG_AUTO = "AUTO"
LANG_EN = "EN"
LANG_HI = "HI"
LANG_TA = "TA"


@dataclass(frozen=True)
class LanguageDetection:
    """Result of language detection."""

    primary: str
    counts: Dict[str, int]


class LanguageNormalizer:
    """Handle language detection, normalization, and synonym expansion."""

    def __init__(self) -> None:
        self._synonyms: Dict[str, str] = {
            "limit kitna": "upi limit",
            "kitna limit": "upi limit",
            "upi limit kitna": "upi limit",
            "statement send pannunga": "statement request",
            "emi calc karna": "emi calculate",
            "emi calculator": "emi calculate",
            "emi enna": "emi calculate",
            "block card": "block card",
            "card block pannunga": "block card",
            "emi 5l": "emi calculate",
            "statement bhejo": "statement request",
        }

    @staticmethod
    def detect_language(text: str) -> LanguageDetection:
        counts = {LANG_EN: 0, LANG_HI: 0, LANG_TA: 0}
        for ch in text:
            code = ord(ch)
            if DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]:
                counts[LANG_HI] += 1
            elif TAMIL_RANGE[0] <= code <= TAMIL_RANGE[1]:
                counts[LANG_TA] += 1
            elif ch.isalpha():
                counts[LANG_EN] += 1
        primary = max(counts.items(), key=lambda item: item[1])[0]
        if all(value == 0 for value in counts.values()):
            primary = LANG_EN
        return LanguageDetection(primary=primary, counts=counts)

    def normalize(self, text: str) -> str:
        """Normalize user text for retrieval."""
        text = re.sub(r"\s+", " ", text).strip()
        tokens: Iterable[str] = re.split(r"(\s+)", text)
        normalized_parts = []
        for token in tokens:
            if token.isspace():
                normalized_parts.append(" ")
                continue
            if self._contains_indic(token):
                normalized_parts.append(token)
            else:
                normalized_parts.append(token.lower())
        normalized = "".join(normalized_parts)
        normalized = self._apply_synonyms(normalized)
        return normalized

    def _contains_indic(self, token: str) -> bool:
        return any(
            DEVANAGARI_RANGE[0] <= ord(ch) <= DEVANAGARI_RANGE[1]
            or TAMIL_RANGE[0] <= ord(ch) <= TAMIL_RANGE[1]
            for ch in token
        )

    def _apply_synonyms(self, text: str) -> str:
        for source, target in self._synonyms.items():
            text = re.sub(rf"\b{re.escape(source)}\b", target, text)
        return text


__all__ = [
    "LanguageDetection",
    "LanguageNormalizer",
    "LANG_AUTO",
    "LANG_EN",
    "LANG_HI",
    "LANG_TA",
]
