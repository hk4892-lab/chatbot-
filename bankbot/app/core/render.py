"""Utilities for formatting responses."""
from __future__ import annotations

from typing import Dict, List


def format_answer(answer: str, citations: List[Dict[str, str]]) -> str:
    if not citations:
        return answer
    citation_text = ", ".join(f"{item['title']} ({item['id']})" for item in citations)
    return f"{answer}\n\nSources: {citation_text}"


__all__ = ["format_answer"]
