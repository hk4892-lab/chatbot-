"""Deterministic PII redaction utilities."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

TOKEN_PREFIX = "PII"


@dataclass
class RedactionResult:
    text: str
    mapping: Dict[str, str]


class Redactor:
    """Redact PII using deterministic reversible tokens."""

    def __init__(self) -> None:
        self.patterns: List[Tuple[str, re.Pattern[str]]] = [
            ("PHONE", re.compile(r"(\+91[- ]?)?[6-9]\d{9}")),
            (
                "EMAIL",
                re.compile(
                    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
                ),
            ),
            ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
            (
                "CARD",
                re.compile(r"\b(?:\d[ -]?){12,19}\b"),
            ),
            ("ACCOUNT", re.compile(r"\b\d{9,18}\b")),
            (
                "AADHAAR",
                re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
            ),
        ]

    def redact(self, text: str) -> RedactionResult:
        mapping: Dict[str, str] = {}
        redacted = text
        for key, pattern in self.patterns:
            redacted, mapping = self._apply(redacted, pattern, key, mapping)
        return RedactionResult(text=redacted, mapping=mapping)

    def _apply(
        self,
        text: str,
        pattern: re.Pattern[str],
        token_type: str,
        mapping: Dict[str, str],
    ) -> Tuple[str, Dict[str, str]]:
        count = 0
        def replacement(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            token = self._build_token(token_type, count)
            mapping[token] = match.group(0)
            return token

        return pattern.sub(replacement, text), mapping

    @staticmethod
    def _build_token(token_type: str, count: int) -> str:
        return f"<{TOKEN_PREFIX}:{token_type}:{count}>"


def detokenize(token: str, mapping: Dict[str, str]) -> str:
    """Return the original value for a given token."""
    return mapping.get(token, token)


def detokenize_many(tokens: Iterable[str], mapping: Dict[str, str]) -> List[str]:
    return [detokenize(token, mapping) for token in tokens]


def scrub_log_line(line: Dict[str, object]) -> Dict[str, object]:
    """Ensure no raw PII is written to logs."""
    clean_line = dict(line)
    if "redaction_tokens" in clean_line:
        clean_line["redaction_tokens"] = list(clean_line["redaction_tokens"])
    return clean_line


def write_audit_log(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(scrub_log_line(record), ensure_ascii=False) + "\n")


__all__ = ["Redactor", "RedactionResult", "detokenize", "detokenize_many", "write_audit_log"]
