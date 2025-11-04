from __future__ import annotations

import re

CARD_RE = re.compile(r"\b(?:\d[ -]?){16}\b")
ACCOUNT_RE = re.compile(r"\b\d{10,12}\b")
PHONE_RE = re.compile(r"\b(?:\+?91[ \-]?)?[6-9]\d{9}\b")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def _mask(pattern: re.Pattern[str], text: str, token: str) -> str:
    return pattern.sub(token, text)


def redact(text: str) -> str:
    sanitized = text
    sanitized = _mask(CARD_RE, sanitized, "[REDACTED:CARD]")
    sanitized = _mask(ACCOUNT_RE, sanitized, "[REDACTED:ACCOUNT]")
    sanitized = _mask(PHONE_RE, sanitized, "[REDACTED:PHONE]")
    sanitized = _mask(EMAIL_RE, sanitized, "[REDACTED:EMAIL]")
    return sanitized


def safe_for_log(text: str) -> str:
    return redact(text)
