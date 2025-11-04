from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

ESCALATE_PATTERNS = [
    re.compile(r"\bpin\b", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"otp", re.IGNORECASE),
    re.compile(r"kyc", re.IGNORECASE),
    re.compile(r"upload", re.IGNORECASE),
]

TOOL_KEYWORDS = {
    "emi": [re.compile(r"\bemi\b", re.IGNORECASE), re.compile(r"loan", re.IGNORECASE)],
    "block_card": [
        re.compile(r"block", re.IGNORECASE),
        re.compile(r"lost card", re.IGNORECASE),
        re.compile(r"stolen", re.IGNORECASE),
    ],
    "interest_rates": [
        re.compile(r"interest", re.IGNORECASE),
        re.compile(r"rate", re.IGNORECASE),
        re.compile(r"savings", re.IGNORECASE),
    ],
}

ESCALATE_MESSAGE = (
    "I’m sorry, but I can’t assist with that request. For sensitive banking help, please "
    "contact customer support directly."
)


@dataclass
class PolicyDecision:
    action: str
    tool: Optional[str] = None
    message: Optional[str] = None


def _has_pattern(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def detect_tool(text: str, intents: Iterable[str]) -> Optional[str]:
    for tool, patterns in TOOL_KEYWORDS.items():
        if _has_pattern(text, patterns):
            return tool
    for intent in intents:
        if intent in {"emi_calculator", "loan_planning"}:
            return "emi"
        if intent in {"card_services", "block_card"}:
            return "block_card"
        if intent in {"interest_rates", "rate_info", "savings_rate"}:
            return "interest_rates"
    return None


def should_escalate(text: str) -> bool:
    return _has_pattern(text, ESCALATE_PATTERNS)


def decide_action(
    text: str,
    hits: list[dict],
    slm_enabled: bool,
    threshold: float,
) -> PolicyDecision:
    if should_escalate(text):
        return PolicyDecision(action="escalate", message=ESCALATE_MESSAGE)

    intents = [hit.get("intent", "") for hit in hits]
    tool = detect_tool(text, intents)
    if tool:
        return PolicyDecision(action="tool", tool=tool)

    best_score = hits[0]["score"] if hits else 0.0
    if hits and best_score >= threshold:
        return PolicyDecision(action="rag")

    if slm_enabled and hits:
        return PolicyDecision(action="slm_rag")

    return PolicyDecision(action="clarify")
