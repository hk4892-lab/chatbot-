"""Dialogue policy for routing user requests."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .language import LanguageNormalizer
from .retrieval import SearchResult
from ..tools import tools as tool_module

DEFAULT_THRESHOLD = 0.18


tool_patterns = {
    "emi": re.compile(
        r"emi(?:\s+calculate|\s+calc|)?\s*(?:p\s*=\s*(?P<P>\d+(?:\.\d+)?))?"
        r"[\s,;]*r\s*=\s*(?P<R>\d+(?:\.\d+)?)"
        r"[\s,;]*n\s*=\s*(?P<N>\d+)",
        re.IGNORECASE,
    ),
    "emi_simple": re.compile(
        r"emi\s+(?P<P>\d+(?:\.\d+)?)(?:l|lac|lakh)?\s+(?P<R>\d+(?:\.\d+)?)%?\s+(?P<N>\d+)",
        re.IGNORECASE,
    ),
    "block_card": re.compile(r"block\s+(?:my\s+)?card", re.IGNORECASE),
    "fetch_rate": re.compile(r"(interest|rate)\s+for\s+(?P<product>[a-zA-Z ]+)", re.IGNORECASE),
}


@dataclass
class PolicyDecision:
    route: str
    answer: Optional[str]
    citations: List[Dict[str, str]]
    confidence: float
    tool_result: Optional[Dict[str, object]] = None
    clarifying_question: Optional[str] = None


class Policy:
    """Main entry point for dialogue policy."""

    def __init__(self, normalizer: LanguageNormalizer) -> None:
        self.normalizer = normalizer

    def decide(
        self,
        message: str,
        lang: str,
        results: List[SearchResult],
        threshold: float = DEFAULT_THRESHOLD,
        redaction_map: Optional[Dict[str, str]] = None,
    ) -> PolicyDecision:
        redaction_map = redaction_map or {}
        normalized_message = self.normalizer.normalize(message)

        tool_decision = self._maybe_tool_call(normalized_message, redaction_map)
        if tool_decision:
            return tool_decision

        if not results:
            return PolicyDecision(
                route="ask",
                answer="Could you provide more details about your request?",
                citations=[],
                confidence=0.0,
                clarifying_question="I couldn't find anything relevant. Maybe mention the product or service?",
            )

        top_result = results[0]
        confidence = top_result.score
        if confidence < threshold:
            return PolicyDecision(
                route="ask",
                answer="I might not have the right details yet. Could you clarify your question?",
                citations=[],
                confidence=confidence,
                clarifying_question="Try adding keywords like card type, channel, or timing.",
            )

        return PolicyDecision(
            route="answer",
            answer=top_result.document.content,
            citations=[
                {
                    "id": res.document.doc_id,
                    "title": res.document.title,
                }
                for res in results[:3]
            ],
            confidence=confidence,
        )

    def _maybe_tool_call(
        self,
        normalized_message: str,
        redaction_map: Dict[str, str],
    ) -> Optional[PolicyDecision]:
        emi_match = tool_patterns["emi"].search(normalized_message)
        if emi_match:
            params = emi_match.groupdict()
            result = tool_module.calculate_emi(
                P=float(params["P"]),
                annual_rate_percent=float(params["R"]),
                months=int(params["N"]),
            )
            return PolicyDecision(
                route="tool",
                answer="Calculated EMI displayed below.",
                citations=[],
                confidence=1.0,
                tool_result={"emi": result},
            )

        emi_simple_match = tool_patterns["emi_simple"].search(normalized_message)
        if emi_simple_match:
            params = emi_simple_match.groupdict()
            principal = float(params["P"]) * (100000 if "l" in params["P"].lower() else 1)
            result = tool_module.calculate_emi(
                P=principal,
                annual_rate_percent=float(params["R"]),
                months=int(params["N"]),
            )
            return PolicyDecision(
                route="tool",
                answer="Calculated EMI displayed below.",
                citations=[],
                confidence=1.0,
                tool_result={"emi": result},
            )

        if tool_patterns["block_card"].search(normalized_message):
            ticket = tool_module.block_card("<TOKENIZED_CARD>", "user_request")
            return PolicyDecision(
                route="tool",
                answer="Card block request submitted. Our team will follow up with the reference below.",
                citations=[],
                confidence=1.0,
                tool_result={"ticket_id": ticket},
            )

        rate_match = tool_patterns["fetch_rate"].search(normalized_message)
        if rate_match:
            product = rate_match.group("product").strip()
            result = tool_module.fetch_rate(product)
            return PolicyDecision(
                route="tool",
                answer="Here is the latest indicative rate information.",
                citations=[],
                confidence=1.0,
                tool_result=result,
            )

        return None


__all__ = ["Policy", "PolicyDecision", "DEFAULT_THRESHOLD"]
