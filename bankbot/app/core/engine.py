"""Reusable chat engine for BankBot."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .language import LANG_EN, LANG_HI, LANG_TA, LanguageNormalizer
from .policy import DEFAULT_THRESHOLD, Policy, PolicyDecision
from .redaction import Redactor, RedactionResult, write_audit_log
from .retrieval import KnowledgeBase, Retriever, SearchResult


@dataclass
class ChatEngineResult:
    """Structured output from a chat turn."""

    lang: str
    decision: PolicyDecision
    results: List[SearchResult]
    redaction: RedactionResult


class ChatEngine:
    """Coordinate normalization, retrieval, and policy steps for a chat turn."""

    def __init__(
        self,
        data_path: Path,
        *,
        log_path: Optional[Path] = None,
    ) -> None:
        self.normalizer = LanguageNormalizer()
        self.policy = Policy(self.normalizer)
        self.redactor = Redactor()
        self.kb = KnowledgeBase(data_path)
        self.log_path = log_path
        self._sentence_transformers_available: Optional[bool] = None

    def run(
        self,
        message: str,
        *,
        threshold: Optional[float] = None,
        lang: Optional[str] = None,
        use_sentence_transformers: bool = False,
    ) -> ChatEngineResult:
        if not message:
            raise ValueError("message cannot be empty")

        detection = self.normalizer.detect_language(message)
        resolved_lang = lang if lang in {LANG_EN, LANG_HI, LANG_TA} else detection.primary

        redaction_result = self.redactor.redact(message)
        normalized = self.normalizer.normalize(redaction_result.text)

        retriever = Retriever(self.kb, use_sentence_transformers=use_sentence_transformers)
        results = retriever.search(normalized)

        threshold_value = threshold if threshold is not None else DEFAULT_THRESHOLD
        decision = self.policy.decide(
            message=redaction_result.text,
            lang=resolved_lang,
            results=results,
            threshold=threshold_value,
            redaction_map=redaction_result.mapping,
        )

        if self.log_path is not None:
            write_audit_log(
                self.log_path,
                {
                    "ts": time.time(),
                    "lang": resolved_lang,
                    "route": decision.route,
                    "topdoc_id": results[0].document.doc_id if results else None,
                    "top_score": results[0].score if results else 0.0,
                    "redaction_tokens": redaction_result.mapping.keys(),
                    "tools_called": list(decision.tool_result.keys()) if decision.tool_result else [],
                },
            )

        return ChatEngineResult(
            lang=resolved_lang,
            decision=decision,
            results=results,
            redaction=redaction_result,
        )

    @property
    def sentence_transformers_available(self) -> bool:
        if self._sentence_transformers_available is None:
            retriever = Retriever(self.kb, use_sentence_transformers=True)
            self._sentence_transformers_available = retriever.use_sentence_transformers
        return self._sentence_transformers_available


__all__ = ["ChatEngine", "ChatEngineResult"]
