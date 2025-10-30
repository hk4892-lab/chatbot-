"""FastAPI application exposing chatbot endpoints."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .core.language import (
    LANG_AUTO,
    LANG_EN,
    LANG_HI,
    LANG_TA,
    LanguageNormalizer,
)
from .core.policy import Policy, PolicyDecision, DEFAULT_THRESHOLD
from .core.redaction import Redactor, write_audit_log
from .core.render import format_answer
from .core.retrieval import KnowledgeBase, Retriever


class ChatRequest(BaseModel):
    message: str
    threshold: Optional[float] = None
    use_sentence_transformers: Optional[bool] = False
    lang: Optional[str] = None


class ChatResponse(BaseModel):
    lang: str
    route: str
    answer: Optional[str]
    citations: List[Dict[str, str]]
    tool_result: Optional[Dict[str, object]]
    confidence: float
    clarifying_question: Optional[str] = None


@dataclass
class SessionState:
    history: List[Dict[str, object]] = field(default_factory=list)
    redaction_map: Dict[str, str] = field(default_factory=dict)


class SessionManager:
    def __init__(self) -> None:
        self.state = SessionState()

    def update(self, user_message: str, decision: PolicyDecision, mapping: Dict[str, str]) -> None:
        self.state.history.append(
            {
                "user": user_message,
                "decision": decision.route,
            }
        )
        self.state.redaction_map.update(mapping)


DATA_PATH = Path(__file__).resolve().parent / "data"
LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit.log"

normalizer = LanguageNormalizer()
policy = Policy(normalizer)
redactor = Redactor()
kb = KnowledgeBase(DATA_PATH)
session_manager = SessionManager()

app = FastAPI(title="BankBot", version="1.0.0")


def _build_retriever(use_sentence_transformers: bool = False) -> Retriever:
    return Retriever(kb=kb, use_sentence_transformers=use_sentence_transformers)


@app.post("/chat/turn", response_model=ChatResponse)
async def chat_turn(request: ChatRequest) -> ChatResponse:
    if not request.message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    detection = normalizer.detect_language(request.message)
    lang = request.lang or detection.primary
    if lang not in {LANG_EN, LANG_HI, LANG_TA}:
        lang = detection.primary

    redaction_result = redactor.redact(request.message)
    normalized = normalizer.normalize(redaction_result.text)

    retriever = _build_retriever(use_sentence_transformers=bool(request.use_sentence_transformers))
    results = retriever.search(normalized)

    threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD
    decision = policy.decide(
        message=redaction_result.text,
        lang=lang,
        results=results,
        threshold=threshold,
        redaction_map=redaction_result.mapping,
    )

    session_manager.update(request.message, decision, redaction_result.mapping)

    answer = decision.answer
    if answer:
        answer = format_answer(answer, decision.citations)

    write_audit_log(
        LOG_PATH,
        {
            "ts": time.time(),
            "lang": lang,
            "route": decision.route,
            "topdoc_id": results[0].document.doc_id if results else None,
            "top_score": results[0].score if results else 0.0,
            "redaction_tokens": redaction_result.mapping.keys(),
            "tools_called": list(decision.tool_result.keys()) if decision.tool_result else [],
        },
    )

    return ChatResponse(
        lang=lang,
        route=decision.route,
        answer=answer,
        citations=decision.citations,
        tool_result=decision.tool_result,
        confidence=decision.confidence,
        clarifying_question=decision.clarifying_question,
    )


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
async def config() -> Dict[str, object]:
    return {
        "languages": [LANG_AUTO, LANG_EN, LANG_HI, LANG_TA],
        "threshold_default": DEFAULT_THRESHOLD,
        "sentence_transformers_available": _build_retriever(False).use_sentence_transformers,
        "history_length": len(session_manager.state.history),
    }


__all__ = ["app"]
