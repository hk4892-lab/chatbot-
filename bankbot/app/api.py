"""FastAPI application exposing chatbot endpoints."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .core.engine import ChatEngine
from .core.language import LANG_AUTO, LANG_EN, LANG_HI, LANG_TA
from .core.policy import PolicyDecision, DEFAULT_THRESHOLD
from .core.redaction import write_audit_log
from .core.render import format_answer


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

session_manager = SessionManager()
engine = ChatEngine(data_path=DATA_PATH)

app = FastAPI(title="BankBot", version="1.0.0")


@app.post("/chat/turn", response_model=ChatResponse)
async def chat_turn(request: ChatRequest) -> ChatResponse:
    if not request.message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    engine_result = engine.run(
        request.message,
        threshold=request.threshold,
        lang=request.lang,
        use_sentence_transformers=bool(request.use_sentence_transformers),
    )

    decision = engine_result.decision
    session_manager.update(request.message, decision, engine_result.redaction.mapping)

    answer = decision.answer
    if answer:
        answer = format_answer(answer, decision.citations)

    write_audit_log(
        LOG_PATH,
        {
            "ts": time.time(),
            "lang": engine_result.lang,
            "route": decision.route,
            "topdoc_id": engine_result.results[0].document.doc_id if engine_result.results else None,
            "top_score": engine_result.results[0].score if engine_result.results else 0.0,
            "redaction_tokens": engine_result.redaction.mapping.keys(),
            "tools_called": list(decision.tool_result.keys()) if decision.tool_result else [],
        },
    )

    return ChatResponse(
        lang=engine_result.lang,
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
        "sentence_transformers_available": engine.sentence_transformers_available,
        "history_length": len(session_manager.state.history),
    }


__all__ = ["app"]
