from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config, rag, slm
from .language import detect_language
from .policy import PolicyDecision, decide_action
from .redaction import safe_for_log
from .tools import block_card_ticket, emi_calculator, interest_rates

LOGGER = logging.getLogger("bankbot")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)

app = FastAPI(title="BankBot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: Optional[int] = None
    max_new_tokens: int = 256
    temperature: float = 0.2


class ChatResponse(BaseModel):
    reply: str
    lang: str
    source: str


@app.on_event("startup")
async def startup_event() -> None:
    rag.build_or_load_index()


@app.get("/")
def root() -> dict[str, object]:
    return {"ok": True, "msg": "BankBot API up. See /docs"}


@app.get("/healthz")
def healthz() -> dict[str, object]:
    store = rag.build_or_load_index()
    rag_ready = store.is_ready()
    slm_ready = bool(slm.get_pipe()) if config.USE_SLM else False
    return {
        "ok": True,
        "rag_index": rag_ready,
        "slm": slm_ready,
        "embed_model": config.EMBED_MODEL,
    }


def _latest_user_message(messages: List[ChatMessage]) -> ChatMessage:
    for message in reversed(messages):
        if message.role == "user":
            return message
    raise HTTPException(status_code=400, detail="A user message is required")


def _run_tool(decision: PolicyDecision, lang: str, user_text: str) -> ChatResponse:
    if decision.tool == "emi":
        result = emi_calculator(500000.0, 10.5, 60)
        reply = (
            "Estimated EMI is ₹{emi} with total payment ₹{total}."
            " Adjust the loan amount or tenure for other scenarios."
        ).format(emi=result["emi"], total=result["total_payment"])
        return ChatResponse(reply=reply, lang=lang, source="tool")
    if decision.tool == "block_card":
        ticket = block_card_ticket("Customer", "1234", "lost card")
        reply = (
            f"Your card block request is logged with ticket {ticket['ticket_id']} and status"
            " {ticket['status']}. Our team will reach out shortly."
        )
        return ChatResponse(reply=reply, lang=lang, source="tool")
    if decision.tool == "interest_rates":
        product = "savings"
        if "loan" in user_text.lower():
            product = "home_loan"
        rate_info = interest_rates(product)
        reply = (
            f"Current {rate_info['product']} interest rate is {rate_info['rate_percent']}%."
            " Contact support for personalised offers."
        )
        return ChatResponse(reply=reply, lang=lang, source="tool")
    return ChatResponse(
        reply="Could you clarify how I can assist with that?",
        lang=lang,
        source="clarify",
    )


def _compose_rag_reply(hits: list[dict]) -> str:
    if not hits:
        return ""
    top = hits[0]
    reply = top.get("answer") or top.get("text")
    intent = top.get("intent")
    if intent:
        reply = f"{reply} (Intent: {intent})"
    return reply


def _slm_context(hits: list[dict], top_k: int) -> str:
    snippets = []
    for hit in hits[:top_k]:
        intent = hit.get("intent", "")
        answer = hit.get("answer", "")
        snippets.append(f"- ({intent}) {answer}")
    return "Context:\n" + "\n".join(snippets)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages are required")
    user_message = _latest_user_message(request.messages)
    lang = detect_language(user_message.content)
    redacted = safe_for_log(user_message.content)
    LOGGER.info("chat_request lang=%s text=%s", lang, redacted)

    top_k = request.top_k or config.TOP_K
    hits = rag.retrieve(user_message.content, top_k)
    decision = decide_action(user_message.content, hits, config.USE_SLM, config.RAG_SCORE_THRESHOLD)

    if decision.action == "escalate":
        return ChatResponse(reply=decision.message or "Please contact support.", lang=lang, source="escalate")

    if decision.action == "tool" and decision.tool:
        return _run_tool(decision, lang, user_message.content)

    if decision.action == "rag":
        reply = _compose_rag_reply(hits)
        if not reply:
            return ChatResponse(reply="Could you provide more details?", lang=lang, source="clarify")
        return ChatResponse(reply=reply, lang=lang, source="rag")

    if decision.action == "slm_rag" and config.USE_SLM:
        context = _slm_context(hits, top_k)
        system_prompt = f"{config.DEFAULT_SYSTEM_PROMPT}\n{context}"
        generated = slm.generate(
            system_prompt,
            [message.model_dump() for message in request.messages],
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        if generated:
            return ChatResponse(reply=generated, lang=lang, source="slm_rag")

    return ChatResponse(
        reply="Could you share a bit more detail so I can help effectively?",
        lang=lang,
        source="clarify",
    )
