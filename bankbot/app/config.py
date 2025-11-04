from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
INDEX_DIR = DATA_DIR / "vector_index"
LOG_PATH = Path(os.getenv("LOG_PATH", APP_DIR.parent / "logs" / "audit.log"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

USE_SLM = bool(int(os.getenv("USE_SLM", "0")))
MODEL_ID = os.getenv("MODEL_ID", "microsoft/phi-3-mini-4k-instruct")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
TOP_K = int(os.getenv("TOP_K", "4"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.32"))

DEFAULT_SYSTEM_PROMPT = (
    "You are a multilingual banking assistant. Use only the provided context to answer. "
    "If unsure, ask briefly for clarification. Never fabricate account details. Respond "
    "concisely (2-5 sentences) and match the user's language."
)


def to_dict() -> dict[str, Any]:
    return {
        "use_slm": USE_SLM,
        "model_id": MODEL_ID,
        "embed_model": EMBED_MODEL,
        "host": HOST,
        "port": PORT,
        "top_k": TOP_K,
        "rag_score_threshold": RAG_SCORE_THRESHOLD,
        "log_path": str(LOG_PATH),
    }
