from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from . import config
from .embeddings import embed_queries
from .language import normalize
from .vector_store import VectorStore

LOGGER = logging.getLogger("bankbot.rag")

KB_ITEMS: List[Dict[str, Any]] = []
VECTOR_STORE = VectorStore()
INDEX_PATH = config.INDEX_DIR


def load_kb() -> List[Dict[str, Any]]:
    global KB_ITEMS
    if KB_ITEMS:
        return KB_ITEMS
    data_dir = config.DATA_DIR
    kb_files = [data_dir / "kb_en.json", data_dir / "kb_ta.json", data_dir / "kb_hi.json"]
    items: List[Dict[str, Any]] = []
    for kb_file in kb_files:
        if not kb_file.exists():
            LOGGER.warning("KB file missing: %s", kb_file)
            continue
        with kb_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
            items.extend(payload)
    KB_ITEMS = items
    return KB_ITEMS


def _prepare_passage(item: Dict[str, Any]) -> str:
    return f"Question: {item['q']} Answer: {item['a']}"


def build_or_load_index() -> VectorStore:
    if VECTOR_STORE.is_ready():
        return VECTOR_STORE
    items = load_kb()
    if INDEX_PATH.exists():
        if VECTOR_STORE.load(INDEX_PATH):
            LOGGER.info("Loaded vector index from disk")
            return VECTOR_STORE
    passages = [_prepare_passage(item) for item in items]
    metas = [
        {
            "intent": item.get("intent", ""),
            "lang": item.get("lang", ""),
            "q": item.get("q", ""),
            "a": item.get("a", ""),
        }
        for item in items
    ]
    VECTOR_STORE.build(passages, metas)
    VECTOR_STORE.save(INDEX_PATH)
    return VECTOR_STORE


def retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    store = build_or_load_index()
    normalized = normalize(query)
    if not normalized:
        return []
    query_vec = embed_queries([normalized])[0]
    results = store.search(query_vec, top_k)
    formatted: List[Dict[str, Any]] = []
    for result in results:
        meta = result["meta"]
        formatted.append(
            {
                "text": result["text"],
                "score": result["score"],
                "intent": meta.get("intent", ""),
                "lang": meta.get("lang", ""),
                "question": meta.get("q", ""),
                "answer": meta.get("a", ""),
            }
        )
    return formatted
