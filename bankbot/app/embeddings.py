from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

import numpy as np

from . import config

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback when library missing
    SentenceTransformer = None  # type: ignore[assignment]

LOGGER = logging.getLogger("bankbot.embeddings")


class _FallbackEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, sentences: Iterable[str], **_: object) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for sentence in sentences:
            rng = np.random.default_rng(abs(hash(sentence)) % (2**32))
            vec = rng.standard_normal(self.dim)
            vectors.append(vec)
        return np.vstack(vectors)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@lru_cache(maxsize=1)
def get_embedder() -> object:
    if SentenceTransformer is None:
        LOGGER.warning("sentence-transformers unavailable; using fallback embeddings")
        return _FallbackEmbedder()

    try:
        model = SentenceTransformer(config.EMBED_MODEL)
        return model
    except Exception as exc:  # pragma: no cover - network/device errors
        LOGGER.warning("Failed to load embedding model %s (%s); using fallback.", config.EMBED_MODEL, exc)
        return _FallbackEmbedder()


def _prepare_inputs(texts: Iterable[str], is_query: bool) -> list[str]:
    prepared: list[str] = []
    if config.EMBED_MODEL.lower().startswith("intfloat/multilingual-e5"):
        prefix = "query: " if is_query else "passage: "
    else:
        prefix = ""
    for text in texts:
        prepared.append(f"{prefix}{text}")
    return prepared


def embed_queries(texts: Iterable[str]) -> np.ndarray:
    embedder = get_embedder()
    prepared = _prepare_inputs(list(texts), is_query=True)
    vectors = np.array(embedder.encode(prepared, convert_to_numpy=True))  # type: ignore[attr-defined]
    return _normalize(vectors)


def embed_passages(texts: Iterable[str]) -> np.ndarray:
    embedder = get_embedder()
    prepared = _prepare_inputs(list(texts), is_query=False)
    vectors = np.array(embedder.encode(prepared, convert_to_numpy=True))  # type: ignore[attr-defined]
    return _normalize(vectors)
