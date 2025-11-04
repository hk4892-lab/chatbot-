from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

import numpy as np

from .embeddings import embed_passages

LOGGER = logging.getLogger("bankbot.vector_store")

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


class VectorStore:
    def __init__(self) -> None:
        self.index: Any | None = None
        self.passages: list[str] = []
        self.metas: list[dict[str, Any]] = []
        self.use_faiss = faiss is not None
        self.vectors: np.ndarray | None = None

    def is_ready(self) -> bool:
        if self.use_faiss:
            return self.index is not None and len(self.passages) == len(self.metas)
        return self.vectors is not None and len(self.passages) == len(self.metas)

    def build(self, passages: List[str], metas: List[dict[str, Any]]) -> None:
        if len(passages) != len(metas):
            raise ValueError("Passages and metas length mismatch")
        vectors = embed_passages(passages)
        self.passages = passages
        self.metas = metas
        if self.use_faiss:
            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vectors.astype(np.float32))
            self.index = index
            self.vectors = None
        else:
            self.vectors = vectors
            self.index = None
        LOGGER.info("VectorStore built with %d passages (faiss=%s)", len(passages), self.use_faiss)

    def save(self, path: str | Path) -> None:
        if not self.is_ready():
            raise RuntimeError("VectorStore not built")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta_payload = []
        for passage, meta in zip(self.passages, self.metas):
            entry = dict(meta)
            entry["text"] = passage
            meta_payload.append(entry)
        with (path / "metas.json").open("w", encoding="utf-8") as fh:
            json.dump(meta_payload, fh, ensure_ascii=False, indent=2)
        if self.use_faiss:
            faiss.write_index(self.index, str(path / "index.faiss"))  # type: ignore[arg-type]
        else:
            np.save(path / "vectors.npy", self.vectors)
        LOGGER.info("VectorStore saved to %s", path)

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        meta_file = path / "metas.json"
        if not meta_file.exists():
            return False
        with meta_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.passages = [item["text"] for item in payload]
        self.metas = [{k: v for k, v in item.items() if k != "text"} for item in payload]
        if self.use_faiss:
            index_file = path / "index.faiss"
            if not index_file.exists():
                LOGGER.warning("FAISS index missing at %s; rebuilding required", index_file)
                return False
            self.index = faiss.read_index(str(index_file))  # type: ignore[arg-type]
            self.vectors = None
        else:
            vectors_file = path / "vectors.npy"
            if not vectors_file.exists():
                LOGGER.warning("Vector cache missing at %s; rebuilding required", vectors_file)
                return False
            self.vectors = np.load(vectors_file)
            self.index = None
        LOGGER.info("VectorStore loaded from %s", path)
        return True

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("VectorStore is not ready")
        top_k = min(top_k, len(self.passages))
        if top_k == 0:
            return []
        if self.use_faiss:
            scores, indices = self.index.search(query_vector[np.newaxis, :].astype(np.float32), top_k)  # type: ignore[union-attr]
            scores = scores[0]
            indices = indices[0]
        else:
            vectors = self.vectors
            assert vectors is not None
            scores = vectors @ query_vector
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        results: list[dict[str, Any]] = []
        for idx, score in zip(indices, scores):
            if idx == -1:
                continue
            results.append(
                {
                    "score": float(score),
                    "meta": self.metas[int(idx)],
                    "text": self.passages[int(idx)],
                }
            )
        return results
