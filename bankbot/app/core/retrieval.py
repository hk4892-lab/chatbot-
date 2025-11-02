"""Knowledge base retrieval utilities."""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_st_spec = importlib.util.find_spec("sentence_transformers")
SentenceTransformer = None
if _st_spec is not None:  # pragma: no cover - optional dependency
    SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer  # type: ignore

_sklearn_root = importlib.util.find_spec("sklearn")
_sklearn_spec = None
if _sklearn_root is not None:
    _sklearn_spec = importlib.util.find_spec("sklearn.feature_extraction.text")
if _sklearn_spec is not None:
    TfidfVectorizer = importlib.import_module("sklearn.feature_extraction.text").TfidfVectorizer  # type: ignore
    cosine_similarity = importlib.import_module("sklearn.metrics.pairwise").cosine_similarity  # type: ignore
else:
    TfidfVectorizer = None
    cosine_similarity = None


class SimpleTfidfVectorizer:
    """Fallback TF-IDF vectorizer supporting unigrams and bigrams."""

    def __init__(self) -> None:
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}
        self.documents_: List[Dict[str, float]] = []

    def fit_transform(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        tokenized = [self._tokenize(text) for text in texts]
        doc_counts: Counter[str] = Counter()
        for tokens in tokenized:
            doc_counts.update(set(tokens))
        total_docs = len(texts)
        self.idf_ = {token: math.log((1 + total_docs) / (1 + count)) + 1 for token, count in doc_counts.items()}
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(doc_counts))}
        self.documents_ = [self._build_vector(tokens) for tokens in tokenized]
        return self.documents_

    def transform(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        return [self._build_vector(self._tokenize(text)) for text in texts]

    def _build_vector(self, tokens: Sequence[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        if not counts:
            return {}
        max_tf = max(counts.values())
        vector = {
            token: (count / max_tf) * self.idf_.get(token, 0.0)
            for token, count in counts.items()
            if token in self.idf_
        }
        return vector

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[\w']+", text.lower())
        bigrams = [" ".join(pair) for pair in zip(tokens, tokens[1:])]
        return tokens + bigrams


def cosine_similarity_dicts(query: Dict[str, float], docs: List[Dict[str, float]]) -> List[float]:
    def norm(vec: Dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vec.values()))

    query_norm = norm(query)
    if query_norm == 0:
        return [0.0 for _ in docs]
    scores: List[float] = []
    for doc in docs:
        dot = sum(query.get(token, 0.0) * weight for token, weight in doc.items())
        denom = query_norm * norm(doc)
        scores.append(dot / denom if denom else 0.0)
    return scores


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    tags: List[str]
    language: str


@dataclass
class SearchResult:
    document: Document
    score: float


class KnowledgeBase:
    """Load documents from JSON files."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    @cached_property
    def documents(self) -> List[Document]:
        docs: List[Document] = []
        for language_file in self.data_path.glob("kb_*.json"):
            language = language_file.stem.split("_")[-1].upper()
            with language_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for item in payload:
                docs.append(
                    Document(
                        doc_id=item["id"],
                        title=item["title"],
                        content=item["content"],
                        tags=item.get("tags", []),
                        language=language,
                    )
                )
        return docs


class Retriever:
    """Retrieve documents using TF-IDF or sentence transformers."""

    def __init__(
        self,
        kb: KnowledgeBase,
        use_sentence_transformers: bool = False,
    ) -> None:
        self.kb = kb
        self.use_sentence_transformers = use_sentence_transformers and SentenceTransformer is not None
        self._encoder = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix: Optional[List[Dict[str, float]]] = None
        if self.use_sentence_transformers:
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")  # pragma: no cover - heavy

    def _ensure_tfidf(self) -> None:
        if self._tfidf_vectorizer is None:
            texts = [doc.content for doc in self.kb.documents]
            if TfidfVectorizer is not None and cosine_similarity is not None:
                self._tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
                self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
            else:
                vectorizer = SimpleTfidfVectorizer()
                self._tfidf_vectorizer = vectorizer  # type: ignore[assignment]
                self._tfidf_matrix = vectorizer.fit_transform(texts)

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        if self.use_sentence_transformers and self._encoder is not None:
            return self._semantic_search(query, limit)
        return self._tfidf_search(query, limit)

    def _tfidf_search(self, query: str, limit: int) -> List[SearchResult]:
        self._ensure_tfidf()
        assert self._tfidf_vectorizer is not None
        assert self._tfidf_matrix is not None
        if TfidfVectorizer is not None and cosine_similarity is not None:
            query_vec = self._tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]
            paired = list(enumerate(scores))
        else:
            query_vec = self._tfidf_vectorizer.transform([query])[0]
            scores = cosine_similarity_dicts(query_vec, self._tfidf_matrix)
            paired = list(enumerate(scores))
        paired.sort(key=lambda item: item[1], reverse=True)
        results: List[SearchResult] = []
        for idx, score in paired[:limit]:
            document = self.kb.documents[int(idx)]
            results.append(SearchResult(document=document, score=float(score)))
        return results

    def _semantic_search(self, query: str, limit: int) -> List[SearchResult]:
        if self._encoder is None:
            return []
        embeddings = self._encoder.encode([doc.content for doc in self.kb.documents])
        query_embedding = self._encoder.encode([query])[0]
        import numpy as np  # type: ignore

        scores = importlib.import_module("sklearn.metrics.pairwise").cosine_similarity([query_embedding], embeddings)[0]
        indices = np.argsort(scores)[::-1][:limit]
        results: List[SearchResult] = []
        for idx in indices:
            document = self.kb.documents[int(idx)]
            results.append(SearchResult(document=document, score=float(scores[idx])))
        return results


__all__ = ["Document", "KnowledgeBase", "Retriever", "SearchResult"]
