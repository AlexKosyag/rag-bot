"""Поиск: эмбеддинг запроса -> top-k по cosine в ChromaDB (плоский, exact)."""
from __future__ import annotations

from dataclasses import dataclass

from src.embedder import Embedder
from src.store import Hit, VectorStore


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]

    @property
    def max_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def gap(self) -> float:
        """Разрыв top1-top2: маленький -> «пять одинаковых», сигнал слабее."""
        if len(self.hits) < 2:
            return 1.0
        return self.hits[0].score - self.hits[1].score


def retrieve(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
) -> RetrievalResult:
    """top_k по косинусной близости. doc_ids — опциональный фильтр по документам."""
    vector = embedder.encode_query(query)
    where = None
    if doc_ids:
        where = {"doc_id": {"$in": doc_ids}}
    hits = store.query(vector, top_k, where=where)
    return RetrievalResult(query=query, hits=hits)
