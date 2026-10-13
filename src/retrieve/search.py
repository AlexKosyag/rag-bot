"""Поиск: эмбеддинг запроса -> child-абзацы в Chroma -> группировка в parent-чанки.

Small-to-big: индекс эмбеддится по абзацам (score выше и разделяемее —
длинный чанк «разбавляет» сходство), а в контекст LLM отдаётся
parent-чанк ~500 токенов с перекрытием (см. docs/pipeline.md).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.embedder import AnyEmbedder
from src.store import Hit, Registry, VectorStore


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
    embedder: AnyEmbedder,
    store: VectorStore,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    registry: Registry | None = None,
) -> RetrievalResult:
    """top_k parent-чанков по косинусной близости лучших child-фрагментов.

    doc_ids — опциональный фильтр по документам.
    Без registry (юнит-тесты/заглушки) возвращаются child-хиты как есть.
    """
    vector = embedder.encode_query(query)
    where = None
    if doc_ids:
        where = {"doc_id": {"$in": doc_ids}}
    # Пул кандидатов шире top_k: child'ы одного parent занимают несколько мест
    pool = store.query(vector, max(top_k * 4, top_k), where=where)
    if registry is None:
        return RetrievalResult(query=query, hits=pool[:top_k])

    # Группировка child -> parent по лучшему score родителя
    best: dict[str, Hit] = {}
    for h in pool:
        cur = best.get(h.chunk_id)
        if cur is None or h.score > cur.score:
            best[h.chunk_id] = h
    ordered = sorted(best.values(), key=lambda h: -h.score)[:top_k]

    parents = registry.get_chunks([h.chunk_id for h in ordered])
    hits: list[Hit] = []
    for rank, h in enumerate(ordered):
        parent = parents.get(h.chunk_id)
        hits.append(
            Hit(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                doc_name=h.doc_name,
                text=parent["text"] if parent else h.text,
                heading=(parent["heading"] if parent else h.heading) or "—",
                page=(int(parent["page"] or 0) if parent else (h.page or 0)) or None,
                score=h.score,  # score лучшего child-фрагмента
                rank=rank,
            )
        )
    return RetrievalResult(query=query, hits=hits)
