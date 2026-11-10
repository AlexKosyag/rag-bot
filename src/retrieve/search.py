"""Поиск: эмбеддинг запроса -> child-абзацы -> parent-чанки (+ гибрид, + реранкер).

Small-to-big: индекс эмбеддится по абзацам (score выше и разделяемее —
длинный чанк «разбавляет» сходство), а в контекст LLM отдаётся
parent-чанк ~500 токенов с перекрытием (см. docs/pipeline.md).

Гибрид: векторный поиск + FTS5-BM25 (SQLite) сливаются через RRF —
точные термины («ГОСТ 8.586», «2-НДФЛ») не тонут в семантике.
Кандидаты, найденные только лексикой, дотягиваются до векторного score
дополнительным запросом к Chroma — порог τ остаётся валидным.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.embedder import AnyEmbedder
from src.store import Hit, Registry, VectorStore

RRF_K_DEFAULT = 60  # стандартная константа Reciprocal Rank Fusion


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]

    @property
    def max_score(self) -> float:
        return max((h.score for h in self.hits), default=0.0)

    @property
    def gap(self) -> float:
        """Разрыв top1-top2: маленький -> «пять одинаковых», сигнал слабее."""
        if len(self.hits) < 2:
            return 1.0
        s = sorted((h.score for h in self.hits), reverse=True)
        return s[0] - s[1]


def rrf_fuse(*rankings: list[str], k: int = RRF_K_DEFAULT) -> list[str]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: -scores[cid])


def retrieve(
    query: str,
    embedder: AnyEmbedder,
    store: VectorStore,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
    registry: Registry | None = None,
    hybrid: bool = False,
    rrf_k: int = RRF_K_DEFAULT,
    reranker=None,
) -> RetrievalResult:
    """top_k parent-чанков по косинусной близости лучших child-фрагментов.

    doc_ids — опциональный фильтр по документам.
    hybrid — слить векторный поиск с FTS5-BM25 через RRF (нужен registry).
    reranker — опциональный переупорядочиватель кандидатов (меняет порядок,
    не score — порог τ остаётся валидным).
    Без registry (юнит-тесты/заглушки) возвращаются child-хиты как есть.
    """
    vector = embedder.encode_query(query)
    where = None
    if doc_ids:
        where = {"doc_id": {"$in": doc_ids}}
    # Пул кандидатов шире top_k: child'ы одного parent занимают несколько мест
    pool = store.query(vector, max(top_k * 4, top_k), where=where)
    if registry is None:
        hits = pool[:top_k]
        return RetrievalResult(query=query, hits=hits)

    # Группировка child -> parent по лучшему score родителя
    best: dict[str, Hit] = {}
    for h in pool:
        cur = best.get(h.chunk_id)
        if cur is None or h.score > cur.score:
            best[h.chunk_id] = h
    order = [h.chunk_id for h in sorted(best.values(), key=lambda h: -h.score)]

    if hybrid:
        # FTS5-BM25 по parent-чанкам (SQLite), RRF-слияние двух рейтингов
        lex_order = registry.search_lexical(query, k=max(top_k * 4, 20),
                                            doc_ids=doc_ids)
        order = rrf_fuse(order, lex_order, k=rrf_k)

    ordered_ids = order[:top_k]

    # Кандидаты из лексики без векторного score — дотягиваем из Chroma
    # через get()+косинус (баг Chroma с $in-фильтром в query()),
    # чтобы score (и порог τ) оставались косинусными и сравнимыми.
    missing = [cid for cid in ordered_ids if cid not in best]
    if missing:
        scores = store.scores_for_chunk_ids(vector, missing)
        for cid, s in scores.items():
            cur = best.get(cid)
            if cur is None or s > cur.score:
                best[cid] = Hit(
                    chunk_id=cid, doc_id=cur.doc_id if cur else "",
                    doc_name=cur.doc_name if cur else "",
                    text=cur.text if cur else "",
                    heading=cur.heading if cur else "—",
                    page=cur.page if cur else None,
                    score=s, rank=0,
                )

    parents = registry.get_chunks(ordered_ids)
    hits: list[Hit] = []
    for rank, cid in enumerate(ordered_ids):
        parent = parents.get(cid)
        vec = best.get(cid)
        hits.append(
            Hit(
                chunk_id=cid,
                doc_id=(parent["doc_id"] if parent else vec.doc_id) if (parent or vec) else "",
                doc_name=(parent["doc_name"] if parent and parent["doc_name"]
                          else (vec.doc_name if vec else "")),
                text=parent["text"] if parent else (vec.text if vec else ""),
                heading=(parent["heading"] if parent else (vec.heading if vec else "—")) or "—",
                page=(int(parent["page"] or 0) if parent else (vec.page if vec else 0)) or None,
                score=vec.score if vec else 0.0,  # score лучшего child-фрагмента
                rank=rank,
            )
        )

    if reranker is not None:
        hits = reranker.rerank(query, hits)
    return RetrievalResult(query=query, hits=hits)
