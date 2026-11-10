"""Реранжирование кандидатов после гибридного поиска.

Два бэкенда под единым интерфейсом:
- HeuristicReranker — без ML: точные термины запроса в тексте/заголовке
  + косинус (для CPU-only и сред без Hugging Face);
- CrossEncoderReranker — cross-encoder `BAAI/bge-reranker-v2-m3`
  (stage 2 по stack.md), грузится лениво, при недоступности HF
  фабрика откатывается на эвристический.

Реранкер меняет только ПОРЯДОК кандидатов; поле `score` остаётся
косинусным (child-score) — порог τ после калибровки остаётся валидным.
"""
from __future__ import annotations

import re
from dataclasses import replace

from src.store import Hit

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _terms(query: str, min_len: int = 3) -> list[str]:
    return [
        t for t in _WORD_RE.findall(query.lower()) if len(t) >= min_len
    ]


def _overlap(query: str, text: str) -> float:
    """Доля термов запроса, встречающихся в тексте (подстрочно, без морфологии)."""
    q_terms = _terms(query, min_len=3)
    if not q_terms:
        return 0.0
    t = text.lower()
    hit = sum(1 for term in q_terms if term in t)
    return hit / len(q_terms)


class HeuristicReranker:
    """Эвристика: w·cosine + (1-w)·лексическое перекрытие термов.

    Хорошо поднимает документы с точными терминами («ГОСТ 8.586»,
    «2-НДФЛ», «форма Б-1»), которые векторный поиск «размывает».
    """

    def __init__(self, w_vec: float = 0.6, w_lex: float = 0.4):
        self.w_vec = w_vec
        self.w_lex = w_lex

    def score(self, query: str, hit: Hit) -> float:
        lex = 0.5 * _overlap(query, hit.text) + 0.5 * _overlap(query, hit.heading)
        return self.w_vec * hit.score + self.w_lex * lex

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        scored = sorted(hits, key=lambda h: -self.score(query, h))
        return [
            replace(h, rank=i) for i, h in enumerate(scored)
        ]


class CrossEncoderReranker:
    """Cross-encoder bge-reranker-v2-m3 (требует весов модели)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str | None = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device or None)

    def score(self, query: str, hit: Hit) -> float:
        return float(self.model.predict([(query, hit.text)])[0])

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        pairs = [(query, f"{h.heading}\n{h.text}") for h in hits]
        scores = self.model.predict(pairs)
        scored = sorted(zip(scores, hits), key=lambda p: -float(p[0]))
        return [replace(h, rank=i) for i, (_, h) in enumerate(scored)]


def create_reranker(kind: str, device: str = "cpu"):
    """Фабрика: off | heuristic | cross_encoder (с откатом на heuristic)."""
    if kind == "off":
        return None
    if kind == "cross_encoder":
        try:
            return CrossEncoderReranker(device=device)
        except Exception:
            # HF недоступен / нет весов -> эвристика (документируется в отчёте)
            return HeuristicReranker()
    return HeuristicReranker()
