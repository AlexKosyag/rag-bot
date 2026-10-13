"""Порог отказа: отвечать или сказать «Нет в документах».

Калибровка tau на dev-наборе обязательна (docs/evaluation.md): порог «на глаз»
вырождается либо в отказы на всё, либо в галлюцинации.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.retrieve.search import RetrievalResult

REFUSAL_TEXT = "Нет в документах."


@dataclass
class Decision:
    answer: bool                 # True -> генерировать ответ, False -> отказ
    reason: str                  # ok | below_threshold | empty | low_margin
    max_score: float
    gap: float
    tau: float

    @property
    def refusal_text(self) -> str:
        return REFUSAL_TEXT if not self.answer else ""


def decide(result: RetrievalResult, tau: float, delta: float = 0.0) -> Decision:
    """max score < tau -> отказ; низкий разрыв top1-top2 -> ответ с пометкой."""
    if not result.hits:
        return Decision(False, "empty", 0.0, 0.0, tau)

    max_score, gap = result.max_score, result.gap
    if max_score < tau:
        return Decision(False, "below_threshold", max_score, gap, tau)
    if delta and gap < delta:
        # Отвечаем, но сигнал слабый — UI может показать больше фрагментов.
        return Decision(True, "low_margin", max_score, gap, tau)
    return Decision(True, "ok", max_score, gap, tau)
