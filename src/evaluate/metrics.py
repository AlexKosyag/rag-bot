"""Метрики оценки: retrieval (Hit@k, MRR, recall) и refusal (парные)."""
from __future__ import annotations

import statistics


def hit_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    """1.0, если хотя бы один эталонный документ в top-k."""
    top = {d.lower() for d in ranked[:k]}
    return 1.0 if any(g.lower() in top for g in gold) else 0.0


def mrr_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    """ reciprocal rank первого совпадения в top-k """
    gold_l = {g.lower() for g in gold}
    for i, doc in enumerate(ranked[:k], 1):
        if doc.lower() in gold_l:
            return 1.0 / i
    return 0.0


def context_recall(ranked: list[str], gold: list[str], k: int) -> float:
    """Доля эталонных документов, найденных в top-k."""
    if not gold:
        return 0.0
    top = {d.lower() for d in ranked[:k]}
    found = sum(1 for g in gold if g.lower() in top)
    return found / len(gold)


def aggregate_retrieval(rows: list[dict], k: int) -> dict:
    """rows: {ranked: [...], gold: [...]}. Возвращает средние по набору."""
    if not rows:
        return {"hit": 0.0, "mrr": 0.0, "context_recall": 0.0, "n": 0}
    return {
        "hit": round(statistics.fmean(hit_at_k(r["ranked"], r["gold"], k) for r in rows), 4),
        "mrr": round(statistics.fmean(mrr_at_k(r["ranked"], r["gold"], k) for r in rows), 4),
        "context_recall": round(
            statistics.fmean(context_recall(r["ranked"], r["gold"], k) for r in rows), 4
        ),
        "n": len(rows),
    }


def refusal_metrics(rows: list[dict]) -> dict:
    """Парные метрики отказов.

    rows: {expect_refusal: bool, refused: bool}.
    correct_refusal = доля «вне базы», где отказали;
    false_refusal   = доля «в базе есть», где отказали (опасна оптимизацией
                      «отказываться на всё» — поэтому обе обязательно).
    """
    out_scope = [r for r in rows if r["expect_refusal"]]
    in_scope = [r for r in rows if not r["expect_refusal"]]
    correct = (
        sum(1 for r in out_scope if r["refused"]) / len(out_scope) if out_scope else None
    )
    false = (
        sum(1 for r in in_scope if r["refused"]) / len(in_scope) if in_scope else None
    )
    return {
        "correct_refusal": round(correct, 4) if correct is not None else None,
        "false_refusal": round(false, 4) if false is not None else None,
        "n_out_of_scope": len(out_scope),
        "n_in_scope": len(in_scope),
    }


def percentile(values: list[float], p: float) -> float:
    """Процентиль (p50/p95) без зависимостей."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return round(s[idx], 3)
