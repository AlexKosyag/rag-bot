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


def calibrate_threshold(
    rows: list[dict], grid: list[float] | None = None
) -> tuple[list[dict], float | None]:
    """Подбор tau: максимум F1 между корректным ответом и корректным отказом.

    rows: [{"score": float (max score), "expect_refusal": bool}, ...] — оценки
    на dev-наборе. Positive-класс задачи детекции отказа: «вне базы» и отказали.
    Возвращает (таблица по сетке, лучший tau).
    """
    if grid is None:
        grid = [round(x / 100, 2) for x in range(30, 81)]
    n_out = sum(1 for r in rows if r["expect_refusal"])
    n_in = len(rows) - n_out

    table: list[dict] = []
    for tau in grid:
        refused = [r["score"] < tau for r in rows]
        tp = sum(1 for r, f in zip(rows, refused) if r["expect_refusal"] and f)
        fp = sum(1 for r, f in zip(rows, refused) if not r["expect_refusal"] and f)
        fn = sum(1 for r, f in zip(rows, refused) if r["expect_refusal"] and not f)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        table.append({
            "tau": tau,
            "correct_refusal": round(tp / n_out, 4) if n_out else None,
            "false_refusal": round(fp / n_in, 4) if n_in else None,
            "f1": round(f1, 4),
        })
    best = max(table, key=lambda t: t["f1"])
    return table, (best["tau"] if best["f1"] > 0 else None)
