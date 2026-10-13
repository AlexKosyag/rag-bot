"""Тесты метрик: Hit@k, MRR, recall, refusal, percentile, калибровка tau."""
from src.evaluate.metrics import (
    aggregate_retrieval,
    calibrate_threshold,
    context_recall,
    hit_at_k,
    mrr_at_k,
    percentile,
    refusal_metrics,
)


def test_hit_at_k():
    assert hit_at_k(["a.md", "b.md"], ["b.md"], k=5) == 1.0
    assert hit_at_k(["a.md", "b.md"], ["c.md"], k=5) == 0.0
    assert hit_at_k(["a.md"], ["b.md"], k=0) == 0.0
    assert hit_at_k(["A.MD"], ["a.md"], k=1) == 1.0  # регистр не важен


def test_mrr_at_k():
    assert mrr_at_k(["a.md", "b.md"], ["b.md"], k=5) == 0.5
    assert mrr_at_k(["a.md", "b.md", "c.md"], ["c.md"], k=5) == 1 / 3
    assert mrr_at_k(["a.md"], ["z.md"], k=5) == 0.0


def test_context_recall():
    assert context_recall(["a.md", "b.md"], ["a.md", "b.md"], 5) == 1.0
    assert context_recall(["a.md", "b.md"], ["a.md", "c.md"], 5) == 0.5
    assert context_recall([], ["a.md"], 5) == 0.0
    assert context_recall(["a.md"], [], 5) == 0.0


def test_aggregate_retrieval():
    rows = [
        {"ranked": ["x.md", "gold.md"], "gold": ["gold.md"]},
        {"ranked": ["y.md", "z.md"], "gold": ["gold.md"]},
    ]
    agg = aggregate_retrieval(rows, k=5)
    assert agg["hit"] == 0.5
    assert agg["mrr"] == 0.25
    assert agg["context_recall"] == 0.5
    assert agg["n"] == 2


def test_refusal_metrics_pairwise():
    rows = [
        {"expect_refusal": True, "refused": True},    # верный отказ
        {"expect_refusal": True, "refused": False},   # промах
        {"expect_refusal": False, "refused": False},  # верный ответ
        {"expect_refusal": False, "refused": True},   # ложный отказ
    ]
    m = refusal_metrics(rows)
    assert m["correct_refusal"] == 0.5
    assert m["false_refusal"] == 0.5
    assert m["n_out_of_scope"] == 2
    assert m["n_in_scope"] == 2


def test_refusal_metrics_empty_groups():
    m = refusal_metrics([{"expect_refusal": False, "refused": False}])
    assert m["correct_refusal"] is None
    assert m["false_refusal"] == 0.0


def test_percentile():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(vals, 50) == 30.0
    assert percentile(vals, 95) == 50.0
    assert percentile([], 50) == 0.0


def test_calibrate_perfect_separation():
    # in-scope >= 0.6, out-of-scope <= 0.4: любой tau из (0.4, 0.6] идеален
    rows = [
        {"score": 0.8, "expect_refusal": False},
        {"score": 0.6, "expect_refusal": False},
        {"score": 0.4, "expect_refusal": True},
        {"score": 0.3, "expect_refusal": True},
    ]
    table, best = calibrate_threshold(rows)
    assert best is not None
    assert 0.4 < best <= 0.6
    best_rows = [r for r in table if r["tau"] == best]
    assert best_rows[0]["f1"] == 1.0
    assert best_rows[0]["correct_refusal"] == 1.0
    assert best_rows[0]["false_refusal"] == 0.0


def test_calibrate_with_overlap_picks_tradeoff():
    # out=0.45, in=0.55: перекрытие; лучший tau разделяет максимум возможного
    rows = [
        {"score": 0.55, "expect_refusal": False},
        {"score": 0.45, "expect_refusal": True},
    ]
    table, best = calibrate_threshold(rows, grid=[0.40, 0.50, 0.60])
    # при tau=0.50: in отвечен (0.55>=0.5), out отказал (0.45<0.5) -> f1=1
    assert best == 0.50
    assert len(table) == 3


def test_calibrate_no_separation():
    # все score одинаковые: отказов нет ни при каком tau выше значения
    rows = [
        {"score": 0.5, "expect_refusal": False},
        {"score": 0.5, "expect_refusal": True},
    ]
    _, best = calibrate_threshold(rows, grid=[0.30, 0.40])
    assert best is None  # f1 = 0: отказов не происходит вовсе
