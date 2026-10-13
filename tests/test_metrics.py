"""Тесты метрик: Hit@k, MRR, recall, refusal, percentile."""
from src.evaluate.metrics import (
    aggregate_retrieval,
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
