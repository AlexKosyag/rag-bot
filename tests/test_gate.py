"""Тесты порога отказа."""
from src.retrieve.gate import decide
from src.retrieve.search import RetrievalResult
from src.store import Hit


def _result(scores: list[float]) -> RetrievalResult:
    hits = [
        Hit(
            chunk_id=f"c{i}", doc_id="d", doc_name="doc.md", text="t",
            heading="h", page=None, score=s, rank=i,
        )
        for i, s in enumerate(scores)
    ]
    return RetrievalResult(query="q", hits=hits)


def test_refusal_below_threshold():
    d = decide(_result([0.40, 0.35]), tau=0.55)
    assert not d.answer
    assert d.reason == "below_threshold"
    assert "Нет в документах" in d.refusal_text


def test_answer_above_threshold():
    d = decide(_result([0.81, 0.74]), tau=0.55)
    assert d.answer
    assert d.reason == "ok"
    assert d.refusal_text == ""


def test_empty_index_refuses():
    d = decide(_result([]), tau=0.55)
    assert not d.answer
    assert d.reason == "empty"


def test_low_margin_flagged_but_answers():
    d = decide(_result([0.70, 0.695]), tau=0.55, delta=0.05)
    assert d.answer
    assert d.reason == "low_margin"
    assert d.gap < 0.05


def test_boundary_equals_tau_answers():
    d = decide(_result([0.55]), tau=0.55)
    assert d.answer  # отказ строго при score < tau
