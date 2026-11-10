"""Тесты реранкера и query rewriting."""
from src.generate.rewrite import build_rewrite_messages, extract_rewritten
from src.retrieve.reranker import HeuristicReranker, create_reranker
from src.store import Hit


def _hit(cid: str, text: str, heading: str, score: float) -> Hit:
    return Hit(chunk_id=cid, doc_id="d", doc_name="a.md", text=text,
               heading=heading, page=None, score=score, rank=0)


def test_heuristic_reranker_boosts_exact_terms():
    reranker = HeuristicReranker(w_vec=0.5, w_lex=0.5)
    hits = [
        _hit("c1", "семантически близкий текст про документы", "Раздел", 0.80),
        _hit("c2", "справка 2-НДФЛ выдаётся в бухгалтерии", "Справки", 0.70),
    ]
    out = reranker.rerank("справка 2-НДФЛ", hits)
    # точное совпадение терминов поднимает c2 выше более похожего c1
    assert out[0].chunk_id == "c2"
    assert out[0].rank == 0


def test_heuristic_reranker_keeps_score():
    reranker = HeuristicReranker()
    hits = [_hit("c1", "текст", "H", 0.77)]
    out = reranker.rerank("запрос", hits)
    assert out[0].score == 0.77  # score не перезаписывается


def test_create_reranker_off():
    assert create_reranker("off") is None


def test_create_reranker_heuristic():
    assert isinstance(create_reranker("heuristic"), HeuristicReranker)


def test_build_rewrite_messages_with_history():
    history = [
        {"role": "user", "content": "Сколько дней отпуск?"},
        {"role": "assistant", "content": "28 календарных дней."},
    ]
    msgs = build_rewrite_messages("а в командировке?", history)
    assert msgs[0]["role"] == "system"
    assert "История диалога" in msgs[1]["content"]
    assert "а в командировке?" in msgs[1]["content"]
    assert "Сколько дней отпуск?" in msgs[1]["content"]


def test_build_rewrite_messages_empty_history():
    msgs = build_rewrite_messages("вопрос", [])
    assert "(история пуста)" in msgs[1]["content"]


def test_extract_rewritten():
    assert extract_rewritten("Каковы правила командировок?", "fallback") == \
        "Каковы правила командировок?"
    assert extract_rewritten("«Правила командировок»", "fallback") == \
        "Правила командировок"
    assert extract_rewritten("", "fallback") == "fallback"
    assert extract_rewritten(None, "fallback") == "fallback"
    assert extract_rewritten("а", "fallback") == "fallback"
    assert extract_rewritten("x" * 600, "fallback") == "fallback"
