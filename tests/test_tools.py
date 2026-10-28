"""Тесты диспетчера инструментов агентного поиска."""
import numpy as np

from src.generate.tools import (
    SEARCH_TOOL,
    SEARCH_TOOL_NAME,
    ToolCall,
    execute_search,
    merge_hits,
)
from src.store import Hit, Registry, VectorStore


def _vec(seed: int) -> list[float]:
    v = np.zeros(8, dtype=np.float32)
    v[seed] = 1.0
    return v


class _FakeEmbedder:
    def __init__(self, query_vec: list[float] | None = None):
        self._qvec = _vec(0) if query_vec is None else query_vec

    def encode_query(self, query: str):
        return self._qvec


def _make_store(tmp_path) -> tuple[VectorStore, Registry]:
    store = VectorStore(tmp_path / "chroma")
    reg = Registry(tmp_path / "registry.db")
    store.upsert(
        ids=["d1:0:0", "d1:0:1"],
        texts=["отпуск 28 дней", "прочее"],
        embeddings=[_vec(0), _vec(3)],
        metadatas=[
            {"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "ustav.md",
             "heading": "Отпуск", "page": 0},
            {"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "ustav.md",
             "heading": "Отпуск", "page": 0},
        ],
    )
    reg.save_chunks([
        {"id": "d1:0", "doc_id": "d1", "chunk_index": 0,
         "heading": "Отпуск", "page": 0, "text": "Ежегодный отпуск 28 календарных дней"},
    ])
    return store, reg


def test_tool_schema():
    assert SEARCH_TOOL["type"] == "function"
    fn = SEARCH_TOOL["function"]
    assert fn["name"] == SEARCH_TOOL_NAME
    assert "query" in fn["parameters"]["required"]


def test_execute_search_returns_hits(tmp_path):
    store, reg = _make_store(tmp_path)
    call = ToolCall(name=SEARCH_TOOL_NAME, args={"query": "отпуск"})
    outcome = execute_search(call, _FakeEmbedder(), store, reg,
                             top_k_default=5, tau=0.50)
    assert outcome.decision.answer
    assert outcome.decision.max_score > 0.50
    assert len(outcome.result.hits) >= 1
    text = outcome.render()
    assert "релевантно" in text
    assert "отпуск 28" in text


def test_execute_search_below_threshold_renders_warning(tmp_path):
    store, reg = _make_store(tmp_path)
    # вектор, ортогональный хранимому -> score ~0.0 < tau
    embedder = _FakeEmbedder(query_vec=_vec(7))
    call = ToolCall(name=SEARCH_TOOL_NAME, args={"query": "xyzzy"})
    outcome = execute_search(call, embedder, store, reg,
                             top_k_default=5, tau=0.99)
    text = outcome.render()
    assert "ниже порога" in text


def test_execute_search_empty_query(tmp_path):
    store, reg = _make_store(tmp_path)
    call = ToolCall(name=SEARCH_TOOL_NAME, args={})
    outcome = execute_search(call, _FakeEmbedder(), store, reg,
                             top_k_default=5, tau=0.50)
    assert len(outcome.result.hits) == 0
    assert "нет" in outcome.render().lower()


def test_execute_search_clamps_top_k(tmp_path):
    store, reg = _make_store(tmp_path)
    call = ToolCall(name=SEARCH_TOOL_NAME, args={"query": "отпуск", "top_k": 99})
    outcome = execute_search(call, _FakeEmbedder(), store, reg,
                             top_k_default=5, tau=0.50)
    assert outcome.call.args["top_k"] == 10


def test_merge_hits_dedup_by_chunk_id():
    h1 = Hit(chunk_id="c1", doc_id="d", doc_name="a.md", text="t1",
             heading="h", page=None, score=0.9, rank=0)
    h2 = Hit(chunk_id="c1", doc_id="d", doc_name="a.md", text="t1",
             heading="h", page=None, score=0.7, rank=1)  # тот же chunk_id, ниже
    h3 = Hit(chunk_id="c2", doc_id="d", doc_name="a.md", text="t2",
             heading="h", page=None, score=0.8, rank=2)
    merged = merge_hits([h1, h2], [h3])
    assert [h.chunk_id for h in merged] == ["c1", "c2"]
    assert merged[0].score == 0.9
    assert merged[0].rank == 0
