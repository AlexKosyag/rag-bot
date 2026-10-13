"""Parent-child (small-to-big): child-хиты группируются в parent-чанки."""
import numpy as np
import pytest

from src.retrieve.search import retrieve
from src.store import Registry, VectorStore


def _vec(seed: int) -> list[float]:
    v = np.zeros(8, dtype=np.float32)
    v[seed] = 1.0
    return v


class _FakeEmbedder:
    def encode_query(self, query: str):
        return _vec(0)


def test_retrieve_groups_children_into_parents(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    reg = Registry(tmp_path / "registry.db")

    # d1:0 — сильный child (vec0 == запрос) и слабый; d1:1 — второй parent
    store.upsert(
        ids=["d1:0:0", "d1:0:1", "d1:1:0"],
        texts=["сильный абзац", "прочее", "чужой абзац"],
        embeddings=[_vec(0), _vec(3), _vec(1)],
        metadatas=[
            {"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "a.md",
             "heading": "—", "page": 0},
            {"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "a.md",
             "heading": "—", "page": 0},
            {"chunk_id": "d1:1", "doc_id": "d1", "doc_name": "a.md",
             "heading": "—", "page": 0},
        ],
    )
    reg.save_chunks([
        {"id": "d1:0", "doc_id": "d1", "chunk_index": 0,
         "heading": "Раздел 1", "page": 0, "text": "полный текст чанка один"},
        {"id": "d1:1", "doc_id": "d1", "chunk_index": 1,
         "heading": "Раздел 2", "page": 0, "text": "полный текст чанка два"},
    ])

    res = retrieve("запрос", _FakeEmbedder(), store, top_k=5, registry=reg)

    # child-дубль d1:0:1 не даёт второго хита: группировка по parent
    assert [h.chunk_id for h in res.hits] == ["d1:0", "d1:1"]
    # в контекст уходит parent-текст, а не child-абзац
    assert res.hits[0].text == "полный текст чанка один"
    assert res.hits[0].heading == "Раздел 1"
    # score — от лучшего child родителя
    assert res.hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert res.max_score == res.hits[0].score
    assert res.gap == pytest.approx(res.hits[0].score - res.hits[1].score)


def test_retrieve_without_registry_returns_children(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.upsert(
        ids=["d1:0:0"],
        texts=["абзац"],
        embeddings=[_vec(0)],
        metadatas=[{"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "a.md",
                    "heading": "—", "page": 0}],
    )
    res = retrieve("запрос", _FakeEmbedder(), store, top_k=5)
    assert res.hits[0].text == "абзац"
