"""Тесты гибридного поиска: FTS5-BM25 + RRF-слияние."""
import numpy as np
import pytest

from src.retrieve.search import retrieve, rrf_fuse
from src.store import Registry, VectorStore


def _vec(seed: int) -> list[float]:
    v = np.zeros(8, dtype=np.float32)
    v[seed] = 1.0
    return v


class _FakeEmbedder:
    def encode_query(self, query: str):
        return _vec(0)


def test_rrf_fuse_interleaves():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = rrf_fuse(a, b, k=60)
    # x и y встречаются в обоих списках — выше z и w
    assert set(fused[:2]) == {"x", "y"}
    assert set(fused) == {"x", "y", "z", "w"}


def test_rrf_fuse_single_ranking():
    assert rrf_fuse(["a", "b"], k=60) == ["a", "b"]


def test_lexical_search_finds_exact_terms(tmp_path):
    reg = Registry(tmp_path / "registry.db")
    doc_id = reg.upsert("gost.md", "sha1", "ready", chunks_count=2)
    reg.save_chunks([
        {"id": f"{doc_id}:0", "doc_id": doc_id, "chunk_index": 0,
         "heading": "Поверка", "page": 0,
         "text": "ГОСТ 8.586.1-2014 устанавливает периодичность поверки"},
        {"id": f"{doc_id}:1", "doc_id": doc_id, "chunk_index": 1,
         "heading": "Отпуск", "page": 0,
         "text": "Ежегодный оплачиваемый отпуск"},
    ])
    found = reg.search_lexical("ГОСТ 8.586 поверка", k=5)
    assert found == [f"{doc_id}:0"]


def test_lexical_search_doc_filter(tmp_path):
    reg = Registry(tmp_path / "registry.db")
    d1 = reg.upsert("a.md", "s1", "ready", chunks_count=1)
    d2 = reg.upsert("b.md", "s2", "ready", chunks_count=1)
    reg.save_chunks([
        {"id": f"{d1}:0", "doc_id": d1, "chunk_index": 0,
         "heading": "H", "page": 0, "text": "общий термин командировка"},
        {"id": f"{d2}:0", "doc_id": d2, "chunk_index": 0,
         "heading": "H", "page": 0, "text": "общий термин командировка"},
    ])
    found = reg.search_lexical("командировка", k=5, doc_ids=[d2])
    assert found == [f"{d2}:0"]


def test_lexical_search_empty_query(tmp_path):
    reg = Registry(tmp_path / "registry.db")
    assert reg.search_lexical("а и", k=5) == []


def test_hybrid_retrieval_finds_lexical_only_hit(tmp_path):
    """Хит, не найденный вектором, поднимается лексикой и получает score."""
    store = VectorStore(tmp_path / "chroma")
    reg = Registry(tmp_path / "registry.db")

    # Векторный запрос совпадает с d1:0 (vec0); d2:0 — почти ортогонален
    # (доля vec0 = 0.3), но содержит точный термин «2-НДФЛ».
    v2 = np.zeros(8, dtype=np.float32)
    v2[0], v2[1] = 0.3, 0.95
    store.upsert(
        ids=["d1:0:0", "d2:0:0"],
        texts=["отпуск работника", "справка 2-НДФЛ выдаётся"],
        embeddings=[_vec(0), v2],
        metadatas=[
            {"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "a.md",
             "heading": "—", "page": 0},
            {"chunk_id": "d2:0", "doc_id": "d2", "doc_name": "b.md",
             "heading": "—", "page": 0},
        ],
    )
    reg.save_chunks([
        {"id": "d1:0", "doc_id": "d1", "chunk_index": 0,
         "heading": "Отпуск", "page": 0, "text": "отпуск работника"},
        {"id": "d2:0", "doc_id": "d2", "chunk_index": 0,
         "heading": "Справки", "page": 0, "text": "справка 2-НДФЛ выдаётся"},
    ])

    # чистый вектор: d2:0 не попадает в топ при top_k=1
    r_vec = retrieve("справка 2-НДФЛ", _FakeEmbedder(), store,
                     top_k=1, registry=reg, hybrid=False)
    assert [h.chunk_id for h in r_vec.hits] == ["d1:0"]

    # гибрид: лексика находит d2:0, RRF поднимает его
    r_hyb = retrieve("справка 2-НДФЛ", _FakeEmbedder(), store,
                     top_k=1, registry=reg, hybrid=True)
    assert [h.chunk_id for h in r_hyb.hits] == ["d2:0"]
    # score — косинусный (дотянут из Chroma): 0.3 / |v2|
    expected = 0.3 / (0.3 ** 2 + 0.95 ** 2) ** 0.5
    assert r_hyb.hits[0].score == pytest.approx(expected, abs=1e-5)


def test_hybrid_hit_has_doc_name(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    reg = Registry(tmp_path / "registry.db")
    store.upsert(
        ids=["d1:0:0"],
        texts=["текст"],
        embeddings=[_vec(0)],
        metadatas=[{"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "a.md",
                    "heading": "H", "page": 0}],
    )
    reg.save_chunks([
        {"id": "d1:0", "doc_id": "d1", "chunk_index": 0,
         "heading": "H", "page": 0, "text": "текст"},
    ])
    r = retrieve("текст", _FakeEmbedder(), store, top_k=1,
                 registry=reg, hybrid=True)
    assert r.hits[0].doc_name == "a.md"
