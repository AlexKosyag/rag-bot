"""Smoke-тест векторного хранилища: upsert/query/delete на фиктивных векторах."""
from pathlib import Path

import numpy as np
import pytest

from src.store import Registry, VectorStore


@pytest.fixture()
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma")


def _vec(seed: int) -> list[float]:
    v = np.zeros(8, dtype=np.float32)
    v[seed] = 1.0
    return v


def test_upsert_and_query(store: VectorStore):
    store.upsert(
        ids=["d1:0", "d2:0"],
        texts=["текст первый", "текст второй"],
        embeddings=[_vec(0), _vec(1)],
        metadatas=[
            {"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "a.md",
             "heading": "H", "page": 0},
            {"chunk_id": "d2:0", "doc_id": "d2", "doc_name": "b.md",
             "heading": "H", "page": 0},
        ],
    )
    assert store.count() == 2

    hits = store.query(_vec(0), k=2)
    assert hits[0].doc_name == "a.md"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)  # cosine similarity
    assert hits[0].page is None                            # page=0 -> None


def test_delete_doc(store: VectorStore):
    store.upsert(
        ids=["d1:0"],
        texts=["т"],
        embeddings=[_vec(0)],
        metadatas=[{"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "a.md",
                    "heading": "H", "page": 0}],
    )
    store.delete_doc("d1")
    assert store.count() == 0


def test_query_filter_by_doc(store: VectorStore):
    store.upsert(
        ids=["d1:0", "d2:0"],
        texts=["т1", "т2"],
        embeddings=[_vec(0), _vec(0)],
        metadatas=[
            {"chunk_id": "d1:0", "doc_id": "d1", "doc_name": "a.md",
             "heading": "H", "page": 0},
            {"chunk_id": "d2:0", "doc_id": "d2", "doc_name": "b.md",
             "heading": "H", "page": 0},
        ],
    )
    hits = store.query(_vec(0), k=5, where={"doc_id": "d1"})
    assert len(hits) == 1
    assert hits[0].doc_id == "d1"


def test_registry_lifecycle(tmp_path: Path):
    reg = Registry(tmp_path / "registry.db")
    doc_id = reg.upsert("a.md", "sha1", "ready", chunks_count=3)
    assert reg.get_by_sha("sha1")["id"] == doc_id
    assert reg.list()[0]["status"] == "ready"
    assert reg.total_chunks() == 3

    # обновление статуса по id
    reg.upsert("a.md", "sha1", "error", error="boom", doc_id=doc_id)
    row = reg.get_by_sha("sha1")
    assert row["status"] == "error"
    assert row["error"] == "boom"

    reg.delete(doc_id)
    assert reg.list() == []


def test_registry_chunks_lifecycle(tmp_path: Path):
    reg = Registry(tmp_path / "registry.db")
    doc_id = reg.upsert("a.md", "sha1", "ready", chunks_count=2)
    reg.save_chunks([
        {"id": f"{doc_id}:0", "doc_id": doc_id, "chunk_index": 0,
         "heading": "Раздел 1", "page": 0, "text": "текст первого чанка"},
        {"id": f"{doc_id}:1", "doc_id": doc_id, "chunk_index": 1,
         "heading": "Раздел 2", "page": 2, "text": "текст второго чанка"},
    ])

    got = reg.get_chunks([f"{doc_id}:0", f"{doc_id}:1", "missing"])
    assert set(got) == {f"{doc_id}:0", f"{doc_id}:1"}
    assert got[f"{doc_id}:1"]["page"] == 2
    assert got[f"{doc_id}:0"]["text"] == "текст первого чанка"

    # delete_chunks — при переиндексации; delete — каскадно на удалении документа
    reg.delete_chunks(doc_id)
    assert reg.get_chunks([f"{doc_id}:0"]) == {}

    reg.save_chunks([
        {"id": f"{doc_id}:0", "doc_id": doc_id, "chunk_index": 0,
         "heading": "H", "page": 0, "text": "T"},
    ])
    reg.delete(doc_id)
    assert reg.get_chunks([f"{doc_id}:0"]) == {}
