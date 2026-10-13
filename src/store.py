"""Хранилища: ChromaDB (чанки) + SQLite (реестр документов)."""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

COLLECTION = "chunks"


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    heading: str
    page: int | None
    score: float          # cosine similarity = 1 - distance
    rank: int


class VectorStore:
    """Персистентное векторное хранилище (embedded Chroma, cosine)."""

    def __init__(self, chroma_dir: Path | str):
        import chromadb

        self.dir = Path(chroma_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.dir))
        self._col = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids, texts, embeddings, metadatas) -> None:
        self._col.upsert(
            ids=list(ids),
            documents=list(texts),
            embeddings=[list(map(float, e)) for e in embeddings],
            metadatas=list(metadatas),
        )

    def query(self, vector, k: int, where: dict | None = None) -> list[Hit]:
        res = self._col.query(
            query_embeddings=[list(map(float, vector))],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for i, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ):
            page = int(meta.get("page", 0)) or None
            hits.append(
                Hit(
                    chunk_id=meta["chunk_id"],
                    doc_id=meta["doc_id"],
                    doc_name=meta["doc_name"],
                    text=doc or "",
                    heading=meta.get("heading", "—"),
                    page=page,
                    score=1.0 - float(dist),  # cosine distance -> similarity
                    rank=i,
                )
            )
        return hits

    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._col = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )


class Registry:
    """SQLite-реестр документов: статусы, хэши, дедупликация."""

    def __init__(self, db_path: Path | str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                chunks_count INTEGER DEFAULT 0,
                error TEXT,
                uploaded_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_by_sha(self, sha256: str) -> sqlite3.Row | None:
        cur = self._conn.execute(
            "SELECT * FROM documents WHERE sha256 = ? ORDER BY uploaded_at DESC LIMIT 1",
            (sha256,),
        )
        return cur.fetchone()

    def get_by_name(self, name: str) -> sqlite3.Row | None:
        cur = self._conn.execute(
            "SELECT * FROM documents WHERE name = ? ORDER BY uploaded_at DESC LIMIT 1",
            (name,),
        )
        return cur.fetchone()

    def upsert(
        self,
        name: str,
        sha256: str,
        status: str,
        chunks_count: int = 0,
        error: str | None = None,
        doc_id: str | None = None,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        doc_id = doc_id or uuid.uuid4().hex[:16]
        self._conn.execute(
            """
            INSERT INTO documents (id, name, sha256, status, chunks_count, error, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                chunks_count=excluded.chunks_count,
                error=excluded.error
            """,
            (doc_id, name, sha256, status, chunks_count, error, now),
        )
        self._conn.commit()
        return doc_id

    def delete(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()

    def list(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    def total_chunks(self) -> int:
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(chunks_count), 0) AS s FROM documents WHERE status='ready'"
        )
        return int(cur.fetchone()["s"])
