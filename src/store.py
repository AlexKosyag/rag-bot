"""Хранилища: ChromaDB (чанки) + SQLite (реестр документов + FTS5-индекс)."""
from __future__ import annotations

import re
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
        try:
            self._col.upsert(
                ids=list(ids),
                documents=list(texts),
                embeddings=[list(map(float, e)) for e in embeddings],
                metadatas=list(metadatas),
            )
        except Exception as exc:
            if "dimension" in str(exc).lower():
                raise RuntimeError(
                    "Размерность эмбеддингов не совпадает с уже индексированными "
                    "данными (сменили модель эмбеддингов?). Выполните полную "
                    "переиндексацию: python -m src.ingest --force"
                ) from exc
            raise

    def query(self, vector, k: int, where: dict | None = None) -> list[Hit]:
        try:
            res = self._col.query(
                query_embeddings=[list(map(float, vector))],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if "dimension" in str(exc).lower():
                raise RuntimeError(
                    "Запрос не проходит: размерность эмбеддинга не совпадает с "
                    "базой (сменили модель?). Переиндексация: "
                    "python -m src.ingest --force"
                ) from exc
            raise
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

    def scores_for_chunk_ids(self, vector, chunk_ids: list[str]) -> dict[str, float]:
        """Косинусный score лучших child-фрагментов для указанных parent-чанков.

        Используется для кандидатов из лексики без векторного score.
        Обходит баг Chroma с $in-фильтром в query(): берём embeddings через get()
        и считаем косинус вручную.
        """
        if not chunk_ids:
            return {}
        import numpy as np

        try:
            res = self._col.get(
                where={"chunk_id": {"$in": list(chunk_ids)}},
                include=["embeddings", "metadatas"],
            )
        except Exception:
            return {}
        vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        out: dict[str, float] = {}
        embeddings = res.get("embeddings")
        metadatas = res.get("metadatas")
        if embeddings is None or metadatas is None:
            return out
        for emb, meta in zip(embeddings, metadatas):
            if emb is None:
                continue
            e = np.asarray(emb, dtype=np.float32)
            n = np.linalg.norm(e)
            s = float(np.dot(vec, e / n)) if n > 0 else 0.0
            cid = meta.get("chunk_id", "")
            out[cid] = max(out.get(cid, 0.0), s)
        return out

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._col = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )


class Registry:
    """SQLite-реестр: документы + родительские чанки (тексты для контекста).

    Векторы дочерних фрагментов живут в Chroma; полный текст родительского
    чанка (~500 токенов) хранится здесь и подставляется в контекст LLM
    после группировки найденных абзацев (small-to-big).
    """

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
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                heading TEXT,
                page INTEGER,
                text TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)"
        )
        self._init_fts()
        self._conn.commit()

    # ------------------------------------------------------- FTS5 (лексика)
    def _init_fts(self) -> None:
        """FTS5-индекс по parent-чанкам (триграммный токенизатор — русский).

        Внешнее содержимое (content='chunks') + триггеры синхронизации;
        при рассинхроне (существующие БД) — автоматический rebuild.
        """
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                heading, text,
                content='chunks', content_rowid='rowid',
                tokenize='trigram'
            )
            """
        )
        for name, sql in (
            ("chunks_fts_ai", """
                CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, heading, text)
                    VALUES (new.rowid, new.heading, new.text);
                END
            """),
            ("chunks_fts_ad", """
                CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, heading, text)
                    VALUES ('delete', old.rowid, old.heading, old.text);
                END
            """),
            ("chunks_fts_au", """
                CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, heading, text)
                    VALUES ('delete', old.rowid, old.heading, old.text);
                    INSERT INTO chunks_fts(rowid, heading, text)
                    VALUES (new.rowid, new.heading, new.text);
                END
            """),
        ):
            self._conn.execute(sql)
        n_chunks = self._conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        n_fts = self._conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        if n_chunks != n_fts:
            self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")

    def search_lexical(self, query: str, k: int = 20,
                       doc_ids: list[str] | None = None) -> list[str]:
        """BM25-поиск по parent-чанкам -> chunk_id по убыванию релевантности.

        Запрос разбивается на слова (>=3 символов — минимум триграмм);
        слова объединяются OR — важен любой термин («ГОСТ 8.586», «2-НДФЛ»).
        """
        tokens = [
            t for t in re.findall(r"\w+", query.lower(), flags=re.UNICODE)
            if len(t) >= 3
        ]
        if not tokens:
            return []
        match = " OR ".join(f'"{t.replace(chr(34), chr(34) * 2)}"' for t in tokens)
        sql = (
            "SELECT c.id FROM chunks_fts "
            "JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? "
        )
        params: list = [match]
        if doc_ids:
            marks = ",".join("?" * len(doc_ids))
            sql += f"AND c.doc_id IN ({marks}) "
            params.extend(doc_ids)
        sql += "ORDER BY bm25(chunks_fts) LIMIT ?"
        params.append(k)
        cur = self._conn.execute(sql, params)
        return [r[0] for r in cur.fetchall()]

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
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    # ----------------------------------------------------------- чанки
    def save_chunks(self, rows: list[dict]) -> None:
        """Пакетный upsert родительских чанков.

        rows: {"id", "doc_id", "chunk_index", "heading", "page", "text"}
        """
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT INTO chunks (id, doc_id, chunk_index, heading, page, text)
            VALUES (:id, :doc_id, :chunk_index, :heading, :page, :text)
            ON CONFLICT(id) DO UPDATE SET
                heading=excluded.heading, page=excluded.page, text=excluded.text
            """,
            rows,
        )
        self._conn.commit()

    def get_chunks(self, ids: list[str]) -> dict[str, sqlite3.Row]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        cur = self._conn.execute(
            f"SELECT c.*, d.name AS doc_name FROM chunks c "
            f"LEFT JOIN documents d ON d.id = c.doc_id "
            f"WHERE c.id IN ({marks})", list(ids)
        )
        return {r["id"]: r for r in cur.fetchall()}

    def delete_chunks(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
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
