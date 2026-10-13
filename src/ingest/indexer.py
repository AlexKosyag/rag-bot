"""Индексация: парсинг -> чанкинг -> эмбеддинг -> Chroma + реестр."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import Config
from src.embedder import Embedder
from src.ingest.chunker import chunk_document
from src.ingest.parser import ParseError, parse_file
from src.store import Registry, VectorStore

log = logging.getLogger(__name__)

SUPPORTED = {".md", ".pdf", ".docx"}


@dataclass
class IndexResult:
    path: str
    doc_id: str | None
    status: str            # ready | skipped | error
    chunks: int = 0
    detail: str = ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def index_file(
    cfg: Config,
    path: Path | str,
    embedder: Embedder,
    store: VectorStore,
    registry: Registry,
    force: bool = False,
) -> IndexResult:
    """Индексирует один файл. Идемпотентно: дубликат по sha256 пропускается."""
    path = Path(path)
    sha = _sha256(path)

    if not force:
        existing = registry.get_by_sha(sha)
        if existing is not None and existing["status"] == "ready":
            return IndexResult(str(path), existing["id"], "skip", existing["chunks_count"],
                               "уже индексирован")

    # Файл с тем же именем, но другим содержимым -> старые чанки удаляем.
    old = registry.get_by_name(path.name)
    old_id = old["id"] if old else None
    if old_id:
        store.delete_doc(old_id)

    try:
        doc = parse_file(path)
    except ParseError as exc:
        registry.upsert(path.name, sha, "error", error=str(exc), doc_id=old_id)
        return IndexResult(str(path), old_id, "error", detail=str(exc))

    chunks = chunk_document(
        doc.text,
        size_tokens=cfg.chunking.size_tokens,
        overlap_tokens=cfg.chunking.overlap_tokens,
        min_tokens=cfg.chunking.min_tokens,
        token_counter=embedder.count_tokens,
    )
    if not chunks:
        registry.upsert(path.name, sha, "error", error="нет текста", doc_id=old_id)
        return IndexResult(str(path), old_id, "error", detail="нет текста")

    doc_id = registry.upsert(path.name, sha, "indexing", doc_id=old_id)

    texts = [c.text for c in chunks]
    embeddings = embedder.encode_passages(texts)
    metadatas = [
        {
            "chunk_id": f"{doc_id}:{c.index}",
            "doc_id": doc_id,
            "doc_name": path.name,
            "chunk_index": c.index,
            "heading": (c.heading or "—")[:250],
            "page": doc.page_for_offset(c.start) if doc.source_type == "pdf" else 0,
            "source_type": doc.source_type,
        }
        for c in chunks
    ]
    store.upsert(
        ids=[m["chunk_id"] for m in metadatas],
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    registry.upsert(path.name, sha, "ready", chunks_count=len(chunks), doc_id=doc_id)
    log.info("indexed %s -> %d chunks", path.name, len(chunks))
    return IndexResult(str(path), doc_id, "ready", len(chunks))


def index_directory(
    cfg: Config,
    directory: Path | str,
    embedder: Embedder,
    store: VectorStore,
    registry: Registry,
    force: bool = False,
) -> list[IndexResult]:
    directory = Path(directory)
    results = []
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        log.warning("в %s нет файлов PDF/DOCX/MD", directory)
    for path in files:
        try:
            results.append(index_file(cfg, path, embedder, store, registry, force=force))
        except Exception as exc:  # не роняем весь прогон из-за одного файла
            log.exception("ошибка индексации %s", path)
            results.append(IndexResult(str(path), None, "error", detail=str(exc)))
    return results
