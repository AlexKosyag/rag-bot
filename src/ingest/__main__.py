"""CLI индексации: python -m src.ingest [--path data/raw] [--force]."""
from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.embedder import create_embedder
from src.ingest.indexer import index_directory
from src.store import Registry, VectorStore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Индексация документов в ChromaDB")
    parser.add_argument("--path", default=None, help="каталог с документами")
    parser.add_argument("--force", action="store_true", help="полная переиндексация")
    args = parser.parse_args()

    cfg = load_config()
    embedder = create_embedder(cfg)
    store = VectorStore(cfg.paths.chroma_dir)
    registry = Registry(cfg.paths.registry_db)

    print(f"Модель эмбеддингов: {cfg.models.embedding} (dim={embedder.dim})")
    results = index_directory(
        cfg, args.path or cfg.paths.raw_dir, embedder, store, registry, force=args.force
    )
    for r in results:
        mark = {"ready": "+", "skip": "=", "error": "!"}.get(r.status, "?")
        print(f" [{mark}] {r.path}: {r.status} {r.chunks} чанков {r.detail}")
    ready = sum(1 for r in results if r.status == "ready")
    print(
        f"Готово: {ready} проиндексировано; "
        f"parent-чанков: {registry.total_chunks()}, "
        f"child-фрагментов в Chroma: {store.count()}"
    )


if __name__ == "__main__":
    main()
