"""Быстрые retrieval-эксперименты для docs/evaluation.md (без генерации LLM).

Сводная таблица: размер чанка (256/500/1024) × top_k (3/5/8) →
Hit@k / MRR / Context Recall + метрики отказов при tau из config.yaml.

- 500 токенов — базовый индекс (data/chroma), переиспользуется;
- 256/1024 — переиндексация демо-корпуса в data/exp/chroma-{size},
  главный индекс не трогаем.

Запуск: python -m src.evaluate.experiments [--qa data/qa/test.json]
        [--reindex] [--out reports/experiments.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.config import ROOT, load_config
from src.engine import RagEngine
from src.evaluate.__main__ import load_qa
from src.evaluate.metrics import aggregate_retrieval, refusal_metrics
from src.ingest.indexer import index_directory

KS = (3, 5, 8)
CHUNK_SIZES = (256, 500, 1024)


def _run(engine: RagEngine, items: list[dict], k_max: int, tau: float) -> dict:
    """Один проход по QA: ранги документов (для Hit@k) + решения об отказе."""
    retrieval_rows: list[dict] = []
    refusal_rows: list[dict] = []
    for item in items:
        prepared = engine.prepare(item["question"], top_k=k_max, tau=tau)
        ranked = [h.doc_name for h in prepared.retrieval.hits]
        refusal_rows.append({
            "expect_refusal": item["expect_refusal"],
            "refused": not prepared.decision.answer,
        })
        if item["gold_docs"]:
            retrieval_rows.append({"ranked": ranked, "gold": item["gold_docs"]})
    return {
        "retrieval": {str(k): aggregate_retrieval(retrieval_rows, k) for k in KS},
        "refusal": refusal_metrics(refusal_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперименты: чанкинг × top_k")
    parser.add_argument("--qa", default=None, help="путь к QA-набору (json)")
    parser.add_argument("--k-max", type=int, default=max(KS),
                        help="сколько parent-чанков извлекать (метрики@k ≤ k-max)")
    parser.add_argument("--reindex", action="store_true",
                        help="пересобрать экспериментальные индексы с нуля")
    parser.add_argument("--out", default=None, help="куда записать отчёт json")
    args = parser.parse_args()

    cfg = load_config()
    qa_path = Path(args.qa) if args.qa else cfg.paths.qa_dir / "test.json"
    items = load_qa(qa_path)
    tau = cfg.retrieval.threshold
    out_path = (Path(args.out) if args.out
                else cfg.paths.reports_dir / "experiments.json")

    exp_dir = ROOT / "data" / "exp"
    runs: list[dict] = []
    for size in CHUNK_SIZES:
        if size == cfg.chunking.size_tokens:
            engine = RagEngine.from_config(cfg)  # базовый индекс, без переиндексации
            index_state = str(cfg.paths.chroma_dir)
        else:
            cfg_s = replace(
                cfg,
                chunking=replace(cfg.chunking, size_tokens=size),
                paths=replace(
                    cfg.paths,
                    chroma_dir=exp_dir / f"chroma-{size}",
                    registry_db=exp_dir / f"registry-{size}.db",
                ),
            )
            engine = RagEngine.from_config(cfg_s)
            if args.reindex or engine.store.count() == 0:
                results = index_directory(
                    cfg_s, cfg_s.paths.raw_dir, engine.embedder,
                    engine.store, engine.registry, force=args.reindex,
                )
                errors = [r for r in results if r.status == "error"]
                if errors:
                    print(f"  ! ошибки индексации ({size}): "
                          f"{[e.detail for e in errors]}")
            index_state = str(cfg_s.paths.chroma_dir)

        n_children = engine.store.count()
        print(f"chunk={size}: child-фрагментов {n_children}, индекс {index_state}")
        run = _run(engine, items, args.k_max, tau)
        run.update(chunk_tokens=size, n_children=n_children, index=index_state)
        runs.append(run)

    report = {
        "params": {
            "qa": str(qa_path), "tau": tau, "k_max": args.k_max,
            "embedding": cfg.models.embedding, "n_items": len(items),
            "note": "retrieval+refusal, без генерации LLM; tau откалиброван "
                    "при chunk=500",
        },
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown-таблица для docs/evaluation.md
    print("\n| chunk | k | Hit@k | MRR | Context Recall "
          "| Correct Refusal | False Refusal |")
    print("|---|---|---|---|---|---|---|")
    for run in runs:
        rf = run["refusal"]
        for k in KS:
            if str(k) not in run["retrieval"]:
                continue
            m = run["retrieval"][str(k)]
            print(f"| {run['chunk_tokens']} | {k} | {m['hit']:.2f} | {m['mrr']:.2f} "
                  f"| {m['context_recall']:.2f} | {rf['correct_refusal']:.2f} "
                  f"| {rf['false_refusal']:.2f} |")
    print(f"Отчёт: {out_path}")


if __name__ == "__main__":
    main()
