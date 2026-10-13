"""CLI оценки: python -m src.evaluate [--qa data/qa/test.json] [--llm] [--judge].

Без флагов считает retrieval-метрики и метрики отказов (быстро, без LLM).
--llm   — генерация ответов (Faithfulness требует также --judge).
--judge — LLM-as-judge для Faithfulness.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import load_config
from src.engine import RagEngine
from src.evaluate.judge import faithfulness_score
from src.evaluate.metrics import aggregate_retrieval, percentile, refusal_metrics
from src.generate.answer import build_context


def load_qa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    for item in items:
        item.setdefault("gold_docs", [])
        item.setdefault("expect_refusal", False)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества RAG")
    parser.add_argument("--qa", default=None, help="путь к QA-набору (json)")
    parser.add_argument("--k", type=int, default=None, help="top-k ретривера")
    parser.add_argument("--tau", type=float, default=None, help="порог отказа")
    parser.add_argument("--llm", action="store_true", help="генерировать ответы")
    parser.add_argument("--judge", action="store_true", help="Faithfulness (LLM-as-judge)")
    parser.add_argument("--out", default=None, help="куда записать отчёт json")
    args = parser.parse_args()
    if args.judge:
        args.llm = True

    cfg = load_config()
    engine = RagEngine.from_config(cfg)
    k = args.k or cfg.retrieval.top_k
    tau = args.tau if args.tau is not None else cfg.retrieval.threshold

    qa_path = Path(args.qa) if args.qa else cfg.paths.qa_dir / "test.json"
    items = load_qa(qa_path)
    out_path = Path(args.out) if args.out else cfg.paths.reports_dir / "metrics.json"

    print(f"QA: {qa_path} ({len(items)} вопросов), top_k={k}, tau={tau}")

    retrieval_rows: list[dict] = []
    refusal_rows: list[dict] = []
    latencies: list[float] = []
    faithfulness: list[int] = []
    model_refusals = 0
    generated = 0

    for i, item in enumerate(items, 1):
        q = item["question"]
        prepared = engine.prepare(q, top_k=k, tau=tau)
        ranked = [h.doc_name for h in prepared.retrieval.hits]
        refused = not prepared.decision.answer

        refusal_rows.append({"expect_refusal": item["expect_refusal"], "refused": refused})
        if item["gold_docs"]:
            retrieval_rows.append({"ranked": ranked, "gold": item["gold_docs"]})

        status = f"refused({prepared.decision.reason})" if refused else \
            f"answer(max={prepared.decision.max_score:.3f})"

        if args.llm and prepared.decision.answer:
            t0 = time.perf_counter()
            answer = engine.answer(prepared)
            latencies.append((time.perf_counter() - t0) * 1000)
            generated += 1
            if engine.answer_is_refusal(answer):
                model_refusals += 1
            if args.judge and not item["expect_refusal"]:
                score = faithfulness_score(
                    q,
                    build_context(prepared.retrieval.hits),
                    answer,
                    engine.llm,
                )
                faithfulness.append(score)
            status += " [llm]"

        print(f"  [{i}/{len(items)}] {item['id']}: {status}")

    report: dict = {
        "params": {
            "qa": str(qa_path), "k": k, "tau": tau,
            "embedding": cfg.models.embedding, "llm": cfg.models.llm,
            "chunk_tokens": cfg.chunking.size_tokens, "n_items": len(items),
        },
        "retrieval": aggregate_retrieval(retrieval_rows, k),
        "refusal": refusal_metrics(refusal_rows),
    }
    if args.llm:
        report["generation"] = {
            "generated": generated,
            "model_refusals_on_answerable": model_refusals,
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
            },
        }
    if args.judge:
        report["generation"]["faithfulness_mean"] = (
            round(sum(faithfulness) / len(faithfulness) / 2, 4) if faithfulness else None
        )
        report["generation"]["faithfulness_n"] = len(faithfulness)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Отчёт: {out_path}")


if __name__ == "__main__":
    main()
