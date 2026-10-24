"""CLI оценки: python -m src.evaluate [--qa data/qa/test.json] [--llm] [--judge].

Без флагов считает retrieval-метрики и метрики отказов (быстро, без LLM).
--llm        — генерация ответов (Faithfulness требует также --judge).
--judge      — LLM-as-judge для Faithfulness.
--correctness — judge-правильность ответов относительно gold.
--baseline   — baseline «Qwen без RAG»: без контекста и порога.
--calibrate  — подбор tau по сетке (dev.json), без генерации.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import load_config
from src.engine import RagEngine
from src.evaluate.judge import correctness_score, faithfulness_score
from src.evaluate.metrics import (
    aggregate_retrieval,
    calibrate_threshold,
    percentile,
    refusal_metrics,
)
from src.generate.answer import build_baseline_messages, build_context


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
    parser.add_argument(
        "--calibrate", action="store_true",
        help="подбор tau по сетке (по умолчанию QA = dev.json), без генерации",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="baseline «Qwen без RAG»: без контекста и порога (подразумевает --llm)",
    )
    parser.add_argument(
        "--correctness", action="store_true",
        help="judge-правильность ответов относительно gold (нужен --llm)",
    )
    args = parser.parse_args()
    if args.judge or args.baseline or args.correctness:
        args.llm = True

    cfg = load_config()
    engine = RagEngine.from_config(cfg)
    k = args.k or cfg.retrieval.top_k
    tau = args.tau if args.tau is not None else cfg.retrieval.threshold

    # ------------------------------------------------------ baseline (no RAG)
    if args.baseline:
        qa_path = Path(args.qa) if args.qa else cfg.paths.qa_dir / "test.json"
        items = load_qa(qa_path)
        print(f"Baseline (Qwen без RAG): {qa_path} ({len(items)} вопросов)")
        refusal_rows: list[dict] = []
        latencies: list[float] = []
        correctness: list[int] = []
        for i, item in enumerate(items, 1):
            t0 = time.perf_counter()
            answer = engine.llm.complete(build_baseline_messages(item["question"]))
            latencies.append((time.perf_counter() - t0) * 1000)
            refused = engine.answer_is_refusal(answer)
            refusal_rows.append(
                {"expect_refusal": item["expect_refusal"], "refused": refused}
            )
            status = f"refused({latencies[-1]:.0f} ms)" if refused else \
                f"answer({latencies[-1]:.0f} ms)"
            if args.judge and not item["expect_refusal"]:
                if refused:
                    correctness.append(0)
                else:
                    correctness.append(correctness_score(
                        item["question"], item.get("gold_answer", ""),
                        answer, engine.llm,
                    ))
            print(f"  [{i}/{len(items)}] {item['id']}: {status}")
        report = {
            "params": {
                "qa": str(qa_path), "mode": "baseline_no_rag",
                "llm": cfg.models.llm, "n_items": len(items),
            },
            "refusal": refusal_metrics(refusal_rows),
            "generation": {
                "generated": len(items) - sum(r["refused"] for r in refusal_rows),
                "model_refusals": sum(r["refused"] for r in refusal_rows),
                "latency_ms": {
                    "p50": percentile(latencies, 50),
                    "p95": percentile(latencies, 95),
                },
            },
        }
        if args.judge:
            report["generation"]["correctness_mean"] = (
                round(sum(correctness) / len(correctness) / 2, 4)
                if correctness else None
            )
            report["generation"]["correctness_n"] = len(correctness)
        out_path = Path(args.out) if args.out else \
            cfg.paths.reports_dir / "metrics-baseline.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Отчёт: {out_path}")
        return

    # ---------------------------------------------------------- калибровка
    if args.calibrate:
        qa_path = Path(args.qa) if args.qa else cfg.paths.qa_dir / "dev.json"
        items = load_qa(qa_path)
        print(f"Калибровка tau: {qa_path} ({len(items)} вопросов), top_k={k}")
        scores: list[dict] = []
        for i, item in enumerate(items, 1):
            prepared = engine.prepare(item["question"], top_k=k)
            scores.append({
                "score": round(prepared.decision.max_score, 4),
                "expect_refusal": item["expect_refusal"],
            })
            print(f"  [{i}/{len(items)}] {item['id']}: score={scores[-1]['score']:.3f}")
        table, best_tau = calibrate_threshold(scores)
        for row in table:
            mark = " <== лучший" if row["tau"] == best_tau else ""
            if row["f1"] > 0 or mark:
                print(f"  tau={row['tau']:.2f} correct={row['correct_refusal']} "
                      f"false={row['false_refusal']} f1={row['f1']}{mark}")
        out = cfg.paths.reports_dir / "tau-calibration.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"qa": str(qa_path), "k": k, "best_tau": best_tau,
                        "scores": scores, "grid": table},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Лучший tau={best_tau} (перенесите в config.yaml: "
              f"retrieval.threshold). Таблица: {out}")
        return

    qa_path = Path(args.qa) if args.qa else cfg.paths.qa_dir / "test.json"
    items = load_qa(qa_path)
    out_path = Path(args.out) if args.out else cfg.paths.reports_dir / "metrics.json"

    print(f"QA: {qa_path} ({len(items)} вопросов), top_k={k}, tau={tau}")

    retrieval_rows: list[dict] = []
    refusal_rows: list[dict] = []
    latencies: list[float] = []
    faithfulness: list[int] = []
    correctness: list[int] = []
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
            if args.correctness and not item["expect_refusal"]:
                correctness.append(correctness_score(
                    q, item.get("gold_answer", ""), answer, engine.llm,
                ))
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
    if args.correctness:
        report["generation"]["correctness_mean"] = (
            round(sum(correctness) / len(correctness) / 2, 4) if correctness else None
        )
        report["generation"]["correctness_n"] = len(correctness)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Отчёт: {out_path}")


if __name__ == "__main__":
    main()
