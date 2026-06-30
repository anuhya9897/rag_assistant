"""
Evaluate RAG retrieval (precision@k / recall@k) and optionally answer token P/R/F1 vs gold_answer.

  cd rag_assistant   # repository root
  python scripts/eval_rag.py --eval eval/example_gold.json -k 5
  python scripts/eval_rag.py --eval eval/example_gold.json -k 5 --with-llm --model llama3.1:8b

Requires kb.duckdb and sentence-transformers. --with-llm also requires Ollama.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean

# Repo root = parent of scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from backend.app.eval_metrics import (  # noqa: E402
    answer_token_precision_recall_f1,
    retrieval_precision_recall_at_k,
)
from backend.app.rag_service import DEFAULT_LLM, ask, retrieve  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="RAG eval: retrieval P/R@k and optional answer F1")
    p.add_argument("--eval", default="eval/example_gold.json", help="Path to gold JSON")
    p.add_argument("-k", type=int, default=5, help="Top-k for retrieval metrics")
    p.add_argument("--similarity-min", type=float, default=None, help="Optional cosine floor")
    p.add_argument("--with-llm", action="store_true", help="Run ask() for items with gold_answer")
    p.add_argument("--model", default=DEFAULT_LLM, help="Ollama model when --with-llm")
    p.add_argument("--output", default=None, help="Write aggregate metrics JSON to this path")
    args = p.parse_args()

    with open(os.path.join(_PROJECT_ROOT, args.eval), encoding="utf-8") as f:
        spec = json.load(f)
    items = spec.get("items") or []
    if not items:
        print("No items in eval file.", file=sys.stderr)
        return 1

    rows_out = []
    precs: list[float] = []
    recalls: list[float] = []
    ans_precs: list[float] = []
    ans_recalls: list[float] = []
    ans_f1s: list[float] = []

    for it in items:
        qid = it.get("id", "?")
        question = it.get("question") or ""
        gold_src = it.get("relevant_source_contains") or []
        gold_answer = (it.get("gold_answer") or "").strip()

        retrieved = retrieve(
            question,
            k=args.k,
            similarity_min=args.similarity_min,
        )
        rp, rr = retrieval_precision_recall_at_k(retrieved, gold_src, args.k)
        row = {
            "id": qid,
            "question": question,
            "retrieval_precision_at_k": rp,
            "retrieval_recall_at_k": rr,
        }
        if rp is not None and rr is not None:
            precs.append(rp)
            recalls.append(rr)

        if args.with_llm and gold_answer:
            out = ask(
                question=question,
                k=args.k,
                model=args.model,
                similarity_min=args.similarity_min,
            )
            pred = out.get("answer") or ""
            ap, ar, af1 = answer_token_precision_recall_f1(pred, gold_answer)
            row["answer_precision"] = ap
            row["answer_recall"] = ar
            row["answer_f1"] = af1
            ans_precs.append(ap)
            ans_recalls.append(ar)
            ans_f1s.append(af1)
        elif gold_answer and not args.with_llm:
            row["answer_precision"] = None
            row["answer_recall"] = None
            row["answer_f1"] = None
            row["answer_note"] = "gold_answer present; re-run with --with-llm to score answers"

        rows_out.append(row)
        pr = "n/a" if rp is None else f"{rp:.3f}"
        re = "n/a" if rr is None else f"{rr:.3f}"
        line = f"[{qid}] retrieval P@{args.k}={pr} R@{args.k}={re}"
        if row.get("answer_f1") is not None:
            line += f" | answer P={row['answer_precision']:.3f} R={row['answer_recall']:.3f} F1={row['answer_f1']:.3f}"
        print(line)

    summary = {
        "eval_file": args.eval,
        "k": args.k,
        "with_llm": args.with_llm,
        "model": args.model if args.with_llm else None,
        "count": len(items),
        "macro_retrieval_precision_at_k": mean(precs) if precs else None,
        "macro_retrieval_recall_at_k": mean(recalls) if recalls else None,
        "macro_answer_precision": mean(ans_precs) if ans_precs else None,
        "macro_answer_recall": mean(ans_recalls) if ans_recalls else None,
        "macro_answer_f1": mean(ans_f1s) if ans_f1s else None,
        "per_item": rows_out,
    }
    print("---")
    print(
        f"macro retrieval P@{args.k}={summary['macro_retrieval_precision_at_k']!s} "
        f"R@{args.k}={summary['macro_retrieval_recall_at_k']!s}"
    )
    if ans_f1s:
        print(
            f"macro answer token P={summary['macro_answer_precision']:.3f} "
            f"R={summary['macro_answer_recall']:.3f} F1={summary['macro_answer_f1']:.3f}"
        )

    if args.output:
        out_path = args.output if os.path.isabs(args.output) else os.path.join(_PROJECT_ROOT, args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
