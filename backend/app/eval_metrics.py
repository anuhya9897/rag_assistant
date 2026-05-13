"""
Evaluation helpers for RAG: retrieval precision/recall @k and answer token P/R/F1.

Retrieval: gold is a list of substrings; a retrieved row is relevant if its source path
contains any gold substring (case-insensitive). Recall = fraction of gold substrings
matched by at least one of the top-k rows. Precision@k = relevant rows in top-k / k.

Answer: multiset token overlap (word tokens) between model output and gold_answer reference.
"""
from __future__ import annotations

import re
from collections import Counter


def _token_counter(text: str) -> Counter[str]:
    return Counter(re.findall(r"\w+", (text or "").lower()))


def answer_token_precision_recall_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    """
    Token-level precision, recall, and F1 using multiset overlap (Counter &).
    """
    p = _token_counter(prediction)
    g = _token_counter(reference)
    pred_total = sum(p.values())
    gold_total = sum(g.values())
    if gold_total == 0:
        return (0.0, 0.0, 0.0)
    overlap = sum((p & g).values())
    if overlap == 0:
        return (0.0, 0.0, 0.0)
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return (precision, recall, f1)


def retrieval_precision_recall_at_k(
    rows: list[tuple],
    gold_substrings: list[str],
    k: int,
) -> tuple[float | None, float | None]:
    """
    rows: list of (source, chunk_id, text, score) from retrieve().
    gold_substrings: each string should appear in a relevant source path.
    Returns (precision@k, recall@k). (None, None) if gold_substrings is empty.
    """
    if not gold_substrings:
        return (None, None)
    topk = rows[:k]
    if not topk:
        return (0.0, 0.0)

    gold_l = [g.lower() for g in gold_substrings]

    def row_relevant(source: str) -> bool:
        s = (source or "").lower()
        return any(g in s for g in gold_l)

    rel_in_topk = sum(1 for row in topk if row_relevant(str(row[0])))
    precision = rel_in_topk / len(topk)

    matched_gold = 0
    for g in gold_l:
        for row in topk:
            if g in (str(row[0]) or "").lower():
                matched_gold += 1
                break
    recall = matched_gold / len(gold_substrings)
    return (precision, recall)
