# src/evaluation/metrics.py
"""
Standard IR evaluation metrics.

All functions follow the TREC convention:
  - relevance_judgments: dict mapping query_id → set of relevant paper_ids
  - retrieved: list of paper_ids in ranked order (best first)

Metrics implemented:
  - MRR@k     : Mean Reciprocal Rank
  - NDCG@k    : Normalized Discounted Cumulative Gain
  - Recall@k  : Fraction of relevant docs retrieved in top-k
  - Precision@k
  - Cross-modal Hit Rate (custom metric for this project)
"""

import math
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from loguru import logger


@dataclass
class EvalResult:
    mrr_at_k:         dict[int, float]
    ndcg_at_k:        dict[int, float]
    recall_at_k:      dict[int, float]
    precision_at_k:   dict[int, float]
    cross_modal_hr:   float | None   # None if no cross-modal queries in test set
    avg_latency_ms:   float
    num_queries:      int


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of first relevant document, 0 if none found."""
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for d in retrieved_k if d in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for d in retrieved_k if d in relevant)
    return hits / len(relevant)


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain at k (binary relevance)."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ideal_dcg_at_k(relevant: set[str], k: int) -> float:
    """Ideal DCG: all relevant docs at top positions."""
    n_rel = min(len(relevant), k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    idcg = ideal_dcg_at_k(relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


def evaluate(
    queries: list[dict],
    relevance_judgments: dict[str, set[str]],
    retrieval_fn,
    k_values: list[int] = None,
    latencies: list[float] = None,
) -> EvalResult:
    """
    Run full evaluation over a test query set.

    queries: list of dicts with keys 'query_id', 'text', 'query_type'
    relevance_judgments: {query_id: {relevant_paper_id, ...}}
    retrieval_fn: callable(query) → list[str] of paper_ids in ranked order
    """
    k_values = k_values or [1, 5, 10]
    latencies = latencies or []

    mrr_scores:       dict[int, list[float]] = defaultdict(list)
    ndcg_scores:      dict[int, list[float]] = defaultdict(list)
    recall_scores:    dict[int, list[float]] = defaultdict(list)
    precision_scores: dict[int, list[float]] = defaultdict(list)
    cross_modal_hits: list[float] = []

    for q in queries:
        qid = q["query_id"]
        relevant = relevance_judgments.get(qid, set())
        if not relevant:
            continue

        retrieved = retrieval_fn(q)  # list of paper_ids

        for k in k_values:
            mrr_scores[k].append(reciprocal_rank(retrieved[:k], relevant))
            ndcg_scores[k].append(ndcg_at_k(retrieved, relevant, k))
            recall_scores[k].append(recall_at_k(retrieved, relevant, k))
            precision_scores[k].append(precision_at_k(retrieved, relevant, k))

        # Cross-modal hit rate: for cross_modal queries, did the correct paper appear in top-10?
        if q.get("query_type") == "cross_modal":
            hit = 1.0 if any(pid in relevant for pid in retrieved[:10]) else 0.0
            cross_modal_hits.append(hit)

    def avg(scores: dict[int, list]) -> dict[int, float]:
        return {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in scores.items()}

    return EvalResult(
        mrr_at_k=avg(mrr_scores),
        ndcg_at_k=avg(ndcg_scores),
        recall_at_k=avg(recall_scores),
        precision_at_k=avg(precision_scores),
        cross_modal_hr=(
            round(sum(cross_modal_hits) / len(cross_modal_hits), 4)
            if cross_modal_hits else None
        ),
        avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        num_queries=len(queries),
    )


def print_eval_report(result: EvalResult):
    """Pretty-print evaluation results to console."""
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Queries evaluated: {result.num_queries}")
    print(f"Avg latency:       {result.avg_latency_ms} ms")
    print()
    for k in sorted(result.mrr_at_k.keys()):
        print(f"  MRR@{k:<3}      = {result.mrr_at_k[k]:.4f}")
    print()
    for k in sorted(result.ndcg_at_k.keys()):
        print(f"  NDCG@{k:<3}     = {result.ndcg_at_k[k]:.4f}")
    print()
    for k in sorted(result.recall_at_k.keys()):
        print(f"  Recall@{k:<3}   = {result.recall_at_k[k]:.4f}")
    print()
    for k in sorted(result.precision_at_k.keys()):
        print(f"  Precision@{k:<3} = {result.precision_at_k[k]:.4f}")
    if result.cross_modal_hr is not None:
        print()
        print(f"  Cross-modal hit rate@10 = {result.cross_modal_hr:.4f}")
    print("=" * 50)
