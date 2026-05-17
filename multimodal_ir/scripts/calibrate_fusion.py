#!/usr/bin/env python3
# scripts/calibrate_fusion.py
"""
Calibrates the optimal α and β fusion weights for each query type
using a grid search over a held-out validation set.

This is the 'novel contribution' part of the project —
most IR systems use fixed weights; we learn them per query type.

Usage:
    python scripts/calibrate_fusion.py
    python scripts/calibrate_fusion.py --metric ndcg --k 10
"""

import sys
import json
import itertools
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.search_engine import SearchEngine
from src.evaluation.metrics import evaluate
from src.db import get_session, Paper, Figure
from src.config import cfg
from loguru import logger


ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BETA_VALUES  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _load_or_generate_queries(query_type: str, n: int) -> tuple[list[dict], dict]:
    """Load test queries of a specific type."""
    from scripts.evaluate import _auto_generate_test_queries, _auto_generate_cross_modal_queries

    if query_type == "text":
        return _auto_generate_test_queries(n)
    elif query_type == "cross_modal":
        return _auto_generate_cross_modal_queries(n)
    else:
        raise ValueError(f"Unknown query type: {query_type}")


@click.command()
@click.option("--metric",    default="ndcg",  type=click.Choice(["ndcg", "mrr", "recall"]))
@click.option("--k",         default=10,      help="Cut-off for metric (e.g. NDCG@k)")
@click.option("--n-queries", default=30,      help="Validation queries per type")
@click.option("--output",    default="configs/calibrated_weights.json")
def main(metric, k, n_queries, output):
    engine = SearchEngine.get()
    engine.load()

    results_by_type = {}

    for qtype in ["text", "cross_modal"]:
        logger.info(f"\n=== Calibrating for query type: {qtype} ===")
        queries, judgments = _load_or_generate_queries(qtype, n_queries)

        if not queries:
            logger.warning(f"No {qtype} queries available, skipping")
            continue

        best_score  = -1.0
        best_alpha  = cfg.fusion.default_alpha
        best_beta   = cfg.fusion.default_beta

        grid_results = []

        for alpha, beta in itertools.product(ALPHA_VALUES, BETA_VALUES):
            # Constrain: alpha + beta <= 1.0 (gamma=BM25 takes the rest)
            if alpha + beta > 1.01:
                continue

            def rfn(q, _a=alpha, _b=beta):
                r = engine.search(text=q["text"], alpha=_a, beta=_b)
                return [x.paper_id for x in r.results]

            res = evaluate(
                queries=queries,
                relevance_judgments=judgments,
                retrieval_fn=rfn,
                k_values=[k],
            )

            score = {
                "ndcg":   res.ndcg_at_k.get(k, 0),
                "mrr":    res.mrr_at_k.get(k, 0),
                "recall": res.recall_at_k.get(k, 0),
            }[metric]

            grid_results.append({"alpha": alpha, "beta": beta, "score": round(score, 4)})

            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_beta  = beta

        logger.success(
            f"[{qtype}] Best {metric.upper()}@{k} = {best_score:.4f} "
            f"at α={best_alpha:.1f}, β={best_beta:.1f}"
        )
        results_by_type[qtype] = {
            "best_alpha": best_alpha,
            "best_beta":  best_beta,
            f"best_{metric}@{k}": best_score,
            "grid": grid_results,
        }

    # Save calibrated weights
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results_by_type, f, indent=2)

    logger.success(f"Calibrated weights saved → {output}")
    print("\n=== CALIBRATED WEIGHTS ===")
    for qtype, res in results_by_type.items():
        print(f"  {qtype}: α={res['best_alpha']}, β={res['best_beta']}")
    print("\nUpdate configs/config.yaml with these values for best performance.")


if __name__ == "__main__":
    main()
