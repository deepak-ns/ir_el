#!/usr/bin/env python3
# scripts/evaluate.py
"""
Step 4 (optional but recommended for report): Run evaluation over a test query set.

Creates test_queries.json and relevance_judgments.json automatically if they don't exist,
using papers already in the DB (for a quick sanity-check evaluation).

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --queries data/test_queries.json
    python scripts/evaluate.py --compare-modes    # Compare text-only vs figure-only vs hybrid
"""

import sys
import json
import time
import click
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.search_engine import SearchEngine
from src.evaluation.metrics import evaluate, print_eval_report
from src.db import get_session, Paper
from src.config import cfg
from loguru import logger


def _auto_generate_test_queries(n: int = 50) -> tuple[list[dict], dict]:
    """
    Auto-generate test queries from DB papers for a quick evaluation.
    Strategy: use the first 20 words of each abstract as the query;
    the correct paper is the only relevant doc (closed-set eval).
    Not ideal for real IR eval, but sufficient for a course demo.
    """
    session = get_session()
    papers = session.query(Paper).all()
    session.close()

    if not papers:
        logger.error("No papers in DB")
        sys.exit(1)

    random.shuffle(papers)
    selected = papers[:min(n, len(papers))]

    queries = []
    judgments = {}

    for i, paper in enumerate(selected):
        abstract = paper.abstract or ""
        # Use first 15 words as query
        words = abstract.split()[:15]
        query_text = " ".join(words)

        qid = f"q_{i:04d}"
        queries.append({
            "query_id": qid,
            "text": query_text,
            "query_type": "text",
            "expected_paper_id": paper.paper_id,
        })
        judgments[qid] = {paper.paper_id}

    return queries, judgments


def _auto_generate_cross_modal_queries(n: int = 20) -> tuple[list[dict], dict]:
    """Generate synthetic cross-modal queries using figure captions."""
    from src.db import Figure
    session = get_session()
    figures = session.query(Figure).filter(
        Figure.caption.isnot(None),
        Figure.caption != "Figure 1",
    ).limit(n * 2).all()
    session.close()

    queries = []
    judgments = {}
    seen_papers = set()

    for fig in figures:
        if fig.paper_id in seen_papers:
            continue
        caption = fig.caption or ""
        if len(caption) < 20:
            continue

        qid = f"cm_{len(queries):04d}"
        queries.append({
            "query_id": qid,
            "text": caption[:100],
            "query_type": "cross_modal",
            "expected_paper_id": fig.paper_id,
        })
        judgments[qid] = {fig.paper_id}
        seen_papers.add(fig.paper_id)

        if len(queries) >= n:
            break

    return queries, judgments


@click.command()
@click.option("--queries",       default=None,  help="Path to test_queries.json")
@click.option("--judgments",     default=None,  help="Path to relevance_judgments.json")
@click.option("--auto-generate", is_flag=True,  default=True, help="Auto-generate queries from DB")
@click.option("--compare-modes", is_flag=True,  help="Compare text-only / figure-only / hybrid modes")
@click.option("--n-queries",     default=50,    help="Number of auto-generated queries")
@click.option("--output",        default="data/eval_results.json", help="Save results JSON")
def main(queries, judgments, auto_generate, compare_modes, n_queries, output):
    # Load or auto-generate queries
    if queries and Path(queries).exists():
        with open(queries) as f:
            test_queries = json.load(f)
        with open(judgments) as f:
            relevance_judgments = {k: set(v) for k, v in json.load(f).items()}
        logger.info(f"Loaded {len(test_queries)} queries from {queries}")
    else:
        logger.info(f"Auto-generating {n_queries} test queries from DB...")
        test_queries, relevance_judgments = _auto_generate_test_queries(n_queries)

        # Add cross-modal queries
        cm_queries, cm_judgments = _auto_generate_cross_modal_queries(20)
        test_queries.extend(cm_queries)
        relevance_judgments.update(cm_judgments)

        # Save for reuse
        Path(cfg.evaluation.test_queries_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.evaluation.test_queries_path, "w") as f:
            json.dump(test_queries, f, indent=2)
        with open(cfg.evaluation.relevance_judgments_path, "w") as f:
            json.dump({k: list(v) for k, v in relevance_judgments.items()}, f, indent=2)
        logger.info(f"Saved {len(test_queries)} queries ({len(cm_queries)} cross-modal)")

    # Load search engine
    engine = SearchEngine.get()
    engine.load()

    # ── Hybrid evaluation (default) ─────────────────────────────────────────
    latencies = []

    def retrieval_fn(query: dict) -> list[str]:
        t0 = time.time()
        resp = engine.search(text=query["text"])
        latencies.append((time.time() - t0) * 1000)
        return [r.paper_id for r in resp.results]

    logger.info("Running hybrid evaluation...")
    result = evaluate(
        queries=test_queries,
        relevance_judgments=relevance_judgments,
        retrieval_fn=retrieval_fn,
        k_values=cfg.evaluation.k_values,
        latencies=latencies,
    )
    print("\n[HYBRID MODE]")
    print_eval_report(result)

    # ── Compare modes ────────────────────────────────────────────────────────
    if compare_modes:
        modes = [
            ("Text-only (α=1, β=0)",   1.0, 0.0),
            ("Figure-only (α=0, β=1)", 0.0, 1.0),
            ("Hybrid (α=0.6, β=0.4)",  0.6, 0.4),
        ]
        comparison = {}
        for label, a, b in modes:
            lat = []
            def rfn(q, _a=a, _b=b, _lat=lat):
                t0 = time.time()
                r = engine.search(text=q["text"], alpha=_a, beta=_b)
                _lat.append((time.time() - t0) * 1000)
                return [x.paper_id for x in r.results]

            res = evaluate(
                queries=test_queries,
                relevance_judgments=relevance_judgments,
                retrieval_fn=rfn,
                k_values=[10],
                latencies=lat,
            )
            comparison[label] = {
                "MRR@10":    res.mrr_at_k.get(10, 0),
                "NDCG@10":   res.ndcg_at_k.get(10, 0),
                "Recall@10": res.recall_at_k.get(10, 0),
                "Latency ms": res.avg_latency_ms,
            }
            print(f"\n[{label}]")
            print_eval_report(res)

        # Pretty comparison table
        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        header = f"{'Mode':<35} {'MRR@10':>8} {'NDCG@10':>8} {'Recall@10':>10} {'Latency':>9}"
        print(header)
        print("-" * 70)
        for label, metrics in comparison.items():
            print(
                f"{label:<35} "
                f"{metrics['MRR@10']:>8.4f} "
                f"{metrics['NDCG@10']:>8.4f} "
                f"{metrics['Recall@10']:>10.4f} "
                f"{metrics['Latency ms']:>8.1f}ms"
            )
        print("=" * 70)

    # Save results
    out_data = {
        "hybrid": {
            "mrr": result.mrr_at_k,
            "ndcg": result.ndcg_at_k,
            "recall": result.recall_at_k,
            "precision": result.precision_at_k,
            "cross_modal_hr": result.cross_modal_hr,
            "avg_latency_ms": result.avg_latency_ms,
            "num_queries": result.num_queries,
        }
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out_data, f, indent=2)
    logger.success(f"Evaluation results saved → {output}")


if __name__ == "__main__":
    main()
