# src/fusion/score_fusion.py
"""
Fuses page-level text, page-level BM25, and optional figure retrieval scores.

Text search now indexes complete PDFs page by page. The fused result is still
returned at the PDF level, with the strongest matching page attached for UI
display.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from src.retrieval.faiss_index import SearchResult
from src.retrieval.bm25_retriever import BM25Result
from src.db import get_session, Paper, Figure, DocumentPage
from src.config import cfg


RRF_K = 60


@dataclass
class FusedResult:
    paper_id: str
    paper: dict
    fused_score: float
    text_score: float
    figure_score: float
    bm25_score: float
    matched_page: dict | None = None
    matched_figures: list[dict] = field(default_factory=list)
    rank: int = 0


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _remember_best_page(
    paper_best_page: dict[str, tuple[int, float]],
    page_scores: dict[int, float],
    page: DocumentPage,
    score: float,
):
    page_scores[page.id] += score
    current = paper_best_page.get(page.paper_id)
    if current is None or page_scores[page.id] > current[1]:
        paper_best_page[page.paper_id] = (page.id, page_scores[page.id])


def fuse_results(
    text_results: list[SearchResult],
    figure_results: list[SearchResult],
    bm25_results: list[BM25Result],
    alpha: float = None,
    beta: float = None,
    gamma: float = 0.2,
    top_k: int = None,
    session=None,
) -> list[FusedResult]:
    """
    Fuse retriever outputs into ranked PDF results.

    Text FAISS IDs and BM25 IDs are DocumentPage database IDs. Figure IDs remain
    Figure database IDs. Scores are accumulated per PDF, while retaining the best
    page that produced the match.
    """
    alpha = alpha if alpha is not None else cfg.fusion.default_alpha
    beta = beta if beta is not None else cfg.fusion.default_beta
    top_k = top_k or cfg.retrieval.final_k

    total_w = alpha + beta + gamma
    alpha /= total_w
    beta /= total_w
    gamma /= total_w

    close_session = session is None
    if session is None:
        session = get_session()

    paper_text_score: dict[str, float] = defaultdict(float)
    paper_figure_score: dict[str, float] = defaultdict(float)
    paper_bm25_score: dict[str, float] = defaultdict(float)
    page_scores: dict[int, float] = defaultdict(float)
    paper_best_page: dict[str, tuple[int, float]] = {}

    if text_results:
        page_ids = [r.faiss_id for r in text_results]
        pages = session.query(DocumentPage).filter(DocumentPage.id.in_(page_ids)).all()
        id_to_page = {p.id: p for p in pages}
        for r in text_results:
            page = id_to_page.get(r.faiss_id)
            if not page:
                continue
            score = _rrf_score(r.rank)
            paper_text_score[page.paper_id] += score
            _remember_best_page(paper_best_page, page_scores, page, score)

    if figure_results:
        fig_ids = [r.faiss_id for r in figure_results]
        figures = session.query(Figure).filter(Figure.id.in_(fig_ids)).all()
        figure_id_to_paper = {f.id: f.paper_id for f in figures}
        for r in figure_results:
            pid = figure_id_to_paper.get(r.faiss_id)
            if pid:
                paper_figure_score[pid] += _rrf_score(r.rank)

    bm25_page_ids = []
    legacy_bm25_paper_ids = []
    for r in bm25_results:
        try:
            bm25_page_ids.append(int(r.paper_id))
        except (TypeError, ValueError):
            legacy_bm25_paper_ids.append(r)

    if bm25_page_ids:
        pages = session.query(DocumentPage).filter(DocumentPage.id.in_(bm25_page_ids)).all()
        id_to_page = {p.id: p for p in pages}
        for r in bm25_results:
            try:
                page_id = int(r.paper_id)
            except (TypeError, ValueError):
                continue
            page = id_to_page.get(page_id)
            if not page:
                continue
            score = _rrf_score(r.rank)
            paper_bm25_score[page.paper_id] += score
            _remember_best_page(paper_best_page, page_scores, page, score)

    for r in legacy_bm25_paper_ids:
        paper_bm25_score[r.paper_id] += _rrf_score(r.rank)

    all_paper_ids = (
        set(paper_text_score.keys())
        | set(paper_figure_score.keys())
        | set(paper_bm25_score.keys())
    )

    scored = []
    for pid in all_paper_ids:
        ts = paper_text_score.get(pid, 0.0)
        fs = paper_figure_score.get(pid, 0.0)
        bs = paper_bm25_score.get(pid, 0.0)
        fused = alpha * ts + beta * fs + gamma * bs
        scored.append((pid, fused, ts, fs, bs))

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:top_k]

    top_paper_ids = [s[0] for s in scored]
    papers_q = session.query(Paper).filter(Paper.paper_id.in_(top_paper_ids)).all()
    paper_map = {p.paper_id: p for p in papers_q}

    matched_figure_ids = {r.faiss_id for r in figure_results}
    all_matched_figs = session.query(Figure).filter(Figure.id.in_(matched_figure_ids)).all()
    paper_to_figs: dict[str, list[Figure]] = defaultdict(list)
    for fig in all_matched_figs:
        paper_to_figs[fig.paper_id].append(fig)

    best_page_ids = [v[0] for v in paper_best_page.values()]
    pages_q = session.query(DocumentPage).filter(DocumentPage.id.in_(best_page_ids)).all() if best_page_ids else []
    page_map = {p.id: p for p in pages_q}

    results = []
    for rank, (pid, fused, ts, fs, bs) in enumerate(scored, start=1):
        paper_obj = paper_map.get(pid)
        if not paper_obj:
            continue

        page_id = paper_best_page.get(pid, (None, 0.0))[0]
        page_obj = page_map.get(page_id)
        matched_page = page_obj.to_dict() if page_obj else None
        if matched_page and page_obj:
            matched_page["score"] = round(page_scores.get(page_obj.id, 0.0), 6)

        results.append(FusedResult(
            paper_id=pid,
            paper=paper_obj.to_dict(),
            fused_score=round(fused, 6),
            text_score=round(ts, 6),
            figure_score=round(fs, 6),
            bm25_score=round(bs, 6),
            matched_page=matched_page,
            matched_figures=[f.to_dict() for f in paper_to_figs.get(pid, [])],
            rank=rank,
        ))

    if close_session:
        session.close()

    logger.debug(f"Fused {len(all_paper_ids)} candidates into top {len(results)}")
    return results
