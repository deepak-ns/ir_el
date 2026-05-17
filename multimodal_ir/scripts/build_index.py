#!/usr/bin/env python3
# scripts/build_index.py
"""
Step 3: Build FAISS text + figure indices and BM25 index from the SQLite corpus.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --text-only      # Skip figure encoding (faster)
    python scripts/build_index.py --figure-only
"""

import sys
import json
import numpy as np
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_session, Paper, Figure, DocumentPage, init_db
from src.embedding.text_encoder import TextEncoder
from src.embedding.figure_encoder import FigureEncoder
from src.retrieval.faiss_index import DualFAISSIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.config import cfg
from loguru import logger


@click.command()
@click.option("--text-only",   is_flag=True, help="Build only text (SBERT) index")
@click.option("--figure-only", is_flag=True, help="Build only figure (CLIP) index")
@click.option("--skip-bm25",   is_flag=True, help="Skip BM25 index")
def main(text_only, figure_only, skip_bm25):
    init_db()
    session = get_session()

    papers  = session.query(Paper).all()
    pages   = session.query(DocumentPage).filter(DocumentPage.text.isnot(None)).all()
    figures = session.query(Figure).filter(Figure.image_path.isnot(None)).all()
    session.close()

    if not papers:
        logger.error("No papers in DB. Run build_corpus.py first.")
        sys.exit(1)

    logger.info(f"Found {len(papers)} PDFs, {len(pages)} pages, {len(figures)} figures in DB")

    dual_index = DualFAISSIndex()

    # ── TEXT INDEX ──────────────────────────────────────────────────────────
    if not figure_only:
        logger.info("=== Building page text index (SBERT) ===")
        text_encoder = TextEncoder()

        if not pages:
            logger.warning("No page rows found; falling back to one text record per existing paper")
            session = get_session()
            for p in papers:
                page = DocumentPage(
                    paper_id=p.paper_id,
                    page_number=1,
                    text=p.abstract or p.title or "",
                    image_path="",
                )
                session.add(page)
            session.commit()
            pages = session.query(DocumentPage).filter(DocumentPage.text.isnot(None)).all()
            session.close()

        page_texts = [p.text or "" for p in pages]
        page_db_ids = [p.id for p in pages]

        vectors = text_encoder.encode(page_texts)

        # Save raw vectors
        vec_path = f"{cfg.data.indices_dir}/text_embeddings.npy"
        Path(cfg.data.indices_dir).mkdir(parents=True, exist_ok=True)
        np.save(vec_path, vectors)

        # Update page records with FAISS row ids
        session = get_session()
        for i, page in enumerate(pages):
            db_page = session.query(DocumentPage).get(page.id)
            if db_page:
                db_page.text_faiss_id = i
        session.commit()
        session.close()

        dual_index.build_text_index(vectors, page_db_ids)
        logger.success("Page text index built")

    # ── FIGURE INDEX ────────────────────────────────────────────────────────
    if not text_only and figures:
        logger.info("=== Building figure index (CLIP) ===")
        figure_encoder = FigureEncoder()

        # Filter to figures with existing image files
        valid_figures = [
            f for f in figures
            if f.image_path and Path(f.image_path).exists()
        ]
        logger.info(f"{len(valid_figures)} figures with valid image files")

        if valid_figures:
            image_paths  = [f.image_path for f in valid_figures]
            figure_db_ids = [f.id for f in valid_figures]

            fig_vectors = figure_encoder.encode_images(image_paths)

            # Update figure records with FAISS row ids
            session = get_session()
            for i, fig in enumerate(valid_figures):
                db_fig = session.query(Figure).get(fig.id)
                if db_fig:
                    db_fig.figure_faiss_id = i
            session.commit()
            session.close()

            dual_index.build_figure_index(fig_vectors, figure_db_ids)
            logger.success("Figure index built")
        else:
            logger.warning("No valid figure images found — building empty figure index")
            # Build a dummy figure index so the search engine loads without error
            dummy_vecs = np.zeros((1, cfg.embedding.figure.dim), dtype=np.float32)
            dual_index.build_figure_index(dummy_vecs, [0])

    elif not text_only:
        logger.warning("No figures in DB — building placeholder figure index")
        dummy_vecs = np.zeros((1, cfg.embedding.figure.dim), dtype=np.float32)
        dual_index.build_figure_index(dummy_vecs, [0])

    # ── SAVE FAISS ──────────────────────────────────────────────────────────
    dual_index.save()
    logger.success(f"FAISS indices saved → {cfg.data.indices_dir}/")

    # ── BM25 ────────────────────────────────────────────────────────────────
    if not skip_bm25:
        logger.info("=== Building page BM25 index ===")
        bm25 = BM25Retriever()
        session = get_session()
        pages_for_bm25 = session.query(DocumentPage).filter(DocumentPage.text.isnot(None)).all()
        session.close()

        bm25.build(
            abstracts=[p.text or "" for p in pages_for_bm25],
            paper_ids=[str(p.id) for p in pages_for_bm25],
        )
        bm25.save(BM25Retriever.default_path())
        logger.success("BM25 index saved")

    logger.success("All indices ready. Run: python scripts/run.py")


if __name__ == "__main__":
    main()
