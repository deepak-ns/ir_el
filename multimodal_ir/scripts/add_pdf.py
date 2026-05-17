#!/usr/bin/env python3
# scripts/add_pdf.py
"""
Add a single PDF of your choice to the corpus and re-index it.

Usage:
    python scripts/add_pdf.py --pdf path/to/paper.pdf
    python scripts/add_pdf.py --pdf paper.pdf --title "My Paper" --authors "Alice, Bob" --year 2024
    python scripts/add_pdf.py --pdf paper.pdf   # title/authors auto-extracted from PDF
"""

import sys
import uuid
import json
import numpy as np
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db, get_session, Paper, Figure, DocumentPage
from src.ingestion.pdf_parser import parse_pdf
from src.embedding.text_encoder import TextEncoder
from src.embedding.figure_encoder import FigureEncoder
from src.retrieval.faiss_index import DualFAISSIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.config import cfg
from loguru import logger


@click.command()
@click.option("--pdf",     required=True, help="Path to your PDF file")
@click.option("--title",   default=None,  help="Paper title (auto-extracted if not given)")
@click.option("--authors", default=None,  help="Comma-separated author names")
@click.option("--year",    default=None,  type=int, help="Publication year")
@click.option("--paper-id",default=None,  help="Custom ID (auto-generated if not given)")
def main(pdf, title, authors, year, paper_id):
    pdf_path = Path(pdf)
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        logger.error("Only PDF files are supported")
        sys.exit(1)

    # Generate a unique paper_id if not given
    paper_id = paper_id or f"user_{uuid.uuid4().hex[:8]}"
    authors_list = [a.strip() for a in authors.split(",")] if authors else []

    logger.info(f"Processing: {pdf_path.name}  →  paper_id={paper_id}")

    # ── Step 1: Parse PDF ────────────────────────────────────────────────────
    init_db()
    session = get_session()

    # Check if already exists
    existing = session.query(Paper).filter_by(paper_id=paper_id).first()
    if existing:
        logger.warning(f"paper_id={paper_id} already in DB. Use --paper-id to assign a different one.")
        session.close()
        sys.exit(1)

    metadata = {
        "title":   title,
        "authors": authors_list,
        "year":    year,
    }

    parsed = parse_pdf(
        pdf_path=str(pdf_path),
        paper_id=paper_id,
        metadata=metadata,
    )

    # Override with user-supplied values where provided
    if title:
        parsed.title = title
    if authors_list:
        parsed.authors = authors_list
    if year:
        parsed.year = year

    logger.info(f"Title:    {parsed.title}")
    logger.info(f"Abstract: {parsed.abstract[:120]}...")
    logger.info(f"Figures extracted: {len(parsed.figures)}")

    # ── Step 2: Store in DB ──────────────────────────────────────────────────
    paper_record = Paper(
        paper_id=parsed.paper_id,
        title=parsed.title,
        abstract=parsed.abstract,
        authors=parsed.authors,
        year=parsed.year,
        venue=parsed.venue,
        doi=parsed.doi,
        pdf_path=str(pdf_path),
    )
    session.add(paper_record)
    session.flush()

    for page in parsed.pages:
        session.add(DocumentPage(
            paper_id=parsed.paper_id,
            page_number=page.page_number,
            text=page.text,
            image_path=page.image_path,
        ))

    for fig in parsed.figures:
        fig_record = Figure(
            paper_id=parsed.paper_id,
            figure_number=fig.figure_number,
            caption=fig.caption,
            image_path=fig.image_path,
            page_number=fig.page_number,
            width=fig.width,
            height=fig.height,
        )
        session.add(fig_record)
    session.flush()

    session.commit()
    logger.success(f"Stored paper and {len(parsed.figures)} figures in DB")

    # ── Step 3: Embed and add to FAISS indices ───────────────────────────────
    # Reload existing indices
    dual_index = DualFAISSIndex()
    index_dir = Path(cfg.data.indices_dir)

    text_index_exists   = (index_dir / "text_index.faiss").exists()
    figure_index_exists = (index_dir / "figure_index.faiss").exists()

    if not text_index_exists:
        logger.warning("No existing FAISS index found. Building from scratch...")
        _build_full_index(session, dual_index)
    else:
        dual_index.load()
        _append_to_index(dual_index, paper_record, parsed, session)

    # ── Step 4: Rebuild BM25 (it doesn't support incremental updates) ────────
    logger.info("Rebuilding BM25 index...")
    session = get_session()
    all_pages = session.query(DocumentPage).filter(DocumentPage.text.isnot(None)).all()
    session.close()

    bm25 = BM25Retriever()
    bm25.build(
        abstracts=[p.text or "" for p in all_pages],
        paper_ids=[str(p.id) for p in all_pages],
    )
    bm25.save(BM25Retriever.default_path())

    logger.success(f"\n✓ '{parsed.title}' is now searchable!")
    logger.success(f"  Restart the API (python scripts/run.py) and search for it.")


def _append_to_index(dual_index, paper_record, parsed, session):
    """Add a single paper's vectors to existing FAISS indices."""
    import faiss
    import numpy as np

    # -- Text embeddings for every PDF page
    text_encoder = TextEncoder()
    session = get_session()
    pages = session.query(DocumentPage).filter_by(paper_id=paper_record.paper_id).order_by(DocumentPage.page_number).all()
    page_texts = [p.text or "" for p in pages]
    if not page_texts:
        page_texts = [parsed.abstract or parsed.title or ""]
    vecs = text_encoder.encode(page_texts)

    for i, page in enumerate(pages):
        new_text_row = dual_index.text_index.index.ntotal
        dual_index.text_index.index.add(vecs[i:i+1])
        dual_index.text_index.id_map.append(page.id)
        page.text_faiss_id = new_text_row

    db_paper = session.query(Paper).filter_by(paper_id=paper_record.paper_id).first()
    if db_paper and pages:
        db_paper.text_faiss_id = pages[0].text_faiss_id
    session.commit()
    session.close()
    logger.info(f"Added {len(pages)} page text vectors")

    # -- Figure embeddings
    valid_figs = session.query(Figure).filter_by(paper_id=parsed.paper_id).all()
    valid_figs = [f for f in valid_figs if f.image_path and Path(f.image_path).exists()]
    if valid_figs:
        figure_encoder = FigureEncoder()
        fig_vecs = figure_encoder.encode_images([f.image_path for f in valid_figs])

        session = get_session()
        for i, fig in enumerate(valid_figs):
            new_fig_row = dual_index.figure_index.index.ntotal
            dual_index.figure_index.index.add(fig_vecs[i:i+1])
            dual_index.figure_index.id_map.append(fig.id)

            # Update figure record
            from src.db import Figure as FigureModel
            db_fig = session.query(FigureModel).get(fig.id)
            if db_fig:
                db_fig.figure_faiss_id = new_fig_row
        session.commit()
        session.close()
        logger.info(f"Added {len(valid_figs)} figure vectors")

    # Save updated indices
    dual_index.save()
    logger.success("FAISS indices updated and saved")


def _build_full_index(session, dual_index):
    """Full index build (used when no index exists yet)."""
    from scripts.build_index import main as build_main
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(build_main, [])
    if result.exit_code != 0:
        logger.error(f"Index build failed: {result.output}")
        sys.exit(1)


if __name__ == "__main__":
    main()
