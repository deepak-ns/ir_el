# src/ingestion/corpus_builder.py
"""
Orchestrates the full ingestion pipeline:
  1. Load paper metadata (from S2ORC API or local JSON)
  2. Parse each PDF for abstract + figures
  3. Store everything in SQLite via SQLAlchemy
"""

from pathlib import Path
from loguru import logger
from tqdm import tqdm

from src.db import init_db, get_session, Paper, Figure, DocumentPage
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.s2orc_loader import load_metadata
from src.config import cfg


def ingest_from_metadata(
    metadata_path: str,
    pdf_dir: str,
    db_path: str = None,
):
    """
    Main ingestion entry point.
    Reads metadata JSON, finds matching PDFs, parses and stores in DB.
    """
    engine = init_db(db_path)
    session = get_session(engine)

    papers_meta = load_metadata(metadata_path)
    logger.info(f"Starting ingestion of {len(papers_meta)} papers")

    ingested = 0
    skipped = 0

    for meta in tqdm(papers_meta, desc="Ingesting papers"):
        paper_id = meta["paperId"]

        # Skip if already in DB
        existing = session.query(Paper).filter_by(paper_id=paper_id).first()
        if existing:
            skipped += 1
            continue

        pdf_path = Path(pdf_dir) / f"{paper_id}.pdf"
        if not pdf_path.exists():
            # No PDF downloaded — still store metadata with abstract from API
            paper_record = Paper(
                paper_id=paper_id,
                title=meta.get("title", "Unknown"),
                abstract=meta.get("abstract", ""),
                authors=[a["name"] for a in meta.get("authors", [])],
                year=meta.get("year"),
                venue=meta.get("venue", ""),
                doi=meta.get("externalIds", {}).get("DOI"),
            )
            session.add(paper_record)
            session.flush()
            session.add(DocumentPage(
                paper_id=paper_id,
                page_number=1,
                text=meta.get("abstract", "") or meta.get("title", "Unknown"),
                image_path="",
            ))
            ingested += 1
            continue

        try:
            parsed = parse_pdf(
                pdf_path=str(pdf_path),
                paper_id=paper_id,
                metadata={
                    "title": meta.get("title"),
                    "abstract": meta.get("abstract"),
                    "authors": [a["name"] for a in meta.get("authors", [])],
                    "year": meta.get("year"),
                    "venue": meta.get("venue"),
                    "doi": meta.get("externalIds", {}).get("DOI"),
                },
            )

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
            session.flush()  # get paper_record.id before adding figures

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

            ingested += 1

        except Exception as e:
            logger.error(f"Failed to ingest {paper_id}: {e}")
            session.rollback()
            continue

        if ingested % 100 == 0:
            session.commit()
            logger.info(f"Committed {ingested} papers so far")

    session.commit()
    session.close()
    logger.success(f"Ingestion complete: {ingested} ingested, {skipped} skipped")


def ingest_demo_data(db_path: str = None):
    """
    Populate DB with a small hardcoded demo dataset (no PDFs needed).
    Useful for development and testing without downloading real papers.
    """
    engine = init_db(db_path)
    session = get_session(engine)

    demo_papers = [
        {
            "paper_id": "demo_001",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "venue": "NeurIPS",
        },
        {
            "paper_id": "demo_002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "year": 2019,
            "venue": "NAACL",
        },
        {
            "paper_id": "demo_003",
            "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
            "abstract": "While the Transformer architecture has become the de-facto standard for NLP tasks, its applications to computer vision remain limited. We show that a pure Transformer applied directly to sequences of image patches can perform very well on image classification tasks with large amounts of data.",
            "authors": ["Alexey Dosovitskiy"],
            "year": 2021,
            "venue": "ICLR",
        },
        {
            "paper_id": "demo_004",
            "title": "Learning Transferable Visual Models From Natural Language Supervision",
            "abstract": "We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million image-text pairs collected from the internet. CLIP models can be applied to any visual classification benchmark by simply providing the names of the visual categories.",
            "authors": ["Alec Radford", "Jong Wook Kim"],
            "year": 2021,
            "venue": "ICML",
        },
        {
            "paper_id": "demo_005",
            "title": "Deep Residual Learning for Image Recognition",
            "abstract": "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
            "authors": ["Kaiming He", "Xiangyu Zhang"],
            "year": 2016,
            "venue": "CVPR",
        },
    ]

    for p in demo_papers:
        existing = session.query(Paper).filter_by(paper_id=p["paper_id"]).first()
        if not existing:
            session.add(Paper(**p))
            session.flush()
            session.add(DocumentPage(
                paper_id=p["paper_id"],
                page_number=1,
                text=p["abstract"],
                image_path="",
            ))

    session.commit()
    session.close()
    logger.success(f"Demo data loaded: {len(demo_papers)} papers")
