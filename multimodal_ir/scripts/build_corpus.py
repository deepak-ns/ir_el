#!/usr/bin/env python3
# scripts/build_corpus.py
"""
Step 2: Parse downloaded PDFs, extract figures, store everything in SQLite.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --demo   # Use built-in demo data (no PDFs needed)
"""

import sys
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.corpus_builder import ingest_from_metadata, ingest_demo_data
from src.db import init_db
from src.config import cfg
from loguru import logger


@click.command()
@click.option("--demo", is_flag=True, help="Load small demo dataset instead of real PDFs")
@click.option("--metadata", default=None, help="Path to papers_metadata.json")
@click.option("--pdf-dir",  default=None, help="Directory containing downloaded PDFs")
def main(demo, metadata, pdf_dir):
    if demo:
        logger.info("Loading demo data (5 papers, no PDFs required)")
        ingest_demo_data()
        logger.success("Demo corpus ready. Run: python scripts/build_index.py")
        return

    metadata_path = metadata or f"{cfg.data.raw_dir}/papers_metadata.json"
    pdf_directory = pdf_dir or f"{cfg.data.raw_dir}/pdfs"

    if not Path(metadata_path).exists():
        logger.error(f"Metadata not found: {metadata_path}")
        logger.info("Run: python scripts/download_data.py first")
        sys.exit(1)

    logger.info(f"Building corpus from {metadata_path}")
    ingest_from_metadata(
        metadata_path=metadata_path,
        pdf_dir=pdf_directory,
    )
    logger.success("Corpus built. Run: python scripts/build_index.py")


if __name__ == "__main__":
    main()
