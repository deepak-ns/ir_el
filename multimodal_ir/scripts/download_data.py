#!/usr/bin/env python3
# scripts/download_data.py
"""
Step 1: Download paper metadata from Semantic Scholar API.
Saves to data/raw/papers_metadata.json

Usage:
    python scripts/download_data.py --limit 5000
    python scripts/download_data.py --limit 500 --field "Computer Science"
"""

import sys
import click
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.s2orc_loader import fetch_papers_by_field, save_metadata, download_pdfs_batch
from src.config import cfg
from loguru import logger


@click.command()
@click.option("--limit",    default=1000,              help="Number of papers to fetch")
@click.option("--field",    default="Computer Science", help="Field of study filter")
@click.option("--api-key",  default=None,              help="Semantic Scholar API key (optional, increases rate limit)")
@click.option("--pdfs/--no-pdfs", default=True,        help="Also download open-access PDFs")
@click.option("--max-pdfs", default=200,               help="Max PDFs to download")
def main(limit, field, api_key, pdfs, max_pdfs):
    metadata_path = f"{cfg.data.raw_dir}/papers_metadata.json"
    pdf_dir       = f"{cfg.data.raw_dir}/pdfs"

    logger.info(f"Fetching {limit} papers in field: {field}")
    papers = fetch_papers_by_field(field_of_study=field, limit=limit, api_key=api_key)
    save_metadata(papers, metadata_path)

    if pdfs and papers:
        logger.info(f"Downloading up to {max_pdfs} open-access PDFs...")
        download_pdfs_batch(papers, pdf_dir, max_downloads=max_pdfs)

    logger.success(f"Done. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
