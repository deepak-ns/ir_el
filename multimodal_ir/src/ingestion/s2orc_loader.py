# src/ingestion/s2orc_loader.py
"""
Downloads and prepares a subset of the S2ORC (Semantic Scholar Open Research Corpus).
Uses the Semantic Scholar API to get paper metadata + abstracts without needing the full 500GB dump.
For a student project, we work with the public API (free, rate-limited to 100 req/s with key).

Alternative: use the Hugging Face datasets mirror of S2ORC for smaller subsets.
"""

import time
import json
import requests
from pathlib import Path
from typing import Iterator
from loguru import logger
from tqdm import tqdm

from src.config import cfg

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,title,abstract,authors,year,venue,externalIds,openAccessPdf"


def _get_headers(api_key: str = None) -> dict:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def fetch_papers_by_field(
    field_of_study: str = "Computer Science",
    limit: int = 1000,
    api_key: str = None,
) -> list[dict]:
    """
    Fetch paper metadata from Semantic Scholar API.
    Returns list of paper dicts with abstract, title, authors, year.
    """
    papers = []
    offset = 0
    batch_size = min(100, limit)
    headers = _get_headers(api_key)

    logger.info(f"Fetching up to {limit} papers (field: {field_of_study})")

    with tqdm(total=limit, desc="Fetching papers") as pbar:
        while len(papers) < limit:
            params = {
                "fields": S2_FIELDS,
                "limit": batch_size,
                "offset": offset,
                "fieldsOfStudy": field_of_study,
            }
            try:
                resp = requests.get(
                    f"{S2_API_BASE}/paper/search",
                    params=params,
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code == 429:
                    logger.warning("Rate limited. Sleeping 10s...")
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                data = resp.json()
                batch = data.get("data", [])
                if not batch:
                    break

                # Filter: only papers with abstracts and open-access PDFs
                valid = [
                    p for p in batch
                    if p.get("abstract") and p.get("openAccessPdf")
                ]
                papers.extend(valid)
                pbar.update(len(valid))
                offset += batch_size

                # Be respectful of rate limits even without a key
                time.sleep(0.5)

            except requests.RequestException as e:
                logger.error(f"API error: {e}")
                time.sleep(5)

    logger.info(f"Fetched {len(papers)} valid papers")
    return papers[:limit]


def save_metadata(papers: list[dict], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(papers, f, indent=2)
    logger.info(f"Saved metadata for {len(papers)} papers → {out_path}")


def load_metadata(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def download_pdf(paper: dict, out_dir: str) -> str | None:
    """
    Download a single open-access PDF.
    Returns local path if successful, None otherwise.
    """
    pdf_info = paper.get("openAccessPdf")
    if not pdf_info or not pdf_info.get("url"):
        return None

    url = pdf_info["url"]
    paper_id = paper["paperId"]
    out_path = Path(out_dir) / f"{paper_id}.pdf"

    if out_path.exists():
        return str(out_path)

    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(out_path)
    except Exception as e:
        logger.debug(f"PDF download failed for {paper_id}: {e}")
        return None


def download_pdfs_batch(
    papers: list[dict],
    out_dir: str,
    max_downloads: int = 500,
) -> dict[str, str]:
    """
    Download PDFs for a batch of papers.
    Returns dict: paper_id → local_pdf_path
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    downloaded = 0

    for paper in tqdm(papers, desc="Downloading PDFs"):
        if downloaded >= max_downloads:
            break
        path = download_pdf(paper, out_dir)
        if path:
            results[paper["paperId"]] = path
            downloaded += 1
        time.sleep(0.2)  # polite delay

    logger.info(f"Downloaded {len(results)} PDFs → {out_dir}")
    return results
