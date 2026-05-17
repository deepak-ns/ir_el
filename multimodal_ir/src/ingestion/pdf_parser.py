# src/ingestion/pdf_parser.py
"""
Parses research paper PDFs to extract:
  - Title, abstract, authors (from metadata or heuristic parsing)
  - All figures as PNG images
  - Figure captions (text immediately following each figure)

Uses PyMuPDF (fitz) for fast, dependency-light PDF handling.
"""

import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from PIL import Image
import io

from src.config import cfg


@dataclass
class ParsedFigure:
    figure_number: int
    caption: str
    image_path: str
    page_number: int
    width: int
    height: int


@dataclass
class ParsedPage:
    page_number: int
    text: str
    image_path: str


@dataclass
class ParsedPaper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: Optional[int]
    venue: Optional[str]
    doi: Optional[str]
    pages: list[ParsedPage] = field(default_factory=list)
    figures: list[ParsedFigure] = field(default_factory=list)


def _clean_text(text: str) -> str:
    """Remove hyphenation, normalize whitespace."""
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _extract_abstract(text: str) -> str:
    """
    Heuristic: find text between 'Abstract' header and next section header.
    Works for the majority of CS/ML papers.
    """
    patterns = [
        r"(?i)abstract[.\s\n]+(.+?)(?=\n\s*\n\s*[1I]\s+[A-Z]|\nintroduction|\nkeywords)",
        r"(?i)abstract[.\s\n]+(.+?)(?=\n[A-Z][a-z]+ [A-Z])",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m and len(m.group(1)) > 100:
            return _clean_text(m.group(1)[:2000])
    # Fallback: return first 512 chars after page 1 header region
    lines = text.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if re.match(r"(?i)abstract", line.strip()):
            body_start = i + 1
            break
    snippet = " ".join(lines[body_start : body_start + 30])
    return _clean_text(snippet[:1500])


def _extract_figure_caption(page_text: str, fig_num: int) -> str:
    """
    Find caption for Figure N on a page.
    Captions typically start with 'Figure N:' or 'Fig. N'.
    """
    patterns = [
        rf"(?i)fig(?:ure)?\.?\s*{fig_num}[:\.\s]+(.+?)(?=fig(?:ure)?\.?\s*\d|table\s*\d|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, page_text, re.DOTALL)
        if m:
            return _clean_text(m.group(1)[:400])
    return f"Figure {fig_num}"


def extract_figures_from_pdf(
    pdf_path: str,
    paper_id: str,
    figures_dir: str = None,
    max_figures: int = None,
) -> list[ParsedFigure]:
    """
    Extract embedded images from a PDF and pair them with captions.
    Images are saved as PNGs under figures_dir/paper_id/.
    """
    figures_dir = Path(figures_dir or cfg.data.figures_dir)
    max_figures = max_figures or cfg.data.max_figures_per_paper
    out_dir = figures_dir / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_figures = []
    doc = fitz.open(pdf_path)
    fig_count = 0

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            if fig_count >= max_figures:
                break

            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = base_image["image"]
            img_ext = base_image["ext"]

            # Filter out tiny images (logos, icons, page decorations)
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            if w < 80 or h < 80 or w * h < 10000:
                continue

            fig_num = fig_count + 1
            caption = _extract_figure_caption(page_text, fig_num)

            img_filename = f"fig_{fig_num:03d}.png"
            img_path = out_dir / img_filename
            img.save(str(img_path), "PNG")

            parsed_figures.append(ParsedFigure(
                figure_number=fig_num,
                caption=caption,
                image_path=str(img_path),
                page_number=page_num + 1,
                width=w,
                height=h,
            ))
            fig_count += 1

    doc.close()
    logger.debug(f"[{paper_id}] Extracted {len(parsed_figures)} figures")
    return parsed_figures


def extract_pages_from_pdf(
    pdf_path: str,
    paper_id: str,
    pages_dir: str = None,
) -> list[ParsedPage]:
    """
    Extract full text from every PDF page and render a page image for UI preview.
    """
    pages_dir = Path(pages_dir or getattr(cfg.data, "pages_dir", "data/pages"))
    out_dir = pages_dir / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_pages = []
    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        text = _clean_text(page.get_text())

        img_path = out_dir / f"page_{page_number:03d}.png"
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            pix.save(str(img_path))
        except Exception as e:
            logger.warning(f"[{paper_id}] Could not render page {page_number}: {e}")
            img_path = Path("")

        parsed_pages.append(ParsedPage(
            page_number=page_number,
            text=text,
            image_path=str(img_path) if img_path else "",
        ))

    doc.close()
    logger.debug(f"[{paper_id}] Extracted text from {len(parsed_pages)} pages")
    return parsed_pages


def parse_pdf(pdf_path: str, paper_id: str, metadata: dict = None) -> ParsedPaper:
    """
    Full parse of a single research paper PDF.
    metadata dict can supply title/authors/year/venue if already known (e.g. from S2ORC).
    """
    metadata = metadata or {}
    doc = fitz.open(pdf_path)

    # Get full text from first 3 pages for abstract extraction
    full_text = ""
    for page in doc[:3]:
        full_text += page.get_text()
    doc.close()

    abstract = metadata.get("abstract") or _extract_abstract(full_text)
    title = metadata.get("title") or _extract_title(full_text)

    pages = extract_pages_from_pdf(pdf_path, paper_id)
    figures = extract_figures_from_pdf(pdf_path, paper_id)

    return ParsedPaper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=metadata.get("authors", []),
        year=metadata.get("year"),
        venue=metadata.get("venue"),
        doi=metadata.get("doi"),
        pages=pages,
        figures=figures,
    )


def _extract_title(text: str) -> str:
    """Heuristic: first non-empty line is usually the title."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return lines[0][:200]
    return "Unknown Title"
