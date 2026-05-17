# api/main.py
"""
FastAPI REST API for the multi-modal IR system.

Endpoints:
  GET  /health          → liveness check
  POST /search          → text or cross-modal search
  POST /search/figure   → image upload search
  GET  /paper/{id}      → full paper record
  GET  /figure/{id}     → figure image (served as static file)
  GET  /stats           → index statistics
"""

import io
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.search_engine import SearchEngine
from src.db import init_db, get_session, Paper, Figure, DocumentPage
from src.config import cfg


# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Full-PDF Search",
    description="Search uploaded PDFs by full-page text using SBERT, BM25, and FAISS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve figure images as static files
figures_path = Path(cfg.data.figures_dir)
if figures_path.exists():
    app.mount("/figures", StaticFiles(directory=str(figures_path)), name="figures")

pages_path = Path(getattr(cfg.data, "pages_dir", "data/pages"))
pages_path.mkdir(parents=True, exist_ok=True)
app.mount("/pages", StaticFiles(directory=str(pages_path)), name="pages")


# ── Pydantic models ─────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="Text weight override")
    beta:  Optional[float] = Field(None, ge=0.0, le=1.0, description="Figure weight override")
    top_k: int = Field(10, ge=1, le=50)


class FigureOut(BaseModel):
    id: int
    paper_id: str
    figure_number: int
    caption: str
    image_url: str


class PaperOut(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: Optional[int]
    venue: Optional[str]
    doi: Optional[str]


class PageOut(BaseModel):
    id: int
    paper_id: str
    page_number: int
    text: str
    image_url: str
    score: float = 0.0


class ResultItem(BaseModel):
    rank: int
    paper: PaperOut
    fused_score: float
    text_score: float
    figure_score: float
    bm25_score: float
    matched_page: Optional[PageOut] = None
    matched_figures: list[FigureOut]
    score_breakdown: dict  # for UI visualization


class SearchResponse(BaseModel):
    results: list[ResultItem]
    query_type: str
    routing_explanation: str
    latency_ms: float
    alpha: float
    beta: float
    total_candidates: int
    query: str


# ── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Starting up — loading search engine...")
    engine = SearchEngine.get()
    engine.load()
    logger.success("API ready")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _figure_to_out(fig: dict) -> FigureOut:
    img_path = fig.get("image_path", "")
    # Convert local path to URL path
    rel = Path(img_path).relative_to(cfg.data.figures_dir) if img_path else None
    url = f"/figures/{rel}" if rel else ""
    return FigureOut(
        id=fig["id"],
        paper_id=fig["paper_id"],
        figure_number=fig.get("figure_number", 0),
        caption=fig.get("caption", ""),
        image_url=url,
    )


def _page_to_out(page: dict | None) -> PageOut | None:
    if not page:
        return None

    img_path = page.get("image_path", "")
    rel = None
    if img_path:
        try:
            rel = Path(img_path).resolve().relative_to(Path(getattr(cfg.data, "pages_dir", "data/pages")).resolve())
        except ValueError:
            rel = Path(img_path).name

    return PageOut(
        id=page["id"],
        paper_id=page["paper_id"],
        page_number=page.get("page_number", 0),
        text=(page.get("text", "") or "")[:1200],
        image_url=f"/pages/{rel}" if rel else "",
        score=page.get("score", 0.0),
    )


def _build_result_item(r) -> ResultItem:
    total = r.text_score + r.figure_score + r.bm25_score or 1.0
    return ResultItem(
        rank=r.rank,
        paper=PaperOut(**r.paper),
        fused_score=r.fused_score,
        text_score=r.text_score,
        figure_score=r.figure_score,
        bm25_score=r.bm25_score,
        matched_page=_page_to_out(r.matched_page),
        matched_figures=[_figure_to_out(f) for f in r.matched_figures],
        score_breakdown={
            "text_pct":   round(r.text_score   / total * 100, 1),
            "figure_pct": round(r.figure_score / total * 100, 1),
            "bm25_pct":   round(r.bm25_score   / total * 100, 1),
        },
    )


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    engine = SearchEngine.get()
    return {"status": "ok", "ready": engine._ready}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Text and cross-modal search."""
    engine = SearchEngine.get()
    if not engine._ready:
        raise HTTPException(503, "Search engine not ready yet")

    resp = engine.search(
        text=req.query,
        alpha=req.alpha,
        beta=req.beta,
        top_k=req.top_k,
    )

    return SearchResponse(
        results=[_build_result_item(r) for r in resp.results],
        query_type=resp.query_type,
        routing_explanation=resp.routing_explanation,
        latency_ms=resp.latency_ms,
        alpha=resp.alpha,
        beta=resp.beta,
        total_candidates=resp.total_candidates,
        query=req.query,
    )


@app.post("/search/figure", response_model=SearchResponse)
async def search_figure(
    file: UploadFile = File(...),
    caption: Optional[str] = Query(None),
    alpha: Optional[float] = Query(None),
    beta:  Optional[float] = Query(None),
    top_k: int = Query(10),
):
    """Image upload search."""
    engine = SearchEngine.get()
    if not engine._ready:
        raise HTTPException(503, "Search engine not ready yet")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    resp = engine.search(
        text=caption,
        image=image,
        alpha=alpha,
        beta=beta,
        top_k=top_k,
    )

    return SearchResponse(
        results=[_build_result_item(r) for r in resp.results],
        query_type=resp.query_type,
        routing_explanation=resp.routing_explanation,
        latency_ms=resp.latency_ms,
        alpha=resp.alpha,
        beta=resp.beta,
        total_candidates=resp.total_candidates,
        query=caption or "(image query)",
    )


@app.get("/paper/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: str):
    session = get_session()
    paper = session.query(Paper).filter_by(paper_id=paper_id).first()
    session.close()
    if not paper:
        raise HTTPException(404, f"Paper {paper_id} not found")
    return PaperOut(**paper.to_dict())


@app.get("/stats")
def stats():
    session = get_session()
    n_papers  = session.query(Paper).count()
    n_pages   = session.query(DocumentPage).count()
    n_figures = session.query(Figure).count()
    session.close()
    engine = SearchEngine.get()
    return {
        "papers_in_db": n_papers,
        "pages_in_db": n_pages,
        "figures_in_db": n_figures,
        "text_index_size": engine.dual_index.text_index.index.ntotal if engine._ready else 0,
        "figure_index_size": engine.dual_index.figure_index.index.ntotal if engine._ready else 0,
    }

# ── PDF ingestion endpoint ────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    paper_id:       str
    title:          str
    abstract:       str
    pages_indexed:  int
    figures_found:  int
    message:        str


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    file:    UploadFile = File(...),
    title:   Optional[str] = Query(None),
    authors: Optional[str] = Query(None),   # comma-separated
    year:    Optional[int] = Query(None),
):
    """
    Upload a PDF, parse it, embed it, and add it to the live search index.
    The paper is searchable immediately after this call returns.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    contents = await file.read()
    if len(contents) < 1000:
        raise HTTPException(400, "File too small to be a valid PDF")

    import uuid, tempfile, shutil, numpy as np
    from pathlib import Path as P
    from src.ingestion.pdf_parser import parse_pdf
    from src.retrieval.bm25_retriever import BM25Retriever

    paper_id = f"user_{uuid.uuid4().hex[:8]}"
    authors_list = [a.strip() for a in authors.split(",")] if authors else []

    # Save to temp file for parsing
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        init_db()
        session = get_session()

        upload_dir = P(cfg.data.raw_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        stored_pdf_path = upload_dir / f"{paper_id}.pdf"
        shutil.copyfile(tmp_path, stored_pdf_path)

        parsed = parse_pdf(
            pdf_path=tmp_path,
            paper_id=paper_id,
            metadata={"title": title, "authors": authors_list, "year": year},
        )
        if title:    parsed.title   = title
        if authors_list: parsed.authors = authors_list
        if year:     parsed.year    = year

        # Store in DB
        paper_record = Paper(
            paper_id=parsed.paper_id,
            title=parsed.title,
            abstract=parsed.abstract,
            authors=parsed.authors,
            year=parsed.year,
            doi=parsed.doi,
            pdf_path=str(stored_pdf_path),
        )
        session.add(paper_record)
        session.flush()

        page_records = []
        for page in parsed.pages:
            pr = DocumentPage(
                paper_id=parsed.paper_id,
                page_number=page.page_number,
                text=page.text,
                image_path=page.image_path,
            )
            session.add(pr)
            page_records.append(pr)
        session.flush()

        fig_records = []
        for fig in parsed.figures:
            fr = Figure(
                paper_id=parsed.paper_id,
                figure_number=fig.figure_number,
                caption=fig.caption,
                image_path=fig.image_path,
                page_number=fig.page_number,
                width=fig.width,
                height=fig.height,
            )
            session.add(fr)
            fig_records.append(fr)
        session.flush()
        session.commit()

        # Embed and add to live FAISS index
        engine = SearchEngine.get()
        if engine._ready:
            # Text vectors for every page in the uploaded PDF
            if page_records:
                text_vecs = engine.text_encoder.encode([p.text or "" for p in page_records])
                for i, page in enumerate(page_records):
                    new_row = engine.dual_index.text_index.index.ntotal
                    engine.dual_index.text_index.index.add(text_vecs[i:i+1])
                    engine.dual_index.text_index.id_map.append(page.id)
                    page.text_faiss_id = new_row

                db_paper = session.query(Paper).filter_by(paper_id=paper_id).first()
                if db_paper:
                    db_paper.text_faiss_id = page_records[0].text_faiss_id
            session.commit()

            # Figure vectors
            valid_fig_records = [
                f for f in fig_records
                if f.image_path and P(f.image_path).exists()
            ]
            if valid_fig_records:
                fig_vecs = engine.figure_encoder.encode_images([f.image_path for f in valid_fig_records])
                for i, fig in enumerate(valid_fig_records):
                    fig_row = engine.dual_index.figure_index.index.ntotal
                    engine.dual_index.figure_index.index.add(fig_vecs[i:i+1])
                    engine.dual_index.figure_index.id_map.append(fig.id)
                    fig.figure_faiss_id = fig_row
                session.commit()

            # Persist updated indices
            engine.dual_index.save()

            # Rebuild BM25
            all_pages = session.query(DocumentPage).filter(DocumentPage.text.isnot(None)).all()
            engine.bm25.build(
                abstracts=[p.text or "" for p in all_pages],
                paper_ids=[str(p.id) for p in all_pages],
            )
            engine.bm25.save(BM25Retriever.default_path())

        session.close()

        return IngestResponse(
            paper_id=paper_id,
            title=parsed.title,
            abstract=next((p.text for p in parsed.pages if p.text), parsed.abstract)[:300],
            pages_indexed=len(parsed.pages),
            figures_found=len(parsed.figures),
            message=f"Successfully indexed {len(parsed.pages)} pages. Search for any text in this PDF now.",
        )

    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        P(tmp_path).unlink(missing_ok=True)
