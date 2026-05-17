# src/fusion/search_engine.py
"""
SearchEngine: the single entry point for all retrieval.

Wires together:
  QueryRouter → encodes query → DualFAISSIndex + BM25 → ScoreFusion → results

Holds encoders and indices in memory after first load (singleton pattern).
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from PIL.Image import Image

from src.embedding.text_encoder import TextEncoder
from src.embedding.figure_encoder import FigureEncoder
from src.retrieval.faiss_index import DualFAISSIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.fusion.query_router import route_query, QueryType, RoutedQuery
from src.fusion.score_fusion import fuse_results, FusedResult
from src.config import cfg


@dataclass
class SearchResponse:
    results: list[FusedResult]
    query_type: str
    routing_explanation: str
    latency_ms: float
    alpha: float
    beta: float
    total_candidates: int


class SearchEngine:
    _instance: "SearchEngine | None" = None

    def __init__(self):
        self.text_encoder:   TextEncoder    | None = None
        self.figure_encoder: FigureEncoder  | None = None
        self.dual_index:     DualFAISSIndex | None = None
        self.bm25:           BM25Retriever  | None = None
        self._ready = False

    @classmethod
    def get(cls) -> "SearchEngine":
        """Singleton accessor — avoids reloading models on every request."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        """Load all models and indices. Call once at startup."""
        if self._ready:
            return
        logger.info("Loading SearchEngine components...")

        self.text_encoder   = TextEncoder()
        self.figure_encoder = FigureEncoder()

        self.dual_index = DualFAISSIndex()
        self.dual_index.load()

        self.bm25 = BM25Retriever()
        bm25_path = BM25Retriever.default_path()
        if Path(bm25_path).exists():
            self.bm25.load(bm25_path)
        else:
            logger.warning("BM25 index not found — keyword fallback disabled")

        self._ready = True
        logger.success("SearchEngine ready")

    def search(
        self,
        text: str | None = None,
        image: "Image | None" = None,
        alpha: float | None = None,
        beta: float | None = None,
        top_k: int | None = None,
    ) -> SearchResponse:
        """
        Execute a multi-modal search.

        Args:
            text:  text query (required for TEXT / CROSS_MODAL)
            image: PIL Image (for FIGURE queries)
            alpha: override text fusion weight (0.0–1.0)
            beta:  override figure fusion weight (0.0–1.0)
            top_k: number of results to return

        Returns:
            SearchResponse with ranked FusedResult list + metadata
        """
        assert self._ready, "Call engine.load() before searching"
        assert text or image, "Provide text, image, or both"

        t0 = time.time()
        top_k = top_k or cfg.retrieval.final_k

        # 1. Route the query
        routed: RoutedQuery = route_query(text=text, image=image)
        # Allow caller to override weights (e.g. from UI slider)
        used_alpha = alpha if alpha is not None else routed.alpha
        used_beta  = beta  if beta  is not None else routed.beta

        # 2. Encode and retrieve
        text_results   = []
        figure_results = []
        bm25_results   = []

        if routed.query_type == QueryType.TEXT:
            query_vec      = self.text_encoder.encode_query(text)
            text_results   = self.dual_index.search_text(query_vec)
            if self.bm25.bm25:
                bm25_results = self.bm25.search(text)

        elif routed.query_type == QueryType.CROSS_MODAL:
            # Text description → CLIP text encoder → figure index
            query_vec      = self.figure_encoder.encode_text(text)
            figure_results = self.dual_index.search_figures(query_vec)
            # Also search text index for the same query via SBERT
            text_query_vec = self.text_encoder.encode_query(text)
            text_results   = self.dual_index.search_text(text_query_vec)
            if self.bm25.bm25:
                bm25_results = self.bm25.search(text)

        elif routed.query_type == QueryType.FIGURE:
            import numpy as np
            img_arr = self.figure_encoder.encode_images([routed.image])
            figure_results = self.dual_index.search_figures(img_arr)
            # Optionally also search with caption text if provided
            if text:
                text_query_vec = self.text_encoder.encode_query(text)
                text_results   = self.dual_index.search_text(text_query_vec)

        # 3. Fuse scores
        fused = fuse_results(
            text_results=text_results,
            figure_results=figure_results,
            bm25_results=bm25_results,
            alpha=used_alpha,
            beta=used_beta,
            top_k=top_k,
        )

        latency_ms = round((time.time() - t0) * 1000, 1)

        return SearchResponse(
            results=fused,
            query_type=routed.query_type.value,
            routing_explanation=routed.explanation,
            latency_ms=latency_ms,
            alpha=used_alpha,
            beta=used_beta,
            total_candidates=len(text_results) + len(figure_results),
        )
