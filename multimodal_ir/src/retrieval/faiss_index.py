# src/retrieval/faiss_index.py
"""
Builds, saves, and queries FAISS IVF-PQ indices for both text and figure embeddings.

Index type: IVF100,PQ8
  - IVF (Inverted File): divides the space into 100 Voronoi cells for fast candidate selection
  - PQ (Product Quantization): compresses each vector into 8 sub-quantizer codes (~4KB per 1M vecs)
  - At query time, only `nprobe=10` cells are scanned — ~40x faster than brute force

Two separate indices are maintained:
  text_index   : paper abstracts (768-d, SBERT)
  figure_index : figure images (512-d, CLIP)
"""

import faiss
import numpy as np
import json
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from typing import Optional

from src.config import cfg


@dataclass
class SearchResult:
    faiss_id: int       # row index in the FAISS index
    score: float        # inner product similarity (higher = more similar)
    rank: int


class FAISSIndex:
    """Wraps a single FAISS IVF-PQ index with load/save/search functionality."""

    def __init__(self, dim: int, index_type: str = "IVF100,PQ8", nprobe: int = 10):
        self.dim = dim
        self.index_type = index_type
        self.nprobe = nprobe
        self.index: Optional[faiss.Index] = None
        self.id_map: list[int] = []  # maps FAISS internal row → DB id

    def build(self, vectors: np.ndarray, ids: list[int]):
        """
        Train and populate the index from a numpy matrix.
        vectors: shape (N, dim), dtype float32, L2-normalized
        ids: list of length N mapping row → external DB id
        """
        assert vectors.dtype == np.float32, "Vectors must be float32"
        assert vectors.shape[1] == self.dim, f"Expected dim={self.dim}, got {vectors.shape[1]}"
        assert len(vectors) == len(ids)

        N = len(vectors)
        logger.info(f"Building FAISS index ({self.index_type}) for {N} vectors of dim {self.dim}")

        # For small datasets, fall back to a flat exact index
        if N < 1000:
            logger.warning(f"Only {N} vectors — using flat exact index instead of IVF-PQ")
            self.index = faiss.IndexFlatIP(self.dim)  # Inner Product (cosine for normalized vecs)
        else:
            quantizer = faiss.IndexFlatIP(self.dim)
            # Parse PQ sub-quantizers from index_type string e.g. "IVF100,PQ8"
            n_cells = int(self.index_type.split(",")[0].replace("IVF", ""))
            pq_m = int(self.index_type.split("PQ")[1])

            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                n_cells,
                pq_m,
                8,  # bits per sub-quantizer code
            )
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT

            logger.info("Training IVF-PQ index...")
            # Train on all data (or a random subset if huge)
            train_vecs = vectors if N <= 100_000 else vectors[np.random.choice(N, 100_000, replace=False)]
            self.index.train(train_vecs)
            logger.info("Training complete")

        self.index.add(vectors)
        self.id_map = list(ids)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        logger.success(f"Index built: {self.index.ntotal} vectors")

    def search(self, query_vec: np.ndarray, top_k: int = 100) -> list[SearchResult]:
        """
        Search for top_k nearest neighbours.
        query_vec: shape (1, dim), float32, L2-normalized
        Returns list of SearchResult sorted by score descending.
        """
        assert self.index is not None, "Index not built or loaded"
        query_vec = query_vec.astype(np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]

        scores, indices = self.index.search(query_vec, top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx == -1:  # FAISS returns -1 for unfilled slots
                continue
            db_id = self.id_map[idx] if idx < len(self.id_map) else idx
            results.append(SearchResult(faiss_id=db_id, score=float(score), rank=rank + 1))

        return results

    def save(self, index_path: str, ids_path: str = None):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        ids_path = ids_path or index_path.replace(".faiss", "_ids.json")
        with open(ids_path, "w") as f:
            json.dump(self.id_map, f)
        logger.success(f"Saved FAISS index → {index_path}")

    def load(self, index_path: str, ids_path: str = None):
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        ids_path = ids_path or index_path.replace(".faiss", "_ids.json")
        with open(ids_path) as f:
            self.id_map = json.load(f)
        logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors) from {index_path}")


class DualFAISSIndex:
    """
    Convenience wrapper holding both the text and figure FAISS indices.
    This is what the retrieval and fusion layers interact with.
    """

    TEXT_INDEX_FILE = "text_index.faiss"
    FIGURE_INDEX_FILE = "figure_index.faiss"

    def __init__(self, indices_dir: str = None):
        self.indices_dir = Path(indices_dir or cfg.data.indices_dir)
        fi = cfg.retrieval.faiss

        self.text_index = FAISSIndex(
            dim=cfg.embedding.text.dim,
            index_type=fi.text.index_type,
            nprobe=fi.text.nprobe,
        )
        self.figure_index = FAISSIndex(
            dim=cfg.embedding.figure.dim,
            index_type=fi.figure.index_type,
            nprobe=fi.figure.nprobe,
        )

    def build_text_index(self, vectors: np.ndarray, paper_ids: list[int]):
        self.text_index.build(vectors, paper_ids)

    def build_figure_index(self, vectors: np.ndarray, figure_ids: list[int]):
        self.figure_index.build(vectors, figure_ids)

    def save(self):
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.text_index.save(str(self.indices_dir / self.TEXT_INDEX_FILE))
        self.figure_index.save(str(self.indices_dir / self.FIGURE_INDEX_FILE))

    def load(self):
        self.text_index.load(str(self.indices_dir / self.TEXT_INDEX_FILE))
        self.figure_index.load(str(self.indices_dir / self.FIGURE_INDEX_FILE))

    @property
    def is_loaded(self) -> bool:
        return (
            self.text_index.index is not None
            and self.figure_index.index is not None
        )

    def search_text(self, query_vec: np.ndarray, top_k: int = None) -> list[SearchResult]:
        top_k = top_k or cfg.retrieval.top_k
        return self.text_index.search(query_vec, top_k)

    def search_figures(self, query_vec: np.ndarray, top_k: int = None) -> list[SearchResult]:
        top_k = top_k or cfg.retrieval.top_k
        return self.figure_index.search(query_vec, top_k)
