# src/retrieval/bm25_retriever.py
"""
BM25 sparse retrieval over paper abstracts.
Used as:
  1. A fallback for exact keyword queries
  2. A component in hybrid fusion (sparse + dense)

BM25 formula:
    score(q, d) = Σ IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))
    where k1=1.5, b=0.75 (Robertson defaults)
"""

import json
import re
import pickle
from pathlib import Path
from loguru import logger
from rank_bm25 import BM25Okapi
from dataclasses import dataclass

from src.config import cfg


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    # Remove very short tokens
    return [t for t in tokens if len(t) > 1]


@dataclass
class BM25Result:
    paper_id: str
    score: float
    rank: int


class BM25Retriever:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.paper_ids: list[str] = []
        self.k1 = cfg.retrieval.bm25.k1
        self.b = cfg.retrieval.bm25.b

    def build(self, abstracts: list[str], paper_ids: list[str]):
        """
        Build BM25 index from a list of abstracts.
        abstracts and paper_ids must be parallel lists.
        """
        assert len(abstracts) == len(paper_ids)
        logger.info(f"Building BM25 index over {len(abstracts)} documents")

        tokenized = [_tokenize(a) for a in abstracts]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self.paper_ids = list(paper_ids)
        logger.success("BM25 index built")

    def search(self, query: str, top_k: int = None) -> list[BM25Result]:
        """
        Search for top_k most relevant papers for a text query.
        Returns list of BM25Result sorted by score descending.
        """
        assert self.bm25 is not None, "BM25 index not built or loaded"
        top_k = top_k or cfg.retrieval.top_k

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                break
            results.append(BM25Result(
                paper_id=self.paper_ids[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))
        return results

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "paper_ids": self.paper_ids}, f)
        logger.success(f"BM25 index saved → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.paper_ids = data["paper_ids"]
        logger.info(f"BM25 index loaded from {path}")

    @classmethod
    def default_path(cls) -> str:
        return str(Path(cfg.data.indices_dir) / "bm25.pkl")
