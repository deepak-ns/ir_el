# src/fusion/query_router.py
"""
Classifies incoming queries into one of three types:
  - TEXT      : pure text query, optimize for full-PDF page retrieval
  - FIGURE    : image upload, search figure index only
  - CROSS_MODAL: text description of a visual (e.g. "U-shaped loss curve")
                 → encode with CLIP text encoder → search figure index

Classification heuristics (no ML model needed for a course project):
  - If an image is provided → FIGURE
  - If the query contains visual/graph keywords → CROSS_MODAL
  - Otherwise → TEXT

The router also returns the recommended fusion weights (α, β)
that were calibrated in scripts/calibrate_fusion.py.
"""

import re
from enum import Enum
from dataclasses import dataclass
from PIL.Image import Image

from src.config import cfg


class QueryType(str, Enum):
    TEXT        = "text"
    FIGURE      = "figure"
    CROSS_MODAL = "cross_modal"


# Keywords that strongly suggest the user is describing a visual artifact
VISUAL_KEYWORDS = [
    r"\bcurve\b", r"\bplot\b", r"\bgraph\b", r"\bchart\b", r"\bfigure\b",
    r"\bdiagram\b", r"\bvisuali[sz]ation\b", r"\bheatmap\b", r"\bscatter\b",
    r"\bbar chart\b", r"\bline graph\b", r"\btable\b", r"\baxis\b",
    r"\bx-axis\b", r"\by-axis\b", r"\bcolumn chart\b", r"\bpie chart\b",
    r"\baccuracy curve\b", r"\bloss curve\b", r"\bconfusion matrix\b",
    r"\barchitecture diagram\b", r"\bflow(chart)?\b", r"\bblueprint\b",
    r"\bshap(ed)?\b", r"u-shaped", r"bell.shaped", r"sigmoid",
    r"\btrend\b", r"\bepoch\b", r"\blearning curve\b",
]

_VISUAL_PATTERN = re.compile("|".join(VISUAL_KEYWORDS), re.IGNORECASE)


@dataclass
class RoutedQuery:
    query_type: QueryType
    text: str | None          # original text query
    image: Image | None       # PIL Image if provided
    alpha: float              # text index weight
    beta: float               # figure index weight
    explanation: str          # human-readable routing reason (shown in UI)


def route_query(
    text: str | None = None,
    image: "Image | None" = None,
) -> RoutedQuery:
    """
    Route a query and assign fusion weights.

    Priority:
      1. Image provided → FIGURE query
      2. Text matches visual keywords → CROSS_MODAL
      3. Default → TEXT
    """
    if text is None and image is None:
        raise ValueError("route_query requires at least one of: text or image")

    fc = cfg.fusion

    if image is not None:
        return RoutedQuery(
            query_type=QueryType.FIGURE,
            text=text,
            image=image,
            alpha=fc.figure_query_alpha,
            beta=fc.figure_query_beta,
            explanation="Image uploaded — searching figure index with CLIP embeddings.",
        )

    if text and _VISUAL_PATTERN.search(text):
        return RoutedQuery(
            query_type=QueryType.CROSS_MODAL,
            text=text,
            image=None,
            alpha=fc.cross_modal_alpha,
            beta=fc.cross_modal_beta,
            explanation=(
                "Visual keywords detected — encoding query with CLIP text encoder "
                "to search figure index cross-modally."
            ),
        )

    return RoutedQuery(
        query_type=QueryType.TEXT,
        text=text,
        image=None,
        alpha=fc.text_query_alpha,
        beta=fc.text_query_beta,
        explanation="Text query - searching full PDF pages with Sentence-BERT and BM25.",
    )
