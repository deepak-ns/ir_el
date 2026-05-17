# tests/test_query_router.py
"""Unit tests for the query router / classifier."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.fusion.query_router import route_query, QueryType


class TestQueryRouter:
    def test_plain_text_query(self):
        routed = route_query(text="attention mechanisms for long sequences")
        assert routed.query_type == QueryType.TEXT
        assert routed.alpha > routed.beta  # text-heavy weighting

    def test_cross_modal_curve(self):
        routed = route_query(text="U-shaped validation loss curve over training epochs")
        assert routed.query_type == QueryType.CROSS_MODAL

    def test_cross_modal_graph(self):
        routed = route_query(text="bar chart comparing model accuracy across datasets")
        assert routed.query_type == QueryType.CROSS_MODAL

    def test_cross_modal_plot(self):
        routed = route_query(text="scatter plot of embedding clusters")
        assert routed.query_type == QueryType.CROSS_MODAL

    def test_cross_modal_diagram(self):
        routed = route_query(text="architecture diagram with encoder decoder blocks")
        assert routed.query_type == QueryType.CROSS_MODAL

    def test_figure_upload(self):
        from PIL import Image
        dummy_img = Image.new("RGB", (100, 100))
        routed = route_query(text=None, image=dummy_img)
        assert routed.query_type == QueryType.FIGURE
        assert routed.beta > routed.alpha  # figure-heavy weighting

    def test_figure_upload_with_text(self):
        from PIL import Image
        dummy_img = Image.new("RGB", (100, 100))
        routed = route_query(text="confusion matrix", image=dummy_img)
        # Image takes priority even when text has visual keywords
        assert routed.query_type == QueryType.FIGURE

    def test_weights_are_positive(self):
        for text in [
            "transformer models",
            "loss curve during training",
            None,
        ]:
            from PIL import Image
            img = Image.new("RGB", (10, 10)) if text is None else None
            routed = route_query(text=text, image=img)
            assert routed.alpha >= 0
            assert routed.beta  >= 0

    def test_no_query_raises(self):
        with pytest.raises(Exception):
            route_query(text=None, image=None)
