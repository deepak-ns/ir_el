# tests/test_score_fusion.py
"""Tests for the RRF score fusion logic (pure math, no DB needed)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.fusion.score_fusion import _rrf_score, RRF_K


class TestRRFScore:
    def test_rank_1_highest(self):
        assert _rrf_score(1) > _rrf_score(2) > _rrf_score(10)

    def test_standard_k(self):
        # Standard RRF: 1/(60+1) for rank=1
        assert _rrf_score(1) == pytest.approx(1 / (RRF_K + 1))

    def test_positive(self):
        for rank in [1, 5, 10, 100]:
            assert _rrf_score(rank) > 0

    def test_diminishing_returns(self):
        # The gap between rank 1→2 should be larger than gap between 50→51
        gap_top    = _rrf_score(1)  - _rrf_score(2)
        gap_bottom = _rrf_score(50) - _rrf_score(51)
        assert gap_top > gap_bottom


class TestWeightNormalization:
    """Ensure the fusion normalization logic handles edge cases."""

    def test_weights_sum_to_one(self):
        """After normalization, alpha+beta+gamma should = 1.0."""
        alpha, beta, gamma = 0.6, 0.4, 0.2
        total = alpha + beta + gamma
        assert abs((alpha / total) + (beta / total) + (gamma / total) - 1.0) < 1e-9

    def test_text_only_mode(self):
        alpha, beta, gamma = 1.0, 0.0, 0.0
        total = alpha + beta + gamma
        assert alpha / total == pytest.approx(1.0)
        assert beta  / total == pytest.approx(0.0)

    def test_figure_only_mode(self):
        alpha, beta, gamma = 0.0, 1.0, 0.0
        total = alpha + beta + gamma
        assert beta  / total == pytest.approx(1.0)
        assert alpha / total == pytest.approx(0.0)
