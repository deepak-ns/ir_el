# tests/test_metrics.py
"""Unit tests for IR evaluation metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.evaluation.metrics import (
    reciprocal_rank,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    dcg_at_k,
    ideal_dcg_at_k,
)


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_not_found(self):
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant(self):
        # First hit at position 2
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)

    def test_empty_retrieved(self):
        assert reciprocal_rank([], {"a"}) == 0.0


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_half(self):
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4) == 0.5

    def test_zero(self):
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_k_larger_than_retrieved(self):
        # k=10 but only 3 retrieved — still divides by k
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=10) == pytest.approx(0.3)


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "x", "x", "x"], {"a", "b"}, k=4) == pytest.approx(0.5)

    def test_zero(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0


class TestNDCG:
    def test_perfect(self):
        # When top-1 is relevant, DCG == IDCG for k=1
        assert ndcg_at_k(["a"], {"a"}, k=1) == 1.0

    def test_worst_case(self):
        assert ndcg_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_ordering_matters(self):
        # Retrieved: relevant at 1 and 3 vs relevant at 2 and 3
        score_good = ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3)
        score_bad  = ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3)
        assert score_good > score_bad

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


class TestDCG:
    def test_single_relevant_at_1(self):
        import math
        assert dcg_at_k(["a"], {"a"}, k=1) == pytest.approx(1.0 / math.log2(2))

    def test_no_relevant(self):
        assert dcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


class TestIdealDCG:
    def test_single_relevant(self):
        import math
        assert ideal_dcg_at_k({"a"}, k=1) == pytest.approx(1.0 / math.log2(2))

    def test_more_relevant_than_k(self):
        # Only k slots available
        assert ideal_dcg_at_k({"a", "b", "c", "d"}, k=2) == ideal_dcg_at_k({"a", "b"}, k=2)
