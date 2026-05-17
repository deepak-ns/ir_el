# tests/test_integration.py
"""
Integration test: runs the full pipeline end-to-end with demo data.
No PDFs or GPU needed. Verifies ingestion → index → search → results.

Run with:
    pytest tests/test_integration.py -v
"""

import sys
import tempfile
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def tmp_data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("ir_test_data")


@pytest.fixture(scope="module")
def demo_db(tmp_data_dir):
    """Set up a temporary SQLite DB with demo papers."""
    db_path = str(tmp_data_dir / "test.db")

    # Patch config to use temp paths
    with patch("src.config.cfg") as mock_cfg:
        mock_cfg.data.db_path        = db_path
        mock_cfg.data.figures_dir    = str(tmp_data_dir / "figures")
        mock_cfg.data.indices_dir    = str(tmp_data_dir / "indices")
        mock_cfg.data.max_figures_per_paper = 5
        mock_cfg.embedding.text.model_name  = "all-mpnet-base-v2"
        mock_cfg.embedding.text.dim         = 768
        mock_cfg.embedding.text.batch_size  = 8
        mock_cfg.embedding.text.max_seq_length = 512
        mock_cfg.embedding.figure.model_name = "openai/clip-vit-base-patch32"
        mock_cfg.embedding.figure.dim        = 512
        mock_cfg.embedding.figure.batch_size = 8
        mock_cfg.retrieval.faiss.text.index_type  = "Flat"
        mock_cfg.retrieval.faiss.text.nprobe      = 1
        mock_cfg.retrieval.faiss.figure.index_type = "Flat"
        mock_cfg.retrieval.faiss.figure.nprobe     = 1
        mock_cfg.retrieval.top_k   = 10
        mock_cfg.retrieval.final_k = 5
        mock_cfg.retrieval.bm25.k1 = 1.5
        mock_cfg.retrieval.bm25.b  = 0.75
        mock_cfg.fusion.default_alpha      = 0.6
        mock_cfg.fusion.default_beta       = 0.4
        mock_cfg.fusion.text_query_alpha   = 0.85
        mock_cfg.fusion.text_query_beta    = 0.15
        mock_cfg.fusion.figure_query_alpha = 0.2
        mock_cfg.fusion.figure_query_beta  = 0.8
        mock_cfg.fusion.cross_modal_alpha  = 0.45
        mock_cfg.fusion.cross_modal_beta   = 0.55

        from src.ingestion.corpus_builder import ingest_demo_data
        ingest_demo_data(db_path=db_path)

    return db_path


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestIngestion:
    def test_demo_data_loaded(self, demo_db):
        from src.db import get_session, Paper
        session = get_session(engine=None)  # Will use patched cfg

        # Just test the module works
        from src.db import init_db
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.db import Base

        engine = init_db(demo_db)
        Session = sessionmaker(bind=engine)
        sess = Session()
        count = sess.query(Paper).count()
        sess.close()
        assert count == 5, f"Expected 5 demo papers, got {count}"

    def test_papers_have_abstracts(self, demo_db):
        from src.db import init_db
        from src.db import Paper
        from sqlalchemy.orm import sessionmaker

        engine = init_db(demo_db)
        Session = sessionmaker(bind=engine)
        sess = Session()
        papers = sess.query(Paper).all()
        sess.close()

        for p in papers:
            assert p.abstract, f"Paper {p.paper_id} has no abstract"
            assert len(p.abstract) > 20


class TestBM25:
    def test_build_and_search(self, demo_db):
        from src.db import init_db, Paper
        from src.retrieval.bm25_retriever import BM25Retriever
        from sqlalchemy.orm import sessionmaker

        engine = init_db(demo_db)
        Session = sessionmaker(bind=engine)
        sess = Session()
        papers = sess.query(Paper).all()
        sess.close()

        bm25 = BM25Retriever()
        bm25.build(
            abstracts=[p.abstract for p in papers],
            paper_ids=[p.paper_id for p in papers],
        )

        results = bm25.search("attention transformer", top_k=5)
        assert len(results) > 0
        assert all(r.score > 0 for r in results)
        assert results[0].rank == 1

    def test_returns_ranked_order(self, demo_db):
        from src.db import init_db, Paper
        from src.retrieval.bm25_retriever import BM25Retriever
        from sqlalchemy.orm import sessionmaker

        engine = init_db(demo_db)
        Session = sessionmaker(bind=engine)
        sess = Session()
        papers = sess.query(Paper).all()
        sess.close()

        bm25 = BM25Retriever()
        bm25.build([p.abstract for p in papers], [p.paper_id for p in papers])

        results = bm25.search("image recognition visual", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"


class TestFAISSIndex:
    def test_build_and_search_flat(self):
        from src.retrieval.faiss_index import FAISSIndex

        dim = 64
        N = 20
        idx = FAISSIndex(dim=dim, index_type="Flat", nprobe=1)

        vecs = np.random.randn(N, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        ids = list(range(N))

        idx.build(vecs, ids)
        assert idx.index.ntotal == N

        results = idx.search(vecs[0:1], top_k=5)
        assert len(results) == 5
        # First result should be the query itself (score ≈ 1.0)
        assert results[0].faiss_id == 0
        assert results[0].score > 0.99

    def test_save_and_load(self, tmp_data_dir):
        from src.retrieval.faiss_index import FAISSIndex

        dim = 32
        idx = FAISSIndex(dim=dim)
        vecs = np.random.randn(10, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        idx.build(vecs, list(range(10)))

        save_path = str(tmp_data_dir / "test.faiss")
        idx.save(save_path)

        idx2 = FAISSIndex(dim=dim)
        idx2.load(save_path)
        assert idx2.index.ntotal == 10

        r1 = idx.search(vecs[0:1], top_k=3)
        r2 = idx2.search(vecs[0:1], top_k=3)
        assert [r.faiss_id for r in r1] == [r.faiss_id for r in r2]


class TestQueryRouter:
    def test_text_query_routing(self):
        from src.fusion.query_router import route_query, QueryType
        r = route_query(text="BERT language model pre-training")
        assert r.query_type == QueryType.TEXT

    def test_cross_modal_routing(self):
        from src.fusion.query_router import route_query, QueryType
        r = route_query(text="U-shaped accuracy curve over epochs")
        assert r.query_type == QueryType.CROSS_MODAL

    def test_figure_routing(self):
        from src.fusion.query_router import route_query, QueryType
        from PIL import Image
        img = Image.new("RGB", (50, 50))
        r = route_query(image=img)
        assert r.query_type == QueryType.FIGURE
