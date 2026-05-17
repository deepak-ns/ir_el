# src/embedding/text_encoder.py
"""
Encodes paper abstracts (and text queries) into dense vectors using Sentence-BERT.
Model: all-mpnet-base-v2  →  768-dimensional L2-normalized embeddings.

Usage:
    encoder = TextEncoder()
    vectors = encoder.encode(["abstract 1", "abstract 2"])   # shape (N, 768)
    encoder.encode_and_save(texts, ids, "data/indices/text_embeddings.npy")
"""

import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import cfg


class TextEncoder:
    def __init__(self, model_name: str = None):
        model_name = model_name or cfg.embedding.text.model_name
        logger.info(f"Loading text encoder: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = cfg.embedding.text.max_seq_length
        self.dim = cfg.embedding.text.dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = None,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings → numpy array of shape (N, dim).
        Vectors are L2-normalized for cosine similarity via inner product.
        """
        batch_size = batch_size or cfg.embedding.text.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_and_save(
        self,
        texts: list[str],
        paper_ids: list[str],
        out_path: str,
        ids_path: str = None,
    ) -> np.ndarray:
        """
        Encode texts and save to .npy file.
        Also saves paper_id order to a parallel .json file.
        Returns the embedding matrix.
        """
        logger.info(f"Encoding {len(texts)} abstracts...")
        vectors = self.encode(texts)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, vectors)
        logger.success(f"Saved text embeddings {vectors.shape} → {out_path}")

        if ids_path is None:
            ids_path = out_path.replace(".npy", "_ids.json")
        import json
        with open(ids_path, "w") as f:
            json.dump(paper_ids, f)
        logger.success(f"Saved paper ID mapping → {ids_path}")

        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns shape (1, dim)."""
        return self.encode([query], show_progress=False)
