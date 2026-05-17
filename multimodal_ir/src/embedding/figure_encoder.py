# src/embedding/figure_encoder.py

"""
Encodes research paper figures (images) and their captions into a shared
512-dimensional CLIP embedding space.

Supports:
- image embeddings
- caption embeddings
- cross-modal text queries

Robust against:
- invalid images
- NaNs/Infs
- transformers version differences
- preprocessing failures
"""

import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.config import cfg


class FigureEncoder:
    def __init__(self, model_name: str = None):
        model_name = model_name or cfg.embedding.figure.model_name

        logger.info(f"Loading CLIP model: {model_name}")

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.dim = cfg.embedding.figure.dim

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {self.device}")

        self.model = self.model.to(self.device)
        self.model.eval()

    # ─────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────
    def _safe_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        L2 normalize safely.
        Prevents divide-by-zero + NaN propagation.
        """

        tensor = torch.nan_to_num(
            tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        norms = tensor.norm(dim=-1, keepdim=True)

        norms = torch.clamp(norms, min=1e-12)

        return tensor / norms

    def _extract_tensor_output(self, outputs):
        """
        Handle transformers version differences.
        """

        # Most common case
        if isinstance(outputs, torch.Tensor):
            return outputs

        # Some versions return wrapped objects
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output

        # Fallback
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state.mean(dim=1)

        raise ValueError(
            f"Unexpected CLIP output type: {type(outputs)}"
        )

    # ─────────────────────────────────────────────
    # Encode Images
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = None,
    ) -> np.ndarray:
        """
        Encode image paths into CLIP embeddings.

        Returns:
            np.ndarray shape (N, 512)
        """

        batch_size = batch_size or cfg.embedding.figure.batch_size

        all_embeddings = []

        for i in tqdm(
            range(0, len(image_paths), batch_size),
            desc="Encoding figures",
        ):
            batch_paths = image_paths[i : i + batch_size]

            images = []
            valid_mask = []

            # ─────────────────────────────────────────
            # Load images safely
            # ─────────────────────────────────────────
            for p in batch_paths:
                try:
                    path = Path(p)

                    if not path.exists():
                        raise FileNotFoundError(f"Missing image: {p}")

                    img = Image.open(path).convert("RGB")

                    # Reject tiny images
                    if img.width < 10 or img.height < 10:
                        raise ValueError(
                            f"Tiny image: {img.width}x{img.height}"
                        )

                    images.append(img)
                    valid_mask.append(True)

                except Exception as e:
                    logger.warning(f"Failed to load image {p}: {e}")

                    # Neutral placeholder image
                    images.append(
                        Image.new(
                            "RGB",
                            (224, 224),
                            color=(128, 128, 128),
                        )
                    )

                    valid_mask.append(False)

            # ─────────────────────────────────────────
            # Preprocess
            # ─────────────────────────────────────────
            try:
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )

                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                }

            except Exception as e:
                logger.error(f"CLIP preprocessing failed: {e}")

                zero_vecs = np.zeros(
                    (len(images), self.dim),
                    dtype=np.float32,
                )

                all_embeddings.append(zero_vecs)
                continue

            # ─────────────────────────────────────────
            # Encode
            # ─────────────────────────────────────────
            try:
                outputs = self.model.get_image_features(**inputs)

                image_features = self._extract_tensor_output(outputs)

            except Exception as e:
                logger.error(f"CLIP image encoding failed: {e}")

                zero_vecs = np.zeros(
                    (len(images), self.dim),
                    dtype=np.float32,
                )

                all_embeddings.append(zero_vecs)
                continue

            # ─────────────────────────────────────────
            # Debug info
            # ─────────────────────────────────────────
            logger.debug(
                f"Image feature tensor shape: {image_features.shape}"
            )

            logger.debug(
                f"HAS NAN: {torch.isnan(image_features).any().item()}"
            )

            logger.debug(
                f"HAS INF: {torch.isinf(image_features).any().item()}"
            )

            # ─────────────────────────────────────────
            # Normalize safely
            # ─────────────────────────────────────────
            image_features = self._safe_normalize(image_features)

            batch_emb = (
                image_features.cpu()
                .numpy()
                .astype(np.float32)
            )

            # Zero-out invalid images
            for j, valid in enumerate(valid_mask):
                if not valid:
                    batch_emb[j] = 0.0

            all_embeddings.append(batch_emb)

        # ─────────────────────────────────────────────
        # Final combine
        # ─────────────────────────────────────────────
        if not all_embeddings:
            logger.warning("No figure embeddings generated")

            return np.zeros(
                (1, self.dim),
                dtype=np.float32,
            )

        return np.vstack(all_embeddings)

    # ─────────────────────────────────────────────
    # Encode Captions
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def encode_captions(
        self,
        captions: list[str],
        batch_size: int = None,
    ) -> np.ndarray:
        """
        Encode figure captions into CLIP text embeddings.
        """

        batch_size = batch_size or cfg.embedding.figure.batch_size

        all_embeddings = []

        for i in tqdm(
            range(0, len(captions), batch_size),
            desc="Encoding captions",
        ):
            batch = captions[i : i + batch_size]

            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            try:
                outputs = self.model.get_text_features(**inputs)

                text_features = self._extract_tensor_output(outputs)

            except Exception as e:
                logger.error(f"CLIP caption encoding failed: {e}")

                zero_vecs = np.zeros(
                    (len(batch), self.dim),
                    dtype=np.float32,
                )

                all_embeddings.append(zero_vecs)
                continue

            text_features = self._safe_normalize(text_features)

            all_embeddings.append(
                text_features.cpu()
                .numpy()
                .astype(np.float32)
            )

        return np.vstack(all_embeddings)

    # ─────────────────────────────────────────────
    # Encode Query Text
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text query into CLIP text space.
        """

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model.get_text_features(**inputs)

        features = self._extract_tensor_output(outputs)

        features = self._safe_normalize(features)

        return (
            features.cpu()
            .numpy()
            .astype(np.float32)
        )

    # ─────────────────────────────────────────────
    # Encode + Save
    # ─────────────────────────────────────────────
    def encode_and_save(
        self,
        image_paths: list[str],
        figure_ids: list[int],
        out_path: str,
        ids_path: str = None,
    ) -> np.ndarray:
        """
        Encode images and save embeddings + ID mappings.
        """

        vectors = self.encode_images(image_paths)

        Path(out_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(out_path, vectors)

        logger.success(
            f"Saved figure embeddings {vectors.shape} → {out_path}"
        )

        if ids_path is None:
            ids_path = out_path.replace(".npy", "_ids.json")

        with open(ids_path, "w") as f:
            json.dump(figure_ids, f)

        logger.success(
            f"Saved figure ID mapping → {ids_path}"
        )

        return vectors