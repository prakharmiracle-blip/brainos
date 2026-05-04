"""
BrainOS — Embedding Engine
Wraps sentence-transformers to produce dense vectors.
Model runs entirely locally — no API key needed.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.utils.logger import log


class EmbeddingEngine:
    def __init__(self, model_name: str = settings.embedding_model):
        log.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        log.info(f"Embedding model ready. Dimension: {self.dim}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.
        Returns shape (N, dim) float32 array.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine similarity via dot product
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (dim,)."""
        return self.embed([text])[0]


# Singleton — loaded once at startup
embedding_engine = EmbeddingEngine()
