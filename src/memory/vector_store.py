"""
BrainOS — FAISS Vector Store
Manages the FAISS index + a parallel JSON metadata store.
Thread-safe writes via a simple lock.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import settings
from src.utils.logger import log


FAISS_FILE = "index.faiss"
META_FILE = "metadata.json"


class VectorStore:
    def __init__(self, index_path: str = settings.faiss_index_path):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self.index: faiss.IndexFlatIP  # inner-product (cosine via normalised vecs)
        self.metadata: list[dict[str, Any]] = []  # parallel list to FAISS rows

        self._load_or_create()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray, meta_list: list[dict[str, Any]]) -> list[int]:
        """
        Add embeddings + metadata to the store.
        Returns list of internal FAISS ids assigned.
        """
        assert len(vectors) == len(meta_list), "vectors and meta_list must be same length"

        with self._lock:
            start_id = self.index.ntotal
            self.index.add(vectors)
            self.metadata.extend(meta_list)
            self._save()

        ids = list(range(start_id, start_id + len(vectors)))
        log.debug(f"Stored {len(vectors)} vectors. Total in index: {self.index.ntotal}")
        return ids

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find top-k nearest neighbours.
        Returns list of metadata dicts with added 'score' field.
        """
        if self.index.ntotal == 0:
            log.warning("Vector store is empty — nothing to search.")
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for unfilled slots
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results

    def count(self) -> int:
        return self.index.ntotal

    def get_recent(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the n most recently added entries (by list order)."""
        return self.metadata[-n:] if self.metadata else []

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_or_create(self):
        faiss_path = self.index_path / FAISS_FILE
        meta_path = self.index_path / META_FILE

        if faiss_path.exists() and meta_path.exists():
            log.info(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(faiss_path))
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            log.info(f"Loaded {self.index.ntotal} vectors from disk.")
        else:
            log.info("Creating new FAISS index (IndexFlatIP).")
            self.index = faiss.IndexFlatIP(settings.embedding_dim)
            self.metadata = []

    def _save(self):
        """Persist index and metadata to disk (call within lock)."""
        faiss.write_index(self.index, str(self.index_path / FAISS_FILE))
        with open(self.index_path / META_FILE, "w") as f:
            json.dump(self.metadata, f, default=str, indent=2)


# Singleton
vector_store = VectorStore()
