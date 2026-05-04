"""
BrainOS — Text Preprocessor
Cleans raw text and splits it into overlapping chunks for embedding.
"""

from __future__ import annotations

import re
from src.config import settings
from src.utils.logger import log


class TextProcessor:
    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Basic text normalisation."""
        text = text.strip()
        text = re.sub(r"\r\n|\r", "\n", text)          # normalise newlines
        text = re.sub(r"\n{3,}", "\n\n", text)          # collapse blank lines
        text = re.sub(r"[ \t]{2,}", " ", text)          # collapse spaces
        text = re.sub(r"[^\S\n]+", " ", text)           # trailing spaces per line
        return text

    def chunk(self, text: str) -> list[str]:
        """
        Split text into overlapping word-based chunks.
        Returns a list of chunk strings.
        """
        text = self.clean(text)
        words = text.split()

        if not words:
            return []

        chunks: list[str] = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end == len(words):
                break

            start += self.chunk_size - self.chunk_overlap

        log.debug(f"Chunked text into {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def process(self, text: str) -> list[str]:
        """Full pipeline: clean → chunk."""
        cleaned = self.clean(text)
        return self.chunk(cleaned)


# Singleton
text_processor = TextProcessor()
