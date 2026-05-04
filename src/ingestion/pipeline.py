"""
BrainOS — Ingestion Pipeline
Single entry point for all input types.
Routes to the right ingester → processes → embeds → stores.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import settings
from src.memory.vector_store import vector_store
from src.processing.embedder import embedding_engine
from src.processing.text_processor import text_processor
from src.utils.logger import log
from src.utils.models import InputType, MemoryEntry


class IngestionPipeline:

    def ingest_text(
        self,
        text: str,
        input_type: InputType = InputType.TEXT,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Ingest raw text (already extracted).
        Returns list of entry IDs stored.
        """
        metadata = metadata or {}
        chunks = text_processor.process(text)
        if not chunks:
            log.warning("Ingestion produced 0 chunks — skipping.")
            return []

        entries: list[MemoryEntry] = []
        for i, chunk in enumerate(chunks):
            entry = MemoryEntry(
                content=chunk,
                input_type=input_type,
                source=source,
                metadata=metadata,
                chunk_index=i,
                total_chunks=len(chunks),
            )
            entries.append(entry)

        # Embed all chunks at once
        texts = [e.content for e in entries]
        vectors = embedding_engine.embed(texts)

        # Build metadata dicts for the vector store
        meta_list = [
            {
                "id": e.id,
                "content": e.content,
                "input_type": e.input_type,
                "source": e.source,
                "timestamp": e.timestamp.isoformat(),
                "chunk_index": e.chunk_index,
                "total_chunks": e.total_chunks,
                **e.metadata,
            }
            for e in entries
        ]

        vector_store.add(vectors, meta_list)
        ids = [e.id for e in entries]
        log.info(f"Ingested {len(ids)} chunks (type={input_type}, source='{source}').")
        return ids

    def ingest_voice(self, audio_path: str | Path, metadata: dict[str, Any] | None = None) -> list[str]:
        """Transcribe audio then ingest as text."""
        from src.ingestion.voice import voice_ingester
        text = voice_ingester.transcribe(audio_path)
        return self.ingest_text(
            text,
            input_type=InputType.VOICE,
            source=str(audio_path),
            metadata=metadata,
        )

    def ingest_url(self, url: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Fetch URL content then ingest."""
        from src.ingestion.document import document_ingester
        text = document_ingester.fetch_url(url)
        return self.ingest_text(
            text,
            input_type=InputType.URL,
            source=url,
            metadata=metadata,
        )

    def ingest_pdf(self, pdf_path: str | Path, metadata: dict[str, Any] | None = None) -> list[str]:
        """Extract PDF text then ingest."""
        from src.ingestion.document import document_ingester
        text = document_ingester.read_pdf(pdf_path)
        return self.ingest_text(
            text,
            input_type=InputType.PDF,
            source=str(pdf_path),
            metadata=metadata,
        )


# Singleton
ingestion_pipeline = IngestionPipeline()
