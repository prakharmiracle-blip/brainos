"""
BrainOS — Core Data Models
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InputType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    URL = "url"
    PDF = "pdf"


class MemoryEntry(BaseModel):
    """A single piece of ingested memory."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    input_type: InputType
    source: str = ""           # original filename, URL, etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Set after embedding
    chunk_index: int = 0
    total_chunks: int = 1


class IngestRequest(BaseModel):
    """API request body for /ingest (text/URL path)."""

    content: str
    input_type: InputType = InputType.TEXT
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    success: bool
    entry_ids: list[str]
    chunks_stored: int
    message: str = ""


class AskRequest(BaseModel):
    """API request body for /ask."""

    query: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    query: str


class SummaryResponse(BaseModel):
    date: str
    insights: list[str]
    key_learnings: list[str]
    important_notes: list[str]
    raw_summary: str
