"""
BrainOS — Extended Models for NotebookLM Features
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ---------------------------------------------------------------------------
# Source Document Models
# ---------------------------------------------------------------------------

class SourceDocument(BaseModel):
    """Represents a single source document in the library"""
    id: str
    title: str
    input_type: str  # text, pdf, voice, url
    upload_date: Optional[str] = None
    file_size: int = 0
    chunk_count: int = 0
    tags: List[str] = Field(default_factory=list)
    topic: Optional[str] = None
    status: str = "active"  # active, archived
    preview: str = ""


class SourceListResponse(BaseModel):
    """Response for /sources endpoint"""
    sources: List[SourceDocument]
    total: int
    filtered: bool = False


class SourceDetailResponse(BaseModel):
    """Detailed information about a specific document"""
    id: str
    title: str
    input_type: str
    upload_date: Optional[str]
    file_size: int
    chunk_count: int
    tags: List[str]
    topic: Optional[str]
    status: str
    content_preview: str = ""
    chunk_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceUpdateRequest(BaseModel):
    """Request to update source metadata"""
    tags: Optional[List[str]] = None
    status: Optional[str] = None  # active, archived
    topic: Optional[str] = None


# ---------------------------------------------------------------------------
# Topic Models
# ---------------------------------------------------------------------------

class TopicResponse(BaseModel):
    """Represents a topic/category"""
    name: str
    document_count: int
    document_ids: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Filtered Chat Models
# ---------------------------------------------------------------------------

class FilteredAskRequest(BaseModel):
    """Request to chat with filtered sources"""
    query: str
    top_k: int = 5
    source_ids: Optional[List[str]] = None  # Specific document IDs
    tags: Optional[List[str]] = None  # Filter by tags
    input_types: Optional[List[str]] = None  # Filter by type (pdf, voice, etc)


# ---------------------------------------------------------------------------
# Insights Models
# ---------------------------------------------------------------------------

class InsightResponse(BaseModel):
    """Extracted insights from documents"""
    key_points: List[str] = Field(default_factory=list)
    main_topics: List[str] = Field(default_factory=list)
    important_dates: List[Dict[str, str]] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    summary: str = ""
    source_ids: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Enhanced IngestResponse
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    """Response after ingesting content"""
    success: bool
    entry_ids: List[str]
    chunks_stored: int
    message: str
    source_id: Optional[str] = None  # NEW: Document ID for library reference
