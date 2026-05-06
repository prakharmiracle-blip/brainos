"""
BrainOS — Enhanced FastAPI Application with NotebookLM Features
NEW Endpoints: 
- /sources (library management)
- /sources/{id} (individual document management)
- /topics (auto-discovered topics)
- /ask/filtered (chat with specific sources)
- /export/insights (extract key learnings)
"""

from __future__ import annotations

import os
import uuid
import json
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agents.chat_agent import brain_chat_agent
from src.agents.summary_agent import summary_agent
from src.config import settings
from src.ingestion.pipeline import ingestion_pipeline
from src.memory.vector_store import vector_store
from src.utils.logger import log
from src.utils.models import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    InputType,
    SummaryResponse,
)

# Import new models for NotebookLM features
from src.utils.notebook_models import (
    SourceDocument,
    SourceListResponse,
    SourceDetailResponse,
    TopicResponse,
    FilteredAskRequest,
    InsightResponse,
    SourceUpdateRequest,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BrainOS API - Enhanced",
    description="Your AI-powered personal memory with NotebookLM-style organization.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(Path(settings.upload_dir) / "sources", exist_ok=True)

# Document metadata storage (in production, use SQLite/Postgres)
METADATA_FILE = Path(settings.upload_dir) / "documents_metadata.json"

def load_metadata():
    """Load document metadata from JSON file"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    """Save document metadata to JSON file"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# Health / Stats
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "version": "2.0.0", "features": ["sources", "topics", "filtered_chat"]}


@app.get("/stats", tags=["System"])
def stats():
    metadata = load_metadata()
    return {
        "total_memory_chunks": vector_store.count(),
        "total_documents": len(metadata),
        "embedding_model": settings.embedding_model,
        "llm_provider": settings.llm_provider,
        "whisper_model": settings.whisper_model,
    }


# ---------------------------------------------------------------------------
# 🆕 DOCUMENT LIBRARY — /sources
# ---------------------------------------------------------------------------

@app.get("/sources", response_model=SourceListResponse, tags=["Document Library"])
def list_sources(
    tag: Optional[str] = Query(None, description="Filter by tag"),
    input_type: Optional[str] = Query(None, description="Filter by type (text, pdf, voice, url)"),
    search: Optional[str] = Query(None, description="Search in title/content"),
):
    """
    Get all uploaded source documents with metadata.
    Supports filtering by tag, type, and search.
    """
    try:
        metadata = load_metadata()
        sources = []
        
        for doc_id, doc_data in metadata.items():
            # Apply filters
            if tag and tag not in doc_data.get('tags', []):
                continue
            if input_type and doc_data.get('input_type') != input_type:
                continue
            if search and search.lower() not in doc_data.get('title', '').lower():
                continue
                
            sources.append(SourceDocument(
                id=doc_id,
                title=doc_data.get('title', 'Untitled'),
                input_type=doc_data.get('input_type', 'text'),
                upload_date=doc_data.get('upload_date'),
                file_size=doc_data.get('file_size', 0),
                chunk_count=doc_data.get('chunk_count', 0),
                tags=doc_data.get('tags', []),
                topic=doc_data.get('topic'),
                status=doc_data.get('status', 'active'),
                preview=doc_data.get('preview', ''),
            ))
        
        # Sort by upload date (newest first)
        sources.sort(key=lambda x: x.upload_date or '', reverse=True)
        
        return SourceListResponse(
            sources=sources,
            total=len(sources),
            filtered=len(sources) < len(metadata),
        )
        
    except Exception as e:
        log.error(f"/sources error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources/{source_id}", response_model=SourceDetailResponse, tags=["Document Library"])
def get_source_detail(source_id: str):
    """Get detailed information about a specific document"""
    try:
        metadata = load_metadata()
        
        if source_id not in metadata:
            raise HTTPException(status_code=404, detail="Source not found")
        
        doc_data = metadata[source_id]
        
        return SourceDetailResponse(
            id=source_id,
            title=doc_data.get('title', 'Untitled'),
            input_type=doc_data.get('input_type', 'text'),
            upload_date=doc_data.get('upload_date'),
            file_size=doc_data.get('file_size', 0),
            chunk_count=doc_data.get('chunk_count', 0),
            tags=doc_data.get('tags', []),
            topic=doc_data.get('topic'),
            status=doc_data.get('status', 'active'),
            content_preview=doc_data.get('content_preview', ''),
            chunk_ids=doc_data.get('chunk_ids', []),
            metadata=doc_data.get('extra_metadata', {}),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/sources/{source_id} error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/sources/{source_id}", tags=["Document Library"])
def update_source(source_id: str, update: SourceUpdateRequest):
    """Update document metadata (tags, status, topic)"""
    try:
        metadata = load_metadata()
        
        if source_id not in metadata:
            raise HTTPException(status_code=404, detail="Source not found")
        
        doc_data = metadata[source_id]
        
        if update.tags is not None:
            doc_data['tags'] = update.tags
        if update.status is not None:
            doc_data['status'] = update.status
        if update.topic is not None:
            doc_data['topic'] = update.topic
            
        metadata[source_id] = doc_data
        save_metadata(metadata)
        
        return {"success": True, "message": "Source updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"PATCH /sources/{source_id} error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sources/{source_id}", tags=["Document Library"])
def delete_source(source_id: str):
    """Delete a source document and its embeddings"""
    try:
        metadata = load_metadata()
        
        if source_id not in metadata:
            raise HTTPException(status_code=404, detail="Source not found")
        
        doc_data = metadata[source_id]
        
        # Delete embeddings from vector store
        chunk_ids = doc_data.get('chunk_ids', [])
        if chunk_ids:
            vector_store.delete_entries(chunk_ids)
        
        # Delete file if exists
        file_path = doc_data.get('file_path')
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()
        
        # Remove from metadata
        del metadata[source_id]
        save_metadata(metadata)
        
        return {"success": True, "message": "Source deleted", "chunks_removed": len(chunk_ids)}
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"DELETE /sources/{source_id} error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 🆕 TOPICS — Auto-discovery
# ---------------------------------------------------------------------------

@app.get("/topics", response_model=List[TopicResponse], tags=["Topics"])
def get_topics():
    """
    Get auto-discovered topics from uploaded documents.
    Uses clustering on embeddings to find natural groupings.
    """
    try:
        metadata = load_metadata()
        
        # Count documents per topic
        topic_counts = {}
        for doc_id, doc_data in metadata.items():
            if doc_data.get('status') == 'active':
                topic = doc_data.get('topic', 'Uncategorized')
                if topic not in topic_counts:
                    topic_counts[topic] = {
                        'name': topic,
                        'document_count': 0,
                        'document_ids': []
                    }
                topic_counts[topic]['document_count'] += 1
                topic_counts[topic]['document_ids'].append(doc_id)
        
        # Convert to response format
        topics = [
            TopicResponse(
                name=data['name'],
                document_count=data['document_count'],
                document_ids=data['document_ids']
            )
            for data in topic_counts.values()
        ]
        
        # Sort by document count
        topics.sort(key=lambda x: x.document_count, reverse=True)
        
        return topics
        
    except Exception as e:
        log.error(f"/topics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 🆕 FILTERED CHAT — Chat with specific sources
# ---------------------------------------------------------------------------

@app.post("/ask/filtered", response_model=AskResponse, tags=["Brain Chat"])
async def ask_brain_filtered(request: FilteredAskRequest):
    """
    Query specific source documents only.
    
    Examples:
    - Chat with only PDF documents
    - Chat with documents tagged "research"
    - Chat with specific document IDs
    """
    try:
        metadata = load_metadata()
        
        # Filter chunk IDs based on criteria
        allowed_chunk_ids = set()
        
        for doc_id, doc_data in metadata.items():
            should_include = True
            
            # Filter by source IDs
            if request.source_ids and doc_id not in request.source_ids:
                should_include = False
            
            # Filter by tags
            if request.tags:
                doc_tags = set(doc_data.get('tags', []))
                if not doc_tags.intersection(request.tags):
                    should_include = False
            
            # Filter by input type
            if request.input_types and doc_data.get('input_type') not in request.input_types:
                should_include = False
            
            if should_include:
                allowed_chunk_ids.update(doc_data.get('chunk_ids', []))
        
        # Perform RAG with filtered chunks
        response = brain_chat_agent.ask_filtered(
            query=request.query,
            top_k=request.top_k,
            allowed_chunk_ids=list(allowed_chunk_ids)
        )
        
        return response
        
    except Exception as e:
        log.error(f"/ask/filtered error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 🆕 INSIGHTS EXTRACTION
# ---------------------------------------------------------------------------

@app.post("/export/insights", response_model=InsightResponse, tags=["Insights"])
async def extract_insights(
    source_ids: List[str] = Query(..., description="Document IDs to analyze"),
):
    """
    Extract key insights, learnings, and important notes from specific documents.
    Similar to NotebookLM's briefing document feature.
    """
    try:
        metadata = load_metadata()
        
        # Gather all chunks from specified sources
        all_chunks = []
        for source_id in source_ids:
            if source_id in metadata:
                chunk_ids = metadata[source_id].get('chunk_ids', [])
                chunks = vector_store.get_chunks_by_ids(chunk_ids)
                all_chunks.extend(chunks)
        
        if not all_chunks:
            raise HTTPException(status_code=404, detail="No content found for specified sources")
        
        # Use LLM to extract structured insights
        from src.agents.insight_agent import insight_agent
        insights = insight_agent.extract(chunks=all_chunks)
        
        return insights
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/export/insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ORIGINAL ENDPOINTS (Enhanced with metadata tracking)
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text_or_url(request: IngestRequest):
    """
    Ingest text, a URL, or plain text content.
    NOW ENHANCED: Automatically creates source document entry.
    """
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        if request.input_type == InputType.URL:
            ids = ingestion_pipeline.ingest_url(
                url=request.content,
                metadata=request.metadata,
            )
            title = request.content[:100]
        else:
            ids = ingestion_pipeline.ingest_text(
                text=request.content,
                input_type=request.input_type,
                source=request.source,
                metadata=request.metadata,
            )
            title = request.content[:50] + "..." if len(request.content) > 50 else request.content
        
        # Save document metadata
        metadata = load_metadata()
        metadata[doc_id] = {
            'id': doc_id,
            'title': title,
            'input_type': request.input_type.value,
            'upload_date': datetime.utcnow().isoformat(),
            'chunk_count': len(ids),
            'chunk_ids': ids,
            'tags': request.metadata.get('tags', []) if request.metadata else [],
            'topic': request.metadata.get('topic', 'General') if request.metadata else 'General',
            'status': 'active',
            'preview': request.content[:200],
            'content_preview': request.content[:500],
        }
        save_metadata(metadata)

        return IngestResponse(
            success=True,
            entry_ids=ids,
            chunks_stored=len(ids),
            message=f"Stored {len(ids)} chunk(s) successfully.",
            source_id=doc_id,  # NEW: Return source ID
        )

    except Exception as e:
        log.error(f"/ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/voice", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_voice(
    file: UploadFile = File(...),
    source: str = Form(default=""),
):
    """
    Upload an audio file (mp3, wav, m4a, ogg, webm).
    NOW ENHANCED: Saves file and tracks as source document.
    """
    doc_id = str(uuid.uuid4())
    suffix = Path(file.filename or "audio").suffix or ".wav"
    save_path = Path(settings.upload_dir) / "sources" / f"{doc_id}{suffix}"

    try:
        # Save upload to permanent location
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        ids = ingestion_pipeline.ingest_voice(
            audio_path=save_path,
            metadata={"original_filename": file.filename},
        )
        
        # Save document metadata
        metadata = load_metadata()
        metadata[doc_id] = {
            'id': doc_id,
            'title': file.filename or 'Voice Recording',
            'input_type': 'voice',
            'upload_date': datetime.utcnow().isoformat(),
            'file_size': len(content),
            'file_path': str(save_path),
            'chunk_count': len(ids),
            'chunk_ids': ids,
            'tags': [],
            'topic': 'Voice Notes',
            'status': 'active',
        }
        save_metadata(metadata)

        return IngestResponse(
            success=True,
            entry_ids=ids,
            chunks_stored=len(ids),
            message=f"Voice transcribed and stored as {len(ids)} chunk(s).",
            source_id=doc_id,
        )

    except Exception as e:
        log.error(f"/ingest/voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(
    file: UploadFile = File(...),
):
    """
    Upload a PDF file for extraction and storage.
    NOW ENHANCED: Saves file and tracks as source document.
    """
    doc_id = str(uuid.uuid4())
    save_path = Path(settings.upload_dir) / "sources" / f"{doc_id}.pdf"

    try:
        # Save to permanent location
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        ids = ingestion_pipeline.ingest_pdf(
            pdf_path=save_path,
            metadata={"original_filename": file.filename},
        )
        
        # Save document metadata
        metadata = load_metadata()
        metadata[doc_id] = {
            'id': doc_id,
            'title': file.filename or 'PDF Document',
            'input_type': 'pdf',
            'upload_date': datetime.utcnow().isoformat(),
            'file_size': len(content),
            'file_path': str(save_path),
            'chunk_count': len(ids),
            'chunk_ids': ids,
            'tags': [],
            'topic': 'Documents',
            'status': 'active',
        }
        save_metadata(metadata)

        return IngestResponse(
            success=True,
            entry_ids=ids,
            chunks_stored=len(ids),
            message=f"PDF extracted and stored as {len(ids)} chunk(s).",
            source_id=doc_id,
        )

    except Exception as e:
        log.error(f"/ingest/pdf error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse, tags=["Brain Chat"])
async def ask_brain(request: AskRequest):
    """
    Query your stored memories using natural language.
    """
    try:
        response = brain_chat_agent.ask(query=request.query, top_k=request.top_k)
        return response
    except Exception as e:
        log.error(f"/ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary", response_model=SummaryResponse, tags=["Summary"])
@app.post("/summary", response_model=SummaryResponse, tags=["Summary"])
async def daily_summary(target_date: str | None = None):
    """Generate a structured daily summary from stored memories."""
    try:
        parsed_date = None
        if target_date:
            parsed_date = date.fromisoformat(target_date)

        result = summary_agent.generate(target_date=parsed_date)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        log.error(f"/summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/recent", tags=["Debug"])
def recent_memories(n: int = 10):
    """Return the N most recently stored memory chunks."""
    entries = vector_store.get_recent(n=n)
    return {"count": len(entries), "entries": entries}
