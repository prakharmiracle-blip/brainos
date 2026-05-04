"""
BrainOS — FastAPI Application
Endpoints: /ingest  /ask  /summary  /health  /stats
"""

from __future__ import annotations

import os
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BrainOS API",
    description="Your AI-powered personal memory and intelligence system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.upload_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Health / Stats
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/stats", tags=["System"])
def stats():
    return {
        "total_memory_chunks": vector_store.count(),
        "embedding_model": settings.embedding_model,
        "llm_provider": settings.llm_provider,
        "whisper_model": settings.whisper_model,
    }


# ---------------------------------------------------------------------------
# /ingest — Text & URL
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text_or_url(request: IngestRequest):
    """
    Ingest text, a URL, or plain text content.

    Examples:
    - Plain text note
    - URL → will be fetched and extracted
    """
    try:
        if request.input_type == InputType.URL:
            ids = ingestion_pipeline.ingest_url(
                url=request.content,
                metadata=request.metadata,
            )
        else:
            ids = ingestion_pipeline.ingest_text(
                text=request.content,
                input_type=request.input_type,
                source=request.source,
                metadata=request.metadata,
            )

        return IngestResponse(
            success=True,
            entry_ids=ids,
            chunks_stored=len(ids),
            message=f"Stored {len(ids)} chunk(s) successfully.",
        )

    except Exception as e:
        log.error(f"/ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /ingest/voice — Audio file upload
# ---------------------------------------------------------------------------

@app.post("/ingest/voice", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_voice(
    file: UploadFile = File(...),
    source: str = Form(default=""),
):
    """
    Upload an audio file (mp3, wav, m4a, ogg, webm).
    Transcribed via Whisper then stored.
    """
    suffix = Path(file.filename or "audio").suffix or ".wav"
    save_path = Path(settings.upload_dir) / f"{uuid.uuid4()}{suffix}"

    try:
        # Save upload to disk
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        ids = ingestion_pipeline.ingest_voice(
            audio_path=save_path,
            metadata={"original_filename": file.filename},
        )

        return IngestResponse(
            success=True,
            entry_ids=ids,
            chunks_stored=len(ids),
            message=f"Voice transcribed and stored as {len(ids)} chunk(s).",
        )

    except Exception as e:
        log.error(f"/ingest/voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if save_path.exists():
            save_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# /ingest/pdf — PDF file upload
# ---------------------------------------------------------------------------

@app.post("/ingest/pdf", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(
    file: UploadFile = File(...),
):
    """Upload a PDF file for extraction and storage."""
    save_path = Path(settings.upload_dir) / f"{uuid.uuid4()}.pdf"

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        ids = ingestion_pipeline.ingest_pdf(
            pdf_path=save_path,
            metadata={"original_filename": file.filename},
        )

        return IngestResponse(
            success=True,
            entry_ids=ids,
            chunks_stored=len(ids),
            message=f"PDF extracted and stored as {len(ids)} chunk(s).",
        )

    except Exception as e:
        log.error(f"/ingest/pdf error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if save_path.exists():
            save_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# /ask — Brain Chat (RAG)
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["Brain Chat"])
async def ask_brain(request: AskRequest):
    """
    Query your stored memories using natural language.

    Examples:
    - "What did I learn about Python last week?"
    - "Summarise my notes on machine learning"
    - "What are my pending tasks?"
    """
    try:
        response = brain_chat_agent.ask(query=request.query, top_k=request.top_k)
        return response
    except Exception as e:
        log.error(f"/ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /summary — Daily Summary Agent
# ---------------------------------------------------------------------------

@app.get("/summary", response_model=SummaryResponse, tags=["Summary"])
@app.post("/summary", response_model=SummaryResponse, tags=["Summary"])
async def daily_summary(target_date: str | None = None):
    """
    Generate a structured daily summary from stored memories.
    Optionally pass ?target_date=YYYY-MM-DD to summarise a specific day.
    """
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


# ---------------------------------------------------------------------------
# /memory/recent — Debug: see recent entries
# ---------------------------------------------------------------------------

@app.get("/memory/recent", tags=["Debug"])
def recent_memories(n: int = 10):
    """Return the N most recently stored memory chunks."""
    entries = vector_store.get_recent(n=n)
    return {"count": len(entries), "entries": entries}
