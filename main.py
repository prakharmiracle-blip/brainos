"""
BrainOS — Main Entry Point
Run with: python main.py  OR  uvicorn main:app --reload
"""

import uvicorn
from src.api.app import app
from src.config import settings
from src.utils.logger import log

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  🧠  BrainOS v1.0 — Starting up")
    log.info(f"  LLM Provider : {settings.llm_provider}")
    log.info(f"  Embedding    : {settings.embedding_model}")
    log.info(f"  FAISS Path   : {settings.faiss_index_path}")
    log.info(f"  Server       : http://{settings.api_host}:{settings.api_port}")
    log.info("=" * 60)

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
