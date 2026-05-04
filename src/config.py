"""
BrainOS — Central Configuration
Loads settings from environment / .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = "openrouter"
    openai_api_key: str = ""
    together_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str = "http://localhost:8000"   # shown in OR dashboard
    openrouter_site_name: str = "BrainOS"
    llm_model: str = "mistralai/mistral-7b-instruct:free"  # free default

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # matches all-MiniLM-L6-v2

    # Storage
    faiss_index_path: str = "data/faiss_index"
    upload_dir: str = "data/uploads"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-me-in-production"

    # Whisper
    whisper_model: str = "base"

    # RAG
    top_k_retrieval: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Summary
    summary_hour: int = 20

    # Logging
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
