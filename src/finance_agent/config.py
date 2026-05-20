"""Typed application configuration loaded from environment / ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed runtime settings sourced from the environment.

    Model names live exclusively here — never hardcoded in Python source.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    google_api_key: SecretStr = Field(..., description="Google AI Studio API key.")
    gemini_model: str = Field(..., description="Chat model identifier, e.g. gemini-2.5-flash.")
    gemini_embedding_model: str = Field(
        ...,
        description="Embedding model identifier, e.g. models/text-embedding-004.",
    )

    filings_text_path: Path = Field(default=Path("data/Nvidia_10K_20240128.txt"))
    faiss_index_path: Path = Field(default=Path(".faiss_index/nvda_10k"))
    chunk_size: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    retrieval_k: int = Field(default=6, ge=1, le=50)

    news_max_articles: int = Field(default=8, ge=1, le=50)
    news_request_timeout: int = Field(default=20, ge=1, le=120)

    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, frozen application settings."""
    return Settings()  # type: ignore[call-arg]
