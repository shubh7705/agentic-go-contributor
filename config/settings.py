"""
config/settings.py — Centralized configuration using Pydantic Settings.
Reads from environment variables / .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenRouter ──────────────────────────────────────────────────────────
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key (set via OPENROUTER_API_KEY env var or .env file)",
    )
    openrouter_model: str = Field(
        default="moonshotai/kimi-k2-0905",
        description="Model identifier on OpenRouter",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter's OpenAI-compatible endpoint",
    )

    # ── GitHub ───────────────────────────────────────────────────────────────
    github_token: Optional[str] = Field(
        default=None,
        description="GitHub personal access token (optional, for higher rate limits)",
    )

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model name",
    )

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(
        default=10,
        description="Number of top files to retrieve",
    )
    embedding_weight: float = Field(
        default=0.6,
        description="Weight for embedding score in hybrid retrieval",
    )
    rg_weight: float = Field(
        default=0.4,
        description="Weight for ripgrep score in hybrid retrieval",
    )

    # ── Repair Loop ──────────────────────────────────────────────────────────
    max_repair_iterations: int = Field(
        default=3,
        description="Maximum number of repair loop iterations",
    )

    # ── File Paths ───────────────────────────────────────────────────────────
    repo_temp_dir: Path = Field(
        default=Path("./tmp/repos"),
        description="Directory for temporarily cloned repositories",
    )
    output_dir: Path = Field(
        default=Path("./outputs"),
        description="Directory for final PR output files",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # ── LLM Parameters ───────────────────────────────────────────────────────
    llm_temperature: float = Field(
        default=0.1,
        description="LLM temperature (low for deterministic code generation)",
    )
    llm_max_tokens: int = Field(
        default=8192,
        description="Maximum output tokens per LLM call",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @field_validator("repo_temp_dir", "output_dir", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        return Path(v)


# Singleton instance — import this everywhere
settings = Settings()  # type: ignore[call-arg]
