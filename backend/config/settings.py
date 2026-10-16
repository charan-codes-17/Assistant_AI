from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Project root calculation:
# backend/config/settings.py -> backend/config -> backend -> root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    # Nebius Token Factory API
    NEBIUS_API_KEY: str = Field(
        default="",
        description="API key for Nebius Token Factory",
    )
    NEBIUS_BASE_URL: str = Field(
        default="https://api.tokenfactory.us-central1.nebius.com/v1/",
        description="Base URL for Nebius Token Factory OpenAI-compatible API",
    )

    # Model IDs (Never hardcoded in logic)
    CHAT_MODEL_FAST: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        description="Fast tier model for routing, extraction, and onboarding",
    )
    CHAT_MODEL_REASONING: str = Field(
        default="Llama-3_1-Nemotron-Ultra-253B-v1",
        description="Reasoning tier model for relevance judgment and intervention",
    )

    # Embedding model and dimension
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model identifier",
    )
    EMBEDDING_DIM: int = Field(
        default=384,
        description="Vector embedding dimension",
    )

    # Telegram bot
    TELEGRAM_BOT_TOKEN: str = Field(
        default="",
        description="Telegram bot token from BotFather",
    )

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/ai_assistant",
        description="PostgreSQL async connection string",
    )

    # Security & Authentication
    JWT_SECRET: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for signing JWT tokens",
    )

    model_config = SettingsConfigDict(
        env_file=(
            str(ROOT_DIR / ".env"),
            str(ROOT_DIR / "config" / ".env"),
            str(Path(__file__).resolve().parent.parent / ".env"),
            ".env",
            "config/.env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

