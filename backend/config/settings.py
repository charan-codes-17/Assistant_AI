from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/config/settings.py -> config -> backend -> repo root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration, loaded from the repo-root .env (and real env vars)."""

    # Environment
    APP_ENV: str = Field(default="dev", description="dev | prod")

    # Nebius Token Factory (OpenAI-compatible)
    NEBIUS_API_KEY: str = Field(default="", description="Nebius Token Factory API key")
    NEBIUS_BASE_URL: str = Field(
        default="https://api.tokenfactory.nebius.com/v1/",
    )

    # Chat model IDs: verify exact IDs in the Nebius console; never hardcode in services.
    CHAT_MODEL_FAST: str = Field(
        description="Fast tier: routing, extraction, onboarding, plain chat",
    )
    CHAT_MODEL_REASONING: str = Field(
        description="Reasoning tier: relevance judgment and intervention",
    )
    # Optional non-reasoning model for the tool-calling step (roadmap §16 Risk 1)
    CHAT_MODEL_TOOLCALL: str = Field(default="", description="Falls back to FAST if empty")

    # Embeddings: dimension MUST match the model; it is baked into the DB schema.
    EMBEDDING_MODEL: str = Field(description="Embedding model ID")
    EMBEDDING_DIM: int = Field(description="Vector dimension of EMBEDDING_MODEL")

    # Telegram
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Dev or deployed bot token")

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/ai_assistant",
    )

    # Auth (needed from Phase 7; no insecure default)
    JWT_SECRET: str = Field(default="", description="Required in Phase 7")

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def require(self, *names: str) -> None:
        """Fail loudly if any named setting is empty. Call at the top of scripts/entrypoints."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(f"Missing required settings in .env: {', '.join(missing)}")


settings = Settings()
