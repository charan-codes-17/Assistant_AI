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
    EMBEDDING_TIMEOUT_SECONDS: float = Field(
        default=4.0,
        description=(
            "Per-attempt wall-clock timeout for the query embedding call in retrieval. One "
            "retry is attempted, so the total embedding budget is EMBEDDING_TIMEOUT_SECONDS * 2; "
            "after that the existing retrieval fail-open applies."
        ),
    )
    EMBEDDING_BATCH_TIMEOUT_SECONDS: float = Field(
        default=20.0,
        description=(
            "Per-attempt wall-clock timeout for all non-query (batch) embedding calls, e.g. "
            "extraction dedupe and edit_record. One retry is attempted, so the total budget is "
            "EMBEDDING_BATCH_TIMEOUT_SECONDS * 2. This is the default when embed_texts is called "
            "without an explicit timeout."
        ),
    )

    # Telegram
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Dev or deployed bot token")

    # Database
    DATABASE_URL: str = Field(description="PostgreSQL async connection string")

    # Deduplication
    DEDUPE_THRESHOLD: float = Field(
        default=0.90,
        description="Cosine similarity threshold for deduplicating knowledge records",
    )

    # Retrieval & Relevance
    RETRIEVAL_TOP_K: int = Field(
        default=5,
        description="Max candidate records to retrieve from vector search",
    )
    RETRIEVAL_MIN_SIMILARITY: float = Field(
        default=0.40,
        description="Minimum cosine similarity for retrieval candidates",
    )
    RELEVANCE_MIN_CONFIDENCE: int = Field(
        default=50,
        description="Minimum confidence score (0-100) for relevance judgment",
    )
    RETRIEVAL_TIMEOUT_SECONDS: float = Field(
        default=15.0,
        description="Wall-clock timeout in seconds for the full retrieve_memory call",
    )

    # Intervention & Scheduler Reminders
    REMINDER_LEAD_MINUTES: int = Field(
        default=60,
        description="Lead time in minutes before task due_at to fire deadline reminders",
    )
    INTERVENTION_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        description="Wall-clock timeout in seconds for the conversation-path intervention LLM call",
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
