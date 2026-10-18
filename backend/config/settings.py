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
        default=5.0,
        description=(
            "Total wall-clock budget for the query embedding call in retrieval. The query path "
            "hedges: one request, plus a second identical request if the first is slow after "
            "EMBEDDING_HEDGE_SECONDS; whichever finishes first wins. When the budget is exhausted "
            "the existing retrieval fail-open applies."
        ),
    )
    EMBEDDING_HEDGE_SECONDS: float = Field(
        default=1.5,
        description=(
            "Seconds with no query-embedding result before the hedge fires a second identical "
            "request concurrently (query path only). Must be below EMBEDDING_TIMEOUT_SECONDS."
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

    # Onboarding (PRD FR17 / FR18)
    ONBOARDING_MAX_CHARS: int = Field(
        default=12000,
        description=(
            "Max characters accepted for an onboarding payload: the pasted import text, or the "
            "combined questionnaire transcript (each answer is capped at 1000 chars first). Larger "
            "input is rejected with an error and nothing is staged; input is never truncated silently."
        ),
    )
    ONBOARDING_MAX_CANDIDATES: int = Field(
        default=40,
        description=(
            "Max candidate records staged from a single onboarding parse. Valid items beyond this "
            "count are dropped and counted in OnboardingResult.dropped.over_cap."
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
