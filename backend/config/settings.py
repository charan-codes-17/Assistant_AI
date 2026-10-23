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
        default=6.0,
        description=(
            "Total wall-clock budget for the query embedding call in retrieval. The query path "
            "hedges (strategy B): a new identical request is fired every EMBEDDING_HEDGE_SECONDS "
            "up to EMBEDDING_HEDGE_MAX_IN_FLIGHT requests, the first success wins and the rest are "
            "cancelled. When the budget is exhausted the existing retrieval fail-open applies."
        ),
    )
    EMBEDDING_HEDGE_SECONDS: float = Field(
        default=0.8,
        description=(
            "Interval between hedged query-embedding requests (query path only): a new identical "
            "request is fired every this many seconds, up to EMBEDDING_HEDGE_MAX_IN_FLIGHT in "
            "flight. Must be below EMBEDDING_TIMEOUT_SECONDS."
        ),
    )
    EMBEDDING_HEDGE_MAX_IN_FLIGHT: int = Field(
        default=4,
        description=(
            "Maximum number of hedged query-embedding requests per query call (query path only). "
            "Also the maximum total attempts: if every attempt fails the call fails."
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
    # Optional at import: the API service must start without it. The bot worker
    # checks it explicitly via settings.require("TELEGRAM_BOT_TOKEN") at startup.
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Dev or deployed bot token")
    INVITE_CODES: str = Field(
        default="",
        description=(
            "Comma-separated invite codes accepted by /start <code>. Compared case-sensitively "
            "with hmac.compare_digest. Empty means nobody can sign up."
        ),
    )
    BOT_MAX_USERS: int = Field(
        default=25,
        description="Max users the bot will sign up; further /start requests are refused.",
    )
    BOT_ENABLED: bool = Field(
        default=False,
        description=(
            "Run the Telegram poller (and the reminder scheduler and the Whisper warm-up) "
            "inside the API process, single-process mode. FALSE by default so that no test "
            "and no local uvicorn run ever starts polling. Set BOT_ENABLED=true to enable."
        ),
    )

    # Whisper STT (local, CPU int8). Model directory = WHISPER_MODEL_ROOT/whisper-<name>.
    WHISPER_MODEL_NAME: str = Field(
        default="small",
        description="Whisper model size directory suffix: 'small' or 'base'.",
    )
    WHISPER_MODEL_ROOT: str = Field(
        default="backend/whisper_models",
        description=(
            "Directory holding the baked Whisper model folders. Relative paths resolve "
            "against the repo root locally; the container sets this to /app/whisper_models."
        ),
    )
    WHISPER_CPU_THREADS: int = Field(
        default=2,
        description="CPU threads used by faster-whisper (int8) transcription.",
    )
    VOICE_MAX_SECONDS: int = Field(
        default=60,
        description=(
            "Voice notes longer than this many seconds are refused without downloading them."
        ),
    )
    VOICE_TRANSCRIBE_TIMEOUT_SECONDS: int = Field(
        default=60,
        description="Hard timeout for a single voice transcription; the voice turn fails friendly.",
    )

    # Database
    DATABASE_URL: str = Field(description="PostgreSQL async connection string")

    # Deduplication
    DEDUPE_THRESHOLD: float = Field(
        default=0.90,
        description="Cosine similarity threshold for deduplicating knowledge records",
    )
    PENDING_DEDUPE_TTL_HOURS: int = Field(
        default=24,
        description=(
            "Pending knowledge records older than this many hours no longer count as duplicates "
            "during dedupe staging (used by stage_candidates, hence extraction and onboarding). "
            "Confirmed rows always count. Stale pending rows are left in place, never deleted."
        ),
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
    REMINDER_SCAN_SECONDS: int = Field(
        default=30,
        description="Interval in seconds between deterministic scheduler reminder scans (used by 5b)",
    )
    REMINDER_MAX_OVERDUE_HOURS: int = Field(
        default=24,
        description="Do not fire a reminder for a task overdue by more than this many hours",
    )
    REMINDER_MAX_PER_USER_PER_TICK: int = Field(
        default=2,
        description="Max reminders selected per user per scheduler scan",
    )
    REMINDER_SNOOZE_MINUTES: int = Field(
        default=30,
        description="Minutes a reminder's Snooze button postpones the task by",
    )
    INTERVENTION_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        description="Wall-clock timeout in seconds for the conversation-path intervention LLM call",
    )

    # Auth (needed from Phase 7; no insecure default)
    JWT_SECRET: str = Field(default="", description="Required in Phase 7")
    JWT_EXPIRE_MINUTES: int = Field(
        default=720, description="JWT lifetime in minutes (default 12 hours)"
    )
    AUTH_COOKIE_NAME: str = Field(
        default="session", description="Name of the httpOnly auth cookie"
    )

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

    @property
    def whisper_model_dir(self) -> Path:
        """Absolute path to the baked model folder: <root>/whisper-<name>."""
        root = Path(self.WHISPER_MODEL_ROOT)
        if not root.is_absolute():
            root = ROOT_DIR / root
        return root / f"whisper-{self.WHISPER_MODEL_NAME}"


settings = Settings()
