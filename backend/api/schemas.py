"""backend/api/schemas.py
Pydantic request/response models for the FastAPI layer (Phase 3).

Rules enforced here:
- Request models carry hard limits (character counts, list lengths, allowed literals).
- Response models never expose embeddings, password hashes, or raw exceptions.
- Orchestrator-owned payloads (memory_used, trace, pending records, intervention) are reused
  directly so the API and the service layer cannot drift apart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.services.intervention import InterventionDecision
from backend.services.onboarding import QUESTIONNAIRE_ANSWER_MAX_CHARS, STATEMENT_MAX_CHARS
from backend.services.orchestrator import MemoryUsedItem, PendingRecord, TraceInfo

# ---------------------------------------------------------------------------
# Field limits (mirrored in CHANGES)
# ---------------------------------------------------------------------------

CHAT_TEXT_MAX = 4000
CHAT_HISTORY_MAX_ITEMS = 50
CHAT_HISTORY_CONTENT_MAX = 2000
MEMORY_EDIT_MAX = 500
MEMORY_BATCH_MAX = 100
TIMELINE_DEFAULT_LIMIT = 50
TIMELINE_MAX_LIMIT = 200
PROFILE_BLOB_MAX = 4000
ONBOARDING_ANSWER_MAX = QUESTIONNAIRE_ANSWER_MAX_CHARS


# ---------------------------------------------------------------------------
# Health / session
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health payload. Never contains secrets, DSNs, or exception text."""

    status: str
    db: bool
    pgvector: bool
    time: str


class SessionNewResponse(BaseModel):
    session_id: int


# ---------------------------------------------------------------------------
# Auth (/auth/login, /auth/logout)
# ---------------------------------------------------------------------------

AUTH_USERNAME_MAX = 100
AUTH_PASSWORD_MAX = 128
AUTH_USERNAME_MIN = 3
AUTH_PASSWORD_MIN = 12
AUTH_INVITE_CODE_MAX = 64
AUTH_USERNAME_PATTERN = r"^[A-Za-z0-9_]+$"


class LoginRequest(BaseModel):
    """Login body with hard caps. The password is never echoed back or logged."""

    username: str = Field(min_length=1, max_length=AUTH_USERNAME_MAX)
    password: str = Field(min_length=1, max_length=AUTH_PASSWORD_MAX)


class LoginResponse(BaseModel):
    """Login success body. Never contains the JWT or the stored password hash."""

    ok: bool
    user_id: int
    username: str | None = None


class LogoutResponse(BaseModel):
    ok: bool


class SignupRequest(BaseModel):
    """Web signup body with hard caps. The password and the invite code are never echoed back or
    logged.

    `invite_code` is length-capped but deliberately NOT min_length-capped: an empty or missing
    code must reach the route so it can return the same generic 403 as a wrong code.
    """

    username: str = Field(
        min_length=AUTH_USERNAME_MIN,
        max_length=AUTH_USERNAME_MAX,
        pattern=AUTH_USERNAME_PATTERN,
    )
    password: str = Field(min_length=AUTH_PASSWORD_MIN, max_length=AUTH_PASSWORD_MAX)
    invite_code: str = Field(default="", max_length=AUTH_INVITE_CODE_MAX)

    @field_validator("invite_code")
    @classmethod
    def _strip_invite_code(cls, value: str) -> str:
        return value.strip()


# ---------------------------------------------------------------------------
# Identity (/me)
# ---------------------------------------------------------------------------

class MeResponse(BaseModel):
    user_id: int
    username: str | None = None
    timezone: str
    quiet_mode: bool


class MeUpdateRequest(BaseModel):
    """Only timezone and quiet_mode are writable; any other field is ignored."""

    model_config = ConfigDict(extra="ignore")

    timezone: str | None = None
    quiet_mode: bool | None = None

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            ZoneInfo(value)
        except Exception as exc:  # noqa: BLE001 - convert to a 422
            raise ValueError(f"{value!r} is not a valid IANA timezone name") from exc
        return value


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=CHAT_HISTORY_CONTENT_MAX)


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=CHAT_TEXT_MAX)
    session_id: int
    history: list[HistoryMessage] = Field(
        default_factory=list, max_length=CHAT_HISTORY_MAX_ITEMS
    )
    source: Literal["text", "voice"] = "text"


class ChatResponse(BaseModel):
    """Immediate reply payload. `intent` is resolved in code only for slash/empty input;
    for every other turn the router settles the intent asynchronously and it is reported
    by the follow-up endpoint instead."""

    turn_id: str
    session_id: int
    intent: str | None = None
    reply: str
    memory_used: list[MemoryUsedItem] = Field(default_factory=list)
    trace: TraceInfo = Field(default_factory=TraceInfo)


class PendingTaskOut(BaseModel):
    """A pending task candidate with its stable index for confirm/discard."""

    index: int
    title: str
    due_at: datetime | None = None
    origin: str


class FollowupResponse(BaseModel):
    status: Literal["pending", "ready"]
    pending_records: list[PendingRecord] = Field(default_factory=list)
    pending_tasks: list[PendingTaskOut] = Field(default_factory=list)
    intervention: InterventionDecision = Field(
        default_factory=lambda: InterventionDecision(action="none", reason="postprocess pending")
    )
    extraction_failed: bool = False
    trace: TraceInfo = Field(default_factory=TraceInfo)


class TaskActionResponse(BaseModel):
    status: Literal["confirmed", "discarded"]
    index: int
    task_id: int | None = None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class RecordActionResponse(BaseModel):
    id: int
    status: str


class EditRequest(BaseModel):
    content: str = Field(min_length=1, max_length=MEMORY_EDIT_MAX)


class BatchRequest(BaseModel):
    record_ids: list[int] = Field(default_factory=list, max_length=MEMORY_BATCH_MAX)


class BatchConfirmResponse(BaseModel):
    confirmed: list[int] = Field(default_factory=list)
    ignored: list[int] = Field(default_factory=list)


class BatchRejectResponse(BaseModel):
    rejected: list[int] = Field(default_factory=list)
    ignored: list[int] = Field(default_factory=list)


class TimelineItem(BaseModel):
    id: int
    content: str
    kind: str
    certainty: str
    source: str
    source_ref: str | None = None
    status: str
    created_at: datetime
    rationale: str | None = None


class TimelineDay(BaseModel):
    date: str
    items: list[TimelineItem] = Field(default_factory=list)


class TaskOut(BaseModel):
    """A user task. The `tasks` table has no created_at/completed_at columns, so neither is exposed."""

    id: int
    title: str
    due_at: datetime | None = None
    status: str
    origin: str


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class ProfileRequest(BaseModel):
    blob: str = Field(default="", max_length=PROFILE_BLOB_MAX)


class ProfileResponse(BaseModel):
    blob: str
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------

class OnboardingImportRequest(BaseModel):
    # Length is validated in the route so oversized input yields 413 (not 422) and makes
    # zero service/LLM calls.
    text: str = Field(default="")


class OnboardingQuestionnaireRequest(BaseModel):
    answers: dict[str, str | None] = Field(default_factory=dict)

    @field_validator("answers")
    @classmethod
    def _validate_answers(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        for key, answer in value.items():
            if answer is not None and len(answer) > ONBOARDING_ANSWER_MAX:
                raise ValueError(
                    f"answer {key!r} exceeds {ONBOARDING_ANSWER_MAX} characters"
                )
        return value


class QuestionOut(BaseModel):
    id: str
    text: str


class OnboardingPromptResponse(BaseModel):
    import_prompt: str
    questions: list[QuestionOut] = Field(default_factory=list)


# STATEMENT_MAX_CHARS is re-exported for documentation/tests parity with the service.
__all__ = [
    "CHAT_TEXT_MAX",
    "CHAT_HISTORY_MAX_ITEMS",
    "CHAT_HISTORY_CONTENT_MAX",
    "MEMORY_EDIT_MAX",
    "MEMORY_BATCH_MAX",
    "TIMELINE_DEFAULT_LIMIT",
    "TIMELINE_MAX_LIMIT",
    "PROFILE_BLOB_MAX",
    "STATEMENT_MAX_CHARS",
    "HealthResponse",
    "SessionNewResponse",
    "MeResponse",
    "MeUpdateRequest",
    "HistoryMessage",
    "ChatRequest",
    "ChatResponse",
    "PendingTaskOut",
    "FollowupResponse",
    "TaskActionResponse",
    "RecordActionResponse",
    "EditRequest",
    "BatchRequest",
    "BatchConfirmResponse",
    "BatchRejectResponse",
    "TimelineItem",
    "TimelineDay",
    "ProfileRequest",
    "ProfileResponse",
    "OnboardingImportRequest",
    "OnboardingQuestionnaireRequest",
    "QuestionOut",
    "OnboardingPromptResponse",
]
