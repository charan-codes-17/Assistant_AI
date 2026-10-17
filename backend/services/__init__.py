"""Backend shared service library."""

from backend.services.embeddings import embed_texts
from backend.services.extraction import (
    CandidateRecord,
    CandidateTask,
    DedupeMatch,
    ExtractionResult,
    StagedResult,
    confirm_task,
    extract_and_stage,
)
from backend.services.llm import (
    LLMEmptyResponse,
    LLMError,
    StrippedMessage,
    chat,
    chat_json,
)
from backend.services.memory_search import search_confirmed
from backend.services.router import RouterResult, classify_turn

__all__ = [
    "embed_texts",
    "chat",
    "chat_json",
    "LLMError",
    "LLMEmptyResponse",
    "StrippedMessage",
    "RouterResult",
    "classify_turn",
    "search_confirmed",
    "CandidateRecord",
    "CandidateTask",
    "ExtractionResult",
    "StagedResult",
    "DedupeMatch",
    "extract_and_stage",
    "confirm_task",
]
