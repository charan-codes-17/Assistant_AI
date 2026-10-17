"""Backend shared service library."""

from backend.services.embeddings import embed_texts
from backend.services.llm import (
    LLMEmptyResponse,
    LLMError,
    StrippedMessage,
    chat,
    chat_json,
)
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
]
