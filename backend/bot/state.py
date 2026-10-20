"""backend/bot/state.py
In-process, per-chat state for the Telegram bot.

Everything here is PROCESS-LOCAL BY DESIGN and is lost on restart:
  - conversation history and the active ConversationSession id, per chat
  - in-flight postprocess background tasks (kept referenced so they are not GC'd)
  - short callback tokens for batch actions (rec:allok:<token>)

Persisted data (KnowledgeRecord rows, Task rows, ConversationSession rows) lives in the
database and is untouched by a restart. Staged task candidates survive because they are stored
as Task rows with status='pending' (see actions.stage_task_candidates).
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field


@dataclass
class ChatState:
    """Mutable per-chat state. Never shared between chats."""

    history: list[dict[str, str]] = field(default_factory=list)
    session_id: int | None = None
    forget_nonce: str | None = None
    forget_expires_at: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def reset(self) -> None:
        """Drop all in-memory state for this chat (used when the account is deleted)."""
        self.history.clear()
        self.session_id = None
        self.forget_nonce = None
        self.forget_expires_at = None


class BotState:
    """Container for all per-chat state plus the background task registry."""

    def __init__(self) -> None:
        self._chats: dict[int, ChatState] = {}
        self.background: set[asyncio.Task] = set()
        self._batch_tokens: dict[str, list[int]] = {}
        self._counter = itertools.count(1)

    def chat(self, chat_id: int) -> ChatState:
        """Return (creating if needed) the state for one chat."""
        state = self._chats.get(chat_id)
        if state is None:
            state = ChatState()
            self._chats[chat_id] = state
        return state

    def new_batch_token(self, record_ids: list[int]) -> str:
        """Register a short token mapped to record ids (used when the csv would exceed 64 bytes)."""
        token = f"b{next(self._counter)}"
        self._batch_tokens[token] = list(record_ids)
        return token

    def resolve_batch_token(self, token: str) -> list[int] | None:
        """Resolve a batch token back to ids, or None if unknown/expired."""
        return self._batch_tokens.get(token)


# Process-wide singleton used by the real handlers. Tests build fresh BotState instances.
state = BotState()
