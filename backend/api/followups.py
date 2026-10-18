"""backend/api/followups.py
In-memory follow-up store for POST /chat turns (Phase 3).

PROCESS-LOCAL BY DESIGN
-----------------------
This store lives in the API process memory only. It assumes a SINGLE API instance. Two
consequences must be understood before deploying:

- Restarting the process loses every pending turn, including its task candidates: a client
  polling /chat/{turn_id}/followup after a restart gets 404 and any un-confirmed promise is
  simply gone.
- A load balancer fanning requests across instances would break the follow-up contract because
  the turn would exist in only one process's memory.

Entries expire after FOLLOWUP_TTL_SECONDS and the store keeps at most FOLLOWUP_MAX_ENTRIES
entries (evicting the oldest). It holds strong references to the background postprocess tasks
so they are never garbage-collected mid-flight, and it cancels them on application shutdown.

The postprocess runner NEVER lets an exception escape the task, so a failing postprocess can
never surface as an unhandled asyncio "task exception was never retrieved" warning.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from backend.services.extraction import CandidateTask
from backend.services.orchestrator import PostResult, ReplyResult

logger = logging.getLogger(__name__)

# Chosen defaults (also listed in CHANGES).
FOLLOWUP_TTL_SECONDS = 30 * 60.0  # 30 minutes
FOLLOWUP_MAX_ENTRIES = 500


@dataclass
class FollowupEntry:
    """State for one chat turn: the immediate reply plus the async postprocess outcome."""

    user_id: int
    turn_id: str
    created_at: float
    reply: ReplyResult
    status: str = "pending"
    post: Optional[PostResult] = None
    task_candidates: list[CandidateTask] = field(default_factory=list)
    confirmed_indices: set[int] = field(default_factory=set)
    discarded_indices: set[int] = field(default_factory=set)
    trace_errors: list[str] = field(default_factory=list)
    extraction_failed: bool = False
    task: Optional[asyncio.Task] = None

    def mark_ready(self, post: PostResult) -> None:
        self.post = post
        self.task_candidates = list(post.pending_tasks)
        self.extraction_failed = bool(post.extraction_failed)
        self.status = "ready"

    def mark_error(self, message: str) -> None:
        self.status = "ready"
        self.extraction_failed = True
        self.trace_errors.append(message)


class FollowupStore:
    """Ordered, TTL-bounded, process-local map keyed by (user_id, turn_id)."""

    def __init__(
        self,
        ttl_seconds: float = FOLLOWUP_TTL_SECONDS,
        max_entries: int = FOLLOWUP_MAX_ENTRIES,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._entries: "OrderedDict[tuple[int, str], FollowupEntry]" = OrderedDict()
        self._tasks: set[asyncio.Task] = set()
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._now = now_fn

    def create(
        self, user_id: int, turn_id: str, reply: ReplyResult
    ) -> FollowupEntry:
        entry = FollowupEntry(
            user_id=user_id,
            turn_id=turn_id,
            created_at=self._now(),
            reply=reply,
        )
        self._entries[(user_id, turn_id)] = entry
        self._evict()
        return entry

    def get(self, user_id: int, turn_id: str) -> Optional[FollowupEntry]:
        self._evict()
        return self._entries.get((user_id, turn_id))

    def register_task(self, entry: FollowupEntry, task: asyncio.Task) -> None:
        entry.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        """Cancel every running postprocess task and drop all state (lifespan shutdown)."""
        pending = [t for t in list(self._tasks) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._entries.clear()

    def _evict(self) -> None:
        now = self._now()
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.created_at > self.ttl_seconds
        ]
        for key in expired:
            self._entries.pop(key, None)
        while len(self._entries) > self.max_entries:
            # Oldest first: OrderedDict preserves insertion order.
            self._entries.popitem(last=False)


# Module-level store used by the routes (single process). Tests can override the dependency.
store = FollowupStore()


def get_followup_store() -> FollowupStore:
    """FastAPI dependency returning the process-local store (overridable in tests)."""
    return store


async def run_postprocess(
    entry: FollowupEntry,
    coro_factory: Callable[[], Awaitable[PostResult]],
) -> None:
    """Await the postprocess coroutine and record its outcome. Never raises."""
    try:
        post = await coro_factory()
    except asyncio.CancelledError:
        # Shutdown path: let cancellation propagate, but the task is already being cancelled.
        raise
    except Exception as exc:  # noqa: BLE001 - swallow everything: no unhandled task exception
        logger.exception("postprocess failed for turn_id=%s", entry.turn_id)
        entry.mark_error(f"postprocess: {exc}")
        return
    entry.mark_ready(post)


def schedule_postprocess(
    store_: FollowupStore,
    entry: FollowupEntry,
    coro_factory: Callable[[], Awaitable[PostResult]],
) -> asyncio.Task:
    """Run postprocess as a background task and keep a strong reference to it."""
    task = asyncio.create_task(run_postprocess(entry, coro_factory))
    store_.register_task(entry, task)
    return task


__all__ = [
    "FOLLOWUP_TTL_SECONDS",
    "FOLLOWUP_MAX_ENTRIES",
    "FollowupEntry",
    "FollowupStore",
    "store",
    "get_followup_store",
    "run_postprocess",
    "schedule_postprocess",
]
