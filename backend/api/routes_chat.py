"""backend/api/routes_chat.py
Chat turn endpoints (Phase 3, PRD §8).

POST /chat runs orchestrator.respond() and returns immediately; postprocess() (extraction +
intervention) is scheduled as a background asyncio task tied to the turn id and polled through
GET /chat/{turn_id}/followup.

The user id is always taken from the identity seam (deps.get_current_user), never from the body,
query, or path. Session ownership is validated before any service call.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from backend.api.deps import CurrentUser, get_current_user
from backend.api.followups import (
    FollowupEntry,
    FollowupStore,
    get_followup_store,
    schedule_postprocess,
)
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    FollowupResponse,
    PendingTaskOut,
    TaskActionResponse,
)
from backend.database import AsyncSessionLocal
from backend.models.models import ConversationSession, User
from backend.services.actions import confirm_task_candidate
from backend.services.intervention import InterventionDecision
from backend.services.orchestrator import postprocess, respond

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


async def _session_belongs_to_user(session_id: int, user_id: int) -> bool:
    """Ownership check in a short-lived session (never held across the LLM calls)."""
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(ConversationSession).where(
                    ConversationSession.id == session_id,
                    ConversationSession.user_id == user_id,
                )
            )
        ).scalar_one_or_none()
        return row is not None


def _build_followup(entry: FollowupEntry) -> FollowupResponse:
    reply = entry.reply
    trace = reply.trace.model_copy(deep=True)
    if entry.post is not None:
        trace.timings = {**reply.trace.timings, **entry.post.trace.timings}
        trace.errors = list(reply.trace.errors) + list(entry.post.trace.errors)
        trace.skipped = entry.post.trace.skipped
        pending_records = list(entry.post.pending_records)
        intervention = entry.post.intervention
    else:
        pending_records = []
        intervention = InterventionDecision(action="none", reason="postprocess pending")
    for message in entry.trace_errors:
        trace.errors.append(message)

    pending_tasks = [
        PendingTaskOut(
            index=index,
            title=candidate.title,
            due_at=candidate.due_at,
            origin=candidate.origin,
        )
        for index, candidate in enumerate(entry.task_candidates)
    ]
    return FollowupResponse(
        status=entry.status,  # type: ignore[arg-type] - constrained to pending|ready
        pending_records=pending_records,
        pending_tasks=pending_tasks,
        intervention=intervention,
        extraction_failed=entry.extraction_failed,
        trace=trace,
    )


@router.post("/chat", response_model=ChatResponse)
async def create_chat_turn(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
    store: FollowupStore = Depends(get_followup_store),
) -> ChatResponse:
    if not await _session_belongs_to_user(request.session_id, current_user.user_id):
        raise HTTPException(status_code=404, detail="not found")

    history = [message.model_dump() for message in request.history]
    reply = await respond(
        current_user.user_id,
        request.text,
        session_id=request.session_id,
        history=history,
        source=request.source,
    )

    entry = store.create(current_user.user_id, reply.turn_id, reply)
    schedule_postprocess(
        store, entry, lambda: postprocess(current_user.user_id, reply)
    )

    return ChatResponse(
        turn_id=reply.turn_id,
        session_id=reply.session_id,
        intent=reply.intent,
        reply=reply.reply,
        memory_used=reply.memory_used,
        trace=reply.trace,
    )


@router.get("/chat/{turn_id}/followup", response_model=FollowupResponse)
async def get_chat_followup(
    turn_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    store: FollowupStore = Depends(get_followup_store),
) -> FollowupResponse:
    # Keyed by (user_id, turn_id): an unknown turn id and another user's turn id are
    # indistinguishable here, so both yield the identical 404.
    entry = store.get(current_user.user_id, turn_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")
    return _build_followup(entry)


@router.post(
    "/chat/{turn_id}/tasks/{index}/confirm", response_model=TaskActionResponse
)
async def confirm_chat_task(
    turn_id: str,
    index: int,
    current_user: CurrentUser = Depends(get_current_user),
    store: FollowupStore = Depends(get_followup_store),
) -> TaskActionResponse:
    entry = store.get(current_user.user_id, turn_id)
    if entry is None or index < 0 or index >= len(entry.task_candidates):
        raise HTTPException(status_code=404, detail="not found")

    candidate = entry.task_candidates[index]
    async with AsyncSessionLocal() as session:
        user = await session.get(User, current_user.user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="not found")
        # confirm_task_candidate is idempotent on (user_id, title, due_at).
        task = await confirm_task_candidate(session, user, candidate)
    entry.confirmed_indices.add(index)
    return TaskActionResponse(status="confirmed", index=index, task_id=task.id)


@router.post(
    "/chat/{turn_id}/tasks/{index}/discard", response_model=TaskActionResponse
)
async def discard_chat_task(
    turn_id: str,
    index: int,
    current_user: CurrentUser = Depends(get_current_user),
    store: FollowupStore = Depends(get_followup_store),
) -> TaskActionResponse:
    entry = store.get(current_user.user_id, turn_id)
    if entry is None or index < 0 or index >= len(entry.task_candidates):
        raise HTTPException(status_code=404, detail="not found")

    # Idempotent: re-discarding the same index is a no-op. Nothing is written to the DB.
    entry.discarded_indices.add(index)
    return TaskActionResponse(status="discarded", index=index, task_id=None)
