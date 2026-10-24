"""backend/api/routes_tasks.py
Task endpoints: list the caller's tasks, and mark one done.

Ownership-scoped like routes_memory: the user id comes from the identity seam, and a missing id
and another user's id produce the identical 404 so ownership cannot be probed. Task transitions
reuse services.actions.mark_task_done (the same helper the Telegram bot uses) — no new status
values are introduced: "open" | "done" | "snoozed" come from the model.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import CurrentUser, get_current_user, get_db
from backend.api.schemas import TaskOut
from backend.models.models import Task, User
from backend.services.actions import NotFoundError, mark_task_done

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])

_NOT_FOUND = "not found"


async def _load_user(session: AsyncSession, user_id: int) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return user


def _to_out(task: Task) -> TaskOut:
    return TaskOut(
        id=task.id,
        title=task.title,
        due_at=task.due_at,
        status=task.status,
        origin=task.origin,
    )


@router.get("/tasks", response_model=list[TaskOut])
async def list_tasks(
    status: Literal["open", "done", "all"] = Query(default="all"),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> list[TaskOut]:
    stmt = select(Task).where(Task.user_id == current_user.user_id)
    if status != "all":
        stmt = stmt.where(Task.status == status)
    stmt = stmt.order_by(Task.due_at.asc().nulls_last(), Task.id.asc())

    rows = (await session.execute(stmt)).scalars().all()
    return [_to_out(row) for row in rows]


@router.post("/tasks/{task_id}/done", response_model=TaskOut)
async def complete_task(
    task_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> TaskOut:
    user = await _load_user(session, current_user.user_id)
    stmt = select(Task).where(Task.id == task_id, Task.user_id == user.id)
    task = (await session.execute(stmt)).scalar_one_or_none()
    if task is None:
        # Missing id and another user's id are indistinguishable on purpose.
        raise HTTPException(status_code=404, detail=_NOT_FOUND)

    if task.status == "open":
        # Reuse the bot's done logic (sets status="done" and commits).
        try:
            task = await mark_task_done(session, user, task_id)
        except NotFoundError:
            raise HTTPException(status_code=404, detail=_NOT_FOUND)
    # Any non-open state (already done, snoozed) is returned unchanged without error.
    return _to_out(task)
