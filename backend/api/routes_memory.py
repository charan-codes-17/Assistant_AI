"""backend/api/routes_memory.py
Memory endpoints (Phase 3, PRD §8): per-record state transitions, batch transitions, and the
Inspector timeline.

Every call is ownership-scoped: the user id comes from the identity seam, and a missing row and
another user's row produce the identical response so ownership cannot be probed. Responses never
include embeddings.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import CurrentUser, get_current_user, get_db
from backend.api.schemas import (
    TIMELINE_DEFAULT_LIMIT,
    TIMELINE_MAX_LIMIT,
    BatchConfirmResponse,
    BatchRejectResponse,
    BatchRequest,
    EditRequest,
    RecordActionResponse,
    TimelineDay,
    TimelineItem,
)
from backend.models.models import KnowledgeRecord, User
from backend.services.actions import (
    NotFoundError,
    confirm_record,
    confirm_records,
    delete_record,
    edit_record,
    reject_record,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memory"])

_NOT_FOUND = "not found"


async def _load_user(session: AsyncSession, user_id: int) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return user


@router.post("/memory/{record_id}/confirm", response_model=RecordActionResponse)
async def confirm_memory(
    record_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> RecordActionResponse:
    user = await _load_user(session, current_user.user_id)
    record = await confirm_record(session, user, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return RecordActionResponse(id=record.id, status=record.status)


@router.post("/memory/{record_id}/reject", response_model=RecordActionResponse)
async def reject_memory(
    record_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> RecordActionResponse:
    user = await _load_user(session, current_user.user_id)
    record = await reject_record(session, user, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return RecordActionResponse(id=record.id, status=record.status)


@router.patch("/memory/{record_id}", response_model=RecordActionResponse)
async def edit_memory(
    record_id: int,
    request: EditRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> RecordActionResponse:
    user = await _load_user(session, current_user.user_id)
    try:
        record = await edit_record(session, user, record_id, request.content)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return RecordActionResponse(id=record.id, status=record.status)


@router.delete("/memory/{record_id}", response_model=RecordActionResponse)
async def delete_memory(
    record_id: int,
    confirm: bool = Query(default=False),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> RecordActionResponse:
    if not confirm:
        # Hard delete is destructive: refuse unless confirm=true is passed explicitly.
        raise HTTPException(status_code=400, detail="confirm=true is required")
    user = await _load_user(session, current_user.user_id)
    try:
        await delete_record(session, user, record_id, confirmed=True)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return RecordActionResponse(id=record_id, status="deleted")


@router.post("/memory/confirm-batch", response_model=BatchConfirmResponse)
async def confirm_memory_batch(
    request: BatchRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> BatchConfirmResponse:
    result = await confirm_records(current_user.user_id, request.record_ids)
    return BatchConfirmResponse(
        confirmed=[record.id for record in result.confirmed_records],
        ignored=result.ignored_ids,
    )


@router.post("/memory/reject-batch", response_model=BatchRejectResponse)
async def reject_memory_batch(
    request: BatchRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> BatchRejectResponse:
    user = await _load_user(session, current_user.user_id)
    unique_ids = list(dict.fromkeys(request.record_ids))
    if not unique_ids:
        return BatchRejectResponse(rejected=[], ignored=[])

    rows = (
        await session.execute(
            select(KnowledgeRecord).where(
                KnowledgeRecord.id.in_(unique_ids),
                KnowledgeRecord.user_id == user.id,
            )
        )
    ).scalars().all()
    by_id = {record.id: record for record in rows}

    rejected: list[int] = []
    ignored: list[int] = []
    for record_id in unique_ids:
        record = by_id.get(record_id)
        if record is None:
            # Not owned or missing: reported the same way (never distinguished).
            ignored.append(record_id)
            continue
        record.status = "rejected"
        rejected.append(record_id)
    await session.commit()
    return BatchRejectResponse(rejected=rejected, ignored=ignored)


@router.get("/memory/timeline", response_model=list[TimelineDay])
async def memory_timeline(
    limit: int = Query(default=TIMELINE_DEFAULT_LIMIT, ge=1, le=TIMELINE_MAX_LIMIT),
    before: datetime | None = Query(default=None),
    status: Literal["confirmed", "pending", "all"] = Query(default="all"),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> list[TimelineDay]:
    stmt = select(KnowledgeRecord).where(
        KnowledgeRecord.user_id == current_user.user_id
    )
    if status == "all":
        # Default view: everything except rejected records.
        stmt = stmt.where(KnowledgeRecord.status != "rejected")
    else:
        stmt = stmt.where(KnowledgeRecord.status == status)
    if before is not None:
        stmt = stmt.where(KnowledgeRecord.created_at < before)
    stmt = stmt.order_by(KnowledgeRecord.created_at.desc()).limit(limit)

    rows = (await session.execute(stmt)).scalars().all()

    try:
        user_tz = ZoneInfo(current_user.timezone)
    except Exception:
        user_tz = ZoneInfo("Asia/Kolkata")

    groups: "OrderedDict[str, list[TimelineItem]]" = OrderedDict()
    for record in rows:
        created = record.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        local_date = created.astimezone(user_tz).date().isoformat()
        groups.setdefault(local_date, []).append(
            TimelineItem(
                id=record.id,
                content=record.content,
                kind=record.kind,
                certainty=record.certainty,
                source=record.source,
                source_ref=record.source_ref,
                status=record.status,
                created_at=record.created_at,
            )
        )

    return [TimelineDay(date=day, items=items) for day, items in groups.items()]
