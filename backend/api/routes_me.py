"""backend/api/routes_me.py
Identity endpoints (Phase 3, PRD §8): GET /me and PATCH /me.

Only `timezone` (a valid IANA name, validated by schemas.MeUpdateRequest) and `quiet_mode` are
writable. Any other field in the PATCH body is ignored (schemas.MeUpdateRequest sets
extra="ignore"), so an attempt to write e.g. username is a silent no-op rather than an error.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import CurrentUser, get_current_user, get_db
from backend.api.schemas import MeResponse, MeUpdateRequest
from backend.models.models import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["me"])

_NOT_FOUND = "not found"


def _to_response(user: User) -> MeResponse:
    return MeResponse(
        user_id=user.id,
        username=user.username,
        timezone=user.timezone,
        quiet_mode=bool(user.quiet_mode),
    )


@router.get("/me", response_model=MeResponse)
async def get_me(
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> MeResponse:
    user = await session.get(User, current_user.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return _to_response(user)


@router.patch("/me", response_model=MeResponse)
async def patch_me(
    request: MeUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> MeResponse:
    user = await session.get(User, current_user.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    if request.timezone is not None:
        user.timezone = request.timezone
    if request.quiet_mode is not None:
        user.quiet_mode = request.quiet_mode
    await session.commit()
    await session.refresh(user)
    return _to_response(user)
