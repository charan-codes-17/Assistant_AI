"""backend/api/routes_profile.py
Profile summary endpoints (Phase 3, PRD §8).

The profile blob is stored and round-tripped, but it is NOT yet used when generating replies:
there is deliberately no LLM merge in Phase 3 (a later phase folds it into the prompt). Any
generated narrative must be treated as pending like any other memory.

No profile ownership can be crossed: the row is keyed by the current user's id.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import CurrentUser, get_current_user, get_db
from backend.api.schemas import ProfileRequest, ProfileResponse
from backend.models.models import ProfileSummary

logger = logging.getLogger(__name__)

router = APIRouter(tags=["profile"])


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    row = await session.get(ProfileSummary, current_user.user_id)
    if row is None:
        return ProfileResponse(blob="", updated_at=None)
    return ProfileResponse(blob=row.blob, updated_at=row.updated_at)


@router.post("/profile", response_model=ProfileResponse)
async def upsert_profile(
    request: ProfileRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    row = await session.get(ProfileSummary, current_user.user_id)
    if row is None:
        row = ProfileSummary(user_id=current_user.user_id, blob=request.blob)
        session.add(row)
    else:
        row.blob = request.blob
    await session.commit()
    await session.refresh(row)
    return ProfileResponse(blob=row.blob, updated_at=row.updated_at)
