"""backend/api/routes_session.py
Conversation session creation (Phase 3, PRD §8).

POST /session/new creates a ConversationSession with channel "web" owned by the current user.
The user id always comes from the identity seam (deps.get_current_user), never from the body.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import CurrentUser, get_current_user, get_db
from backend.api.schemas import SessionNewResponse
from backend.models.models import ConversationSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["session"])


@router.post("/session/new", response_model=SessionNewResponse)
async def create_session(
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> SessionNewResponse:
    row = ConversationSession(user_id=current_user.user_id, channel="web")
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return SessionNewResponse(session_id=row.id)
