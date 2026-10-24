"""backend/api/routes_conversations.py
Server-side chat history endpoints. Backend only: nothing here touches the /chat reply path.

Ownership-scoped like routes_memory/routes_tasks: the user id comes from the identity seam, EVERY
query filters on it, and a missing id and another user's id produce the identical 404 so ownership
cannot be probed. Stored content is untrusted text: it is written and returned verbatim, never
executed and never parsed.

No raw SQL. No LLM call: creating a conversation derives the title in code from the first message.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import CurrentUser, get_current_user, get_db
from backend.api.schemas import (
    CONVERSATION_DEFAULT_LIMIT,
    CONVERSATION_MAX_LIMIT,
    ChatMessageOut,
    ConversationActionResponse,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationOut,
    ConversationRenameRequest,
    MessagePairRequest,
    MessagePairResponse,
)
from backend.models.models import ChatMessage, Conversation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversations"])

_NOT_FOUND = "not found"

# The title is derived from the first message by trimming whitespace and truncating.
_TITLE_FROM_FIRST_MESSAGE_MAX = 60


def _title_from(first_message: str) -> str:
    # Trim surrounding whitespace, then truncate; fall back to the raw message if trimming
    # removed everything (e.g. a whitespace-only first message).
    trimmed = first_message.strip() or first_message
    return trimmed[:_TITLE_FROM_FIRST_MESSAGE_MAX]


def _message_out(message: ChatMessage) -> ChatMessageOut:
    return ChatMessageOut(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


def _conversation_out(conversation: Conversation) -> ConversationOut:
    return ConversationOut(
        id=conversation.id,
        title=conversation.title,
        updated_at=conversation.updated_at,
    )


async def _load_conversation(
    session: AsyncSession, user_id: int, conversation_id: int
) -> Conversation:
    """Load one conversation owned by the caller. Missing or not-owned -> identical 404."""
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
    )
    conversation = (await session.execute(stmt)).scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return conversation


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    limit: int = Query(default=CONVERSATION_DEFAULT_LIMIT, ge=1, le=CONVERSATION_MAX_LIMIT),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> list[ConversationOut]:
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == current_user.user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [_conversation_out(row) for row in rows]


@router.post("/conversations", response_model=ConversationOut)
async def create_conversation(
    request: ConversationCreateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ConversationOut:
    # Title is computed in code (trimmed to 60 chars); no LLM call is made here.
    conversation = Conversation(
        user_id=current_user.user_id,
        title=_title_from(request.first_message),
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return _conversation_out(conversation)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ConversationDetail:
    conversation = await _load_conversation(session, current_user.user_id, conversation_id)
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation.id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[_message_out(row) for row in rows],
    )


@router.post("/conversations/{conversation_id}/messages", response_model=MessagePairResponse)
async def append_messages(
    conversation_id: int,
    request: MessagePairRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> MessagePairResponse:
    conversation = await _load_conversation(session, current_user.user_id, conversation_id)

    # One transaction: both rows plus the updated_at bump commit together.
    user_message = ChatMessage(
        conversation_id=conversation.id, role="user", content=request.user
    )
    assistant_message = ChatMessage(
        conversation_id=conversation.id, role="assistant", content=request.assistant
    )
    session.add_all([user_message, assistant_message])
    conversation.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(user_message)
    await session.refresh(assistant_message)
    await session.refresh(conversation)

    return MessagePairResponse(
        conversation_id=conversation.id,
        messages=[_message_out(user_message), _message_out(assistant_message)],
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationOut)
async def rename_conversation(
    conversation_id: int,
    request: ConversationRenameRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ConversationOut:
    conversation = await _load_conversation(session, current_user.user_id, conversation_id)
    conversation.title = request.title
    await session.commit()
    await session.refresh(conversation)
    return _conversation_out(conversation)


@router.delete("/conversations/{conversation_id}", response_model=ConversationActionResponse)
async def delete_conversation(
    conversation_id: int,
    confirm: bool = Query(default=False),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ConversationActionResponse:
    if not confirm:
        # Destructive: refuse unless confirm=true is passed explicitly.
        raise HTTPException(status_code=400, detail="confirm=true is required")
    conversation = await _load_conversation(session, current_user.user_id, conversation_id)
    # Messages are removed by the ON DELETE CASCADE foreign key.
    await session.delete(conversation)
    await session.commit()
    return ConversationActionResponse(id=conversation_id, status="deleted")
