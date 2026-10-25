"""backend/api/routes_skills.py
Skill endpoints: list skills, run one, create a user-defined skill, delete one.

Ownership-scoped like routes_tasks: the user id comes from the identity seam, and a missing id
and another user's id produce the identical 404 so ownership cannot be probed.

Built-in skills are a Python registry (see services.skills) and never produce database rows, so
they cannot be deleted here. Only the caller's own `user_skills` rows are addressable.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import CurrentUser, get_current_user, get_db
from backend.api.schemas import (
    SKILL_MAX_PER_USER,
    SkillCreateRequest,
    SkillCreatedResponse,
    SkillDeleteResponse,
    SkillOut,
    SkillRunRequest,
    SkillRunResponse,
)
from backend.models.models import User, UserSkill
from backend.services.onboarding import contains_secret
from backend.services.skills import (
    BUILTIN_SKILL_NAMES,
    BUILTIN_SKILLS,
    SkillInputRequiredError,
    SkillNotFoundError,
    SkillRateLimitedError,
    run_skill,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["skills"])

_NOT_FOUND = "not found"
_RESERVED_NAME = "That name belongs to a built-in skill. Pick another name."
_SECRET_TEXT = "That looks like it contains a secret. Remove passwords, codes or long tokens."
_TOO_MANY_SKILLS = "You have reached the maximum number of skills."


async def _load_user(session: AsyncSession, user_id: int) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return user


@router.get("/skills", response_model=list[SkillOut])
async def list_skills(
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> list[SkillOut]:
    """The eight built-ins first, then the caller's own user-defined skills."""
    builtins = [
        SkillOut(
            id=None,
            name=spec.name,
            description=spec.description,
            input=spec.input_mode,  # type: ignore[arg-type]
            builtin=True,
        )
        for spec in BUILTIN_SKILLS.values()
    ]

    stmt = (
        select(UserSkill)
        .where(UserSkill.user_id == current_user.user_id)
        .order_by(UserSkill.name.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    custom = [
        SkillOut(
            id=row.id,
            name=row.name,
            description=row.description,
            input="optional",
            builtin=False,
        )
        for row in rows
    ]
    return builtins + custom


@router.post("/skills/run", response_model=SkillRunResponse)
async def run_skill_route(
    payload: SkillRunRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> SkillRunResponse:
    """Run one skill. Built-in names win; otherwise the caller's own user skills are searched."""
    try:
        result = await run_skill(current_user.user_id, payload.name, payload.input)
    except SkillNotFoundError:
        # Unknown skill and another user's skill are indistinguishable on purpose.
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    except SkillInputRequiredError as exc:
        raise HTTPException(status_code=422, detail=exc.detail)
    except SkillRateLimitedError:
        raise HTTPException(status_code=429, detail="too many skill runs")

    return SkillRunResponse(
        turn_id=result.turn_id,
        skill=result.skill,
        reply=result.reply,
        memory_used=result.memory_used,
        trace=result.trace,
        pending_tasks_count=result.pending_tasks_count,
    )


@router.post("/skills", response_model=SkillCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_skill(
    payload: SkillCreateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> SkillCreatedResponse:
    """Create a user-defined skill. The instructions are never echoed back."""
    await _load_user(session, current_user.user_id)

    if not payload.instructions:
        raise HTTPException(
            status_code=422,
            detail="The instructions cannot be empty.",
        )
    if contains_secret(payload.instructions):
        raise HTTPException(status_code=422, detail=_SECRET_TEXT)
    if payload.name in BUILTIN_SKILL_NAMES:
        raise HTTPException(status_code=422, detail=_RESERVED_NAME)

    existing = (
        await session.execute(
            select(UserSkill.id).where(
                UserSkill.user_id == current_user.user_id,
                UserSkill.name == payload.name,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="That skill name already exists.")

    count = int(
        (
            await session.execute(
                select(func.count())
                .select_from(UserSkill)
                .where(UserSkill.user_id == current_user.user_id)
            )
        ).scalar_one()
    )
    if count >= SKILL_MAX_PER_USER:
        raise HTTPException(status_code=409, detail=_TOO_MANY_SKILLS)

    row = UserSkill(
        user_id=current_user.user_id,
        name=payload.name,
        description=payload.description,
        instructions=payload.instructions,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    return SkillCreatedResponse(
        id=row.id, name=row.name, description=row.description, builtin=False
    )


@router.delete("/skills/{skill_id}", response_model=SkillDeleteResponse)
async def delete_skill(
    skill_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> SkillDeleteResponse:
    """Delete one of the caller's own user-defined skills."""
    stmt = select(UserSkill).where(
        UserSkill.id == skill_id, UserSkill.user_id == current_user.user_id
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        # Missing id and another user's id are indistinguishable on purpose.
        raise HTTPException(status_code=404, detail=_NOT_FOUND)

    await session.delete(row)
    await session.commit()
    return SkillDeleteResponse(id=skill_id, status="deleted")
