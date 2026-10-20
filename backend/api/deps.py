"""backend/api/deps.py
Request dependencies for the FastAPI layer: the identity seam and the DB session helper.

Session policy
--------------
- `get_db` yields ONE AsyncSession per request, used by the simple CRUD endpoints.
- Endpoints that call LLM-backed services (chat, onboarding) do NOT hold a request session
  across those calls: the services open their own short-lived sessions, and no ORM object is
  passed across an asyncio task boundary.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
import logging

from fastapi import Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import settings
from backend.database import AsyncSessionLocal
from backend.models.models import User
from backend.services.auth import decode_token

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CurrentUser:
    """Detached identity object. Deliberately NOT an ORM instance.

    Only ids and scalars cross into the service layer; the services re-load the User row
    inside their own sessions, so nothing here can leak a session or a stale ORM object.
    """

    user_id: int
    timezone: str
    quiet_mode: bool


def _unauthorized() -> HTTPException:
    # Generic body: never reveal whether the header was missing, malformed, or unknown.
    return HTTPException(status_code=401, detail="unauthorized")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """One DB session per request (dependency). Commit before returning it to the pool."""
    async with AsyncSessionLocal() as session:
        yield session


async def get_current_user(
    request: Request,
    x_dev_user_id: str | None = Header(default=None, alias="X-Dev-User-Id"),
) -> CurrentUser:
    """Identity: the auth cookie first; the header seam second (development only).

    ################################################################################
    # 1. If the auth cookie is present it decides the outcome by itself: a valid JWT  #
    #    loads the User from the DB (missing row -> 401); anything else (expired,      #
    #    tampered, alg none, missing sub, weak production secret) is a 401. The        #
    #    X-Dev-User-Id header is NOT consulted when a cookie is present.               #
    # 2. With NO cookie, the X-Dev-User-Id header is honoured ONLY when                #
    #    settings.APP_ENV == "development". In production it is ignored even if        #
    #    present and valid.                                                            #
    # Every rejection keeps the identical 401 body.                                    #
    ################################################################################
    """
    token = request.cookies.get(settings.AUTH_COOKIE_NAME)
    if token:
        user_id = decode_token(token)
        if user_id is None:
            raise _unauthorized()
        async with AsyncSessionLocal() as session:
            user = await session.get(User, user_id)
            if user is None:
                raise _unauthorized()
            return CurrentUser(
                user_id=user.id,
                timezone=user.timezone or "Asia/Kolkata",
                quiet_mode=bool(user.quiet_mode),
            )

    # No cookie: development-only header seam.
    if settings.APP_ENV != "development":
        raise _unauthorized()
    if x_dev_user_id is None:
        raise _unauthorized()
    try:
        user_id = int(x_dev_user_id)
    except (TypeError, ValueError):
        raise _unauthorized()

    # Short-lived session, closed before we return: never held across an LLM call.
    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise _unauthorized()
        return CurrentUser(
            user_id=user.id,
            timezone=user.timezone or "Asia/Kolkata",
            quiet_mode=bool(user.quiet_mode),
        )
