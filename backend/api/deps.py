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

from fastapi import Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import settings
from backend.database import AsyncSessionLocal
from backend.models.models import User

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
    x_dev_user_id: str | None = Header(default=None, alias="X-Dev-User-Id"),
) -> CurrentUser:
    """Interim identity seam for development only.

    ################################################################################
    # LOUD WARNING: DEVELOPER SEAM - DO NOT DEPLOY.                                 #
    # Phase 7 replaces this with real JWT auth. Until then this trusts the raw       #
    # X-Dev-User-Id header and only works when settings.APP_ENV == "development";    #
    # in any other environment (or with a missing / non-integer / unknown header)    #
    # every request is rejected with an identical 401. The user id is NEVER taken    #
    # from the request body, query string, or path.                                  #
    ################################################################################
    """
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
