"""backend/api/routes_auth.py
Minimal web auth (roadmap Phase 7, pulled forward): POST /auth/login and POST /auth/logout.

Nothing here sits behind get_current_user: /auth/* and /health are the only unauthenticated
routes. Login sets an httpOnly, SameSite=Lax cookie; the response body never contains the token,
the password, or the stored hash.

Wrong password, unknown user and a user with a NULL password_hash all return a byte-identical
401, and every one of those paths performs exactly one bcrypt verification (against a dummy hash
when there is no real one) so the timing does not reveal whether the username exists.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.api.schemas import LoginRequest, LoginResponse, LogoutResponse
from backend.config.settings import settings
from backend.models.models import User
from backend.services.auth import (
    auth_configured,
    create_token,
    dummy_hash,
    login_throttle,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

_UNAUTHORIZED = "unauthorized"
_RATE_LIMITED = "too many attempts"
_UNAVAILABLE = "auth unavailable"


def _client_ip(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def _secure_cookie() -> bool:
    return settings.APP_ENV == "production"


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=_secure_cookie(),
        path="/",
        max_age=settings.JWT_EXPIRE_MINUTES * 60,
    )


@router.post("/auth/login", response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    response: Response,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> LoginResponse:
    # Fail closed in production without a strong secret: 503, and never a token.
    if not auth_configured():
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)

    key = (payload.username, _client_ip(request))
    if login_throttle.is_blocked(key):
        raise HTTPException(status_code=429, detail=_RATE_LIMITED)

    user = (
        await session.execute(select(User).where(User.username == payload.username))
    ).scalar_one_or_none()

    # ALWAYS exactly one bcrypt verification, even for an unknown user or a NULL hash.
    stored = user.password_hash if (user is not None and user.password_hash) else dummy_hash()
    password_ok = verify_password(payload.password, stored)

    if user is None or not user.password_hash or not password_ok:
        login_throttle.record_failure(key)
        # Identical body for wrong password, unknown user and NULL password_hash.
        raise HTTPException(status_code=401, detail=_UNAUTHORIZED)

    login_throttle.reset(key)
    _set_session_cookie(response, create_token(user.id))
    return LoginResponse(ok=True, user_id=user.id, username=user.username)


@router.post("/auth/logout", response_model=LogoutResponse)
async def logout(response: Response) -> LogoutResponse:
    """Clear the session cookie. Needs no valid session, so it always succeeds."""
    response.delete_cookie(
        key=settings.AUTH_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=_secure_cookie(),
    )
    return LogoutResponse(ok=True)
