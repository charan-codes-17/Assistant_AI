"""backend/api/routes_auth.py
Minimal web auth (roadmap Phase 7, pulled forward): POST /auth/login, POST /auth/signup and
POST /auth/logout.

Nothing here sits behind get_current_user: /auth/* and /health are the only unauthenticated
routes. Login and signup set an httpOnly, SameSite=Lax cookie; neither response body ever contains
the token, the password, the stored hash or the invite code.

Wrong password, unknown user and a user with a NULL password_hash all return a byte-identical
401, and every one of those paths performs exactly one bcrypt verification (against a dummy hash
when there is no real one) so the timing does not reveal whether the username exists.

Signup is invite-code gated: a wrong, empty or unset code all return the SAME 403, and reaching
the global user cap returns that same 403 too, so a response never reveals whether a code exists.
Throttles key on the normalised username only; X-Forwarded-For is never trusted.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.api.schemas import (
    LoginRequest,
    LoginResponse,
    LogoutResponse,
    SignupRequest,
)
from backend.config.settings import settings
from backend.models.models import User
from backend.services.auth import (
    auth_configured,
    create_token,
    dummy_hash,
    hash_password,
    login_throttle,
    normalize_username,
    signup_throttle,
    verify_password,
)
from backend.services.invites import code_matches, invite_codes

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

_UNAUTHORIZED = "unauthorized"
_RATE_LIMITED = "too many attempts"
_UNAVAILABLE = "auth unavailable"
_INVALID_CODE = "invalid invite code"
_USERNAME_TAKEN = "username unavailable"

# Throttle keys. The user key is scoped to the normalised username; the global key is shared by
# every signup attempt, so once its budget is spent signup is blocked for everyone in the window.
_SIGNUP_USER_KEY = "signup_user"
_SIGNUP_GLOBAL_KEY = "signup_global"


def _record_signup_failure(user_key: tuple[str, ...]) -> None:
    """Count one failed signup against the per-username key and the single global key."""
    signup_throttle.record_failure(user_key)
    signup_throttle.record_failure((_SIGNUP_GLOBAL_KEY, "*"))


def _record_signup_conflict(user_key: tuple[str, ...]) -> None:
    """Count one username conflict against the per-username key ONLY.

    A taken username is not evidence of invite-code probing. Counting it globally would let
    anyone lock signup for every user by repeating one name, so the two 409 paths keep the
    global budget untouched; the invite-code and cap guards still count on both keys.
    """
    signup_throttle.record_failure(user_key)


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
    session: AsyncSession = Depends(get_db),
) -> LoginResponse:
    # Fail closed in production without a strong secret: 503, and never a token.
    if not auth_configured():
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)

    # Keyed on the normalised username only; the client IP is never used (X-Forwarded-For is not
    # trusted).
    normalized = normalize_username(payload.username)
    key = (normalized,)
    if login_throttle.is_blocked(key):
        raise HTTPException(status_code=429, detail=_RATE_LIMITED)

    # Case-insensitive lookup so "Riya" and "riya" resolve to the same account.
    user = (
        await session.execute(
            select(User).where(func.lower(User.username) == normalized).limit(1)
        )
    ).scalars().first()

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


@router.post("/auth/signup", response_model=LoginResponse)
async def signup(
    payload: SignupRequest,
    response: Response,
    session: AsyncSession = Depends(get_db),
) -> LoginResponse:
    # (a) Fail closed in production without a strong secret: 503, same as login.
    if not auth_configured():
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)

    normalized = normalize_username(payload.username)
    user_key = (_SIGNUP_USER_KEY, normalized)

    # (b) Throttle before doing any work: identical 429 body as login. Blocked when either this
    # username or the shared global budget is spent.
    if signup_throttle.is_blocked(user_key) or signup_throttle.is_blocked(
        (_SIGNUP_GLOBAL_KEY, "*")
    ):
        raise HTTPException(status_code=429, detail=_RATE_LIMITED)

    # (c) Invite code. The SAME 403 for a wrong, empty or unset code, so nothing reveals whether
    # a code exists. A failed attempt is counted against the throttle.
    if not code_matches(payload.invite_code, invite_codes()):
        _record_signup_failure(user_key)
        raise HTTPException(status_code=403, detail=_INVALID_CODE)

    # (d) Global user cap, shared with the bot. Deliberately the same 403 as (c).
    total = (await session.execute(select(func.count()).select_from(User))).scalar_one()
    if total >= settings.BOT_MAX_USERS:
        _record_signup_failure(user_key)
        raise HTTPException(status_code=403, detail=_INVALID_CODE)

    # (e) Username taken. Pre-check for a fast, friendly 409. Usernames are stored lower-cased by
    # normalize_username, so the case-sensitive unique index on users.username is what makes the
    # race-safe case-insensitive guarantee for rows written by this route.
    existing = (
        await session.execute(
            select(User.id).where(func.lower(User.username) == normalized).limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        _record_signup_conflict(user_key)
        raise HTTPException(status_code=409, detail=_USERNAME_TAKEN)

    # (f) Create the account. Web accounts are separate from Telegram accounts (telegram_chat_id
    # stays NULL); only a bcrypt hash is stored.
    user = User(
        username=normalized,
        password_hash=hash_password(payload.password),
        telegram_chat_id=None,
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        _record_signup_conflict(user_key)
        raise HTTPException(status_code=409, detail=_USERNAME_TAKEN)
    await session.refresh(user)

    # Signup logs the new account straight in with the same session cookie as login.
    signup_throttle.reset(user_key)
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
