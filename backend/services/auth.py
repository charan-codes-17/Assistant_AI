"""backend/services/auth.py
Password hashing and JWT helpers for the minimal web login (roadmap Phase 7, pulled forward).

Design notes
------------
- Passwords are hashed with `bcrypt` directly (no passlib). Only the hash is ever stored; the
  plaintext password is never logged, returned, or persisted.
- Tokens are HS256 JWTs (PyJWT) with claims `sub` (the user id, as a string), `iat` and `exp`.
  Decoding pins the algorithm list to HS256, so `alg: none` and any other algorithm are rejected
  outright, and requires `exp`/`iat`/`sub` to be present.
- Fail-closed in production: when APP_ENV == "production" and JWT_SECRET is missing or shorter
  than 32 characters, `auth_configured()` is False, so login refuses (503, enforced in the
  route) and every token is rejected (401, enforced here).
- The login and signup throttles are PER-PROCESS and in-memory. Behind more than one API
  instance every replica keeps its own counter, so the effective limit is per replica, not global.
- Both throttles key on the NORMALISED username only; the client IP is never part of a key
  (X-Forwarded-For is not trusted). Signup additionally counts every failure against ONE global
  key, which blocks signup for everyone once that budget is spent.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import bcrypt
import jwt

from backend.config.settings import settings

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"
_MIN_SECRET_CHARS = 32

# A fixed, non-secret password hashed once at import. It exists only so that a login attempt for
# an unknown user (or a user with no password_hash) still performs exactly one bcrypt
# verification, keeping its response time indistinguishable from a wrong-password attempt.
_DUMMY_HASH = bcrypt.hashpw(
    b"timing-equalisation-placeholder", bcrypt.gensalt()
).decode("utf-8")


def hash_password(password: str) -> str:
    """Return a bcrypt hash (starts with "$2b$"). The input is never logged or returned."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """bcrypt check. A malformed stored hash returns False instead of raising."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (TypeError, ValueError):
        return False


def dummy_hash() -> str:
    """The hash used when there is no real stored hash (timing equalisation)."""
    return _DUMMY_HASH


def auth_configured() -> bool:
    """True when JWT auth may be used. Production demands a strong secret; development does not."""
    if settings.APP_ENV == "production":
        return len(settings.JWT_SECRET or "") >= _MIN_SECRET_CHARS
    return True


def normalize_username(username: str) -> str:
    """Canonical username form for storage, lookup and throttling: trimmed, lower-cased.

    Applied on signup (stored value) and on login (lookup), so "Riya" and "riya" are the same
    account and cannot both exist. Only the username is normalised; a password is never altered.
    """
    return username.strip().lower()


def create_token(user_id: int, *, now: Optional[datetime] = None) -> str:
    """Sign an HS256 token for `user_id`. `now` is injectable for tests."""
    issued = now or datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": int(issued.timestamp()),
        "exp": int(
            (issued + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)).timestamp()
        ),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=_ALGORITHM)


def decode_token(token: str) -> Optional[int]:
    """Return the user id carried by a valid token, or None for anything invalid.

    Rejects: not configured (production with a weak/absent secret), `alg: none`, any algorithm
    other than HS256, a bad signature, an expired/not-yet-valid token, and a token with no `sub`.
    """
    if not auth_configured():
        return None
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[_ALGORITHM],
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.InvalidTokenError:
        return None
    subject = payload.get("sub")
    if subject is None:
        return None
    try:
        return int(subject)
    except (TypeError, ValueError):
        return None


class LoginThrottle:
    """Per-process failed-attempt counter keyed by an opaque tuple supplied by the caller.

    The login route keys on the normalised username alone, e.g. `("riya",)`. The signup route keys
    on the normalised username and, separately, on one global key shared by every signup attempt.
    The client IP is deliberately NOT part of any key: X-Forwarded-For is not trusted.

    NOT shared across processes: with more than one API replica each keeps its own window, so the
    real limit is `max_attempts` per replica. `now_fn` is injectable so tests can advance a fake
    clock.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: float = 15 * 60.0,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.now_fn = now_fn
        self._failures: dict[tuple[str, ...], list[float]] = {}

    def _recent(self, key: tuple[str, ...]) -> list[float]:
        cutoff = self.now_fn() - self.window_seconds
        recent = [t for t in self._failures.get(key, []) if t > cutoff]
        if recent:
            self._failures[key] = recent
        else:
            self._failures.pop(key, None)
        return recent

    def is_blocked(self, key: tuple[str, ...]) -> bool:
        return len(self._recent(key)) >= self.max_attempts

    def record_failure(self, key: tuple[str, ...]) -> None:
        self._recent(key)
        self._failures.setdefault(key, []).append(self.now_fn())

    def reset(self, key: tuple[str, ...]) -> None:
        self._failures.pop(key, None)


# Module-level throttles. Tests may replace `now_fn` or the whole object.
# Login: 5 failures per normalised username per 15 minutes.
login_throttle = LoginThrottle()
# Signup: 10 failures per normalised username, AND 10 failures GLOBALLY across every signup
# attempt, per 15 minutes. The global key means a burst of bad attempts can block signup for
# everyone for the rest of the window.
signup_throttle = LoginThrottle(max_attempts=10, window_seconds=15 * 60.0)
