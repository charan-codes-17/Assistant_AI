"""backend/scripts/test_auth.py
Evaluation script for the minimal web auth (roadmap Phase 7, pulled forward).

Usage:
  backend\\venv\\Scripts\\python.exe backend\\scripts\\test_auth.py
  backend\\venv\\Scripts\\python.exe backend\\scripts\\test_auth.py --only A1,A2

The app is exercised IN-PROCESS with httpx.AsyncClient + ASGITransport. No uvicorn, no bot.
settings.APP_ENV / settings.JWT_SECRET / settings.AUTH_COOKIE_NAME are changed in-process only and
restored in a finally block: no environment variable and no .env file is touched. Every user
created here is throwaway and deleted at the end.

Secrets policy: the passwords are generated per run and A12 asserts that no captured response
body or log line contains a password, the bcrypt hash prefix, or any issued token.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx
import jwt
from sqlalchemy import func, select

import backend.api.main as main_module
import backend.api.routes_auth as auth_routes
import backend.services.auth as auth_service
from backend.config.settings import settings
from backend.database import AsyncSessionLocal, engine
from backend.models.models import User
from backend.services.auth import LoginThrottle

DEV = "development"
PROD = "production"
# Obvious non-secret test secret (>= 32 chars). No real value is used or printed.
TEST_SECRET = "test-only-secret-" + "0" * 48
BCRYPT_PREFIX = "$2b$"

CAPTURED_BODIES: List[str] = []
CAPTURED_LOGS: List[str] = []
ISSUED_TOKENS: List[str] = []
PASSWORDS_USED: List[str] = []
CODES_USED: List[str] = []
CREATED_USERNAMES: List[str] = []
SHARED: Dict[str, Any] = {}


class Ctx:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client


class _LogCapture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            CAPTURED_LOGS.append(record.getMessage())
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------

def new_username(label: str) -> str:
    return f"authtest_{label}_{uuid.uuid4().hex[:10]}"


def new_password() -> str:
    password = f"AuthTest!{uuid.uuid4().hex[:12]}"
    PASSWORDS_USED.append(password)
    return password


async def create_user(
    username: str, password: Optional[str] = None, chat_id: Optional[int] = None
) -> int:
    async with AsyncSessionLocal() as session:
        user = User(username=username)
        if chat_id is not None:
            user.telegram_chat_id = chat_id
        if password is not None:
            user.password_hash = auth_service.hash_password(password)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        uid = user.id
    CREATED_USERNAMES.append(username)
    return uid


async def delete_user_row(uid: int) -> None:
    async with AsyncSessionLocal() as session:
        row = await session.get(User, uid)
        if row is not None:
            await session.delete(row)
            await session.commit()


async def delete_created_users() -> None:
    if not CREATED_USERNAMES:
        return
    try:
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(User).where(User.username.in_(list(CREATED_USERNAMES)))
                )
            ).scalars().all()
            for row in rows:
                await session.delete(row)
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"   [WARN] cleanup failed: {exc}")
    CREATED_USERNAMES.clear()


async def login(client: httpx.AsyncClient, username: str, password: str) -> httpx.Response:
    response = await client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    CAPTURED_BODIES.append(response.text)
    return response


def cookie_token(response: httpx.Response) -> Optional[str]:
    name = settings.AUTH_COOKIE_NAME
    try:
        value = response.cookies.get(name)
    except Exception:  # noqa: BLE001
        value = None
    if value:
        return value
    raw = response.headers.get("set-cookie", "")
    marker = f"{name}="
    index = raw.find(marker)
    if index == -1:
        return None
    return raw[index + len(marker):].split(";", 1)[0]


def cookie_attrs(response: httpx.Response) -> List[str]:
    raw = response.headers.get("set-cookie", "")
    return [part.strip().lower() for part in raw.split(";")]


def has_attr(response: httpx.Response, attr: str) -> bool:
    attr = attr.lower()
    return any(part == attr or part.startswith(f"{attr}=") for part in cookie_attrs(response))


def b64url(data: Dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def alg_none_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    header = {"alg": "none", "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    return f"{b64url(header)}.{b64url(payload)}."


def no_sub_token() -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=10)).timestamp())},
        settings.JWT_SECRET,
        algorithm="HS256",
    )


def _run_seed(env: Dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "backend" / "scripts" / "seed_demo_user.py")],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# Signup helpers (Phase 7, web invite-code signup)
# ---------------------------------------------------------------------------

def new_invite_code() -> str:
    """A throwaway invite code for this run. Used as a leak-scan needle; never printed."""
    code = "INV-" + uuid.uuid4().hex[:16]
    CODES_USED.append(code)
    return code


@contextmanager
def fresh_signup_throttle(max_attempts: int = 10, window_seconds: float = 15 * 60.0):
    """Install a throwaway signup throttle driven by a fake clock.

    Signup counts EVERY failure against one global key shared by all attempts, so without this the
    real module-level throttle would accumulate failures across sections and start returning 429.
    Yields `(throttle, clock)`; advance the window with `clock["t"] += ...`.
    """
    original = auth_routes.signup_throttle
    clock = {"t": 0.0}
    throttle = LoginThrottle(
        max_attempts=max_attempts,
        window_seconds=window_seconds,
        now_fn=lambda: clock["t"],
    )
    auth_routes.signup_throttle = throttle
    try:
        yield throttle, clock
    finally:
        auth_routes.signup_throttle = original


async def signup(
    client: httpx.AsyncClient, username: str, password: str, invite_code: str
) -> httpx.Response:
    response = await client.post(
        "/auth/signup",
        json={"username": username, "password": password, "invite_code": invite_code},
    )
    CAPTURED_BODIES.append(response.text)
    return response


async def count_rows(username: str) -> int:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(User.id).where(
                    func.lower(User.username) == auth_service.normalize_username(username)
                )
            )
        ).all()
    return len(rows)


# ---------------------------------------------------------------------------
# A1  correct login
# ---------------------------------------------------------------------------

async def a1(ctx: Ctx) -> tuple:
    username = new_username("a1")
    password = new_password()
    uid = await create_user(username, password)

    r = await login(ctx.client, username, password)
    token = cookie_token(r)
    if token:
        ISSUED_TOKENS.append(token)
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {}

    checks = [
        ("login 200", r.status_code == 200),
        (
            "body ok+ids",
            body.get("ok") is True
            and body.get("user_id") == uid
            and body.get("username") == username,
        ),
        ("cookie set", bool(token)),
        ("HttpOnly", has_attr(r, "httponly")),
        ("SameSite=lax", has_attr(r, "samesite=lax")),
        ("token not in body", bool(token) and token not in r.text),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    return ok, detail


# ---------------------------------------------------------------------------
# A2 / A3 / A3b / A3c  identical 401s + exactly one bcrypt verify each
# ---------------------------------------------------------------------------

async def _login_counting_verify(
    ctx: Ctx, username: str, password: str
) -> tuple[httpx.Response, int]:
    calls = {"n": 0}
    original = auth_routes.verify_password

    def counting(pw: str, stored: str) -> bool:
        calls["n"] += 1
        return original(pw, stored)

    auth_routes.verify_password = counting
    try:
        response = await login(ctx.client, username, password)
    finally:
        auth_routes.verify_password = original
    return response, calls["n"]


async def a2(ctx: Ctx) -> tuple:
    username = new_username("a2")
    password = new_password()
    await create_user(username, password)

    r, verify_calls = await _login_counting_verify(ctx, username, "wrong-password-xyz")
    SHARED["a2_body"] = r.text
    SHARED["a2_calls"] = verify_calls
    ok = r.status_code == 401 and verify_calls == 1
    return ok, f"status={r.status_code}, verify_calls={verify_calls}"


async def a3(ctx: Ctx) -> tuple:
    missing = new_username("a3missing")
    r, verify_calls = await _login_counting_verify(ctx, missing, "unknown-user-pw")
    SHARED["a3_calls"] = verify_calls
    identical = r.text == SHARED.get("a2_body")
    ok = r.status_code == 401 and identical and verify_calls == 1
    return ok, f"status={r.status_code}, identical_to_A2={identical}, verify_calls={verify_calls}"


async def a3b(ctx: Ctx) -> tuple:
    username = new_username("a3null")
    await create_user(username, password=None)  # NULL password_hash
    r, verify_calls = await _login_counting_verify(ctx, username, "null-hash-pw")
    SHARED["a3b_calls"] = verify_calls
    identical = r.text == SHARED.get("a2_body")
    ok = r.status_code == 401 and identical and verify_calls == 1
    return ok, f"status={r.status_code}, identical_to_A2={identical}, verify_calls={verify_calls}"


async def a3c(ctx: Ctx) -> tuple:
    counts = {
        "A2": SHARED.get("a2_calls"),
        "A3": SHARED.get("a3_calls"),
        "A3b": SHARED.get("a3b_calls"),
    }
    ok = all(v == 1 for v in counts.values())
    return ok, ", ".join(f"{k}={v}" for k, v in counts.items())


# ---------------------------------------------------------------------------
# A4  cookie authenticates /me
# ---------------------------------------------------------------------------

async def a4(ctx: Ctx) -> tuple:
    username = new_username("a4")
    password = new_password()
    uid = await create_user(username, password)

    r = await login(ctx.client, username, password)
    token = cookie_token(r)
    if token:
        ISSUED_TOKENS.append(token)
    if not token:
        return False, f"login status={r.status_code}, no cookie"

    r_me = await ctx.client.get("/me", cookies={settings.AUTH_COOKIE_NAME: token})
    CAPTURED_BODIES.append(r_me.text)
    try:
        body = r_me.json()
    except Exception:  # noqa: BLE001
        body = {}
    ok = r_me.status_code == 200 and body.get("user_id") == uid
    return ok, f"login={r.status_code}, me={r_me.status_code}, user_id={body.get('user_id')}"


# ---------------------------------------------------------------------------
# A5  production, no cookie, no header
# ---------------------------------------------------------------------------

async def a5(ctx: Ctx) -> tuple:
    original = settings.APP_ENV
    settings.APP_ENV = PROD
    try:
        r = await ctx.client.get("/me")
    finally:
        settings.APP_ENV = original
    CAPTURED_BODIES.append(r.text)
    return r.status_code == 401, f"status={r.status_code}"


# ---------------------------------------------------------------------------
# A6  expired / tampered / alg-none / no-sub tokens
# ---------------------------------------------------------------------------

async def a6(ctx: Ctx) -> tuple:
    username = new_username("a6")
    password = new_password()
    uid = await create_user(username, password)

    expired = auth_service.create_token(
        uid,
        now=datetime.now(timezone.utc)
        - timedelta(minutes=settings.JWT_EXPIRE_MINUTES + 5),
    )
    valid = auth_service.create_token(uid)
    ISSUED_TOKENS.append(valid)

    head, payload_seg, signature = valid.split(".")
    flipped = "A" if signature[2] != "A" else "B"
    tampered = f"{head}.{payload_seg}.{signature[:2]}{flipped}{signature[3:]}"

    cases = {
        "expired": expired,
        "tampered": tampered,
        "alg_none": alg_none_token(uid),
        "no_sub": no_sub_token(),
    }
    statuses: Dict[str, int] = {}
    for label, token in cases.items():
        r = await ctx.client.get("/me", cookies={settings.AUTH_COOKIE_NAME: token})
        CAPTURED_BODIES.append(r.text)
        statuses[label] = r.status_code

    ok = all(code == 401 for code in statuses.values())
    return ok, ", ".join(f"{k}={v}" for k, v in statuses.items())


# ---------------------------------------------------------------------------
# A7  valid token for a deleted user
# ---------------------------------------------------------------------------

async def a7(ctx: Ctx) -> tuple:
    username = new_username("a7")
    password = new_password()
    uid = await create_user(username, password)
    token = auth_service.create_token(uid)
    ISSUED_TOKENS.append(token)

    await delete_user_row(uid)

    r = await ctx.client.get("/me", cookies={settings.AUTH_COOKIE_NAME: token})
    CAPTURED_BODIES.append(r.text)
    return r.status_code == 401, f"status={r.status_code}"


# ---------------------------------------------------------------------------
# A8  dev header seam vs production header
# ---------------------------------------------------------------------------

async def a8(ctx: Ctx) -> tuple:
    username = new_username("a8")
    uid = await create_user(username, password=None)

    original = settings.APP_ENV
    try:
        settings.APP_ENV = DEV
        r_dev = await ctx.client.get("/me", headers={"X-Dev-User-Id": str(uid)})
        settings.APP_ENV = PROD
        r_prod = await ctx.client.get("/me", headers={"X-Dev-User-Id": str(uid)})
    finally:
        settings.APP_ENV = original
    CAPTURED_BODIES.append(r_dev.text)
    CAPTURED_BODIES.append(r_prod.text)

    ok = r_dev.status_code == 200 and r_prod.status_code == 401
    return ok, f"dev_header={r_dev.status_code}, prod_header={r_prod.status_code}"


# ---------------------------------------------------------------------------
# A9  logout clears the cookie; Secure only in production
# ---------------------------------------------------------------------------

async def a9(ctx: Ctx) -> tuple:
    original_env, original_secret = settings.APP_ENV, settings.JWT_SECRET
    try:
        settings.APP_ENV = DEV
        settings.JWT_SECRET = TEST_SECRET
        r_dev = await ctx.client.post("/auth/logout")
        settings.APP_ENV = PROD
        r_prod = await ctx.client.post("/auth/logout")
    finally:
        settings.APP_ENV = original_env
        settings.JWT_SECRET = original_secret
    CAPTURED_BODIES.append(r_dev.text)
    CAPTURED_BODIES.append(r_prod.text)

    dev_attrs = cookie_attrs(r_dev)
    checks = [
        ("dev logout 200", r_dev.status_code == 200),
        ("dev clears (name+max-age=0)", has_attr(r_dev, settings.AUTH_COOKIE_NAME) and "max-age=0" in dev_attrs),
        ("dev no Secure", not has_attr(r_dev, "secure")),
        ("prod logout 200", r_prod.status_code == 200),
        ("prod Secure", has_attr(r_prod, "secure")),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    return ok, detail


# ---------------------------------------------------------------------------
# A10 production with an empty JWT_SECRET fails closed
# ---------------------------------------------------------------------------

async def a10(ctx: Ctx) -> tuple:
    username = new_username("a10")
    password = new_password()
    uid = await create_user(username, password)
    valid_looking = auth_service.create_token(uid)
    ISSUED_TOKENS.append(valid_looking)

    original_env, original_secret = settings.APP_ENV, settings.JWT_SECRET
    handler = _LogCapture()
    logging.getLogger().addHandler(handler)
    try:
        settings.APP_ENV = PROD
        settings.JWT_SECRET = ""
        r_login = await login(ctx.client, username, password)
        r_me = await ctx.client.get(
            "/me", cookies={settings.AUTH_COOKIE_NAME: valid_looking}
        )
        CAPTURED_BODIES.append(r_me.text)
        # Exercise the real startup path so the "AUTH DISABLED" log line actually fires.
        async with main_module.lifespan(main_module.app):
            pass
    finally:
        logging.getLogger().removeHandler(handler)
        settings.APP_ENV = original_env
        settings.JWT_SECRET = original_secret

    startup_lines = [line for line in CAPTURED_LOGS if "AUTH DISABLED" in line]
    checks = [
        ("login 503", r_login.status_code == 503),
        ("token 401", r_me.status_code == 401),
        ("startup log line", len(startup_lines) >= 1),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    return ok, detail


# ---------------------------------------------------------------------------
# A11 throttle: 5 fails -> 429, window expiry, success resets
# ---------------------------------------------------------------------------

async def a11(ctx: Ctx) -> tuple:
    blocked_user = new_username("a11blocked")
    blocked_pw = new_password()
    await create_user(blocked_user, blocked_pw)

    reset_user = new_username("a11reset")
    reset_pw = new_password()
    await create_user(reset_user, reset_pw)

    clock = {"t": 0.0}
    throttle = LoginThrottle(
        max_attempts=5, window_seconds=15 * 60.0, now_fn=lambda: clock["t"]
    )
    original_throttle = auth_routes.login_throttle
    original_env = settings.APP_ENV
    auth_routes.login_throttle = throttle
    settings.APP_ENV = DEV
    try:
        bad = [
            (await login(ctx.client, blocked_user, "wrong-password-a")).status_code
            for _ in range(5)
        ]
        sixth = (await login(ctx.client, blocked_user, blocked_pw)).status_code

        clock["t"] += 15 * 60.0 + 1
        after_window = (await login(ctx.client, blocked_user, blocked_pw)).status_code

        reset_fails = [
            (await login(ctx.client, reset_user, "wrong-password-b")).status_code
            for _ in range(4)
        ]
        after_success = (await login(ctx.client, reset_user, reset_pw)).status_code
        reset_fails_2 = [
            (await login(ctx.client, reset_user, "wrong-password-c")).status_code
            for _ in range(4)
        ]
        after_reset = (await login(ctx.client, reset_user, reset_pw)).status_code
    finally:
        auth_routes.login_throttle = original_throttle
        settings.APP_ENV = original_env

    checks = [
        ("5 bad -> 401", bad == [401] * 5),
        ("6th correct -> 429", sixth == 429),
        ("after window -> 200", after_window == 200),
        ("4 bad -> 401", reset_fails == [401] * 4 and reset_fails_2 == [401] * 4),
        ("first success -> 200", after_success == 200),
        ("success resets -> 200", after_reset == 200),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    return ok, detail


# ---------------------------------------------------------------------------
# A12 no captured body or log line leaks a password, hash prefix, or token
# ---------------------------------------------------------------------------

async def a12(ctx: Ctx) -> tuple:
    needles: List[tuple] = [("password", p) for p in PASSWORDS_USED]
    needles.append(("hash_prefix", BCRYPT_PREFIX))
    needles += [("token", t) for t in ISSUED_TOKENS]

    leaks: List[str] = []
    for kind, needle in needles:
        if not needle:
            continue
        for body in CAPTURED_BODIES:
            if needle in body:
                leaks.append(f"{kind} in response body")
        for line in CAPTURED_LOGS:
            if needle in line:
                leaks.append(f"{kind} in log line")

    ok = not leaks
    detail = (
        f"scanned {len(CAPTURED_BODIES)} bodies, {len(CAPTURED_LOGS)} log lines, "
        f"{len(needles)} needles"
    )
    if leaks:
        detail += f"; LEAKS={sorted(set(leaks))[:5]}"
    return ok, detail


# ---------------------------------------------------------------------------
# A13 seed script (subprocess, env only for that subprocess)
# ---------------------------------------------------------------------------

async def a13(ctx: Ctx) -> tuple:
    username = new_username("a13")
    password = new_password()
    chat_id = 900_000_000 + (uuid.uuid4().int % 10_000_000)
    CREATED_USERNAMES.append(username)

    env = os.environ.copy()
    env["DEMO_USERNAME"] = username
    env["DEMO_PASSWORD"] = password
    env["DEMO_TELEGRAM_CHAT_ID"] = str(chat_id)

    first = _run_seed(env)
    second = _run_seed(env)

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(select(User).where(User.username == username))
        ).scalars().all()
    row_count = len(rows)
    stored = rows[0].password_hash if rows else None

    env_missing = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DEMO_USERNAME", "DEMO_PASSWORD", "DEMO_TELEGRAM_CHAT_ID")
    }
    missing_run = _run_seed(env_missing)

    env_short = dict(env)
    env_short["DEMO_PASSWORD"] = "short"
    short_run = _run_seed(env_short)

    seed_output = (first.stdout + first.stderr + second.stdout + second.stderr)
    checks = [
        ("first run created", first.returncode == 0 and "created" in first.stdout),
        ("second run updated", second.returncode == 0 and "updated" in second.stdout),
        ("exactly one row", row_count == 1),
        ("hash is bcrypt", bool(stored) and stored.startswith(BCRYPT_PREFIX)),
        ("no password in stdout/stderr", password not in seed_output),
        ("missing vars -> non-zero", missing_run.returncode != 0),
        ("short password -> non-zero", short_run.returncode != 0),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    return ok, detail


# ---------------------------------------------------------------------------
# A14 seeded user logs in end to end
# ---------------------------------------------------------------------------

async def a14(ctx: Ctx) -> tuple:
    username = new_username("a14")
    password = new_password()
    CREATED_USERNAMES.append(username)

    env = os.environ.copy()
    env["DEMO_USERNAME"] = username
    env["DEMO_PASSWORD"] = password
    seeded = _run_seed(env)
    if seeded.returncode != 0:
        return False, f"seed failed rc={seeded.returncode}"

    r = await login(ctx.client, username, password)
    token = cookie_token(r)
    if token:
        ISSUED_TOKENS.append(token)
    if not token:
        return False, f"login status={r.status_code}, no cookie"

    r_me = await ctx.client.get("/me", cookies={settings.AUTH_COOKIE_NAME: token})
    CAPTURED_BODIES.append(r_me.text)
    try:
        body = r_me.json()
    except Exception:  # noqa: BLE001
        body = {}

    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(select(User).where(User.username == username))
        ).scalar_one_or_none()
        expected_uid = row.id if row is not None else None

    ok = (
        r.status_code == 200
        and r_me.status_code == 200
        and expected_uid is not None
        and body.get("user_id") == expected_uid
    )
    return ok, (
        f"seed_rc={seeded.returncode}, login={r.status_code}, "
        f"me={r_me.status_code}, uid_match={body.get('user_id') == expected_uid}"
    )


# ---------------------------------------------------------------------------
# A15 signup (a): valid code + fresh username -> 200, HttpOnly cookie, DB row
# ---------------------------------------------------------------------------

async def a15(ctx: Ctx) -> tuple:
    code = new_invite_code()
    username = new_username("a15")
    password = new_password()
    original_codes = settings.INVITE_CODES
    settings.INVITE_CODES = code
    try:
        with fresh_signup_throttle():
            r = await signup(ctx.client, username, password, code)
    finally:
        settings.INVITE_CODES = original_codes

    token = cookie_token(r)
    if token:
        ISSUED_TOKENS.append(token)
    normalized = auth_service.normalize_username(username)
    CREATED_USERNAMES.append(normalized)

    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(select(User).where(User.username == normalized))
        ).scalar_one_or_none()
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {}

    checks = [
        ("signup 200", r.status_code == 200),
        (
            "body ok+ids",
            body.get("ok") is True
            and row is not None
            and body.get("user_id") == row.id
            and body.get("username") == normalized,
        ),
        ("cookie set", bool(token)),
        ("HttpOnly", has_attr(r, "httponly")),
        ("row exists", row is not None),
        ("username stored", row is not None and row.username == normalized),
        (
            "hash verifies",
            row is not None
            and bool(row.password_hash)
            and auth_service.verify_password(password, row.password_hash),
        ),
        ("chat_id NULL", row is not None and row.telegram_chat_id is None),
        ("token not in body", bool(token) and token not in r.text),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    return ok, detail


# ---------------------------------------------------------------------------
# A16 signup (b): wrong / empty / codes-unset -> identical 403, no row
# ---------------------------------------------------------------------------

async def a16(ctx: Ctx) -> tuple:
    code = new_invite_code()
    wrong_user = new_username("a16w")
    empty_user = new_username("a16e")
    unset_user = new_username("a16u")
    pw_wrong = new_password()
    pw_empty = new_password()

    original_codes = settings.INVITE_CODES
    try:
        with fresh_signup_throttle():
            settings.INVITE_CODES = code
            r_wrong = await signup(ctx.client, wrong_user, pw_wrong, "definitely-not-the-code")
            settings.INVITE_CODES = code
            r_empty = await signup(ctx.client, empty_user, pw_empty, "")
            settings.INVITE_CODES = ""  # INVITE_CODES unset
            r_unset = await signup(ctx.client, unset_user, new_password(), code)
    finally:
        settings.INVITE_CODES = original_codes

    SHARED["signup_invalid_body"] = r_wrong.text
    statuses = [r_wrong.status_code, r_empty.status_code, r_unset.status_code]
    identical = len({r_wrong.text, r_empty.text, r_unset.text}) == 1
    rows_created = (
        await count_rows(wrong_user)
        + await count_rows(empty_user)
        + await count_rows(unset_user)
    )

    checks = [
        ("all three 403", statuses == [403, 403, 403]),
        ("identical body", identical),
        ("no rows created", rows_created == 0),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    return ok, detail


# ---------------------------------------------------------------------------
# A17 signup (c): duplicate username (incl. different case) -> 409, one row
# ---------------------------------------------------------------------------

async def a17(ctx: Ctx) -> tuple:
    code = new_invite_code()
    username = new_username("a17")
    password = new_password()
    other_pw = new_password()

    original_codes = settings.INVITE_CODES
    settings.INVITE_CODES = code
    try:
        with fresh_signup_throttle():
            first = await signup(ctx.client, username, password, code)
            dup_same = await signup(ctx.client, username, other_pw, code)
            dup_case = await signup(ctx.client, username.upper(), other_pw, code)
    finally:
        settings.INVITE_CODES = original_codes

    normalized = auth_service.normalize_username(username)
    CREATED_USERNAMES.append(normalized)
    row_count = await count_rows(username)

    checks = [
        ("first 200", first.status_code == 200),
        ("dup same 409", dup_same.status_code == 409),
        ("dup case 409", dup_case.status_code == 409),
        ("exactly one row", row_count == 1),
        ("409 identical", dup_same.text == dup_case.text and "unavailable" in dup_same.text),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    return ok, detail


# ---------------------------------------------------------------------------
# A18 signup (d): global user cap -> the SAME 403 as an invalid code
# ---------------------------------------------------------------------------

async def a18(ctx: Ctx) -> tuple:
    code = new_invite_code()
    username = new_username("a18")
    password = new_password()
    ref_user = new_username("a18ref")

    original_codes, original_cap = settings.INVITE_CODES, settings.BOT_MAX_USERS
    settings.INVITE_CODES = code
    try:
        with fresh_signup_throttle():
            # Reference: an invalid code, i.e. the same generic 403 that (b) asserts.
            r_ref = await signup(
                ctx.client, ref_user, new_password(), "definitely-not-the-code"
            )
            # Then saturate the cap and retry with the VALID code.
            settings.BOT_MAX_USERS = 0  # any existing user means the cap is already reached
            r_cap = await signup(ctx.client, username, password, code)
    finally:
        settings.INVITE_CODES = original_codes
        settings.BOT_MAX_USERS = original_cap

    row_count = await count_rows(username)
    checks = [
        ("cap -> 403", r_cap.status_code == 403),
        ("same body as invalid code", r_cap.text == r_ref.text),
        ("no row created", row_count == 0),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    return ok, detail


# ---------------------------------------------------------------------------
# A19 signup (e): throttle -> 429; the global key (not the per-username key) is
# what blocks a different username
# ---------------------------------------------------------------------------

async def a19(ctx: Ctx) -> tuple:
    code = new_invite_code()
    user1 = new_username("a19a")
    user2 = new_username("a19b")
    password = new_password()

    original_codes = settings.INVITE_CODES
    settings.INVITE_CODES = code

    original_throttle = auth_routes.signup_throttle
    clock = {"t": 0.0}
    throttle = LoginThrottle(
        max_attempts=10, window_seconds=15 * 60.0, now_fn=lambda: clock["t"]
    )
    auth_routes.signup_throttle = throttle
    try:
        fails = [
            (
                await signup(
                    ctx.client, user1, password, f"nope-{uuid.uuid4().hex[:8]}"
                )
            ).status_code
            for _ in range(10)
        ]
        eleventh = (
            await signup(ctx.client, user1, password, f"nope-{uuid.uuid4().hex[:8]}")
        ).status_code
        other = (
            await signup(ctx.client, user2, password, f"nope-{uuid.uuid4().hex[:8]}")
        ).status_code
        user2_key_blocked = throttle.is_blocked(
            ("signup_user", auth_service.normalize_username(user2))
        )
        global_key_blocked = throttle.is_blocked(("signup_global", "*"))

        clock["t"] += 15 * 60.0 + 1
        after_window = (
            await signup(ctx.client, user1, password, f"nope-{uuid.uuid4().hex[:8]}")
        ).status_code
    finally:
        auth_routes.signup_throttle = original_throttle
        settings.INVITE_CODES = original_codes

    checks = [
        ("10 fails -> 403", fails == [403] * 10),
        ("11th -> 429", eleventh == 429),
        (
            "other user: per-username free, GLOBAL blocks",
            other == 429 and global_key_blocked and not user2_key_blocked,
        ),
        ("after window -> 403", after_window == 403),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    detail += (
        f"; user2_per_username_blocked={user2_key_blocked}, "
        f"global_key_blocked={global_key_blocked}"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# A20 signup (f): the cookie from signup authenticates GET /me
# ---------------------------------------------------------------------------

async def a20(ctx: Ctx) -> tuple:
    code = new_invite_code()
    username = new_username("a20")
    password = new_password()

    original_codes = settings.INVITE_CODES
    settings.INVITE_CODES = code
    try:
        with fresh_signup_throttle():
            r = await signup(ctx.client, username, password, code)
    finally:
        settings.INVITE_CODES = original_codes

    token = cookie_token(r)
    if token:
        ISSUED_TOKENS.append(token)
    normalized = auth_service.normalize_username(username)
    CREATED_USERNAMES.append(normalized)
    if not token:
        return False, f"signup status={r.status_code}, no cookie"

    r_me = await ctx.client.get("/me", cookies={settings.AUTH_COOKIE_NAME: token})
    CAPTURED_BODIES.append(r_me.text)
    try:
        body = r_me.json()
    except Exception:  # noqa: BLE001
        body = {}

    async with AsyncSessionLocal() as session:
        expected_uid = (
            await session.execute(select(User.id).where(User.username == normalized))
        ).scalar_one_or_none()

    ok = (
        r_me.status_code == 200
        and expected_uid is not None
        and body.get("user_id") == expected_uid
    )
    return ok, (
        f"signup={r.status_code}, me={r_me.status_code}, "
        f"uid_match={body.get('user_id') == expected_uid}"
    )


# ---------------------------------------------------------------------------
# A21 signup (g): no captured body or log line leaks a password, invite code,
# hash prefix, or token (mirrors A12, adds codes)
# ---------------------------------------------------------------------------

async def a21(ctx: Ctx) -> tuple:
    needles: List[tuple] = [("password", p) for p in PASSWORDS_USED]
    needles += [("code", c) for c in CODES_USED]
    needles.append(("hash_prefix", BCRYPT_PREFIX))
    needles += [("token", t) for t in ISSUED_TOKENS]

    leaks: List[str] = []
    for kind, needle in needles:
        if not needle:
            continue
        for body in CAPTURED_BODIES:
            if needle in body:
                leaks.append(f"{kind} in response body")
        for line in CAPTURED_LOGS:
            if needle in line:
                leaks.append(f"{kind} in log line")

    ok = not leaks
    detail = (
        f"scanned {len(CAPTURED_BODIES)} bodies, {len(CAPTURED_LOGS)} log lines, "
        f"{len(needles)} needles"
    )
    if leaks:
        detail += f"; LEAKS={sorted(set(leaks))[:5]}"
    return ok, detail


# ---------------------------------------------------------------------------
# A22 signup (h): login still works after the key change
# ---------------------------------------------------------------------------

async def a22(ctx: Ctx) -> tuple:
    username = new_username("a22")
    password = new_password()
    uid = await create_user(username, password)

    r = await login(ctx.client, username, password)
    token = cookie_token(r)
    if token:
        ISSUED_TOKENS.append(token)
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {}

    ok = (
        r.status_code == 200
        and bool(token)
        and body.get("ok") is True
        and body.get("user_id") == uid
    )
    return ok, (
        f"login={r.status_code}, cookie={'yes' if token else 'no'}, "
        f"uid_match={body.get('user_id') == uid}"
    )


# ---------------------------------------------------------------------------
# A23 signup: a NON-ASCII invite code is a plain 403, never a 500
# (hmac.compare_digest raises TypeError on a non-ASCII str)
# ---------------------------------------------------------------------------

async def a23(ctx: Ctx) -> tuple:
    code = new_invite_code()
    user_ref = new_username("a23r")
    user_acc = new_username("a23a")
    user_cjk = new_username("a23c")

    original_codes = settings.INVITE_CODES
    settings.INVITE_CODES = code
    try:
        with fresh_signup_throttle():
            r_ref = await signup(ctx.client, user_ref, new_password(), "definitely-not-the-code")
            r_acc = await signup(ctx.client, user_acc, new_password(), "é")
            r_cjk = await signup(ctx.client, user_cjk, new_password(), "日本")
    finally:
        settings.INVITE_CODES = original_codes

    rows_created = (
        await count_rows(user_ref) + await count_rows(user_acc) + await count_rows(user_cjk)
    )
    checks = [
        ("reference wrong code 403", r_ref.status_code == 403),
        ("non-ASCII 'é' 403 not 500", r_acc.status_code == 403),
        ("non-ASCII '日本' 403 not 500", r_cjk.status_code == 403),
        (
            "identical body to wrong code",
            r_acc.text == r_ref.text and r_cjk.text == r_ref.text,
        ),
        ("no rows created", rows_created == 0),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    detail += f"; statuses={[r_ref.status_code, r_acc.status_code, r_cjk.status_code]}"
    return ok, detail


# ---------------------------------------------------------------------------
# A24 signup: repeated 409s (username taken) must NOT spend the GLOBAL budget,
# so a later signup with a valid code and a fresh username still succeeds
# ---------------------------------------------------------------------------

async def a24(ctx: Ctx) -> tuple:
    code = new_invite_code()
    taken = new_username("a24t")
    winner = new_username("a24w")
    password = new_password()

    original_codes = settings.INVITE_CODES
    settings.INVITE_CODES = code
    try:
        with fresh_signup_throttle():
            first = await signup(ctx.client, taken, password, code)
            duplicates = [
                (await signup(ctx.client, taken, new_password(), code)).status_code
                for _ in range(10)
            ]
            global_blocked_after = auth_routes.signup_throttle.is_blocked(
                ("signup_global", "*")
            )
            user_key_blocked_after = auth_routes.signup_throttle.is_blocked(
                ("signup_user", auth_service.normalize_username(taken))
            )
            late = await signup(ctx.client, winner, new_password(), code)
    finally:
        settings.INVITE_CODES = original_codes

    CREATED_USERNAMES.append(auth_service.normalize_username(taken))
    CREATED_USERNAMES.append(auth_service.normalize_username(winner))
    winner_token = cookie_token(late)
    if winner_token:
        ISSUED_TOKENS.append(winner_token)

    checks = [
        ("first signup 200", first.status_code == 200),
        ("10 duplicates 409", duplicates == [409] * 10),
        ("global key untouched", global_blocked_after is False),
        ("per-username key spent", user_key_blocked_after is True),
        ("later valid signup 200", late.status_code == 200),
    ]
    ok = all(passed for _, passed in checks)
    detail = ", ".join(f"{n}={'ok' if p else 'BAD'}" for n, p in checks)
    detail += (
        f"; dup_statuses={sorted(set(duplicates))}, global_blocked={global_blocked_after}, "
        f"user_key_blocked={user_key_blocked_after}, late={late.status_code}"
    )
    return ok, detail


ORDER = [
    "A1", "A2", "A3", "A3b", "A3c", "A4", "A5", "A6", "A7",
    "A8", "A9", "A10", "A11", "A12", "A13", "A14",
    "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22",
    "A23", "A24",
]

FUNCS = {
    "A1": a1,
    "A2": a2,
    "A3": a3,
    "A3b": a3b,
    "A3c": a3c,
    "A4": a4,
    "A5": a5,
    "A6": a6,
    "A7": a7,
    "A8": a8,
    "A9": a9,
    "A10": a10,
    "A11": a11,
    "A12": a12,
    "A13": a13,
    "A14": a14,
    "A15": a15,
    "A16": a16,
    "A17": a17,
    "A18": a18,
    "A19": a19,
    "A20": a20,
    "A21": a21,
    "A22": a22,
    "A23": a23,
    "A24": a24,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal web auth evaluation (Phase 7)")
    parser.add_argument("--only", default=None, help="Run only sections, e.g. A1 or A1,A2")
    args = parser.parse_args()
    only = None
    if args.only:
        only = [x.strip() for x in args.only.split(",") if x.strip()]

    settings.require("DATABASE_URL")

    print("=" * 100)
    print("PHASE 7 (minimal): WEB AUTH EVALUATION")
    print(f"sections: {only or 'ALL'}")
    print("=" * 100)

    original_env = settings.APP_ENV
    original_secret = settings.JWT_SECRET
    original_cookie = settings.AUTH_COOKIE_NAME
    settings.APP_ENV = DEV
    settings.JWT_SECRET = TEST_SECRET
    settings.AUTH_COOKIE_NAME = "session"

    handler = _LogCapture()
    logging.getLogger().addHandler(handler)

    results: List[tuple] = []
    transport = httpx.ASGITransport(app=main_module.app)
    try:
        for tid in ORDER:
            if only and tid not in only:
                continue
            # A fresh client per section: no cookie jar carries over between sections.
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver", timeout=60.0
            ) as client:
                ctx = Ctx(client)
                try:
                    ok, detail = await FUNCS[tid](ctx)
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    ok, detail = False, f"unhandled exception: {exc!r}"
            status = "PASS" if ok else "FAIL"
            results.append((tid, status, detail))
            print(f"{status} {tid}: {detail}")
    finally:
        logging.getLogger().removeHandler(handler)
        settings.APP_ENV = original_env
        settings.JWT_SECRET = original_secret
        settings.AUTH_COOKIE_NAME = original_cookie
        await delete_created_users()
        await engine.dispose()

    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)
    print("-" * 100)
    print(f"TOTAL: {passed}/{total} PASS")
    print(f"EXIT: {0 if passed == total else 1}")
    print("=" * 100)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
