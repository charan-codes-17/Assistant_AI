"""backend/scripts/seed_demo_user.py
Create or update the single demo web-login account (roadmap Phase 7).

Reads DEMO_USERNAME and DEMO_PASSWORD (required) and DEMO_TELEGRAM_CHAT_ID (optional) from the
environment. Idempotent: if a user already holds that username or that telegram chat id, that row
is updated in place; a second row is never created. Only a bcrypt hash is stored.

Prints exactly "created"/"updated" plus the user id, and never the password or the hash. Exits
non-zero when the required variables are missing or the password is shorter than 12 characters.

Usage (PowerShell):
  $env:DEMO_USERNAME='riya'
  $env:DEMO_PASSWORD='<at least 12 characters>'
  .\\backend\\venv\\Scripts\\python.exe backend\\scripts\\seed_demo_user.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select

from backend.database import AsyncSessionLocal
from backend.models.models import User
from backend.services.auth import hash_password

MIN_PASSWORD_CHARS = 12


async def seed(username: str, password: str, chat_id: int | None) -> tuple[str, int]:
    """Create or update the demo user. Returns ("created"|"updated", user_id)."""
    async with AsyncSessionLocal() as session:
        user = (
            await session.execute(select(User).where(User.username == username))
        ).scalar_one_or_none()
        if user is None and chat_id is not None:
            user = (
                await session.execute(
                    select(User).where(User.telegram_chat_id == chat_id)
                )
            ).scalar_one_or_none()

        created = user is None
        if created:
            user = User(username=username)
            session.add(user)

        user.username = username
        user.password_hash = hash_password(password)
        if chat_id is not None:
            user.telegram_chat_id = chat_id

        await session.commit()
        await session.refresh(user)
        return ("created" if created else "updated"), user.id


def main() -> int:
    username = os.environ.get("DEMO_USERNAME", "").strip()
    password = os.environ.get("DEMO_PASSWORD", "")
    chat_id_raw = os.environ.get("DEMO_TELEGRAM_CHAT_ID", "").strip()

    if not username or not password:
        print("ERROR: DEMO_USERNAME and DEMO_PASSWORD must both be set", file=sys.stderr)
        return 2
    if len(password) < MIN_PASSWORD_CHARS:
        print(
            f"ERROR: DEMO_PASSWORD must be at least {MIN_PASSWORD_CHARS} characters",
            file=sys.stderr,
        )
        return 2

    chat_id: int | None = None
    if chat_id_raw:
        try:
            chat_id = int(chat_id_raw)
        except ValueError:
            print("ERROR: DEMO_TELEGRAM_CHAT_ID must be an integer", file=sys.stderr)
            return 2

    status, user_id = asyncio.run(seed(username, password, chat_id))
    print(f"{status} user_id={user_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
