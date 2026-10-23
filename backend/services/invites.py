"""backend/services/invites.py
Invite-code loading and comparison for the web signup route (roadmap Phase 7).

Copied from backend/bot/handlers.py (`_invite_codes` / `_code_matches`) so the bot and the API
apply the SAME rule: comma-separated codes in settings.INVITE_CODES, compared case-sensitively
with hmac.compare_digest. handlers.py keeps its own private copies and is unchanged.

A code is never logged or returned; callers only ever learn "match" or "no match".
"""

from __future__ import annotations

import hmac
from typing import Sequence

from backend.config.settings import settings


def invite_codes() -> list[str]:
    """The configured invite codes: comma-split, stripped, empties dropped."""
    return [c.strip() for c in (settings.INVITE_CODES or "").split(",") if c.strip()]


def code_matches(code: str, configured: Sequence[str]) -> bool:
    """True when `code` equals one of `configured`. An empty code never matches.

    Both sides are UTF-8 encoded before comparison: hmac.compare_digest raises TypeError on a
    non-ASCII str, so an accented or CJK code must simply fail to match instead of erroring.
    """
    if not code:
        return False
    given = code.encode("utf-8")
    return any(
        hmac.compare_digest(given, candidate.encode("utf-8")) for candidate in configured
    )
