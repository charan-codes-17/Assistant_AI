"""backend/bot/textfmt.py
Plain-text formatting for Telegram replies.

The bot never sets parse_mode, so a small subset of markdown is stripped to plain text,
long messages are split at paragraph/sentence boundaries (Telegram's hard limit is 4096 chars),
and the memory footer / "Show source" button are derived from the reply's memory_used list.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional, Sequence

TELEGRAM_MAX_CHARS = 4096

# Bullets are converted BEFORE emphasis so a leading "* item" is not mistaken for italic.
_STAR_BULLET_RE = re.compile(r"^([ \t]*)\*[ \t]+", re.MULTILINE)
_HEADER_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_UNDER_RE = re.compile(r"__(.+?)__", re.DOTALL)
_ITALIC_STAR_RE = re.compile(r"\*(.+?)\*", re.DOTALL)
_ITALIC_UNDER_RE = re.compile(r"(?<!\w)_(.+?)_(?!\w)", re.DOTALL)
_CODE_RE = re.compile(r"`(.+?)`", re.DOTALL)

_PARA_RE = re.compile(r"\n\n")
_SENT_RE = re.compile(r"[.!?](?=\s)")


def strip_markdown(text: str) -> str:
    """Reduce a small markdown subset to plain text.

    Handles **bold**, *italic*, __bold__, _italic_, `code`, leading # headers, and leading
    '* ' / '- ' bullets (both normalised to '- ').
    """
    if not text:
        return text
    out = _STAR_BULLET_RE.sub(r"\1- ", text)
    out = _HEADER_RE.sub("", out)
    out = _BOLD_RE.sub(r"\1", out)
    out = _BOLD_UNDER_RE.sub(r"\1", out)
    out = _ITALIC_STAR_RE.sub(r"\1", out)
    out = _ITALIC_UNDER_RE.sub(r"\1", out)
    out = _CODE_RE.sub(r"\1", out)
    return out


def _last_boundary(window: str) -> int:
    """Index just past the last paragraph break in window, else the last sentence end, else -1."""
    best = -1
    for match in _PARA_RE.finditer(window):
        best = match.end()
    if best > 0:
        return best
    for match in _SENT_RE.finditer(window):
        best = match.end()
    return best


def chunk_text(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Split text into chunks of at most `limit` chars, preferring paragraph/sentence breaks.

    Chunks are contiguous slices of the input, so "".join(chunk_text(t)) == t.
    """
    if not text:
        return [""]
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + limit, total)
        if end < total:
            idx = _last_boundary(text[start:end])
            if idx > 0:
                end = start + idx
        chunks.append(text[start:end])
        start = end
    return chunks


def pick_top_memory(memory_used: Sequence[Any]) -> Optional[Any]:
    """Return the highest-confidence item, or None when there is no memory."""
    if not memory_used:
        return None
    return max(memory_used, key=lambda item: getattr(item, "confidence", 0) or 0)


def _as_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    dt: Optional[datetime] = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    if dt is None:
        return None
    return f"{dt.day} {dt.strftime('%b %Y')}"


def format_footer(memory_used: Sequence[Any]) -> Optional[str]:
    """Footer for the reply: "relevance NN% · from your note on <d Mon YYYY>".

    Uses the highest-confidence item. The word is always "relevance", never "probability".
    """
    item = pick_top_memory(memory_used)
    if item is None:
        return None
    confidence = getattr(item, "confidence", 0) or 0
    date = _as_date(getattr(item, "created_at", None))
    if date:
        return f"relevance {confidence}% · from your note on {date}"
    return f"relevance {confidence}%"
