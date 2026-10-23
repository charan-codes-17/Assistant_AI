"""backend/bot/handlers.py
Telegram bot handlers (text only, Phase 4a).

Design notes:
  - Services are called IN-PROCESS (orchestrator / actions), never over the HTTP API.
  - Private chats only. Group/channel updates are ignored.
  - The reply is sent BEFORE any confirm prompt; postprocess runs in a background task.
  - Handlers read chat_id/user from the update and never trust callback data for identity.
  - Every handler is wrapped so an exception is logged (no tokens, no message text beyond the
    traceback) and a friendly message is sent instead of crashing the poller.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_, select

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from backend.bot import state as state_module
from backend.bot import textfmt
from backend.config.settings import settings
from backend.database import AsyncSessionLocal
from backend.models.models import ConversationSession, KnowledgeRecord, Task, User
from backend.services import actions as actions_service
from backend.services import audio as audio_service
from backend.services import embeddings as embeddings_service
from backend.services import memory_search
from backend.services import orchestrator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed user-facing strings
# ---------------------------------------------------------------------------
WELCOME_BANNER = (
    "Demo only: please don't send real personal data. "
    "Send /forget any time to delete everything."
)
WELCOME = f"You're signed up. Send me a message and I'll help.\n\n{WELCOME_BANNER}"
WELCOME_BACK = "Welcome back."
SIGNUP_PROMPT = "Please send /start <invite code> to begin."
REFUSAL = "Sorry, I can't sign you up. Please send /start <invite code>."
NOT_AVAILABLE = "That item is no longer available"
VOICE_TOO_LONG = "That voice note is too long. Please keep it under {0} seconds."
VOICE_NO_SPEECH = "I couldn't make out any speech in that. Could you try again?"
VOICE_ERROR = "Sorry, I couldn't process that voice note. Please try again."
VOICE_HEARD = "Heard: {0}"
UNKNOWN_COMMAND = "That command is not available yet."
GENERIC_ERROR = "Something went wrong. Please try again."
EXTRACTION_FAILED = "I couldn't process that note for saving. You can send it again."
NEW_SESSION = "New session started."
QUIET_OFF_TEXT = "Reminders are off. Send /quiet to turn them back on."
QUIET_ON_TEXT = "Reminders are on."
REMINDER_DONE_TEXT = "Marked done ✅"
HELP_TEXT = (
    "Available commands:\n"
    "/start <invite code> - sign up\n"
    "/newsession - start a new conversation session\n"
    "/quiet - turn reminders on or off\n"
    "/memory <topic> - search your saved memories\n"
    "/forget - delete everything and your account\n"
    "/help - show this list"
)
MEMORY_EMPTY = "I don't have anything saved about that yet."
MEMORY_DELETE_PROMPT = "Delete this memory? {0}"
MEMORY_DELETED = "Deleted."
CANCELLED = "Cancelled."
FORGET_WARNING = (
    "This will permanently delete all your memories, tasks and your account. "
    "This cannot be undone."
)
FORGET_CONFIRM = "Are you sure? Everything will be deleted."
FORGET_EXPIRED = "That request expired. Send /forget again."
FORGET_DONE = "Everything has been deleted. Send /start <invite code> to begin again."
FORGET_NONCE_TTL_SECONDS = 120
MEMORY_PREVIEW_CHARS = 100

MAX_INLINE_RECORDS = 3
CALLBACK_DATA_MAX_BYTES = 64
SAVED_CONTENT_CHARS = 300
BATCH_PREVIEW_LINES = 15
BATCH_PREVIEW_CHARS = 80
CONTRADICTION_HEADER = "Heads up: this may conflict with something you told me earlier: {0}."
NOTICE_FIELD_CHARS = 200
_SAVED = "Saved ✅"
_DISCARDED = "Discarded ❌"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _default_state(context: Any) -> state_module.BotState:
    data = getattr(context, "bot_data", None)
    if isinstance(data, dict):
        found = data.get("state")
        if found is not None:
            return found
    return state_module.state


def _is_private(update: Any) -> bool:
    chat = getattr(update, "effective_chat", None)
    return getattr(chat, "type", "private") == "private"


def _invite_codes() -> list[str]:
    return [c.strip() for c in (settings.INVITE_CODES or "").split(",") if c.strip()]


def _code_matches(code: str, configured: Sequence[str]) -> bool:
    if not code:
        return False
    given = code.encode("utf-8")
    return any(
        hmac.compare_digest(given, candidate.encode("utf-8")) for candidate in configured
    )


def _to_int(raw: str) -> Optional[int]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def _user_by_chat(session: Any, chat_id: int) -> Optional[User]:
    stmt = select(User).where(User.telegram_chat_id == chat_id)
    return (await session.execute(stmt)).scalar_one_or_none()


def _append_history(
    history: list[dict[str, str]], user_text: str, assistant_text: Optional[str]
) -> list[dict[str, str]]:
    out = list(history)
    out.append({"role": "user", "content": user_text})
    if assistant_text:
        out.append({"role": "assistant", "content": assistant_text})
    return out[-orchestrator.HISTORY_MAX_MESSAGES:]


def _callbacks(markup: Any) -> list[str]:
    rows = getattr(markup, "inline_keyboard", None) or []
    out: list[str] = []
    for row in rows:
        for button in row:
            cb = getattr(button, "callback_data", None)
            if cb is not None:
                out.append(cb)
    return out


def _format_due(due: Optional[datetime], tz_key: str) -> str:
    if due is None:
        return "no due date"
    try:
        tz = ZoneInfo(tz_key or "UTC")
    except Exception:  # noqa: BLE001 - unknown tz name falls back to UTC
        tz = ZoneInfo("UTC")
    local = due.astimezone(tz)
    return f"{local.strftime('%Y-%m-%d %H:%M')} {local.tzname()} ({tz_key})"


async def _replace_message(
    context: Any, chat_id: int, message_id: int, text: str, reply_markup: Any = None
) -> None:
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=text, reply_markup=reply_markup
        )
    except Exception:  # noqa: BLE001 - editing must never bubble up
        logger.exception("edit_message_text failed for chat_id=%s", chat_id)


def _forget_now() -> float:
    """Monotonic clock seam for the /forget nonce expiry (tests patch this)."""
    return time.monotonic()


def _memory_line(index: int, rec: Any) -> str:
    date = textfmt._as_date(rec.created_at) or "unknown date"
    return f"{index}. {rec.content} ({rec.kind}, {rec.certainty}, from {rec.source}, {date})"


def snooze_text(minutes: int) -> str:
    """Snooze confirmation, singular for exactly one minute."""
    unit = "minute" if minutes == 1 else "minutes"
    return f"Snoozed for {minutes} {unit} ⏰"


def _saved_record_text(record: Any) -> str:
    """Keep what was saved: the record content (bounded) under the Saved header."""
    return f"{_SAVED}\n{record.content[:SAVED_CONTENT_CHARS]}"


def _saved_task_text(task: Any, user: User) -> str:
    """Keep what was saved: the task title and its due time, same format as the prompt."""
    tz_key = user.timezone or "UTC"
    if task.due_at is None:
        return f"{_SAVED}\n{task.title} (no due date)"
    return f"{_SAVED}\n{task.title}, due {_format_due(task.due_at, tz_key)}"


def _batch_preview(records: list[Any]) -> str:
    """Header + up to BATCH_PREVIEW_LINES numbered items, then '...and K more'."""
    lines = [f"{len(records)} memories to review:"]
    for index, rec in enumerate(records[:BATCH_PREVIEW_LINES], start=1):
        lines.append(f"{index}. {rec.content[:BATCH_PREVIEW_CHARS]}")
    extra = len(records) - BATCH_PREVIEW_LINES
    if extra > 0:
        lines.append(f"...and {extra} more")
    return "\n".join(lines)


def contradiction_notice(decision: Any) -> Optional[str]:
    """Plain-text contradiction notice, or None when nothing should be sent.

    Deterministic: built in code from the intervention decision (no LLM call). A notice is sent
    iff the engine flagged a contradiction with a target (contradiction=True AND conflicts_with
    present), regardless of the action directive: contradiction=False and a missing
    conflicts_with both send nothing. Every free-text field is truncated and markdown-stripped.
    """
    if not getattr(decision, "contradiction", False):
        return None
    ref = getattr(decision, "conflicts_with", None)
    if ref is None:
        return None

    summary = getattr(decision, "contradiction_detail", None) or f"your earlier {ref.kind}"
    lines = [CONTRADICTION_HEADER.format(textfmt.strip_markdown(summary)[:NOTICE_FIELD_CHARS])]
    question = getattr(decision, "clarifying_question", None)
    if question:
        lines.append(textfmt.strip_markdown(question)[:NOTICE_FIELD_CHARS])
    return "\n".join(lines)


async def _send_contradiction_notice(context: Any, chat_id: int, decision: Any) -> None:
    """Send the notice (if any) before the confirm prompts; a failure never breaks the turn."""
    notice = contradiction_notice(decision)
    if not notice:
        return
    try:
        await context.bot.send_message(chat_id=chat_id, text=notice)
    except Exception as exc:  # noqa: BLE001 - best-effort; log exception class only, no content
        logger.warning("contradiction notice send failed: %s", type(exc).__name__)


def guarded(fn):  # type: ignore[no-untyped-def]
    """Wrap a handler so exceptions are logged and replaced with a friendly message."""

    async def wrapper(update: Any, context: Any, *args: Any, **kwargs: Any) -> None:
        try:
            return await fn(update, context, *args, **kwargs)
        except Exception:  # noqa: BLE001 - poller must never crash
            logger.exception("handler %s failed", fn.__name__)
            try:
                chat = getattr(update, "effective_chat", None)
                if chat is not None:
                    await context.bot.send_message(chat_id=chat.id, text=GENERIC_ERROR)
            except Exception:  # noqa: BLE001
                logger.exception("failed to send generic error message")

    wrapper.__name__ = fn.__name__
    return wrapper


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------
@guarded
async def cmd_start(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    raw = getattr(getattr(update, "message", None), "text", "") or ""
    parts = raw.split(maxsplit=1)
    code = parts[1].strip() if len(parts) > 1 else ""

    async with AsyncSessionLocal() as session:
        existing = await _user_by_chat(session, chat_id)
        if existing is not None:
            await context.bot.send_message(chat_id=chat_id, text=WELCOME_BACK)
            return

        configured = _invite_codes()
        if not _code_matches(code, configured):
            await context.bot.send_message(chat_id=chat_id, text=REFUSAL)
            return

        total = (await session.execute(select(func.count()).select_from(User))).scalar_one()
        if total >= settings.BOT_MAX_USERS:
            await context.bot.send_message(chat_id=chat_id, text=REFUSAL)
            return

        # All other User columns have defaults; only telegram_chat_id is set.
        session.add(User(telegram_chat_id=chat_id))
        await session.commit()

    await context.bot.send_message(chat_id=chat_id, text=WELCOME)


# ---------------------------------------------------------------------------
# Shared turn pipeline (text and voice)
# ---------------------------------------------------------------------------
async def _run_turn(
    context: Any, chat_id: int, user_id: int, text: str, source: str, st: Any, chat: Any
) -> None:
    """Reply first, then start the background postprocess. Caller must hold chat.lock."""
    if chat.session_id is None:
        async with AsyncSessionLocal() as session:
            sess = ConversationSession(user_id=user_id, channel="telegram")
            session.add(sess)
            await session.commit()
            await session.refresh(sess)
            chat.session_id = sess.id

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    history = list(chat.history)
    reply = await orchestrator.respond(
        user_id, text, session_id=chat.session_id, history=history, source=source
    )
    await _send_reply(context, chat_id, reply)
    chat.history = _append_history(chat.history, text, reply.reply)

    task = asyncio.create_task(
        _postprocess_and_prompt(user_id, chat_id, reply, context, st)
    )
    st.background.add(task)
    task.add_done_callback(st.background.discard)


# ---------------------------------------------------------------------------
# Free-form text -> reply (then background postprocess)
# ---------------------------------------------------------------------------
@guarded
async def on_text(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    text = getattr(getattr(update, "message", None), "text", "") or ""
    if not text.strip():
        return

    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
    if user is None:
        await context.bot.send_message(chat_id=chat_id, text=SIGNUP_PROMPT)
        return

    st = state or _default_state(context)
    chat = st.chat(chat_id)

    async with chat.lock:
        await _run_turn(context, chat_id, user.id, text, "text", st, chat)


async def _send_reply(context: Any, chat_id: int, reply: Any) -> None:
    body = textfmt.strip_markdown(reply.reply or orchestrator.FALLBACK_REPLY)
    footer = textfmt.format_footer(reply.memory_used)
    if footer:
        body = f"{body}\n\n{footer}"

    keyboard = None
    top = textfmt.pick_top_memory(reply.memory_used)
    if top is not None:
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Show source", callback_data=f"src:{top.record_id}")]]
        )

    chunks = textfmt.chunk_text(body)
    for index, chunk in enumerate(chunks):
        await context.bot.send_message(
            chat_id=chat_id,
            text=chunk,
            reply_markup=keyboard if index == len(chunks) - 1 else None,
        )


async def _postprocess_and_prompt(
    user_id: int, chat_id: int, reply: Any, context: Any, st: Any
) -> None:
    try:
        post = await orchestrator.postprocess(user_id, reply)
    except Exception:  # noqa: BLE001 - follow-up is best-effort
        logger.exception("postprocess failed for chat_id=%s", chat_id)
        return

    # Intervention is logged as flags/ids only (never memory content). The contradiction notice is
    # the only user-facing part; it is sent right after the reply and before the confirm prompts.
    decision = post.intervention
    logger.info(
        "intervention: action=%s contradiction=%s conflicts_with=%s target_task_id=%s "
        "surfaced_record_id=%s",
        decision.action,
        decision.contradiction,
        (
            f"{decision.conflicts_with.kind}:{decision.conflicts_with.id}"
            if decision.conflicts_with
            else None
        ),
        decision.target_task_id,
        decision.surfaced_record_id,
    )
    await _send_contradiction_notice(context, chat_id, decision)

    if post.extraction_failed:
        await context.bot.send_message(chat_id=chat_id, text=EXTRACTION_FAILED)

    await _prompt_records(context, chat_id, list(post.pending_records), st)
    await _prompt_tasks(context, chat_id, user_id, list(post.pending_tasks))


async def _prompt_records(context: Any, chat_id: int, records: list[Any], st: Any) -> None:
    if not records:
        return

    if len(records) <= MAX_INLINE_RECORDS:
        for rec in records:
            keyboard = InlineKeyboardMarkup(
                [[
                    InlineKeyboardButton("✅ Save", callback_data=f"rec:ok:{rec.id}"),
                    InlineKeyboardButton("❌ Discard", callback_data=f"rec:no:{rec.id}"),
                ]]
            )
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Save this memory? {rec.content}",
                reply_markup=keyboard,
            )
        return

    ids = [rec.id for rec in records]
    csv = ",".join(str(i) for i in ids)
    save_data = f"rec:allok:{csv}"
    discard_data = f"rec:allno:{csv}"
    if (
        len(save_data.encode("utf-8")) > CALLBACK_DATA_MAX_BYTES
        or len(discard_data.encode("utf-8")) > CALLBACK_DATA_MAX_BYTES
    ):
        token = st.new_batch_token(ids)
        save_data = f"rec:allok:{token}"
        discard_data = f"rec:allno:{token}"
    keyboard = InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ Save all", callback_data=save_data),
            InlineKeyboardButton("❌ Discard all", callback_data=discard_data),
        ]]
    )
    await context.bot.send_message(
        chat_id=chat_id,
        text=_batch_preview(records),
        reply_markup=keyboard,
    )


async def _prompt_tasks(
    context: Any, chat_id: int, user_id: int, candidates: list[Any]
) -> None:
    if not candidates:
        return

    async with AsyncSessionLocal() as session:
        user = await session.get(User, user_id)
        if user is None:
            logger.warning("stage_task_candidates skipped: user_id=%s not found", user_id)
            return
        tz_key = user.timezone or "UTC"
        staged = await actions_service.stage_task_candidates(session, user, candidates)

    for task in staged:
        if task.due_at is None:
            body = f"Save this promise? {task.title} (no due date)"
        else:
            body = f"Save this promise? {task.title}, due {_format_due(task.due_at, tz_key)}"
        keyboard = InlineKeyboardMarkup(
            [[
                InlineKeyboardButton("✅", callback_data=f"tk:ok:{task.id}"),
                InlineKeyboardButton("❌", callback_data=f"tk:no:{task.id}"),
            ]]
        )
        await context.bot.send_message(chat_id=chat_id, text=body, reply_markup=keyboard)


# ---------------------------------------------------------------------------
# Voice notes -> transcript -> the same turn pipeline
# ---------------------------------------------------------------------------
_voice_sem: Optional[asyncio.Semaphore] = None
_voice_sem_loop: Any = None


def _voice_semaphore() -> asyncio.Semaphore:
    """One transcription at a time, per running event loop (a fresh loop gets a fresh lock)."""
    global _voice_sem, _voice_sem_loop
    loop = asyncio.get_running_loop()
    if _voice_sem is None or _voice_sem_loop is not loop:
        _voice_sem = asyncio.Semaphore(1)
        _voice_sem_loop = loop
    return _voice_sem


async def _download_voice(context: Any, voice: Any) -> bytes:
    tg_file = await context.bot.get_file(voice.file_id)
    data = await tg_file.download_as_bytearray()
    return bytes(data)


async def _transcribe_voice(data: bytes) -> str:
    async with _voice_semaphore():
        return await asyncio.wait_for(
            asyncio.to_thread(audio_service.transcribe_bytes, data),
            timeout=settings.VOICE_TRANSCRIBE_TIMEOUT_SECONDS,
        )


@guarded
async def on_voice(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    voice = getattr(getattr(update, "message", None), "voice", None)
    if voice is None:
        return

    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
    if user is None:
        # Unknown chat: no download, no transcription.
        await context.bot.send_message(chat_id=chat_id, text=SIGNUP_PROMPT)
        return

    duration = getattr(voice, "duration", None)
    if duration is not None and duration > settings.VOICE_MAX_SECONDS:
        await context.bot.send_message(
            chat_id=chat_id, text=VOICE_TOO_LONG.format(settings.VOICE_MAX_SECONDS)
        )
        return

    st = state or _default_state(context)
    chat = st.chat(chat_id)

    async with chat.lock:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        t_download = 0.0
        t_transcribe = 0.0
        try:
            started = time.perf_counter()
            data = await _download_voice(context, voice)
            t_download = time.perf_counter() - started
            started = time.perf_counter()
            transcript = await _transcribe_voice(data)
            t_transcribe = time.perf_counter() - started
        except Exception:  # noqa: BLE001 - download/decode/transcribe/timeout all land here
            logger.exception("voice handling failed for chat_id=%s", chat_id)
            await context.bot.send_message(chat_id=chat_id, text=VOICE_ERROR)
            return

        # Privacy: duration, size and timings only; never the transcript or the bytes.
        logger.info(
            "voice turn: chat_id=%s duration=%ss bytes=%s download=%.2fs transcribe=%.2fs",
            chat_id,
            duration,
            len(data),
            t_download,
            t_transcribe,
        )

        text = (transcript or "").strip()
        if not text:
            await context.bot.send_message(chat_id=chat_id, text=VOICE_NO_SPEECH)
            return

        await context.bot.send_message(chat_id=chat_id, text=VOICE_HEARD.format(text))
        await _run_turn(context, chat_id, user.id, text, "voice", st, chat)


# ---------------------------------------------------------------------------
# /newsession and other commands
# ---------------------------------------------------------------------------
@guarded
async def on_newsession(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
        if user is None:
            await context.bot.send_message(chat_id=chat_id, text=SIGNUP_PROMPT)
            return
        sess = ConversationSession(user_id=user.id, channel="telegram")
        session.add(sess)
        await session.commit()
        await session.refresh(sess)
        new_session_id = sess.id

    st = state or _default_state(context)
    chat = st.chat(chat_id)
    chat.history.clear()
    chat.session_id = new_session_id

    await context.bot.send_message(chat_id=chat_id, text=NEW_SESSION)


@guarded
async def cmd_quiet(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
        if user is None:
            await context.bot.send_message(chat_id=chat_id, text=SIGNUP_PROMPT)
            return
        new_value = not user.quiet_mode
        await actions_service.set_quiet_mode(session, user, new_value)

    await context.bot.send_message(
        chat_id=chat_id, text=QUIET_OFF_TEXT if new_value else QUIET_ON_TEXT
    )


# ---------------------------------------------------------------------------
# /help, /memory, /forget (Phase 4c-1)
# ---------------------------------------------------------------------------
@guarded
async def cmd_help(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
    if user is None:
        await context.bot.send_message(chat_id=chat_id, text=SIGNUP_PROMPT)
        return
    await context.bot.send_message(chat_id=chat_id, text=HELP_TEXT)


async def _memory_results(session: Any, user: User, topic: str) -> list[KnowledgeRecord]:
    """Confirmed, non-expired records for this user (newest first, at most 5)."""
    if topic:
        vectors = await embeddings_service.embed_texts(
            [topic], timeout=settings.EMBEDDING_TIMEOUT_SECONDS
        )
        hits = await memory_search.search_confirmed(session, user.id, vectors[0], k=5)
        return [rec for rec, _similarity in hits]

    stmt = (
        select(KnowledgeRecord)
        .where(
            KnowledgeRecord.user_id == user.id,
            KnowledgeRecord.status == "confirmed",
            KnowledgeRecord.superseded_by.is_(None),
            or_(
                KnowledgeRecord.expires_at.is_(None),
                KnowledgeRecord.expires_at > func.now(),
            ),
        )
        .order_by(KnowledgeRecord.created_at.desc(), KnowledgeRecord.id.desc())
        .limit(5)
    )
    return list((await session.execute(stmt)).scalars().all())


@guarded
async def cmd_memory(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    raw = getattr(getattr(update, "message", None), "text", "") or ""
    parts = raw.split(maxsplit=1)
    topic = parts[1].strip() if len(parts) > 1 else ""

    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
        if user is None:
            await context.bot.send_message(chat_id=chat_id, text=SIGNUP_PROMPT)
            return
        records = await _memory_results(session, user, topic)

    # Privacy: ids and counts only, never the record content.
    logger.info(
        "memory list: chat_id=%s count=%s ids=%s",
        chat_id,
        len(records),
        [rec.id for rec in records],
    )

    if not records:
        await context.bot.send_message(chat_id=chat_id, text=MEMORY_EMPTY)
        return

    for index, rec in enumerate(records, start=1):
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🗑 Delete", callback_data=f"mem:del:{rec.id}")]]
        )
        await context.bot.send_message(
            chat_id=chat_id, text=_memory_line(index, rec), reply_markup=keyboard
        )


@guarded
async def cmd_forget(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
    if user is None:
        await context.bot.send_message(chat_id=chat_id, text=SIGNUP_PROMPT)
        return

    st = state or _default_state(context)
    chat = st.chat(chat_id)
    chat.forget_nonce = None
    chat.forget_expires_at = None

    keyboard = InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("Continue", callback_data="fg:1"),
            InlineKeyboardButton("Cancel", callback_data="fg:no"),
        ]]
    )
    await context.bot.send_message(chat_id=chat_id, text=FORGET_WARNING, reply_markup=keyboard)


@guarded
async def on_other_command(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
    if user is None:
        await context.bot.send_message(chat_id=chat_id, text=SIGNUP_PROMPT)
        return
    await context.bot.send_message(chat_id=chat_id, text=UNKNOWN_COMMAND)


# ---------------------------------------------------------------------------
# Callback queries
# ---------------------------------------------------------------------------
@guarded
async def on_callback(update: Any, context: Any, state: Any = None) -> None:
    query = update.callback_query
    if query is None:
        return
    data = query.data or ""
    chat = getattr(query, "message", None)
    chat_id = getattr(chat, "chat_id", None) or getattr(getattr(chat, "chat", None), "id", None)
    message_id = getattr(chat, "message_id", 0)

    await context.bot.answer_callback_query(callback_query_id=query.id)

    if chat_id is None:
        return

    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
        if user is None:
            await _replace_message(context, chat_id, message_id, NOT_AVAILABLE)
            return

        chat_state = (state or _default_state(context)).chat(chat_id)

        if data.startswith("src:"):
            await _show_source(context, session, user, chat_id, data[len("src:"):])
            return

        if data.startswith("rec:ok:"):
            record_id = _to_int(data[len("rec:ok:"):])
            record = (
                await actions_service.confirm_record(session, user, record_id)
                if record_id is not None
                else None
            )
            await _replace_message(
                context,
                chat_id,
                message_id,
                _saved_record_text(record) if record is not None else NOT_AVAILABLE,
            )
            return

        if data.startswith("rec:no:"):
            record_id = _to_int(data[len("rec:no:"):])
            record = (
                await actions_service.reject_record(session, user, record_id)
                if record_id is not None
                else None
            )
            await _replace_message(
                context, chat_id, message_id, _DISCARDED if record is not None else NOT_AVAILABLE
            )
            return

        if data.startswith("rec:allok:"):
            ids = _resolve_batch_ids(state or _default_state(context), data[len("rec:allok:"):])
            result = await actions_service.confirm_records(user.id, ids) if ids else None
            count = len(result.confirmed_records) if result else 0
            await _replace_message(
                context,
                chat_id,
                message_id,
                f"Saved {count} memories ✅" if count else NOT_AVAILABLE,
            )
            return

        if data.startswith("rec:allno:"):
            ids = _resolve_batch_ids(state or _default_state(context), data[len("rec:allno:"):])
            any_rejected = False
            for record_id in ids:
                if await actions_service.reject_record(session, user, record_id) is not None:
                    any_rejected = True
            await _replace_message(
                context, chat_id, message_id, _DISCARDED if any_rejected else NOT_AVAILABLE
            )
            return

        if data.startswith("tk:ok:"):
            task_id = _to_int(data[len("tk:ok:"):])
            task = (
                await actions_service.confirm_pending_task(session, user, task_id)
                if task_id is not None
                else None
            )
            await _replace_message(
                context,
                chat_id,
                message_id,
                _saved_task_text(task, user) if task is not None else NOT_AVAILABLE,
            )
            return

        if data.startswith("tk:no:"):
            task_id = _to_int(data[len("tk:no:"):])
            deleted = (
                await actions_service.discard_pending_task(session, user, task_id)
                if task_id is not None
                else False
            )
            await _replace_message(
                context, chat_id, message_id, _DISCARDED if deleted else NOT_AVAILABLE
            )
            return

        if data.startswith("rm:done:"):
            task_id = _to_int(data[len("rm:done:"):])
            task = None
            if task_id is not None:
                try:
                    task = await actions_service.mark_task_done(session, user, task_id)
                except actions_service.NotFoundError:
                    task = None
            await _replace_message(
                context, chat_id, message_id, REMINDER_DONE_TEXT if task is not None else NOT_AVAILABLE
            )
            return

        if data.startswith("rm:snz:"):
            task_id = _to_int(data[len("rm:snz:"):])
            task = None
            if task_id is not None:
                try:
                    task = await actions_service.snooze_task(
                        session, user, task_id, minutes=settings.REMINDER_SNOOZE_MINUTES
                    )
                except actions_service.NotFoundError:
                    task = None
            await _replace_message(
                context,
                chat_id,
                message_id,
                snooze_text(settings.REMINDER_SNOOZE_MINUTES)
                if task is not None
                else NOT_AVAILABLE,
            )
            return

        if data.startswith("rm:quiet:"):
            task_id = _to_int(data[len("rm:quiet:"):])
            owned = False
            if task_id is not None:
                stmt = select(Task).where(Task.id == task_id, Task.user_id == user.id)
                owned = (await session.execute(stmt)).scalar_one_or_none() is not None
            if owned:
                await actions_service.set_quiet_mode(session, user, True)
                await _replace_message(context, chat_id, message_id, QUIET_OFF_TEXT)
            else:
                await _replace_message(context, chat_id, message_id, NOT_AVAILABLE)
            return

        if data.startswith("mem:del:"):
            record_id = _to_int(data[len("mem:del:"):])
            record = None
            if record_id is not None:
                stmt = select(KnowledgeRecord).where(
                    KnowledgeRecord.id == record_id,
                    KnowledgeRecord.user_id == user.id,
                )
                record = (await session.execute(stmt)).scalar_one_or_none()
            if record is None:
                await _replace_message(context, chat_id, message_id, NOT_AVAILABLE)
                return
            keyboard = InlineKeyboardMarkup(
                [[
                    InlineKeyboardButton("Yes, delete", callback_data=f"mem:yes:{record.id}"),
                    InlineKeyboardButton("Cancel", callback_data=f"mem:no:{record.id}"),
                ]]
            )
            await _replace_message(
                context,
                chat_id,
                message_id,
                MEMORY_DELETE_PROMPT.format(record.content[:MEMORY_PREVIEW_CHARS]),
                reply_markup=keyboard,
            )
            return

        if data.startswith("mem:no:"):
            # Cancel: keep the record; nothing is deleted.
            await _replace_message(context, chat_id, message_id, CANCELLED)
            return

        if data.startswith("mem:yes:"):
            record_id = _to_int(data[len("mem:yes:"):])
            deleted = False
            if record_id is not None:
                try:
                    await actions_service.delete_record(session, user, record_id, confirmed=True)
                    deleted = True
                except actions_service.NotFoundError:
                    deleted = False
            await _replace_message(
                context, chat_id, message_id, MEMORY_DELETED if deleted else NOT_AVAILABLE
            )
            return

        if data == "fg:1":
            nonce = uuid.uuid4().hex
            chat_state.forget_nonce = nonce
            chat_state.forget_expires_at = _forget_now() + FORGET_NONCE_TTL_SECONDS
            keyboard = InlineKeyboardMarkup(
                [[
                    InlineKeyboardButton("Yes, delete everything", callback_data=f"fg:2:{nonce}"),
                    InlineKeyboardButton("Cancel", callback_data="fg:no"),
                ]]
            )
            await _replace_message(
                context, chat_id, message_id, FORGET_CONFIRM, reply_markup=keyboard
            )
            return

        if data == "fg:no":
            chat_state.forget_nonce = None
            chat_state.forget_expires_at = None
            await _replace_message(context, chat_id, message_id, CANCELLED)
            return

        if data.startswith("fg:2:"):
            provided = data[len("fg:2:"):]
            valid = (
                chat_state.forget_nonce is not None
                and chat_state.forget_expires_at is not None
                and _forget_now() <= chat_state.forget_expires_at
                and hmac.compare_digest(
                    provided.encode("utf-8"), chat_state.forget_nonce.encode("utf-8")
                )
            )
            chat_state.forget_nonce = None
            chat_state.forget_expires_at = None
            if not valid:
                await _replace_message(context, chat_id, message_id, FORGET_EXPIRED)
                return
            await actions_service.delete_user_account(session, user, confirmed=True)
            chat_state.reset()
            await _replace_message(context, chat_id, message_id, FORGET_DONE)
            return

        await _replace_message(context, chat_id, message_id, NOT_AVAILABLE)


def _resolve_batch_ids(st: Any, raw: str) -> list[int]:
    """Resolve either a csv of ids or a short batch token into a list of ids."""
    token_ids = st.resolve_batch_token(raw)
    if token_ids is not None:
        return token_ids
    ids: list[int] = []
    for part in raw.split(","):
        value = _to_int(part.strip())
        if value is not None:
            ids.append(value)
    return ids


async def _show_source(
    context: Any, session: Any, user: User, chat_id: int, raw_id: str
) -> None:
    record_id = _to_int(raw_id)
    record = None
    if record_id is not None:
        stmt = select(KnowledgeRecord).where(
            KnowledgeRecord.id == record_id,
            KnowledgeRecord.user_id == user.id,
        )
        record = (await session.execute(stmt)).scalar_one_or_none()

    if record is None:
        await context.bot.send_message(chat_id=chat_id, text=NOT_AVAILABLE)
        return

    date = textfmt._as_date(record.created_at) or "unknown date"
    body = (
        f"Source\nkind: {record.kind}\ncertainty: {record.certainty}\n"
        f"from: {record.source}\ndate: {date}\n\n{record.content}"
    )
    await context.bot.send_message(chat_id=chat_id, text=body)


# ---------------------------------------------------------------------------
# Application error handler
# ---------------------------------------------------------------------------
async def on_error(update: Any, context: Any) -> None:
    logger.error("application error: %s", getattr(context, "error", None), exc_info=getattr(context, "error", None))
    try:
        chat = getattr(update, "effective_chat", None)
        if chat is not None:
            await context.bot.send_message(chat_id=chat.id, text=GENERIC_ERROR)
    except Exception:  # noqa: BLE001
        logger.exception("failed to send generic error message")
