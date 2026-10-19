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
from datetime import datetime
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import func, select

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from backend.bot import state as state_module
from backend.bot import textfmt
from backend.config.settings import settings
from backend.database import AsyncSessionLocal
from backend.models.models import ConversationSession, KnowledgeRecord, User
from backend.services import actions as actions_service
from backend.services import orchestrator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed user-facing strings
# ---------------------------------------------------------------------------
WELCOME_BANNER = "Demo only: please don't send real personal data."
WELCOME = f"You're signed up. Send me a message and I'll help.\n\n{WELCOME_BANNER}"
WELCOME_BACK = "Welcome back."
SIGNUP_PROMPT = "Please send /start <invite code> to begin."
REFUSAL = "Sorry, I can't sign you up. Please send /start <invite code>."
NOT_AVAILABLE = "That item is no longer available"
VOICE_PLACEHOLDER = "Voice notes are not supported yet. Please type your message."
UNKNOWN_COMMAND = "That command is not available yet."
GENERIC_ERROR = "Something went wrong. Please try again."
EXTRACTION_FAILED = "I couldn't process that note for saving. You can send it again."
NEW_SESSION = "New session started."

MAX_INLINE_RECORDS = 3
CALLBACK_DATA_MAX_BYTES = 64
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
    return any(hmac.compare_digest(code, candidate) for candidate in configured)


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


async def _replace_message(context: Any, chat_id: int, message_id: int, text: str) -> None:
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=text, reply_markup=None
        )
    except Exception:  # noqa: BLE001 - editing must never bubble up
        logger.exception("edit_message_text failed for chat_id=%s", chat_id)


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
        if chat.session_id is None:
            async with AsyncSessionLocal() as session:
                sess = ConversationSession(user_id=user.id, channel="telegram")
                session.add(sess)
                await session.commit()
                await session.refresh(sess)
                chat.session_id = sess.id

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        history = list(chat.history)
        reply = await orchestrator.respond(
            user.id, text, session_id=chat.session_id, history=history
        )
        await _send_reply(context, chat_id, reply)
        chat.history = _append_history(chat.history, text, reply.reply)

        task = asyncio.create_task(
            _postprocess_and_prompt(user.id, chat_id, reply, context, st)
        )
        st.background.add(task)
        task.add_done_callback(st.background.discard)


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

    # Intervention is surfaced in the reply path only in a later phase; logging is all we do now.
    logger.info("intervention: %s", post.intervention.model_dump())

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
        text=f"{len(records)} memories to review",
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
# Voice (placeholder until Phase 4b)
# ---------------------------------------------------------------------------
@guarded
async def on_voice(update: Any, context: Any, state: Any = None) -> None:
    if not _is_private(update):
        return
    chat_id = update.effective_chat.id
    async with AsyncSessionLocal() as session:
        user = await _user_by_chat(session, chat_id)
    if user is None:
        await context.bot.send_message(chat_id=chat_id, text=SIGNUP_PROMPT)
        return
    await context.bot.send_message(chat_id=chat_id, text=VOICE_PLACEHOLDER)


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
                context, chat_id, message_id, _SAVED if record is not None else NOT_AVAILABLE
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
            confirmed = bool(result and result.confirmed_records)
            await _replace_message(
                context, chat_id, message_id, _SAVED if confirmed else NOT_AVAILABLE
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
                context, chat_id, message_id, _SAVED if task is not None else NOT_AVAILABLE
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
