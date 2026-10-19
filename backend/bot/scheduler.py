"""backend/bot/scheduler.py
Deterministic reminder scheduler for the bot worker (Roadmap Phase 5.1; PRD FR11/FR20).

Runs an asyncio polling loop alongside the Telegram poller. Each tick:
  1. opens a session and asks reminders.find_due_reminders(now) for the due reminders,
  2. snapshots then atomically claims each one (so two concurrent ticks cannot both send),
  3. sends a plain-text reminder with [Done]/[Snooze]/[Stay quiet] buttons.

DETERMINISTIC: zero LLM calls and zero embedding calls.

Failure policy:
  - Forbidden, or BadRequest mentioning "chat not found" -> PERMANENT: keep the task claimed
    (so it is not retried forever) and log WARNING with the task id and exception class only.
  - Anything else (NetworkError, TimedOut, RetryAfter, unknown) -> TRANSIENT: release the task
    back to its pre-claim snapshot and retry on the next tick.

Privacy: never log task titles, message text or transcripts; only counts and ids.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, Forbidden

from backend.config.settings import settings
from backend.database import AsyncSessionLocal
from backend.models.models import Task
from backend.services import reminders
from backend.services.reminders import ReminderSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TickResult:
    """Outcome counters for one scheduler tick (no titles, no message text)."""

    sent: int = 0
    skipped: int = 0
    failed_transient: int = 0
    failed_permanent: int = 0


def _keyboard(task_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ Done", callback_data=f"rm:done:{task_id}"),
            InlineKeyboardButton("⏰ Snooze", callback_data=f"rm:snz:{task_id}"),
            InlineKeyboardButton("🔕 Stay quiet", callback_data=f"rm:quiet:{task_id}"),
        ]]
    )


def _is_permanent(exc: Exception) -> bool:
    """Forbidden, or a BadRequest whose message says the chat no longer exists."""
    if isinstance(exc, Forbidden):
        return True
    if isinstance(exc, BadRequest):
        return "chat not found" in str(exc).lower()
    return False


async def run_tick(bot: Any, now: datetime) -> TickResult:
    """One reminder scan. Never raises for a single item's send failure."""
    result = TickResult()
    async with AsyncSessionLocal() as session:
        due = await reminders.find_due_reminders(session, now)
        for item in due:
            if item.due_at is None:
                # A reminder line needs a due time; a snoozed task without due_at is skipped.
                result.skipped += 1
                continue

            task = await session.get(Task, item.task_id)
            if task is None:
                result.skipped += 1
                continue
            snapshot = ReminderSnapshot(
                status=task.status,
                snoozed_until=task.snoozed_until,
                last_reminded_at=task.last_reminded_at,
            )

            claimed = await reminders.claim_reminder(session, item.task_id, now)
            if not claimed:
                result.skipped += 1
                continue

            text = reminders.render_reminder_text(item.title, item.due_at, item.timezone, now)
            try:
                await bot.send_message(
                    chat_id=item.telegram_chat_id,
                    text=text,
                    reply_markup=_keyboard(item.task_id),
                )
            except Exception as exc:  # noqa: BLE001 - one bad chat must not stop the tick
                if _is_permanent(exc):
                    result.failed_permanent += 1
                    logger.warning(
                        "Reminder permanently undeliverable for task_id=%s (%s); keeping it claimed.",
                        item.task_id,
                        type(exc).__name__,
                    )
                else:
                    result.failed_transient += 1
                    logger.warning(
                        "Reminder send failed transiently for task_id=%s (%s); releasing for retry.",
                        item.task_id,
                        type(exc).__name__,
                    )
                    await reminders.release_reminder(session, item.task_id, snapshot)
                continue

            result.sent += 1

    return result


async def scheduler_loop(bot: Any) -> None:
    """Poll for due reminders forever. Cancellation propagates cleanly."""
    logger.info("Reminder scheduler loop starting (interval=%ss).", settings.REMINDER_SCAN_SECONDS)
    while True:
        try:
            await run_tick(bot, now=datetime.now(timezone.utc))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a failed tick must never kill the loop
            logger.exception("Reminder scheduler tick failed; continuing.")
        await asyncio.sleep(settings.REMINDER_SCAN_SECONDS)
