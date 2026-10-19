"""backend/services/reminders.py
Deterministic reminder selection and claiming for the scheduler (PRD FR11/FR20, Roadmap §5.1).

Guarantees:
- Purely deterministic: every function takes `now` explicitly and never reads the clock.
- ZERO LLM calls and ZERO embedding calls.
- A task can be claimed at most once across concurrent claimers (single atomic UPDATE ... RETURNING).
- Only the owner's eligible rows are ever returned/claimed; quiet users and users without a
  telegram_chat_id are excluded here (defence in depth on top of the intervention guard, which
  deduplicates on tasks.last_reminded_at).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import settings
from backend.models.models import Task, User


@dataclass
class DueReminder:
    """One task that the scheduler should remind its owner about."""

    task_id: int
    title: str
    due_at: Optional[datetime]
    user_id: int
    telegram_chat_id: int
    timezone: str
    was_snoozed: bool


@dataclass
class ReminderSnapshot:
    """Task state captured before a claim, so release_reminder can restore it exactly."""

    status: str
    snoozed_until: Optional[datetime]
    last_reminded_at: Optional[datetime]


def _eligibility(now: datetime):
    """Shared WHERE predicate: rows a reminder may fire for, given `now`.

    open:    due_at set, never reminded, inside [due_at - LEAD, due_at + MAX_OVERDUE]
    snoozed: snoozed_until <= now, and (no due_at or within MAX_OVERDUE of due_at)
    """
    lead = timedelta(minutes=settings.REMINDER_LEAD_MINUTES)
    overdue = timedelta(hours=settings.REMINDER_MAX_OVERDUE_HOURS)

    open_branch = and_(
        Task.status == "open",
        Task.due_at.is_not(None),
        Task.last_reminded_at.is_(None),
        Task.due_at - lead <= now,
        Task.due_at + overdue >= now,
    )
    snoozed_branch = and_(
        Task.status == "snoozed",
        Task.snoozed_until.is_not(None),
        Task.snoozed_until <= now,
        or_(Task.due_at.is_(None), Task.due_at + overdue >= now),
    )
    return or_(open_branch, snoozed_branch)


async def find_due_reminders(session: AsyncSession, now: datetime) -> List[DueReminder]:
    """Return the reminders due at `now`, oldest due_at first, capped per user per call.

    Excludes: non open/snoozed statuses, quiet-mode users, users with no telegram_chat_id.
    """
    cap = settings.REMINDER_MAX_PER_USER_PER_TICK

    stmt = (
        select(Task, User)
        .join(User, Task.user_id == User.id)
        .where(
            User.quiet_mode.is_(False),
            User.telegram_chat_id.is_not(None),
            _eligibility(now),
        )
        .order_by(Task.due_at.asc(), Task.id.asc())
    )
    rows = (await session.execute(stmt)).all()

    counts: Dict[int, int] = {}
    out: List[DueReminder] = []
    for task, user in rows:
        seen = counts.get(user.id, 0)
        if seen >= cap:
            continue
        counts[user.id] = seen + 1
        out.append(
            DueReminder(
                task_id=task.id,
                title=task.title,
                due_at=task.due_at,
                user_id=user.id,
                telegram_chat_id=user.telegram_chat_id,
                timezone=user.timezone,
                was_snoozed=(task.status == "snoozed"),
            )
        )
    return out


async def claim_reminder(session: AsyncSession, task_id: int, now: datetime) -> bool:
    """Atomically claim a single reminder. Returns True iff this caller won the row.

    One UPDATE ... WHERE <eligibility> RETURNING: two concurrent callers cannot both win.
    On success sets last_reminded_at=now and normalises the row to open (snoozed_until=NULL),
    which also flips a snoozed task to open.
    """
    stmt = (
        update(Task)
        .where(
            Task.id == task_id,
            Task.user_id.in_(
                select(User.id).where(
                    User.quiet_mode.is_(False),
                    User.telegram_chat_id.is_not(None),
                )
            ),
            _eligibility(now),
        )
        .values(last_reminded_at=now, status="open", snoozed_until=None)
        .returning(Task.id)
    )
    result = await session.execute(stmt)
    claimed = result.first() is not None
    await session.commit()
    return claimed


async def release_reminder(
    session: AsyncSession,
    task_id: int,
    previous_state: ReminderSnapshot,
) -> None:
    """Restore a task to its pre-claim state (used when the Telegram send fails)."""
    await session.execute(
        update(Task)
        .where(Task.id == task_id)
        .values(
            status=previous_state.status,
            snoozed_until=previous_state.snoozed_until,
            last_reminded_at=previous_state.last_reminded_at,
        )
    )
    await session.commit()


def render_reminder_text(title: str, due_at: datetime, tz_name: str, now: datetime) -> str:
    """Deterministic plain-text reminder line (no markdown, no command hints).

    "Reminder: <title> (due <HH:MM> <TZ>, in about N min)."
    with "now due" at/under a minute and "overdue by about N min" once past.
    """
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    local_due = due_at.astimezone(tz)
    hhmm = local_due.strftime("%H:%M")
    tz_abbr = local_due.strftime("%Z") or tz_name

    minutes = int(round((due_at - now).total_seconds() / 60.0))
    if due_at <= now:
        when = "now due" if minutes == 0 else f"overdue by about {abs(minutes)} min"
    else:
        when = f"in about {minutes} min"

    return f"Reminder: {title} (due {hhmm} {tz_abbr}, {when})."


__all__ = [
    "DueReminder",
    "ReminderSnapshot",
    "find_due_reminders",
    "claim_reminder",
    "release_reminder",
    "render_reminder_text",
]
