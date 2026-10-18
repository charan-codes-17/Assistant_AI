"""backend/services/actions.py
Action executor service for AI Assistant per PRD §4.4, §10.1 and Roadmap §3.2.
Applies verified directives, memory state transitions (confirm, reject, edit, delete, supersede),
task updates (mark_done, snooze, reschedule, mark_reminded), and quiet mode toggles.

Guarantees:
- Every mutating function verifies the record/task belongs to the user and raises NotFoundError
  (or returns None for confirm/reject) to prevent forged Telegram button callback execution.
- delete_record and apply_directive require confirmed=True, otherwise raising PermissionError.
- reschedule_task requires a timezone-aware datetime.
- Each function commits its own database transaction.
- Re-exports confirm_task from extraction.py.
- confirm_task_candidate persists an extraction CandidateTask (ownership-scoped, idempotent).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import AsyncSessionLocal
from backend.models.models import KnowledgeRecord, Task, User
from backend.services.embeddings import embed_texts
from backend.services.extraction import CandidateTask, confirm_task

logger = logging.getLogger(__name__)


class NotFoundError(Exception):
    """Raised when an entity is not found or does not belong to the user."""
    pass


async def confirm_record(
    session: AsyncSession,
    user: User,
    record_id: int,
) -> Optional[KnowledgeRecord]:
    """Confirm a pending memory record.

    - If record doesn't belong to user or doesn't exist: returns None.
    - If status is rejected: returns None.
    - If already confirmed: returns unchanged (idempotent).
    - If pending: changes status to 'confirmed', commits, and returns record.
    """
    stmt = select(KnowledgeRecord).where(
        KnowledgeRecord.id == record_id,
        KnowledgeRecord.user_id == user.id,
    )
    rec = (await session.execute(stmt)).scalar_one_or_none()
    if rec is None:
        logger.warning("confirm_record: record %s not found for user %s", record_id, user.id)
        return None
    if rec.status == "rejected":
        logger.info("confirm_record: record %s is rejected, returning None", record_id)
        return None
    if rec.status == "confirmed":
        return rec

    rec.status = "confirmed"
    await session.commit()
    await session.refresh(rec)
    return rec


@dataclass
class BatchConfirmResult:
    """Outcome of confirm_records: the rows confirmed, and the ids that were ignored."""

    confirmed_records: List[KnowledgeRecord] = field(default_factory=list)
    ignored_ids: List[int] = field(default_factory=list)


async def confirm_records(
    user_id: int,
    ids: Sequence[int],
) -> BatchConfirmResult:
    """Confirm several pending memory records the user owns, in one transaction.

    - Ownership-scoped: ids that do not exist or are owned by another user are ignored and
      returned in BatchConfirmResult.ignored_ids (never confirmed, never mutated).
    - Rejected records are ignored and reported too.
    - Idempotent: already-confirmed owned records are returned unchanged (no duplicate writes).
    - Opens its own short-lived session/transaction, so it never holds a caller's session.
    """
    result = BatchConfirmResult()
    unique_ids: List[int] = list(dict.fromkeys(ids))
    if not unique_ids:
        return result

    async with AsyncSessionLocal() as session:
        stmt = select(KnowledgeRecord).where(
            KnowledgeRecord.id.in_(unique_ids),
            KnowledgeRecord.user_id == user_id,
        )
        rows = (await session.execute(stmt)).scalars().all()
        by_id = {rec.id: rec for rec in rows}

        for record_id in unique_ids:
            rec = by_id.get(record_id)
            if rec is None or rec.status == "rejected":
                result.ignored_ids.append(record_id)
                continue
            if rec.status != "confirmed":
                rec.status = "confirmed"
            result.confirmed_records.append(rec)

        await session.commit()
        for rec in result.confirmed_records:
            await session.refresh(rec)

    return result


async def reject_record(
    session: AsyncSession,
    user: User,
    record_id: int,
) -> Optional[KnowledgeRecord]:
    """Soft-reject a memory record (pending -> rejected).

    - If record doesn't belong to user or doesn't exist: returns None.
    - Commits and returns the record.
    """
    stmt = select(KnowledgeRecord).where(
        KnowledgeRecord.id == record_id,
        KnowledgeRecord.user_id == user.id,
    )
    rec = (await session.execute(stmt)).scalar_one_or_none()
    if rec is None:
        logger.warning("reject_record: record %s not found for user %s", record_id, user.id)
        return None

    rec.status = "rejected"
    await session.commit()
    await session.refresh(rec)
    return rec


async def edit_record(
    session: AsyncSession,
    user: User,
    record_id: int,
    new_content: str,
) -> KnowledgeRecord:
    """Update content and re-embed with embeddings.embed_texts. Status remains unchanged."""
    stmt = select(KnowledgeRecord).where(
        KnowledgeRecord.id == record_id,
        KnowledgeRecord.user_id == user.id,
    )
    rec = (await session.execute(stmt)).scalar_one_or_none()
    if rec is None:
        raise NotFoundError(f"KnowledgeRecord {record_id} not found for user {user.id}")

    rec.content = new_content
    vecs = await embed_texts([new_content])
    rec.embedding = vecs[0]
    await session.commit()
    await session.refresh(rec)
    return rec


async def delete_record(
    session: AsyncSession,
    user: User,
    record_id: int,
    *,
    confirmed: bool,
) -> None:
    """Hard-delete a memory record. Requires explicit confirmed=True."""
    if not confirmed:
        raise PermissionError("delete_record requires confirmed=True")

    stmt = select(KnowledgeRecord).where(
        KnowledgeRecord.id == record_id,
        KnowledgeRecord.user_id == user.id,
    )
    rec = (await session.execute(stmt)).scalar_one_or_none()
    if rec is None:
        raise NotFoundError(f"KnowledgeRecord {record_id} not found for user {user.id}")

    await session.delete(rec)
    await session.commit()


async def supersede_record(
    session: AsyncSession,
    user: User,
    old_id: int,
    new_id: int,
) -> Tuple[KnowledgeRecord, KnowledgeRecord]:
    """Supersede an existing record with a new one.

    - Confirms new record.
    - Sets old.superseded_by = new_id.
    - Both records must belong to user.
    """
    stmt_old = select(KnowledgeRecord).where(
        KnowledgeRecord.id == old_id,
        KnowledgeRecord.user_id == user.id,
    )
    old_rec = (await session.execute(stmt_old)).scalar_one_or_none()
    if old_rec is None:
        raise NotFoundError(f"Old KnowledgeRecord {old_id} not found for user {user.id}")

    stmt_new = select(KnowledgeRecord).where(
        KnowledgeRecord.id == new_id,
        KnowledgeRecord.user_id == user.id,
    )
    new_rec = (await session.execute(stmt_new)).scalar_one_or_none()
    if new_rec is None:
        raise NotFoundError(f"New KnowledgeRecord {new_id} not found for user {user.id}")

    new_rec.status = "confirmed"
    old_rec.superseded_by = new_id

    await session.commit()
    await session.refresh(old_rec)
    await session.refresh(new_rec)
    return old_rec, new_rec


def _due_at_equal(a: Optional[datetime], b: Optional[datetime]) -> bool:
    """Compare due datetimes as absolute instants (None equals None)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_cmp = a if a.tzinfo is not None else a.replace(tzinfo=timezone.utc)
    b_cmp = b if b.tzinfo is not None else b.replace(tzinfo=timezone.utc)
    return a_cmp.astimezone(timezone.utc) == b_cmp.astimezone(timezone.utc)


async def confirm_task_candidate(
    session: AsyncSession,
    user: User,
    candidate: CandidateTask,
) -> Task:
    """Persist a confirmed extraction task candidate for this user.

    Ownership-scoped (always inserts/returns a row with user.id), own transaction,
    idempotent on (user_id, title, due_at). Never returns or mutates another user's task.
    """
    stmt = select(Task).where(
        Task.user_id == user.id,
        Task.title == candidate.title,
    )
    existing_rows = (await session.execute(stmt)).scalars().all()
    for row in existing_rows:
        if _due_at_equal(row.due_at, candidate.due_at):
            return row

    task = Task(
        user_id=user.id,
        title=candidate.title,
        due_at=candidate.due_at,
        status="open",
        origin=candidate.origin,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def mark_task_done(
    session: AsyncSession,
    user: User,
    task_id: int,
) -> Task:
    """Mark a user task as done."""
    stmt = select(Task).where(
        Task.id == task_id,
        Task.user_id == user.id,
    )
    task = (await session.execute(stmt)).scalar_one_or_none()
    if task is None:
        raise NotFoundError(f"Task {task_id} not found for user {user.id}")

    task.status = "done"
    await session.commit()
    await session.refresh(task)
    return task


async def snooze_task(
    session: AsyncSession,
    user: User,
    task_id: int,
    minutes: int = 60,
    now: Optional[datetime] = None,
) -> Task:
    """Snooze a task for specified minutes.

    Sets status='snoozed', snoozed_until=now+minutes, last_reminded_at=None.
    """
    stmt = select(Task).where(
        Task.id == task_id,
        Task.user_id == user.id,
    )
    task = (await session.execute(stmt)).scalar_one_or_none()
    if task is None:
        raise NotFoundError(f"Task {task_id} not found for user {user.id}")

    ref_now = now or datetime.now(timezone.utc)
    task.status = "snoozed"
    task.snoozed_until = ref_now + timedelta(minutes=minutes)
    task.last_reminded_at = None
    await session.commit()
    await session.refresh(task)
    return task


async def mark_reminded(
    session: AsyncSession,
    task_id: int,
    now: Optional[datetime] = None,
) -> Task:
    """Update last_reminded_at on task (called by scheduler worker)."""
    stmt = select(Task).where(Task.id == task_id)
    task = (await session.execute(stmt)).scalar_one_or_none()
    if task is None:
        raise NotFoundError(f"Task {task_id} not found")

    task.last_reminded_at = now or datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(task)
    return task


async def reschedule_task(
    session: AsyncSession,
    user: User,
    task_id: int,
    new_due_at: datetime,
) -> Task:
    """Reschedule task to a new due_at.

    Requires timezone-aware datetime. Sets status='open', snoozed_until=None, last_reminded_at=None.
    """
    if new_due_at.tzinfo is None:
        raise ValueError("new_due_at must be a timezone-aware datetime")

    stmt = select(Task).where(
        Task.id == task_id,
        Task.user_id == user.id,
    )
    task = (await session.execute(stmt)).scalar_one_or_none()
    if task is None:
        raise NotFoundError(f"Task {task_id} not found for user {user.id}")

    task.due_at = new_due_at
    task.status = "open"
    task.snoozed_until = None
    task.last_reminded_at = None
    await session.commit()
    await session.refresh(task)
    return task


async def apply_directive(
    session: AsyncSession,
    user: User,
    directive: Any,
    *,
    confirmed: bool,
) -> Any:
    """Apply an action directive (mark_done, snooze, reschedule). Requires confirmed=True."""
    if not confirmed:
        raise PermissionError("apply_directive requires confirmed=True")

    d_type = getattr(directive, "type", None) or (directive.get("type") if isinstance(directive, dict) else None)
    d_task_id = getattr(directive, "task_id", None) or (directive.get("task_id") if isinstance(directive, dict) else None)
    d_new_due = getattr(directive, "new_due_at", None) or (directive.get("new_due_at") if isinstance(directive, dict) else None)

    if d_type == "mark_done":
        return await mark_task_done(session, user, d_task_id)
    elif d_type == "snooze":
        return await snooze_task(session, user, d_task_id)
    elif d_type == "reschedule":
        if isinstance(d_new_due, str):
            d_new_due = datetime.fromisoformat(d_new_due)
        return await reschedule_task(session, user, d_task_id, d_new_due)
    else:
        raise ValueError(f"Unsupported directive type: {d_type}")


async def set_quiet_mode(
    session: AsyncSession,
    user: User,
    value: bool,
) -> User:
    """Enable or disable quiet mode for the user."""
    user.quiet_mode = value
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


__all__ = [
    "NotFoundError",
    "BatchConfirmResult",
    "confirm_record",
    "confirm_records",
    "reject_record",
    "edit_record",
    "delete_record",
    "supersede_record",
    "mark_task_done",
    "snooze_task",
    "mark_reminded",
    "reschedule_task",
    "apply_directive",
    "set_quiet_mode",
    "confirm_task",
    "confirm_task_candidate",
]

