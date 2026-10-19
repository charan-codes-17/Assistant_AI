"""backend/scripts/test_reminders.py
Deterministic reminder logic evaluation (Roadmap Phase 5.1; PRD FR11/FR20).

Covers selection, atomic claiming, release/restore, rendering, and mark_reminded ownership.
Uses the real dev DB with synthetic users whose telegram_chat_id is in 960000-969999; those
users are purged at the START and again in a finally block. Fixed `now` values only (no sleeps).

Run:
  backend\\venv\\Scripts\\python.exe -u backend\\scripts\\test_reminders.py
"""

import asyncio
from datetime import datetime, timedelta, timezone
import inspect
import sys
from typing import Any, Dict, List, Optional
import unittest.mock
import uuid
from zoneinfo import ZoneInfo

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import backend.services.embeddings as embeddings_module
import backend.services.llm as llm_module
from backend.config.settings import settings
from backend.models.models import Task, User
from backend.services.actions import (
    NotFoundError,
    mark_reminded,
)
from backend.services.reminders import (
    DueReminder,
    ReminderSnapshot,
    claim_reminder,
    find_due_reminders,
    release_reminder,
    render_reminder_text,
)

CHAT_MIN = 960000
CHAT_MAX = 969999
IST = ZoneInfo("Asia/Kolkata")
FIXED_NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
async def make_user(
    session: AsyncSession,
    chat_id: Optional[int],
    *,
    quiet: bool = False,
    tz: str = "Asia/Kolkata",
    label: str = "u",
) -> User:
    user = User(
        username=f"rem_{label}_{uuid.uuid4().hex[:8]}",
        telegram_chat_id=chat_id,
        timezone=tz,
        quiet_mode=quiet,
        onboarding_state="testing",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def make_task(
    session: AsyncSession,
    user: User,
    title: str,
    *,
    status: str = "open",
    due_at: Optional[datetime] = None,
    snoozed_until: Optional[datetime] = None,
    last_reminded_at: Optional[datetime] = None,
) -> Task:
    task = Task(
        user_id=user.id,
        title=title,
        status=status,
        due_at=due_at,
        snoozed_until=snoozed_until,
        last_reminded_at=last_reminded_at,
        origin="manual",
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def purge_synthetic(session: AsyncSession) -> int:
    """Delete all synthetic reminder users (chat range or rem_ username prefix). Cascade deletes tasks."""
    rows = (
        await session.execute(
            select(User).where(
                (User.telegram_chat_id >= CHAT_MIN) & (User.telegram_chat_id <= CHAT_MAX)
            )
        )
    ).scalars().all()
    rows += (
        await session.execute(select(User).where(User.username.like("rem_%")))
    ).scalars().all()
    for user in rows:
        await session.delete(user)
    await session.commit()
    return len(rows)


def mine(results: List[DueReminder], *users: User) -> List[DueReminder]:
    """Isolate results to the given synthetic users (the dev DB may hold unrelated rows)."""
    ids = {u.id for u in users}
    return [r for r in results if r.user_id in ids]


def add(results: List[Dict[str, Any]], rid: str, name: str, ok: bool, expected: str, actual: str, detail: str = "") -> None:
    results.append(
        {
            "id": rid,
            "name": name,
            "status": "PASS" if ok else "FAIL",
            "expected": expected,
            "actual": actual,
            "detail": detail,
        }
    )


async def main() -> None:
    settings.require("DATABASE_URL")

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    results: List[Dict[str, Any]] = []
    now = FIXED_NOW

    print("=" * 110)
    print("DETERMINISTIC REMINDER LOGIC EVALUATION (Phase 5.1)")
    print(f"Reference now: {now.isoformat()}  |  LEAD={settings.REMINDER_LEAD_MINUTES}m "
          f"MAX_OVERDUE={settings.REMINDER_MAX_OVERDUE_HOURS}h CAP={settings.REMINDER_MAX_PER_USER_PER_TICK}")
    print("=" * 110)

    async with async_session() as session:
        purged = await purge_synthetic(session)
        print(f"PURGE at start: removed {purged} synthetic user(s).\n")

        try:
            # ------------------------------------------------------------------
            # R1: open task due in 30 min, never reminded -> returned
            # ------------------------------------------------------------------
            u1 = await make_user(session, 960001, label="r1")
            t1 = await make_task(session, u1, "R1 task", due_at=now + timedelta(minutes=30))
            res1 = mine(await find_due_reminders(session, now), u1)
            ok1 = any(r.task_id == t1.id for r in res1) and len(res1) == 1
            add(results, "R1", "open task due in 30 min -> returned",
                ok1, "task returned", f"returned={[r.task_id for r in res1]}")

            # ------------------------------------------------------------------
            # R2: open task due in 3h -> not returned; at now+2h10m -> returned
            # ------------------------------------------------------------------
            u2 = await make_user(session, 960002, label="r2")
            t2 = await make_task(session, u2, "R2 task", due_at=now + timedelta(hours=3))
            res2a = mine(await find_due_reminders(session, now), u2)
            res2b = mine(await find_due_reminders(session, now + timedelta(minutes=130)), u2)
            ok2 = (not any(r.task_id == t2.id for r in res2a)) and any(r.task_id == t2.id for r in res2b)
            add(results, "R2", "due in 3h not due; +2h10m later due",
                ok2, "absent now, present at +130m",
                f"now={[r.task_id for r in res2a]}, later={[r.task_id for r in res2b]}")

            # ------------------------------------------------------------------
            # R3: status pending / done -> never returned
            # ------------------------------------------------------------------
            u3 = await make_user(session, 960003, label="r3")
            t3p = await make_task(session, u3, "R3 pending", status="pending", due_at=now + timedelta(minutes=5))
            t3d = await make_task(session, u3, "R3 done", status="done", due_at=now + timedelta(minutes=5))
            res3 = mine(await find_due_reminders(session, now), u3)
            ok3 = (not any(r.task_id == t3p.id for r in res3)) and (not any(r.task_id == t3d.id for r in res3))
            add(results, "R3", "pending/done statuses never returned",
                ok3, "neither returned", f"returned={[r.task_id for r in res3]}")

            # ------------------------------------------------------------------
            # R4: open task with last_reminded_at set -> not returned
            # ------------------------------------------------------------------
            u4 = await make_user(session, 960004, label="r4")
            t4 = await make_task(session, u4, "R4 already reminded",
                                 due_at=now + timedelta(minutes=30),
                                 last_reminded_at=now - timedelta(minutes=1))
            res4 = mine(await find_due_reminders(session, now), u4)
            ok4 = not any(r.task_id == t4.id for r in res4)
            add(results, "R4", "already-reminded open task -> not returned",
                ok4, "not returned", f"returned={[r.task_id for r in res4]}")

            # ------------------------------------------------------------------
            # R5: snoozed future -> not returned; snoozed past -> returned
            # ------------------------------------------------------------------
            u5 = await make_user(session, 960005, label="r5")
            t5 = await make_task(session, u5, "R5 snoozed",
                                 status="snoozed",
                                 due_at=now + timedelta(minutes=30),
                                 snoozed_until=now + timedelta(minutes=10))
            res5a = mine(await find_due_reminders(session, now), u5)
            t5.snoozed_until = now - timedelta(minutes=5)
            session.add(t5)
            await session.commit()
            res5b = mine(await find_due_reminders(session, now), u5)
            ok5 = (not any(r.task_id == t5.id for r in res5a)) and any(
                r.task_id == t5.id and r.was_snoozed for r in res5b
            )
            add(results, "R5", "snooze gate: future hidden, past returned (was_snoozed)",
                ok5, "hidden then returned",
                f"future={[r.task_id for r in res5a]}, past={[r.task_id for r in res5b]}")

            # ------------------------------------------------------------------
            # R6: quiet_mode user -> not returned AND row unchanged
            # ------------------------------------------------------------------
            u6 = await make_user(session, 960006, quiet=True, label="r6")
            t6 = await make_task(session, u6, "R6 quiet", due_at=now + timedelta(minutes=30))
            res6 = mine(await find_due_reminders(session, now), u6)
            await session.refresh(t6)
            ok6 = (not any(r.task_id == t6.id for r in res6)) and t6.last_reminded_at is None
            add(results, "R6", "quiet_mode user excluded, row untouched",
                ok6, "not returned, last_reminded_at NULL",
                f"returned={[r.task_id for r in res6]}, last_reminded_at={t6.last_reminded_at}")

            # ------------------------------------------------------------------
            # R7: user without telegram_chat_id -> not returned
            # ------------------------------------------------------------------
            u7 = await make_user(session, None, label="r7")
            t7 = await make_task(session, u7, "R7 no chat id", due_at=now + timedelta(minutes=30))
            res7 = mine(await find_due_reminders(session, now), u7)
            ok7 = not any(r.task_id == t7.id for r in res7)
            add(results, "R7", "user without telegram_chat_id excluded",
                ok7, "not returned", f"returned={[r.task_id for r in res7]}")

            # ------------------------------------------------------------------
            # R8: overdue > MAX_OVERDUE -> not returned; overdue by less -> returned
            # ------------------------------------------------------------------
            u8 = await make_user(session, 960008, label="r8")
            t8_far = await make_task(session, u8, "R8 far overdue", due_at=now - timedelta(hours=25))
            t8_near = await make_task(session, u8, "R8 near overdue", due_at=now - timedelta(hours=1))
            res8 = mine(await find_due_reminders(session, now), u8)
            ok8 = (not any(r.task_id == t8_far.id for r in res8)) and any(r.task_id == t8_near.id for r in res8)
            add(results, "R8", "MAX_OVERDUE window enforced",
                ok8, "25h old absent, 1h old present",
                f"returned={[r.task_id for r in res8]} (far={t8_far.id}, near={t8_near.id})")

            # ------------------------------------------------------------------
            # R9: five due tasks for one user, cap 2 -> two oldest due
            # ------------------------------------------------------------------
            u9 = await make_user(session, 960009, label="r9")
            offsets = [-50, -40, -30, -20, -10]
            t9 = {}
            for off in offsets:
                t9[off] = await make_task(session, u9, f"R9 -{abs(off)}m", due_at=now + timedelta(minutes=off))
            res9 = mine(await find_due_reminders(session, now), u9)
            expected9 = {t9[-50].id, t9[-40].id}
            ok9 = len(res9) == 2 and {r.task_id for r in res9} == expected9
            add(results, "R9", "per-user cap selects two oldest due",
                ok9, f"ids={sorted(expected9)}", f"returned={[r.task_id for r in res9]}")

            # ------------------------------------------------------------------
            # R10: concurrent claims -> exactly one wins; snoozed claim opens task
            # ------------------------------------------------------------------
            u10 = await make_user(session, 960010, label="r10")
            t10o = await make_task(session, u10, "R10 open", due_at=now + timedelta(minutes=30))
            t10s = await make_task(session, u10, "R10 snoozed", status="snoozed",
                                   due_at=now + timedelta(minutes=30),
                                   snoozed_until=now - timedelta(minutes=5))

            async with async_session() as s1, async_session() as s2:
                r10o = await asyncio.gather(
                    claim_reminder(s1, t10o.id, now),
                    claim_reminder(s2, t10o.id, now),
                )
            async with async_session() as s3, async_session() as s4:
                r10s = await asyncio.gather(
                    claim_reminder(s3, t10s.id, now),
                    claim_reminder(s4, t10s.id, now),
                )
            await session.refresh(t10o)
            await session.refresh(t10s)
            ok10 = (
                sum(1 for x in r10o if x) == 1
                and t10o.last_reminded_at == now
                and sum(1 for x in r10s if x) == 1
                and t10s.status == "open"
                and t10s.snoozed_until is None
                and t10s.last_reminded_at == now
            )
            add(results, "R10", "atomic claim: one winner; snoozed -> open/NULL",
                ok10, "1 True per pair; snoozed normalized",
                f"open={r10o}, snoozed={r10s}, s.status={t10s.status}, "
                f"s.snoozed_until={t10s.snoozed_until}")

            # ------------------------------------------------------------------
            # R11: release_reminder restores open and snoozed to exact prior values
            # ------------------------------------------------------------------
            u11 = await make_user(session, 960011, label="r11")
            t11o = await make_task(session, u11, "R11 open", due_at=now + timedelta(minutes=30))
            snap_o = ReminderSnapshot(status=t11o.status, snoozed_until=t11o.snoozed_until,
                                      last_reminded_at=t11o.last_reminded_at)
            claimed_o = await claim_reminder(session, t11o.id, now)
            await release_reminder(session, t11o.id, snap_o)
            await session.refresh(t11o)

            t11s = await make_task(session, u11, "R11 snoozed", status="snoozed",
                                   due_at=now + timedelta(minutes=30),
                                   snoozed_until=now - timedelta(minutes=5))
            snap_s = ReminderSnapshot(status=t11s.status, snoozed_until=t11s.snoozed_until,
                                      last_reminded_at=t11s.last_reminded_at)
            claimed_s = await claim_reminder(session, t11s.id, now)
            await release_reminder(session, t11s.id, snap_s)
            await session.refresh(t11s)

            ok11 = (
                claimed_o and claimed_s
                and t11o.status == "open" and t11o.snoozed_until is None and t11o.last_reminded_at is None
                and t11s.status == "snoozed"
                and t11s.snoozed_until == snap_s.snoozed_until
                and t11s.last_reminded_at is None
            )
            add(results, "R11", "release_reminder restores exact prior state",
                ok11, "open: open/NULL/NULL; snoozed restored",
                f"open=({t11o.status},{t11o.snoozed_until},{t11o.last_reminded_at}); "
                f"snoozed=({t11s.status},{t11s.snoozed_until},{t11s.last_reminded_at})")

            # ------------------------------------------------------------------
            # R12: render_reminder_text exact strings (Asia/Kolkata)
            # ------------------------------------------------------------------
            now_ist = datetime(2026, 10, 18, 12, 0, 0, tzinfo=IST)
            ttl = "Submit assignment"
            s_a = render_reminder_text(ttl, now_ist + timedelta(minutes=30), "Asia/Kolkata", now_ist)
            exp_a = "Reminder: Submit assignment (due 12:30 IST, in about 30 min)."
            s_b = render_reminder_text(ttl, now_ist, "Asia/Kolkata", now_ist)
            exp_b = "Reminder: Submit assignment (due 12:00 IST, now due)."
            s_c = render_reminder_text(ttl, now_ist - timedelta(minutes=10), "Asia/Kolkata", now_ist)
            exp_c = "Reminder: Submit assignment (due 11:50 IST, overdue by about 10 min)."
            ok12 = (s_a == exp_a) and (s_b == exp_b) and (s_c == exp_c)
            add(results, "R12", "render_reminder_text exact templates",
                ok12, "3 exact strings match",
                f"a={s_a!r} | b={s_b!r} | c={s_c!r}")

            # ------------------------------------------------------------------
            # R13: NO LLM / embedding calls
            # ------------------------------------------------------------------
            patchers = []
            for module in (llm_module, embeddings_module):
                for name in dir(module):
                    if name.startswith("_"):
                        continue
                    attr = getattr(module, name)
                    if inspect.iscoroutinefunction(attr):
                        async def _blocked(*a: Any, _n: str = name, **k: Any) -> Any:
                            raise AssertionError(f"LLM/embedding call must not happen: {_n}")
                        patchers.append(unittest.mock.patch.object(module, name, _blocked))

            u13 = await make_user(session, 960013, label="r13")
            t13 = await make_task(session, u13, "R13 task", due_at=now + timedelta(minutes=30))
            r13_error = ""
            patched_names = []
            try:
                for p in patchers:
                    p.start()
                    patched_names.append(p.attribute)
                await find_due_reminders(session, now)
                await claim_reminder(session, t13.id, now)
                render_reminder_text("R13", now + timedelta(minutes=30), "Asia/Kolkata", now)
            except Exception as exc:  # noqa: BLE001
                r13_error = f"{type(exc).__name__}: {exc}"
            finally:
                for p in patchers:
                    p.stop()
            ok13 = r13_error == ""
            add(results, "R13", "no LLM/embedding calls (all public coroutines blocked)",
                ok13, "no exception", f"error={r13_error or 'none'}, blocked={patched_names}")

            # ------------------------------------------------------------------
            # R14: mark_reminded ownership + status gate
            # ------------------------------------------------------------------
            u14a = await make_user(session, 960014, label="r14a")
            u14b = await make_user(session, 960015, label="r14b")
            t14 = await make_task(session, u14a, "R14 owned open", due_at=now + timedelta(minutes=30))
            t14p = await make_task(session, u14a, "R14 owned pending", status="pending",
                                   due_at=now + timedelta(minutes=30))

            other_raised = False
            try:
                await mark_reminded(session, t14.id, now=now, user=u14b)
            except NotFoundError:
                other_raised = True
            await session.refresh(t14)
            other_noop = t14.last_reminded_at is None

            pending_raised = False
            try:
                await mark_reminded(session, t14p.id, now=now, user=u14a)
            except NotFoundError:
                pending_raised = True
            await session.refresh(t14p)
            pending_noop = t14p.last_reminded_at is None

            t14ok = await make_task(session, u14a, "R14 owner marks open", due_at=now + timedelta(minutes=30))
            marked = await mark_reminded(session, t14ok.id, now=now, user=u14a)

            ok14 = other_raised and other_noop and pending_raised and pending_noop and marked.last_reminded_at == now
            add(results, "R14", "mark_reminded ownership + status gate",
                ok14, "other user/pending -> NotFoundError & no change; owner ok",
                f"other_raised={other_raised}, other_untouched={other_noop}, "
                f"pending_raised={pending_raised}, pending_untouched={pending_noop}, "
                f"owner_marked={marked.last_reminded_at == now}")

        finally:
            print("Cleaning up synthetic reminder users...")
            cleaned = await purge_synthetic(session)
            print(f"Cleanup removed {cleaned} synthetic user(s).")

    await engine.dispose()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print(f"{'ID':<5} | {'CASE':<52} | {'STATUS':<6} | {'EXPECTED':<34} | {'ACTUAL'}")
    print("-" * 110)
    all_passed = True
    for r in results:
        if r["status"] != "PASS":
            all_passed = False
        print(f"{r['id']:<5} | {r['name']:<52} | {r['status']:<6} | {r['expected']:<34} | {r['actual']}")
    print("=" * 110)

    if all_passed:
        print("\nALL REMINDER TESTS PASSED.")
    else:
        print("\nSOME REMINDER TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
