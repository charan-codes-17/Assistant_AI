"""backend/scripts/test_pending_tasks.py
Evaluation for Phase 4a-1: persistent task candidates (status='pending' Task rows).

Tests:
  P1  stage one candidate -> one row: status 'pending', origin 'promise', title and tz-aware
      due_at match; the row is returned.
  P2  stage the same candidate twice -> still exactly one row, same id returned both times.
  P3  pending is invisible to intervention gather_context, while a seeded open task is visible.
  P4  confirm_pending_task: pending -> 'open'; a second call leaves it 'open' and returns it.
  P5  isolation: user B confirming/discarding user A's pending task -> None / False; A unchanged.
  P6  discard: pending row deleted (True); discard on an 'open' row returns False and remains.
  P7  the bug fix: confirm_task_candidate with a matching pending row flips THAT row to 'open'
      and no second row exists.
  P8  restart survival: stage, engine.dispose(), confirm by id through a fresh session -> 'open'.
  P9  staging a candidate that matches an already-open task -> skipped, row count unchanged.
  P10 staging a candidate with due_at=None twice -> one row (None-safe equality).

NO LLM calls. Throwaway users; every row is deleted in a finally block.
"""

import asyncio
from datetime import datetime
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")
from typing import Any, Dict, List, Optional
import uuid
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select

from backend.config.settings import settings
from backend.database import AsyncSessionLocal, engine
from backend.models.models import Task, User
from backend.services.actions import (
    confirm_pending_task,
    confirm_task_candidate,
    discard_pending_task,
    stage_task_candidates,
)
from backend.services.extraction import CandidateTask
from backend.services.intervention import gather_context

IST_TZ = ZoneInfo("Asia/Kolkata")
DUE = datetime(2026, 10, 20, 9, 0, 0, tzinfo=IST_TZ)


def cand(title: str, due_at: Optional[datetime] = None, origin: str = "promise") -> CandidateTask:
    return CandidateTask(title=title, due_at=due_at, origin=origin)


async def new_user(label: str) -> int:
    async with AsyncSessionLocal() as session:
        user = User(
            username=f"ptask_{label}_{uuid.uuid4().hex[:8]}",
            timezone="Asia/Kolkata",
            quiet_mode=False,
            onboarding_state="testing",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.id


async def drop_user(user_id: Optional[int]) -> None:
    if user_id is None:
        return
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, user_id)
            if user is not None:
                await session.delete(user)
                await session.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"   [WARN] Failed to delete test user {user_id}: {exc}")


async def fetch_tasks(user_id: int) -> List[Task]:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(Task).where(Task.user_id == user_id).order_by(Task.id.asc())
            )
        ).scalars().all()
    return list(rows)


def row(rid: str, name: str, expected: str, actual: str, ok: bool, details: str = "") -> Dict[str, Any]:
    return {
        "id": rid,
        "name": name,
        "expected": expected,
        "actual": actual,
        "ok": bool(ok),
        "details": details,
    }


# ---------------------------------------------------------------------------
# P1
# ---------------------------------------------------------------------------
async def p1() -> Dict[str, Any]:
    uid = await new_user("p1")
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            staged = await stage_task_candidates(session, user, [cand("Send Arun the PDF", DUE)])
        rows = await fetch_tasks(uid)
        if len(rows) != 1:
            return row("P1", "stage one candidate", "1 row pending/promise, returned",
                       f"rows={len(rows)}, returned={len(staged)}", False,
                       f"ids={[r.id for r in rows]}")
        r = rows[0]
        ok = (
            r.status == "pending"
            and r.origin == "promise"
            and r.title == "Send Arun the PDF"
            and r.due_at == DUE
            and len(staged) == 1
            and staged[0].id == r.id
        )
        return row("P1", "stage one candidate", "status=pending, origin=promise, due matches, returned",
                   f"status={r.status}, origin={r.origin}, title={r.title!r}, due={r.due_at}, returned={len(staged)}",
                   ok, f"id={r.id}, staged_id={staged[0].id if staged else None}")
    finally:
        await drop_user(uid)


# ---------------------------------------------------------------------------
# P2
# ---------------------------------------------------------------------------
async def p2() -> Dict[str, Any]:
    uid = await new_user("p2")
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            first = await stage_task_candidates(session, user, [cand("Send Arun the PDF", DUE)])
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            second = await stage_task_candidates(session, user, [cand("Send Arun the PDF", DUE)])
        rows = await fetch_tasks(uid)
        ids = [x.id for x in first] + [x.id for x in second]
        ok = len(rows) == 1 and len(first) == 1 and len(second) == 1 and ids[0] == ids[1] == rows[0].id
        return row("P2", "stage same candidate twice", "1 row, same id both times",
                   f"rows={len(rows)}, first_ids={[x.id for x in first]}, second_ids={[x.id for x in second]}",
                   ok, f"ids={[r.id for r in rows]}")
    finally:
        await drop_user(uid)


# ---------------------------------------------------------------------------
# P3
# ---------------------------------------------------------------------------
async def p3() -> Dict[str, Any]:
    uid = await new_user("p3")
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            staged = await stage_task_candidates(session, user, [cand("Pending invisible", DUE)])
            pending_id = staged[0].id
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            open_task = Task(user_id=uid, title="Open visible", due_at=DUE, status="open", origin="manual")
            session.add(open_task)
            await session.commit()
            await session.refresh(open_task)
            open_id = open_task.id
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            ctx = await gather_context(session, user, now=DUE)
        ctx_ids = [t["id"] for t in ctx.tasks]
        ok = (pending_id not in ctx_ids) and (open_id in ctx_ids)
        return row("P3", "pending invisible to gather_context", "pending absent, open present",
                   f"context_ids={ctx_ids}", ok,
                   f"pending_id={pending_id}, open_id={open_id}")
    finally:
        await drop_user(uid)


# ---------------------------------------------------------------------------
# P4
# ---------------------------------------------------------------------------
async def p4() -> Dict[str, Any]:
    uid = await new_user("p4")
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            staged = await stage_task_candidates(session, user, [cand("Confirm me", DUE)])
            task_id = staged[0].id
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            first = await confirm_pending_task(session, user, task_id)
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            second = await confirm_pending_task(session, user, task_id)
        rows = await fetch_tasks(uid)
        ok = (
            first is not None and first.status == "open"
            and second is not None and second.status == "open"
            and second.id == task_id
            and len(rows) == 1 and rows[0].status == "open"
        )
        return row("P4", "confirm_pending_task", "pending->open, idempotent",
                   f"first={first.status if first else None}, second={second.status if second else None}, "
                   f"db_status={rows[0].status if rows else None}",
                   ok, f"task_id={task_id}, returned_id={second.id if second else None}")
    finally:
        await drop_user(uid)


# ---------------------------------------------------------------------------
# P5
# ---------------------------------------------------------------------------
async def p5() -> Dict[str, Any]:
    uid_a = await new_user("p5a")
    uid_b = await new_user("p5b")
    try:
        async with AsyncSessionLocal() as session:
            user_a = await session.get(User, uid_a)
            staged = await stage_task_candidates(session, user_a, [cand("A private", DUE)])
            task_id = staged[0].id
        async with AsyncSessionLocal() as session:
            user_b = await session.get(User, uid_b)
            b_confirm = await confirm_pending_task(session, user_b, task_id)
        async with AsyncSessionLocal() as session:
            user_b = await session.get(User, uid_b)
            b_discard = await discard_pending_task(session, user_b, task_id)
        rows_a = await fetch_tasks(uid_a)
        ok = (b_confirm is None) and (b_discard is False) and len(rows_a) == 1 and rows_a[0].status == "pending"
        return row("P5", "ownership isolation", "B: None / False, A still pending",
                   f"b_confirm={b_confirm}, b_discard={b_discard}, a_status={rows_a[0].status if rows_a else None}",
                   ok, f"a_task_id={task_id}, a_rows={len(rows_a)}")
    finally:
        await drop_user(uid_a)
        await drop_user(uid_b)


# ---------------------------------------------------------------------------
# P6
# ---------------------------------------------------------------------------
async def p6() -> Dict[str, Any]:
    uid = await new_user("p6")
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            staged = await stage_task_candidates(session, user, [cand("Discard pending", DUE)])
            pending_id = staged[0].id
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            ok_discard = await discard_pending_task(session, user, pending_id)
        remaining_after_discard = await fetch_tasks(uid)
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            open_task = Task(user_id=uid, title="Discard open", due_at=DUE, status="open", origin="manual")
            session.add(open_task)
            await session.commit()
            await session.refresh(open_task)
            open_id = open_task.id
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            bad_discard = await discard_pending_task(session, user, open_id)
        remaining_after_open = await fetch_tasks(uid)
        open_still = any(r.id == open_id and r.status == "open" for r in remaining_after_open)
        ok = (
            ok_discard is True
            and len(remaining_after_discard) == 0
            and bad_discard is False
            and open_still
        )
        return row("P6", "discard pending vs open", "pending deleted True; open False, remains",
                   f"pending_discard={ok_discard}, after_pending={len(remaining_after_discard)}, "
                   f"open_discard={bad_discard}, open_still={open_still}",
                   ok, f"pending_id={pending_id}, open_id={open_id}")
    finally:
        await drop_user(uid)


# ---------------------------------------------------------------------------
# P7
# ---------------------------------------------------------------------------
async def p7() -> Dict[str, Any]:
    uid = await new_user("p7")
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            staged = await stage_task_candidates(session, user, [cand("Promote me", DUE)])
            pending_id = staged[0].id
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            confirmed = await confirm_task_candidate(session, user, cand("Promote me", DUE))
        rows = await fetch_tasks(uid)
        ok = (
            confirmed is not None
            and confirmed.id == pending_id
            and confirmed.status == "open"
            and len(rows) == 1
            and rows[0].status == "open"
        )
        return row("P7", "confirm_task_candidate promotes pending", "same row -> open, no 2nd row",
                   f"returned_id={confirmed.id if confirmed else None}, returned_status={confirmed.status if confirmed else None}, "
                   f"rows={len(rows)}",
                   ok, f"pending_id={pending_id}")
    finally:
        await drop_user(uid)


# ---------------------------------------------------------------------------
# P8
# ---------------------------------------------------------------------------
async def p8() -> Dict[str, Any]:
    uid = await new_user("p8")
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            staged = await stage_task_candidates(session, user, [cand("Survive restart", DUE)])
            task_id = staged[0].id
        # New connections only: dispose the pool, no shared Python objects carried over.
        await engine.dispose()
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            confirmed = await confirm_pending_task(session, user, task_id)
        rows = await fetch_tasks(uid)
        ok = (
            confirmed is not None
            and confirmed.id == task_id
            and confirmed.status == "open"
            and len(rows) == 1
            and rows[0].status == "open"
        )
        return row("P8", "restart survival (engine.dispose)", "still pending after reconnect -> open",
                   f"confirmed={confirmed.status if confirmed else None}, rows={len(rows)}, "
                   f"db_status={rows[0].status if rows else None}",
                   ok, f"task_id={task_id}")
    finally:
        await drop_user(uid)


# ---------------------------------------------------------------------------
# P9
# ---------------------------------------------------------------------------
async def p9() -> Dict[str, Any]:
    uid = await new_user("p9")
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            open_task = Task(user_id=uid, title="Already open", due_at=DUE, status="open", origin="manual")
            session.add(open_task)
            await session.commit()
            await session.refresh(open_task)
            open_id = open_task.id
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            staged = await stage_task_candidates(session, user, [cand("Already open", DUE)])
        rows = await fetch_tasks(uid)
        ok = len(staged) == 0 and len(rows) == 1 and rows[0].id == open_id and rows[0].status == "open"
        return row("P9", "stage against existing open", "skipped, row count unchanged",
                   f"returned={len(staged)}, rows={len(rows)}, status={rows[0].status if rows else None}",
                   ok, f"open_id={open_id}")
    finally:
        await drop_user(uid)


# ---------------------------------------------------------------------------
# P10
# ---------------------------------------------------------------------------
async def p10() -> Dict[str, Any]:
    uid = await new_user("p10")
    try:
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            first = await stage_task_candidates(session, user, [cand("No due date", None)])
        async with AsyncSessionLocal() as session:
            user = await session.get(User, uid)
            second = await stage_task_candidates(session, user, [cand("No due date", None)])
        rows = await fetch_tasks(uid)
        ids = [x.id for x in first] + [x.id for x in second]
        ok = len(rows) == 1 and len(ids) == 2 and ids[0] == ids[1] == rows[0].id and rows[0].due_at is None
        return row("P10", "stage due_at=None twice", "1 row (None-safe equality)",
                   f"rows={len(rows)}, ids={ids}, due={rows[0].due_at if rows else 'n/a'}",
                   ok, "")
    finally:
        await drop_user(uid)


async def main() -> None:
    settings.require("DATABASE_URL")

    print("=" * 130)
    print("PHASE 4a-1: PERSISTENT PENDING TASK CANDIDATES EVALUATION")
    print("=" * 130)

    cases = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
    results: List[Dict[str, Any]] = []
    try:
        for case in cases:
            print(f"--- Running {case.__name__.upper()} ---")
            try:
                res = await case()
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                res = row(case.__name__.upper(), case.__name__, "-", "", False,
                          f"unhandled exception: {exc!r}")
            results.append(res)
            print(f"    [{'PASS' if res['ok'] else 'FAIL'}] {res['name']}: {res['actual']}")
    finally:
        await engine.dispose()

    print("\n" + "=" * 130)
    print(f"{'ID':<5} | {'NAME':<40} | {'EXPECTED':<46} | {'ACTUAL':<46} | {'STATUS':<6} | {'DETAILS'}")
    print("-" * 130)
    all_passed = True
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            all_passed = False
        print(f"{r['id']:<5} | {r['name']:<40} | {r['expected']:<46} | {r['actual']:<46} | {status:<6} | {r['details']}")
    print("=" * 130)
    print(f"RESULT: {'ALL PASS' if all_passed else 'FAILURES PRESENT'} ({sum(1 for r in results if r['ok'])}/{len(results)})")


if __name__ == "__main__":
    asyncio.run(main())
