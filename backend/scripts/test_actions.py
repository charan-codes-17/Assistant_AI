"""backend/scripts/test_actions.py
Evaluation script for Action Executor (Phase 3 & 5 / Roadmap §3.2).
Tests:
  1. confirm makes a record appear in search_confirmed
  2. double confirm is idempotent
  3. reject excludes it from search_confirmed, and cannot be re-confirmed
  4. user B cannot confirm/delete/snooze user A's record/task (nothing changes)
  5. delete without confirmed raises PermissionError; delete with confirmed hard-deletes
  6. edit changes content AND the embedding (still length 1024), status unchanged
  7. apply_directive without confirmed raises PermissionError; applies mark_done when confirmed
  8. snooze then advance the clock past snoozed_until -> scheduler path says remind again
  9. reschedule clears last_reminded_at and resets status to open
  10. supersede removes the old record from search_confirmed and keeps the new one

NO LLM calls are made in these tests except where re-embedding.
All throwaway users and rows cleaned up in finally block.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
from typing import Any, Dict, List, Optional
import unittest.mock
import uuid
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config.settings import settings
from backend.models.models import KnowledgeRecord, Task, User
from backend.services.actions import (
    NotFoundError,
    apply_directive,
    confirm_record,
    confirm_task,
    delete_record,
    edit_record,
    mark_reminded,
    mark_task_done,
    reject_record,
    reschedule_task,
    set_quiet_mode,
    snooze_task,
    supersede_record,
)
from backend.services.embeddings import embed_texts
import backend.services.intervention as intervention_module
from backend.services.intervention import ActionDirective, decide_intervention
from backend.services.memory_search import search_confirmed

FIXED_NOW = datetime(2026, 10, 17, 10, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
IST_TZ = ZoneInfo("Asia/Kolkata")


async def create_test_user(session: AsyncSession, label: str) -> User:
    username = f"act_{label}_{uuid.uuid4().hex[:6]}"
    user = User(
        username=username,
        timezone="Asia/Kolkata",
        quiet_mode=False,
        onboarding_state="testing",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def cleanup_user(session: AsyncSession, user_id: Optional[int]) -> None:
    if user_id is None:
        return
    try:
        user = await session.get(User, user_id)
        if user:
            await session.delete(user)
            await session.commit()
    except Exception as exc:
        print(f"   [WARN] Failed to delete test user {user_id}: {exc}")


async def main() -> None:
    settings.require(
        "DATABASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
    )

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    users_to_cleanup: List[int] = []
    results: List[Dict[str, Any]] = []

    print("=" * 110)
    print("ACTION EXECUTOR & MUTATION VERIFICATION EVALUATION")
    print(f"Reference Clock: {FIXED_NOW.strftime('%Y-%m-%d %H:%M:%S %Z (%A)')}")
    print("=" * 110)

    async with async_session() as session:
        user_a: Optional[User] = None
        user_b: Optional[User] = None
        try:
            user_a = await create_test_user(session, "userA")
            user_b = await create_test_user(session, "userB")
            users_to_cleanup.extend([user_a.id, user_b.id])

            print(f"Seeded User A ID {user_a.id}, User B ID {user_b.id}")

            # -----------------------------------------------------------------
            # Test 1: Confirm makes a record appear in search_confirmed
            # -----------------------------------------------------------------
            text_1 = "User is learning algorithms"
            vecs_1 = await embed_texts([text_1])
            rec_1 = KnowledgeRecord(
                user_id=user_a.id,
                kind="knowledge",
                content=text_1,
                status="pending",
                embedding=vecs_1[0],
            )
            session.add(rec_1)
            await session.commit()
            await session.refresh(rec_1)

            # Before confirm: search_confirmed returns 0 matches
            before_matches = await search_confirmed(session, user_a.id, vecs_1[0], k=5)
            has_before = any(r.id == rec_1.id for r, _ in before_matches)

            # Confirm
            confirmed_rec = await confirm_record(session, user_a, rec_1.id)
            after_matches = await search_confirmed(session, user_a.id, vecs_1[0], k=5)
            has_after = any(r.id == rec_1.id for r, _ in after_matches)

            t1_pass = not has_before and has_after and confirmed_rec.status == "confirmed"
            results.append({
                "id": "Test-1",
                "name": "confirm_record -> search_confirmed visible",
                "expected": "before=False, after=True, status=confirmed",
                "actual": f"before={has_before}, after={has_after}, status={confirmed_rec.status if confirmed_rec else None}",
                "status": "PASS" if t1_pass else "FAIL",
                "detail": f"Rec ID {rec_1.id} transitioned pending -> confirmed and indexed",
            })

            # -----------------------------------------------------------------
            # Test 2: Double confirm is idempotent
            # -----------------------------------------------------------------
            double_rec = await confirm_record(session, user_a, rec_1.id)
            all_recs_user_a = (
                await session.execute(
                    select(KnowledgeRecord).where(KnowledgeRecord.user_id == user_a.id, KnowledgeRecord.id == rec_1.id)
                )
            ).scalars().all()
            t2_pass = (
                double_rec is not None
                and double_rec.status == "confirmed"
                and len(all_recs_user_a) == 1
            )
            results.append({
                "id": "Test-2",
                "name": "Double confirm idempotency",
                "expected": "record unchanged, 1 row exists",
                "actual": f"status={double_rec.status if double_rec else None}, count={len(all_recs_user_a)}",
                "status": "PASS" if t2_pass else "FAIL",
                "detail": "Second confirm returned existing record without duplicate writes",
            })

            # -----------------------------------------------------------------
            # Test 3: Reject excludes record and cannot be re-confirmed
            # -----------------------------------------------------------------
            text_3 = "User dislikes loud music"
            vecs_3 = await embed_texts([text_3])
            rec_3 = KnowledgeRecord(
                user_id=user_a.id,
                kind="preference",
                content=text_3,
                status="pending",
                embedding=vecs_3[0],
            )
            session.add(rec_3)
            await session.commit()
            await session.refresh(rec_3)

            rej_rec = await reject_record(session, user_a, rec_3.id)
            matches_after_rej = await search_confirmed(session, user_a.id, vecs_3[0], k=5)
            has_rej_match = any(r.id == rec_3.id for r, _ in matches_after_rej)

            # Attempt to confirm rejected record
            conf_rej = await confirm_record(session, user_a, rec_3.id)

            t3_pass = (
                rej_rec is not None
                and rej_rec.status == "rejected"
                and not has_rej_match
                and conf_rej is None
            )
            results.append({
                "id": "Test-3",
                "name": "reject_record excludes and prevents confirm",
                "expected": "status=rejected, visible=False, confirm_rejected=None",
                "actual": f"status={rej_rec.status if rej_rec else None}, visible={has_rej_match}, conf_rej={conf_rej}",
                "status": "PASS" if t3_pass else "FAIL",
                "detail": "Soft reject properly excluded from search and cannot be confirmed",
            })

            # -----------------------------------------------------------------
            # Test 4: User B cannot confirm/delete/snooze User A's record/task
            # -----------------------------------------------------------------
            text_4 = "User A secret note"
            vecs_4 = await embed_texts([text_4])
            rec_a_owned = KnowledgeRecord(
                user_id=user_a.id,
                kind="knowledge",
                content=text_4,
                status="pending",
                embedding=vecs_4[0],
            )
            task_a_owned = Task(
                user_id=user_a.id,
                title="User A task",
                status="open",
                due_at=FIXED_NOW + timedelta(hours=3),
            )
            session.add_all([rec_a_owned, task_a_owned])
            await session.commit()
            await session.refresh(rec_a_owned)
            await session.refresh(task_a_owned)

            # User B attempts to confirm User A's record -> returns None
            cross_conf = await confirm_record(session, user_b, rec_a_owned.id)

            # User B attempts to delete User A's record -> raises NotFoundError
            cross_del_raised = False
            try:
                await delete_record(session, user_b, rec_a_owned.id, confirmed=True)
            except NotFoundError:
                cross_del_raised = True

            # User B attempts to snooze User A's task -> raises NotFoundError
            cross_snooze_raised = False
            try:
                await snooze_task(session, user_b, task_a_owned.id, minutes=30)
            except NotFoundError:
                cross_snooze_raised = True

            # Verify User A's rows are unchanged
            await session.refresh(rec_a_owned)
            await session.refresh(task_a_owned)
            rec_a_intact = rec_a_owned.status == "pending"
            task_a_intact = task_a_owned.status == "open" and task_a_owned.snoozed_until is None

            t4_pass = (
                cross_conf is None
                and cross_del_raised
                and cross_snooze_raised
                and rec_a_intact
                and task_a_intact
            )
            results.append({
                "id": "Test-4",
                "name": "Cross-user security isolation",
                "expected": "cross_conf=None, del=NotFoundError, snooze=NotFoundError, intact=True",
                "actual": f"cross_conf={cross_conf}, del_raised={cross_del_raised}, snooze_raised={cross_snooze_raised}, intact={rec_a_intact and task_a_intact}",
                "status": "PASS" if t4_pass else "FAIL",
                "detail": "User B cannot manipulate User A records or tasks",
            })

            # -----------------------------------------------------------------
            # Test 5: Delete without confirmed raises PermissionError; delete with confirmed hard-deletes
            # -----------------------------------------------------------------
            unconf_del_raised = False
            try:
                await delete_record(session, user_a, rec_a_owned.id, confirmed=False)
            except PermissionError:
                unconf_del_raised = True

            await delete_record(session, user_a, rec_a_owned.id, confirmed=True)
            deleted_check = await session.get(KnowledgeRecord, rec_a_owned.id)

            t5_pass = unconf_del_raised and (deleted_check is None)
            results.append({
                "id": "Test-5",
                "name": "delete_record confirmation gate & hard delete",
                "expected": "unconfirmed=PermissionError, confirmed=hard deleted",
                "actual": f"unconf_raised={unconf_del_raised}, row_exists={deleted_check is not None}",
                "status": "PASS" if t5_pass else "FAIL",
                "detail": "PermissionError properly enforced; confirmed=True hard-deleted row",
            })

            # -----------------------------------------------------------------
            # Test 6: Edit changes content AND embedding (dim 1024), status unchanged
            # -----------------------------------------------------------------
            text_6_init = "Original content about calculus"
            vecs_6_init = await embed_texts([text_6_init])
            rec_6 = KnowledgeRecord(
                user_id=user_a.id,
                kind="knowledge",
                content=text_6_init,
                status="confirmed",
                embedding=vecs_6_init[0],
            )
            session.add(rec_6)
            await session.commit()
            await session.refresh(rec_6)

            new_text_6 = "Updated content about multivariable calculus and vectors"
            edited_rec = await edit_record(session, user_a, rec_6.id, new_content=new_text_6)

            vec_edited = list(edited_rec.embedding) if edited_rec.embedding is not None else []
            vec_init = list(vecs_6_init[0])
            dim_match = len(vec_edited) == settings.EMBEDDING_DIM
            vec_changed = vec_edited != vec_init

            t6_pass = (
                edited_rec.content == new_text_6
                and edited_rec.status == "confirmed"
                and dim_match
                and vec_changed
            )
            results.append({
                "id": "Test-6",
                "name": "edit_record updates content and re-embeds",
                "expected": f"content updated, dim={settings.EMBEDDING_DIM}, embedding changed, status unchanged",
                "actual": f"dim={len(vec_edited)}, changed={vec_changed}, status={edited_rec.status}",
                "status": "PASS" if t6_pass else "FAIL",
                "detail": f"Content updated and re-embedded with dimension {len(vec_edited)}",
            })

            # -----------------------------------------------------------------
            # Test 7: apply_directive without confirmed raises PermissionError
            # -----------------------------------------------------------------
            task_7 = Task(
                user_id=user_a.id,
                title="Finish homework",
                status="open",
                due_at=FIXED_NOW + timedelta(hours=2),
            )
            session.add(task_7)
            await session.commit()
            await session.refresh(task_7)

            dir_mark_done = ActionDirective(type="mark_done", task_id=task_7.id)

            dir_unconf_raised = False
            try:
                await apply_directive(session, user_a, dir_mark_done, confirmed=False)
            except PermissionError:
                dir_unconf_raised = True

            # Apply with confirmed=True
            await apply_directive(session, user_a, dir_mark_done, confirmed=True)
            await session.refresh(task_7)

            t7_pass = dir_unconf_raised and (task_7.status == "done")
            results.append({
                "id": "Test-7",
                "name": "apply_directive confirmation gate",
                "expected": "unconfirmed=PermissionError, confirmed=mark_done applied",
                "actual": f"unconf_raised={dir_unconf_raised}, status={task_7.status}",
                "status": "PASS" if t7_pass else "FAIL",
                "detail": f"Unconfirmed raised PermissionError; confirmed updated status to {task_7.status}",
            })

            # -----------------------------------------------------------------
            # Test 8: Snooze then advance clock past snoozed_until -> remind again
            # -----------------------------------------------------------------
            t8_now = FIXED_NOW
            task_8 = Task(
                user_id=user_a.id,
                title="Submit assignment",
                status="open",
                due_at=t8_now + timedelta(minutes=45),  # within 60 min lead
            )
            session.add(task_8)
            await session.commit()
            await session.refresh(task_8)

            # Fake chat_json to ensure NO LLM calls in test_actions
            async def fake_llm_actions(*args: Any, **kwargs: Any) -> Any:
                return None

            with unittest.mock.patch.object(intervention_module, "chat_json", side_effect=fake_llm_actions):
                # 1. Eligible initially
                dec8_init = await decide_intervention(
                    session=session,
                    user=user_a,
                    trigger="scheduler",
                    task=task_8,
                    now=t8_now,
                )

                # 2. Snooze for 60 minutes
                await snooze_task(session, user_a, task_8.id, minutes=60, now=t8_now)
                await session.refresh(task_8)

                # 3. Check at +20 mins (still snoozed)
                dec8_mid = await decide_intervention(
                    session=session,
                    user=user_a,
                    trigger="scheduler",
                    task=task_8,
                    now=t8_now + timedelta(minutes=20),
                )

                # 4. Check at +65 mins (snooze expired, overdue/due -> remind again!)
                dec8_after = await decide_intervention(
                    session=session,
                    user=user_a,
                    trigger="scheduler",
                    task=task_8,
                    now=t8_now + timedelta(minutes=65),
                )

            t8_pass = (
                dec8_init.action == "remind"
                and dec8_mid.action == "none"
                and dec8_after.action == "remind"
            )
            results.append({
                "id": "Test-8",
                "name": "snooze_task -> suppressed then reminds after expiry",
                "expected": "init=remind, mid=none, after=remind",
                "actual": f"init={dec8_init.action}, mid={dec8_mid.action}, after={dec8_after.action}",
                "status": "PASS" if t8_pass else "FAIL",
                "detail": f"Snoozed until {task_8.snoozed_until}; re-reminded at +65m",
            })

            # -----------------------------------------------------------------
            # Test 9: Reschedule clears last_reminded_at
            # -----------------------------------------------------------------
            task_9 = Task(
                user_id=user_a.id,
                title="Prepare presentation",
                status="snoozed",
                due_at=FIXED_NOW,
                last_reminded_at=FIXED_NOW - timedelta(minutes=30),
                snoozed_until=FIXED_NOW + timedelta(minutes=30),
            )
            session.add(task_9)
            await session.commit()
            await session.refresh(task_9)

            new_due_9 = FIXED_NOW + timedelta(days=2)
            await reschedule_task(session, user_a, task_9.id, new_due_at=new_due_9)
            await session.refresh(task_9)

            t9_pass = (
                task_9.due_at == new_due_9
                and task_9.status == "open"
                and task_9.snoozed_until is None
                and task_9.last_reminded_at is None
            )
            results.append({
                "id": "Test-9",
                "name": "reschedule_task clears last_reminded_at and resets status",
                "expected": "status=open, snoozed_until=None, last_reminded_at=None",
                "actual": f"status={task_9.status}, snoozed={task_9.snoozed_until}, reminded={task_9.last_reminded_at}",
                "status": "PASS" if t9_pass else "FAIL",
                "detail": f"New due: {task_9.due_at.isoformat()}; state reset cleanly",
            })

            # -----------------------------------------------------------------
            # Test 10: Supersede removes old record from search_confirmed and keeps new
            # -----------------------------------------------------------------
            text_old = "User decided to use SQLite"
            text_new = "User decided to use Postgres for pgvector support"
            vecs_10 = await embed_texts([text_old, text_new])

            rec_old = KnowledgeRecord(
                user_id=user_a.id,
                kind="decision",
                content=text_old,
                status="confirmed",
                embedding=vecs_10[0],
            )
            rec_new = KnowledgeRecord(
                user_id=user_a.id,
                kind="decision",
                content=text_new,
                status="pending",
                embedding=vecs_10[1],
            )
            session.add_all([rec_old, rec_new])
            await session.commit()
            await session.refresh(rec_old)
            await session.refresh(rec_new)

            # Supersede
            await supersede_record(session, user_a, old_id=rec_old.id, new_id=rec_new.id)

            # Search confirmed with both embeddings
            matches_for_old = await search_confirmed(session, user_a.id, vecs_10[0], k=10)
            has_old = any(r.id == rec_old.id for r, _ in matches_for_old)

            matches_for_new = await search_confirmed(session, user_a.id, vecs_10[1], k=10)
            has_new = any(r.id == rec_new.id for r, _ in matches_for_new)

            t10_pass = (not has_old) and has_new and (rec_old.superseded_by == rec_new.id)
            results.append({
                "id": "Test-10",
                "name": "supersede_record replaces in search_confirmed",
                "expected": "old in search=False, new in search=True, superseded_by=new_id",
                "actual": f"has_old={has_old}, has_new={has_new}, superseded_by={rec_old.superseded_by}",
                "status": "PASS" if t10_pass else "FAIL",
                "detail": f"Old ID {rec_old.id} hidden; New ID {rec_new.id} confirmed and visible",
            })

        finally:
            print("\nCleaning up throwaway test users and their records...")
            for uid in users_to_cleanup:
                await cleanup_user(session, uid)
            print("Cleanup completed.")

    # =============================================================
    # Print PASS/FAIL table
    # =============================================================
    print("\n" + "=" * 115)
    print(f"{'TEST ID':<8} | {'TEST NAME':<42} | {'EXPECTED':<30} | {'STATUS':<6} | {'DETAILS'}")
    print("-" * 115)
    all_passed = True
    for r in results:
        status_str = f"[{r['status']}]"
        if r["status"] != "PASS":
            all_passed = False
        print(f"{r['id']:<8} | {r['name']:<42} | {r['expected']:<30} | {status_str:<6} | {r['detail']}")
    print("=" * 115)

    if all_passed:
        print("\nALL ACTION EXECUTOR TESTS PASSED SUCCESSFULLY.")
    else:
        print("\nSOME ACTION EXECUTOR TESTS FAILED. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

