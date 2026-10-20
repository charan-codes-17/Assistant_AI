"""backend/scripts/test_scheduler.py
Phase 5b evaluation for the deterministic reminder scheduler, the reminder buttons and /quiet.

No LLM and no embedding calls are ever allowed here. A fake Bot object records sends/edits; the
real dev DB is used with synthetic users whose telegram_chat_id is in 970000-979999. Those users
are purged at the START and again in a finally block. Fixed `now` values are used everywhere
except S7 (snooze uses the real clock) and S9 (the only sleep, to exercise the loop).

Run:
  backend\\venv\\Scripts\\python.exe -u backend\\scripts\\test_scheduler.py
"""

import asyncio
from datetime import datetime, timedelta, timezone
import inspect
import logging
from pathlib import Path
import sys
import time
from typing import Any, List, Optional
import unittest.mock
import uuid
from zoneinfo import ZoneInfo

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select

import backend.services.embeddings as embeddings_module
import backend.services.llm as llm_module
from backend.bot import handlers, scheduler
from backend.bot.state import BotState
from backend.config.settings import settings
from backend.database import AsyncSessionLocal, engine
from backend.models.models import Task, User
from backend.services import actions, reminders
from telegram.error import Forbidden, NetworkError

CHAT_MIN = 970000
CHAT_MAX = 979999
IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)

# Distinctive titles used in S1/S5/S6 and asserted absent from every captured log record (S11).
TITLE_S1 = "ZqS1SecretTitle"
TITLE_S5 = "ZqS5SecretTitle"
TITLE_S6 = "ZqS6SecretTitle"
PRIVACY_TITLES = (TITLE_S1, TITLE_S5, TITLE_S6)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeChat:
    def __init__(self, chat_id: int, chat_type: str = "private") -> None:
        self.id = chat_id
        self.type = chat_type


class FakeMessage:
    def __init__(self, chat_id: int, message_id: int = 500, text: Optional[str] = None) -> None:
        self.chat = FakeChat(chat_id)
        self.chat_id = chat_id
        self.message_id = message_id
        self.text = text


class FakeCallbackQuery:
    def __init__(self, chat_id: int, data: str, message_id: int = 500) -> None:
        self.id = f"cb-{uuid.uuid4().hex[:6]}"
        self.data = data
        self.message = FakeMessage(chat_id, message_id=message_id)


class FakeUpdate:
    def __init__(self, chat_id: int, text: Optional[str] = None, callback: Any = None) -> None:
        self.effective_chat = FakeChat(chat_id)
        self.effective_user = FakeChat(chat_id)
        self.message = FakeMessage(chat_id, text=text)
        self.callback_query = callback


class FakeContext:
    def __init__(self, bot: "FakeBot", bot_data=None) -> None:
        self.bot = bot
        self.bot_data = bot_data or {}


class FakeBot:
    """Records sends/edits. If `error` is set, every send_message raises it."""

    def __init__(self) -> None:
        self.sent: List[dict] = []
        self.edits: List[dict] = []
        self.answers: List[str] = []
        self.error: Optional[Exception] = None

    async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
        if self.error is not None:
            raise self.error
        self.sent.append({"chat_id": chat_id, "text": text, "reply_markup": reply_markup})
        return None

    async def edit_message_text(self, chat_id, message_id, text, reply_markup=None, **kwargs):
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "text": text})

    async def answer_callback_query(self, callback_query_id, **kwargs):
        self.answers.append(callback_query_id)


def make_env():
    bot = FakeBot()
    st = BotState()
    return bot, FakeContext(bot, {"state": st}), st


def callbacks_of(markup) -> List[str]:
    rows = getattr(markup, "inline_keyboard", None) or []
    out: List[str] = []
    for row in rows:
        for button in row:
            cb = getattr(button, "callback_data", None)
            if cb is not None:
                out.append(cb)
    return out


def sends_to(bot: FakeBot, chat_id: int) -> List[dict]:
    return [m for m in bot.sent if m["chat_id"] == chat_id]


# ---------------------------------------------------------------------------
# Log capture (S11)
# ---------------------------------------------------------------------------
LOG_RECORDS: List[logging.LogRecord] = []


class ListHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        LOG_RECORDS.append(record)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
async def make_user(
    session, chat_id: Optional[int], *, quiet: bool = False, tz: str = "Asia/Kolkata"
) -> User:
    user = User(
        username=f"sched_{uuid.uuid4().hex[:8]}",
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
    session,
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


async def get_task(session, task_id: int) -> Optional[Task]:
    return await session.get(Task, task_id, populate_existing=True)


async def get_user(session, user_id: int) -> Optional[User]:
    return await session.get(User, user_id, populate_existing=True)


async def purge_synthetic(session) -> int:
    rows = (
        await session.execute(
            select(User).where(
                (User.telegram_chat_id >= CHAT_MIN) & (User.telegram_chat_id <= CHAT_MAX)
            )
        )
    ).scalars().all()
    rows += (
        await session.execute(select(User).where(User.username.like("sched_%")))
    ).scalars().all()
    for user in rows:
        await session.delete(user)
    await session.commit()
    return len(rows)


def add(results, rid, name, ok, expected, actual, details=""):
    results.append(
        {
            "id": rid,
            "name": name,
            "ok": bool(ok),
            "expected": expected,
            "actual": actual,
            "details": details,
        }
    )


# ---------------------------------------------------------------------------
# S1
# ---------------------------------------------------------------------------
async def s1(session, results) -> None:
    chat = 970001
    user = await make_user(session, chat)
    due = NOW + timedelta(minutes=30)
    task = await make_task(session, user, TITLE_S1, due_at=due)

    bot = FakeBot()
    result = await scheduler.run_tick(bot, NOW)
    await session.refresh(task)

    sends = sends_to(bot, chat)
    expected_text = reminders.render_reminder_text(TITLE_S1, due, "Asia/Kolkata", NOW)
    text_ok = len(sends) == 1 and sends[0]["text"] == expected_text
    cbs = callbacks_of(sends[0]["reply_markup"]) if sends else []
    expected_cbs = [f"rm:done:{task.id}", f"rm:snz:{task.id}", f"rm:quiet:{task.id}"]
    cbs_ok = cbs == expected_cbs
    cbs_size_ok = all(len(c.encode("utf-8")) <= 64 for c in cbs)
    marked = task.last_reminded_at == NOW
    ok = text_ok and cbs_ok and cbs_size_ok and marked and result.sent >= 1
    add(
        results, "S1", "due task -> one send + 3 buttons + claimed",
        ok, "1 send, exact text, rm:done/snz/quiet, last_reminded_at=NOW",
        f"sends={len(sends)}, text_ok={text_ok}, cbs={cbs}, marked={marked}",
        f"expected_text={expected_text!r}, result.sent={result.sent}, cb_bytes_ok={cbs_size_ok}",
    )


# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------
async def s2(session, results) -> None:
    chat = 970002
    user = await make_user(session, chat)
    due = NOW + timedelta(minutes=30)
    await make_task(session, user, "S2 task", due_at=due)

    bot = FakeBot()
    await scheduler.run_tick(bot, NOW)
    first = len(sends_to(bot, chat))
    await scheduler.run_tick(bot, NOW + timedelta(minutes=1))
    second = len(sends_to(bot, chat))
    ok = first == 1 and second == 1  # exactly one send total
    add(
        results, "S2", "second tick a minute later -> no resend",
        ok, "1 send after tick 1; still 1 after tick 2",
        f"after_tick1={first}, after_tick2={second}",
    )


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------
async def s3(session, results) -> None:
    chat = 970003
    user = await make_user(session, chat, quiet=True)
    due = NOW + timedelta(minutes=30)
    task = await make_task(session, user, "S3 task", due_at=due)

    bot = FakeBot()
    await scheduler.run_tick(bot, NOW)
    await session.refresh(task)
    quiet_sends = len(sends_to(bot, chat))
    unchanged = task.last_reminded_at is None and task.status == "open"

    fresh = await get_user(session, user.id)
    await actions.set_quiet_mode(session, fresh, False)
    await scheduler.run_tick(bot, NOW)
    after_sends = len(sends_to(bot, chat))

    ok = quiet_sends == 0 and unchanged and after_sends == 1
    add(
        results, "S3", "quiet user skipped; unquiet -> one send",
        ok, "0 sends while quiet, row untouched; 1 send after quiet off",
        f"quiet_sends={quiet_sends}, row_unchanged={unchanged}, after_off_sends={after_sends}",
    )


# ---------------------------------------------------------------------------
# S4
# ---------------------------------------------------------------------------
async def s4(session, results) -> None:
    chat = 970004
    user = await make_user(session, chat)
    tp = await make_task(session, user, "S4 pending", status="pending", due_at=NOW + timedelta(minutes=5))
    td = await make_task(session, user, "S4 done", status="done", due_at=NOW + timedelta(minutes=5))

    bot = FakeBot()
    await scheduler.run_tick(bot, NOW)
    await session.refresh(tp)
    await session.refresh(td)
    sends = len(sends_to(bot, chat))
    untouched = (
        tp.last_reminded_at is None and td.last_reminded_at is None
        and tp.status == "pending" and td.status == "done"
    )
    ok = sends == 0 and untouched
    add(
        results, "S4", "pending and done tasks never reminded",
        ok, "0 sends, both rows untouched",
        f"sends={sends}, rows_untouched={untouched}",
    )


# ---------------------------------------------------------------------------
# S5
# ---------------------------------------------------------------------------
async def s5(session, results) -> None:
    chat = 970005
    user = await make_user(session, chat)
    due = NOW + timedelta(minutes=30)
    task = await make_task(session, user, TITLE_S5, due_at=due)

    bot = FakeBot()
    bot.error = NetworkError("net down")
    raised = False
    result = None
    try:
        result = await scheduler.run_tick(bot, NOW)
    except Exception:  # noqa: BLE001
        raised = True
    await session.refresh(task)
    released = task.last_reminded_at is None and task.status == "open" and task.snoozed_until is None

    # Next tick with a working bot: the released task sends once.
    bot2 = FakeBot()
    await scheduler.run_tick(bot2, NOW)
    next_sends = len(sends_to(bot2, chat))

    ok = (
        (not raised) and released and next_sends == 1
        and result is not None and result.failed_transient == 1 and result.sent == 0
    )
    add(
        results, "S5", "transient send failure -> released, retried next tick",
        ok, "no raise; row released; next tick sends once",
        f"raised={raised}, released={released}, next_sends={next_sends}, "
        f"transient={getattr(result, 'failed_transient', None)}",
    )


# ---------------------------------------------------------------------------
# S6
# ---------------------------------------------------------------------------
async def s6(session, results) -> None:
    chat = 970006
    user = await make_user(session, chat)
    due = NOW + timedelta(minutes=30)
    task = await make_task(session, user, TITLE_S6, due_at=due)

    bot = FakeBot()
    bot.error = Forbidden("bot was blocked by the user")
    raised = False
    result = None
    try:
        result = await scheduler.run_tick(bot, NOW)
    except Exception:  # noqa: BLE001
        raised = True
    await session.refresh(task)
    kept_claimed = task.last_reminded_at == NOW and task.status == "open"

    # Next tick (working bot) must NOT retry: the row stays claimed.
    bot2 = FakeBot()
    await scheduler.run_tick(bot2, NOW)
    retry_sends = len(sends_to(bot2, chat))

    ok = (
        (not raised) and kept_claimed and retry_sends == 0
        and result is not None and result.failed_permanent == 1
    )
    add(
        results, "S6", "permanent send failure (Forbidden) -> no retry",
        ok, "no raise; row stays claimed; next tick sends nothing",
        f"raised={raised}, kept_claimed={kept_claimed}, retry_sends={retry_sends}, "
        f"permanent={getattr(result, 'failed_permanent', None)}",
    )


# ---------------------------------------------------------------------------
# S7
# ---------------------------------------------------------------------------
async def s7(session, results) -> None:
    details: List[str] = []
    oks: List[bool] = []

    # --- rm:done ---
    chat_done = 970007
    u_done = await make_user(session, chat_done)
    t_done = await make_task(session, u_done, "S7 done", due_at=NOW + timedelta(minutes=30))
    bot, ctx, st = make_env()
    await handlers.on_callback(
        FakeUpdate(chat_done, callback=FakeCallbackQuery(chat_done, f"rm:done:{t_done.id}")),
        ctx, state=st,
    )
    t_done = await get_task(session, t_done.id)
    done_edit = any(e["text"] == handlers.REMINDER_DONE_TEXT for e in bot.edits)
    done_ok = t_done.status == "done" and done_edit
    oks.append(done_ok)
    details.append(f"done: status={t_done.status}, edit={done_edit}")

    # --- rm:snz (real clock drives snooze; scan at snoozed_until + 1 min) ---
    chat_snz = 970008
    u_snz = await make_user(session, chat_snz)
    real_now = datetime.now(timezone.utc)
    t_snz = await make_task(session, u_snz, "S7 snooze", due_at=real_now + timedelta(minutes=30))
    bot_s, ctx_s, st_s = make_env()
    await handlers.on_callback(
        FakeUpdate(chat_snz, callback=FakeCallbackQuery(chat_snz, f"rm:snz:{t_snz.id}")),
        ctx_s, state=st_s,
    )
    t_snz = await get_task(session, t_snz.id)
    snooze_edit = any(e["text"] == handlers.snooze_text(settings.REMINDER_SNOOZE_MINUTES)
                      for e in bot_s.edits)
    snoozed_ok = (
        t_snz.status == "snoozed"
        and t_snz.snoozed_until is not None
        and abs((t_snz.snoozed_until - real_now).total_seconds() - settings.REMINDER_SNOOZE_MINUTES * 60) < 120
        and snooze_edit
    )
    scan_now = t_snz.snoozed_until + timedelta(minutes=1)
    bot_s2 = FakeBot()
    await scheduler.run_tick(bot_s2, scan_now)
    resend = len(sends_to(bot_s2, chat_snz))
    t_snz = await get_task(session, t_snz.id)
    reopened = t_snz.status == "open" and t_snz.last_reminded_at is not None
    snz_ok = snoozed_ok and resend == 1 and reopened
    oks.append(snz_ok)
    details.append(
        f"snooze: snoozed_ok={snoozed_ok}, resend={resend}, reopened={reopened}"
    )

    # --- rm:snz singular wording when REMINDER_SNOOZE_MINUTES == 1 ---
    chat_snz1 = 970017
    u_snz1 = await make_user(session, chat_snz1)
    t_snz1 = await make_task(session, u_snz1, "S7 snooze one", due_at=NOW + timedelta(minutes=30))
    bot_s1, ctx_s1, st_s1 = make_env()
    with unittest.mock.patch.object(settings, "REMINDER_SNOOZE_MINUTES", 1):
        await handlers.on_callback(
            FakeUpdate(chat_snz1, callback=FakeCallbackQuery(chat_snz1, f"rm:snz:{t_snz1.id}")),
            ctx_s1, state=st_s1,
        )
    singular_edit = any(e["text"] == "Snoozed for 1 minute ⏰" for e in bot_s1.edits)
    oks.append(singular_edit)
    details.append(f"singular (N=1): edit={singular_edit}")

    # --- rm:quiet ---
    chat_q = 970009
    u_q = await make_user(session, chat_q)
    t_q = await make_task(session, u_q, "S7 quiet", due_at=NOW + timedelta(minutes=30))
    bot_q, ctx_q, st_q = make_env()
    await handlers.on_callback(
        FakeUpdate(chat_q, callback=FakeCallbackQuery(chat_q, f"rm:quiet:{t_q.id}")),
        ctx_q, state=st_q,
    )
    u_q_fresh = await get_user(session, u_q.id)
    quiet_edit = any("/quiet" in e["text"] for e in bot_q.edits)
    quiet_ok = u_q_fresh.quiet_mode is True and quiet_edit
    oks.append(quiet_ok)
    details.append(f"quiet: quiet_mode={u_q_fresh.quiet_mode}, edit_mentions=/quiet:{quiet_edit}")

    # --- another user's task id + a missing id ---
    chat_x = 970010
    u_x = await make_user(session, chat_x)
    t_x = await make_task(session, u_x, "S7 other", due_at=NOW + timedelta(minutes=30))
    chat_y = 970011
    await make_user(session, chat_y)
    bot_y, ctx_y, st_y = make_env()
    await handlers.on_callback(
        FakeUpdate(chat_y, callback=FakeCallbackQuery(chat_y, f"rm:done:{t_x.id}")),
        ctx_y, state=st_y,
    )
    other_edit = any(e["text"] == handlers.NOT_AVAILABLE for e in bot_y.edits)
    await handlers.on_callback(
        FakeUpdate(chat_y, callback=FakeCallbackQuery(chat_y, "rm:done:999999999")),
        ctx_y, state=st_y,
    )
    missing_edit = any(e["text"] == handlers.NOT_AVAILABLE for e in bot_y.edits)
    t_x = await get_task(session, t_x.id)
    other_ok = (
        other_edit and missing_edit
        and t_x.status == "open" and t_x.last_reminded_at is None
    )
    oks.append(other_ok)
    details.append(
        f"other: edit={other_edit}, missing_edit={missing_edit}, x_status={t_x.status}, "
        f"x_last_reminded={t_x.last_reminded_at}"
    )

    ok = all(oks)
    add(results, "S7", "rm:done / rm:snz / rm:quiet callbacks + ownership",
        ok, "done->done; snz->snoozed then reopen (N=1 -> 'Snoozed for 1 minute'); quiet->quiet; other's id->not available",
        "; ".join(details))


# ---------------------------------------------------------------------------
# S8
# ---------------------------------------------------------------------------
async def s8(session, results) -> None:
    chat = 970012
    user = await make_user(session, chat, quiet=False)
    bot, ctx, st = make_env()

    await handlers.cmd_quiet(FakeUpdate(chat, text="/quiet"), ctx, state=st)
    u1 = await get_user(session, user.id)
    first_off = u1.quiet_mode is True and bot.sent[-1]["text"] == handlers.QUIET_OFF_TEXT

    await handlers.cmd_quiet(FakeUpdate(chat, text="/quiet"), ctx, state=st)
    u2 = await get_user(session, user.id)
    second_on = u2.quiet_mode is False and bot.sent[-1]["text"] == handlers.QUIET_ON_TEXT

    unknown = 970013
    await handlers.cmd_quiet(FakeUpdate(unknown, text="/quiet"), ctx, state=st)
    unknown_prompt = bot.sent[-1]["text"] == handlers.SIGNUP_PROMPT

    ok = first_off and second_on and unknown_prompt
    add(
        results, "S8", "/quiet toggles off then on; unknown chat -> signup",
        ok, "off text, on text, signup prompt for unknown chat",
        f"first_off={first_off}, second_on={second_on}, unknown_prompt={unknown_prompt}",
    )


# ---------------------------------------------------------------------------
# S9
# ---------------------------------------------------------------------------
async def s9(session, results) -> None:
    chat = 970014
    user = await make_user(session, chat)
    # scheduler_loop reads the real clock, so the task must be due relative to real time.
    real_now = datetime.now(timezone.utc)
    await make_task(session, user, "S9 task", due_at=real_now + timedelta(minutes=30))

    real_find = scheduler.reminders.find_due_reminders
    calls = [0]

    async def flaky(session_arg, now_arg):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("boom on first tick")
        return await real_find(session_arg, now_arg)

    scheduler.reminders.find_due_reminders = flaky
    old_scan = settings.REMINDER_SCAN_SECONDS
    settings.REMINDER_SCAN_SECONDS = 0.05
    bot = FakeBot()
    raised: Optional[BaseException] = None
    loop_task = asyncio.create_task(scheduler.scheduler_loop(bot))
    try:
        # Remote DB round-trips make a tick slower than the 0.05s interval; wait (bounded) for the
        # second tick to actually deliver instead of cancelling a fixed short time later.
        deadline = time.monotonic() + 6.0
        while len(sends_to(bot, chat)) == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
        except BaseException as exc:  # noqa: BLE001
            raised = exc
        scheduler.reminders.find_due_reminders = real_find
        settings.REMINDER_SCAN_SECONDS = old_scan

    sends = len(sends_to(bot, chat))
    ok = raised is None and sends == 1 and calls[0] >= 2
    add(
        results, "S9", "loop survives one failed tick, cancels cleanly",
        ok, "no unexpected exception; second tick sends once",
        f"raised={raised!r}, sends={sends}, find_calls={calls[0]}",
    )


# ---------------------------------------------------------------------------
# S10
# ---------------------------------------------------------------------------
async def s10(session, results) -> None:
    chat = 970015
    user = await make_user(session, chat)
    await make_task(session, user, "S10 task", due_at=NOW + timedelta(minutes=30))

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

    patched_names: List[str] = []
    error = ""
    sends = 0
    bot = FakeBot()
    try:
        for p in patchers:
            p.start()
            patched_names.append(p.attribute)
        await scheduler.run_tick(bot, NOW)
        sends = len(sends_to(bot, chat))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for p in patchers:
            p.stop()

    ok = error == "" and sends == 1
    add(
        results, "S10", "no LLM/embedding calls (all public coroutines blocked)",
        ok, "no exception; one send",
        f"error={error or 'none'}, sends={sends}, blocked={len(patched_names)}",
    )


# ---------------------------------------------------------------------------
# S11
# ---------------------------------------------------------------------------
async def s11(session, results) -> None:
    texts = [r.getMessage() for r in LOG_RECORDS]
    blob = "\n".join(texts)
    leaked = [t for t in PRIVACY_TITLES if t in blob]
    captured_task_id = any("task_id=" in t for t in texts)
    ok = not leaked and captured_task_id
    add(
        results, "S11", "privacy: no task title in any log record",
        ok, "no title leaked; scheduler warnings captured (task_id= present)",
        f"leaked={leaked}, records={len(LOG_RECORDS)}, task_id_seen={captured_task_id}",
    )


# ---------------------------------------------------------------------------
# S12
# ---------------------------------------------------------------------------
async def s12(session, results) -> None:
    chat = 970016
    user = await make_user(session, chat)
    await make_task(session, user, "S12 task", due_at=NOW + timedelta(minutes=30))

    bot = FakeBot()
    r1, r2 = await asyncio.gather(
        scheduler.run_tick(bot, NOW),
        scheduler.run_tick(bot, NOW),
    )
    sends = len(sends_to(bot, chat))
    total_sent = r1.sent + r2.sent
    ok = sends == 1 and total_sent >= 1
    add(
        results, "S12", "two concurrent ticks -> exactly one send",
        ok, "1 send to the chat for the shared task",
        f"sends={sends}, r1.sent={r1.sent}, r2.sent={r2.sent}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def main() -> int:
    settings.require("DATABASE_URL")

    root = logging.getLogger()
    capture = ListHandler()
    root.addHandler(capture)

    print("=" * 130)
    print("PHASE 5b: REMINDER SCHEDULER + BUTTONS + /quiet EVALUATION")
    print(f"Reference now: {NOW.isoformat()}  |  SCAN={settings.REMINDER_SCAN_SECONDS}s "
          f"LEAD={settings.REMINDER_LEAD_MINUTES}m SNOOZE={settings.REMINDER_SNOOZE_MINUTES}m "
          f"CAP={settings.REMINDER_MAX_PER_USER_PER_TICK}")
    print("=" * 130)

    results: List[dict] = []
    cases = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]

    async with AsyncSessionLocal() as session:
        purged = await purge_synthetic(session)
        print(f"PURGE at start: removed {purged} synthetic user(s).\n")
        try:
            for case in cases:
                print(f"--- Running {case.__name__.upper()} ---")
                try:
                    await case(session, results)
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    add(results, case.__name__.upper(), case.__name__, False, "-", "",
                        f"unhandled: {exc!r}")
                last = results[-1]
                print(f"    [{'PASS' if last['ok'] else 'FAIL'}] {last['name']}: {last['actual']}")
        finally:
            print("\nCleaning up synthetic scheduler users...")
            cleaned = await purge_synthetic(session)
            print(f"Cleanup removed {cleaned} synthetic user(s).")

    root.removeHandler(capture)
    await engine.dispose()

    print("\n" + "=" * 130)
    print(f"{'ID':<5} | {'CASE':<46} | {'STATUS':<6} | {'EXPECTED':<58} | {'ACTUAL'}")
    print("-" * 130)
    all_passed = True
    for r in results:
        if not r["ok"]:
            all_passed = False
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{r['id']:<5} | {r['name']:<46} | {status:<6} | {r['expected']:<58} | {r['actual']}")
    print("=" * 130)
    passed = sum(1 for r in results if r["ok"])
    print(f"RESULT: {'ALL PASS' if all_passed else 'FAILURES PRESENT'} ({passed}/{len(results)})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
