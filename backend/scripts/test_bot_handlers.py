"""backend/scripts/test_bot_handlers.py
Phase 4a-2 evaluation for the Telegram bot handlers (text only).

Fakes: a FakeBot recording send_message / edit_message_text / answer_callback_query /
send_chat_action, and simple fake update/context objects. No Telegram network is used.

B1-B10 (and B12) use a mocked orchestrator (no LLM credits). B11 is a REAL end-to-end run with
real models and real services, using the fake Telegram surface.

Throwaway users only; every row is deleted in a finally block.
"""

import asyncio
import importlib
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import uuid
from zoneinfo import ZoneInfo

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import func, select

from backend.bot import handlers, textfmt
from backend.bot.state import BotState
from backend.config.settings import settings
from backend.database import AsyncSessionLocal, engine
from backend.models.models import KnowledgeRecord, Task, User
from backend.services import orchestrator
from backend.services.extraction import CandidateTask
from backend.services.orchestrator import (
    MemoryUsedItem,
    PendingRecord,
    PostResult,
    ReplyResult,
)

IST = ZoneInfo("Asia/Kolkata")

CHAT_IDS: set[int] = set()
_next_chat = [900000]


def next_chat() -> int:
    _next_chat[0] += 1
    CHAT_IDS.add(_next_chat[0])
    return _next_chat[0]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeChat:
    def __init__(self, chat_id: int, chat_type: str = "private") -> None:
        self.id = chat_id
        self.type = chat_type


class FakeMessage:
    def __init__(self, chat_id: int, message_id: int = 1, text=None, voice=None) -> None:
        self.chat = FakeChat(chat_id)
        self.chat_id = chat_id
        self.message_id = message_id
        self.text = text
        self.voice = voice


class FakeTelegramUser:
    def __init__(self, uid: int) -> None:
        self.id = uid


class FakeCallbackQuery:
    def __init__(self, chat_id: int, data: str, message_id: int = 500) -> None:
        self.id = f"cb-{uuid.uuid4().hex[:6]}"
        self.data = data
        self.message = FakeMessage(chat_id, message_id=message_id)


class FakeVoice:
    def __init__(self, duration: int = 3, file_id: str = "voice-file") -> None:
        self.duration = duration
        self.file_id = file_id


class FakeTgFile:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def download_as_bytearray(self) -> bytearray:
        return bytearray(self._data)


class FakeUpdate:
    def __init__(self, chat_id: int, text=None, voice=False, callback=None, duration: int = 3) -> None:
        self.effective_chat = FakeChat(chat_id)
        self.effective_user = FakeTelegramUser(chat_id)
        self.message = FakeMessage(
            chat_id, text=text, voice=FakeVoice(duration=duration) if voice else None
        )
        self.callback_query = callback


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.answers: list[str] = []
        self.actions: list[dict] = []
        self.voice_bytes = b"fake-ogg-bytes"
        self._t0 = time.perf_counter()
        self._mid = 100

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    async def get_file(self, file_id, **kwargs):
        return FakeTgFile(self.voice_bytes)

    async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
        self._mid += 1
        self.sent.append(
            {"chat_id": chat_id, "text": text, "reply_markup": reply_markup, "t": self._now()}
        )
        return FakeMessage(chat_id, message_id=self._mid, text=text)

    async def send_chat_action(self, chat_id, action, **kwargs):
        self.actions.append({"chat_id": chat_id, "action": action, "t": self._now()})

    async def edit_message_text(self, chat_id, message_id, text, reply_markup=None, **kwargs):
        self.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "text": text, "reply_markup": reply_markup}
        )

    async def answer_callback_query(self, callback_query_id, **kwargs):
        self.answers.append(callback_query_id)


class FakeContext:
    def __init__(self, bot: FakeBot, bot_data=None) -> None:
        self.bot = bot
        self.bot_data = bot_data or {}


def make_env() -> tuple[FakeBot, FakeContext, BotState]:
    bot = FakeBot()
    st = BotState()
    ctx = FakeContext(bot, {"state": st})
    return bot, ctx, st


def callbacks_of(markup) -> list[str]:
    rows = getattr(markup, "inline_keyboard", None) or []
    out = []
    for row in rows:
        for button in row:
            cb = getattr(button, "callback_data", None)
            if cb is not None:
                out.append(cb)
    return out


def all_callbacks(bot: FakeBot) -> list[str]:
    out = []
    for msg in bot.sent:
        out.extend(callbacks_of(msg.get("reply_markup")))
    return out


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
async def db_add_user(chat_id: int) -> int:
    async with AsyncSessionLocal() as session:
        user = User(username=f"bott_{uuid.uuid4().hex[:8]}", telegram_chat_id=chat_id,
                    timezone="Asia/Kolkata")
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.id


async def db_user_id_for_chat(chat_id: int) -> int | None:
    async with AsyncSessionLocal() as session:
        user = (
            await session.execute(select(User).where(User.telegram_chat_id == chat_id))
        ).scalar_one_or_none()
        return user.id if user else None


async def db_add_record(user_id: int, content: str, status: str = "pending",
                        kind: str = "knowledge") -> int:
    async with AsyncSessionLocal() as session:
        rec = KnowledgeRecord(
            user_id=user_id, kind=kind, content=content, certainty="tentative",
            source="text", status=status,
        )
        session.add(rec)
        await session.commit()
        await session.refresh(rec)
        return rec.id


async def db_record_status(record_id: int) -> str | None:
    async with AsyncSessionLocal() as session:
        rec = await session.get(KnowledgeRecord, record_id)
        return rec.status if rec else None


async def db_add_pending_task(user_id: int, title: str, due=None) -> int:
    async with AsyncSessionLocal() as session:
        task = Task(user_id=user_id, title=title, due_at=due, status="pending", origin="promise")
        session.add(task)
        await session.commit()
        await session.refresh(task)
        return task.id


async def db_task_status(task_id: int) -> str | None:
    async with AsyncSessionLocal() as session:
        task = await session.get(Task, task_id)
        return task.status if task else "MISSING"


async def db_tasks_for(user_id: int) -> list[Task]:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(select(Task).where(Task.user_id == user_id).order_by(Task.id))
        ).scalars().all()
    return list(rows)


# ---------------------------------------------------------------------------
# Mocked orchestrator
# ---------------------------------------------------------------------------
def make_respond(reply_text="mocked reply", memory_used=None, counter=None):
    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        if counter is not None:
            counter[0] += 1
        return ReplyResult(
            turn_id=f"turn-{uuid.uuid4().hex[:6]}",
            session_id=session_id,
            reply=reply_text,
            memory_used=memory_used or [],
        )
    return respond


def make_postprocess(result_factory=None, delay=0.0):
    async def postprocess(user_id, reply, *, now=None):
        if delay:
            await asyncio.sleep(delay)
        if result_factory is not None:
            return result_factory(reply)
        return PostResult()
    return postprocess


@contextmanager
def mock_orchestrator(respond_fn, postprocess_fn):
    old_r, old_p = orchestrator.respond, orchestrator.postprocess
    orchestrator.respond = respond_fn
    orchestrator.postprocess = postprocess_fn
    try:
        yield
    finally:
        orchestrator.respond, orchestrator.postprocess = old_r, old_p


@contextmanager
def patched_settings(**kwargs):
    old = {k: getattr(settings, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(settings, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(settings, k, v)


async def drain(st: BotState) -> None:
    while st.background:
        await asyncio.gather(*list(st.background), return_exceptions=True)


async def send_text(bot, ctx, st, chat_id, text):
    await handlers.on_text(FakeUpdate(chat_id, text=text), ctx, state=st)


async def press(bot, ctx, st, chat_id, data):
    await handlers.on_callback(FakeUpdate(chat_id, callback=FakeCallbackQuery(chat_id, data)), ctx, state=st)


def row(rid, name, expected, actual, ok, details=""):
    return {"id": rid, "name": name, "expected": expected, "actual": actual,
            "ok": bool(ok), "details": details}


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------
async def b1() -> dict:
    chat = next_chat()
    bot, ctx, st = make_env()
    with patched_settings(INVITE_CODES="TESTCODE", BOT_MAX_USERS=10_000_000):
        await handlers.cmd_start(FakeUpdate(chat, text="/start TESTCODE"), ctx, state=st)
        uid = await db_user_id_for_chat(chat)
        await handlers.cmd_start(FakeUpdate(chat, text="/start TESTCODE"), ctx, state=st)
        uid2 = await db_user_id_for_chat(chat)
    count = 1 if uid == uid2 and uid is not None else 0
    welcome = bot.sent[0]["text"] if bot.sent else ""
    second = bot.sent[1]["text"] if len(bot.sent) > 1 else ""
    ok = (
        uid is not None and uid2 == uid and count == 1
        and "Welcome back" in second
        and "/forget" in welcome
    )
    return row("B1", "/start valid code + welcome back",
               "1 user row; 2nd start = welcome back; banner mentions /forget",
               f"user_id={uid}, rows={count}, second={second!r}, banner_has_forget={'/forget' in welcome}",
               ok, f"welcome={welcome[:60]!r}")


# ---------------------------------------------------------------------------
# B2
# ---------------------------------------------------------------------------
async def b2() -> dict:
    chat = next_chat()
    chat2 = next_chat()
    bot, ctx, st = make_env()
    counter = [0]
    with patched_settings(INVITE_CODES="TESTCODE"):
        with mock_orchestrator(make_respond(counter=counter), make_postprocess()):
            await handlers.cmd_start(FakeUpdate(chat, text="/start WRONG"), ctx, state=st)
            await handlers.cmd_start(FakeUpdate(chat2, text="/start"), ctx, state=st)
            rows_wrong = await db_user_id_for_chat(chat)
            rows_none = await db_user_id_for_chat(chat2)
            await send_text(bot, ctx, st, chat, "hello there")
    ok = rows_wrong is None and rows_none is None and counter[0] == 0
    return row("B2", "wrong/no code + unknown chat",
               "2 refusals, 0 rows, 0 orchestrator calls",
               f"wrong_row={rows_wrong}, no_code_row={rows_none}, orchestrator_calls={counter[0]}",
               ok, "")


# ---------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------
async def b3() -> dict:
    chat1 = next_chat()
    chat2 = next_chat()
    bot, ctx, st = make_env()
    async with AsyncSessionLocal() as session:
        current = (await session.execute(select(func.count()).select_from(User))).scalar_one()
    with patched_settings(INVITE_CODES="TESTCODE", BOT_MAX_USERS=current + 1):
        await handlers.cmd_start(FakeUpdate(chat1, text="/start TESTCODE"), ctx, state=st)
        row1 = await db_user_id_for_chat(chat1)
        await handlers.cmd_start(FakeUpdate(chat2, text="/start TESTCODE"), ctx, state=st)
        row2 = await db_user_id_for_chat(chat2)
    ok = row1 is not None and row2 is None
    return row("B3", "BOT_MAX_USERS cap",
               "chat1 created; chat2 refused, no row",
               f"max={current + 1}, chat1={row1}, chat2={row2}",
               ok, f"baseline_users={current}")


# ---------------------------------------------------------------------------
# B4
# ---------------------------------------------------------------------------
async def b4() -> dict:
    stripped = textfmt.strip_markdown("**x**")
    italic = textfmt.strip_markdown("*y*")
    code = textfmt.strip_markdown("`z`")
    header = textfmt.strip_markdown("# H")
    star_bullet = textfmt.strip_markdown("* item")
    dash_bullet = textfmt.strip_markdown("- item")

    base = ("This is a sentence. " * 40 + "\n\n") * 40  # ~3600+ chars, has paragraphs+sentences
    long_text = (base * 3)[:9000]
    chunks = textfmt.chunk_text(long_text)
    rejoined = "".join(chunks)
    ok = (
        stripped == "x" and italic == "y" and code == "z" and header == "H"
        and star_bullet == "- item" and dash_bullet == "- item"
        and all(len(c) <= 4096 for c in chunks)
        and rejoined == long_text
    )
    return row("B4", "textfmt markdown + chunking",
               "markdown stripped; 9000 chars -> chunks<=4096, lossless",
               f"strip=({stripped},{italic},{code},{header},{star_bullet!r},{dash_bullet!r}), "
               f"chunks={len(chunks)}, max_len={max(len(c) for c in chunks)}, lossless={rejoined == long_text}",
               ok, "")


# ---------------------------------------------------------------------------
# B5
# ---------------------------------------------------------------------------
async def b5() -> dict:
    chat_a = next_chat()
    chat_b = next_chat()
    uid_a = await db_add_user(chat_a)
    uid_b = await db_add_user(chat_b)
    rec_a = await db_add_record(uid_a, "A memory content")
    bot, ctx, st = make_env()

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        return ReplyResult(turn_id=f"turn-{user_id}", session_id=session_id,
                           reply=f"reply for {text}", memory_used=[])

    async def postprocess(user_id, reply, *, now=None):
        await asyncio.sleep(2.0)
        if reply.turn_id == f"turn-{uid_a}":
            return PostResult(pending_records=[
                PendingRecord(id=rec_a, content="A memory content", kind="knowledge",
                              certainty="tentative", source="text")
            ])
        return PostResult()

    with mock_orchestrator(respond, postprocess):
        await send_text(bot, ctx, st, chat_a, "msg-alpha")
        await send_text(bot, ctx, st, chat_b, "msg-beta")
        await drain(st)

    reply_a = next(m for m in bot.sent if m["chat_id"] == chat_a and m["text"].startswith("reply for"))
    confirm_a = next((m for m in bot.sent if m["chat_id"] == chat_a and m["text"].startswith("Save this memory?")), None)
    typing = [a for a in bot.actions if a["chat_id"] == chat_a and a["action"] == "typing"]
    typing_first = bool(typing) and typing[0]["t"] <= reply_a["t"]
    ordering = confirm_a is not None and reply_a["t"] < confirm_a["t"]
    leaked = any(m["chat_id"] == chat_b and "msg-alpha" in m["text"] for m in bot.sent)
    ok = typing_first and ordering and not leaked
    return row("B5", "ordering + typing + no cross-chat leak",
               "typing first; reply before confirm; A never sent to B",
               f"typing_first={typing_first}, reply_t={reply_a['t']:.2f}, "
               f"confirm_t={confirm_a['t'] if confirm_a else None}, leaked={leaked}",
               ok, f"confirm_present={confirm_a is not None}")


# ---------------------------------------------------------------------------
# B6
# ---------------------------------------------------------------------------
async def b6() -> dict:
    chat_a = next_chat()
    chat_b = next_chat()
    uid_a = await db_add_user(chat_a)
    uid_b = await db_add_user(chat_b)
    rec1 = await db_add_record(uid_a, "record one")
    rec2 = await db_add_record(uid_a, "record two")
    rec3 = await db_add_record(uid_a, "record three")
    bot, ctx, st = make_env()

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        return ReplyResult(turn_id="t", session_id=session_id, reply="ok", memory_used=[])

    async def postprocess(user_id, reply, *, now=None):
        return PostResult(pending_records=[
            PendingRecord(id=rec1, content="record one", kind="knowledge", certainty="tentative", source="text"),
            PendingRecord(id=rec2, content="record two", kind="knowledge", certainty="tentative", source="text"),
        ])

    with mock_orchestrator(respond, postprocess):
        await send_text(bot, ctx, st, chat_a, "note")
        await drain(st)
        await press(bot, ctx, st, chat_a, f"rec:ok:{rec1}")
        s1 = await db_record_status(rec1)
        saved_edit = any(e["text"] == "Saved ✅\nrecord one" for e in bot.edits)
        await press(bot, ctx, st, chat_a, f"rec:ok:{rec1}")  # idempotent
        s1b = await db_record_status(rec1)
        await press(bot, ctx, st, chat_a, f"rec:no:{rec2}")
        s2 = await db_record_status(rec2)
        # user B pressing A's record
        before_b = bot.edits[-1]
        await press(bot, ctx, st, chat_b, f"rec:ok:{rec3}")
        s3 = await db_record_status(rec3)
        b_edit = bot.edits[-1]["text"]

    ok = (
        s1 == "confirmed" and s1b == "confirmed" and saved_edit
        and s2 == "rejected" and s3 == "pending" and b_edit == "That item is no longer available"
    )
    return row("B6", "record confirm/discard + isolation",
               "ok->confirmed; ok again unchanged; no->rejected; B blocked",
               f"rec1={s1}/{s1b}, rec2={s2}, rec3={s3}, B_edit={b_edit!r}",
               ok, f"saved_edit={saved_edit}")


# ---------------------------------------------------------------------------
# B7
# ---------------------------------------------------------------------------
async def b7() -> dict:
    chat_a = next_chat()
    chat_b = next_chat()
    uid_a = await db_add_user(chat_a)
    uid_b = await db_add_user(chat_b)
    bot, ctx, st = make_env()
    due = datetime(2026, 10, 25, 9, 0, tzinfo=IST)

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        return ReplyResult(turn_id="t", session_id=session_id, reply="ok", memory_used=[])

    async def postprocess(user_id, reply, *, now=None):
        return PostResult(pending_tasks=[
            CandidateTask(title="Send Arun the PDF", due_at=due, origin="promise"),
            CandidateTask(title="Book the venue", due_at=None, origin="promise"),
        ])

    with mock_orchestrator(respond, postprocess):
        await send_text(bot, ctx, st, chat_a, "i'll do two things")
        await drain(st)
        tasks = await db_tasks_for(uid_a)
        pending = [t for t in tasks if t.status == "pending"]
        promise_msgs = [m for m in bot.sent if m["text"].startswith("Save this promise?")]
        t_first = pending[0].id if pending else None
        t_second = pending[1].id if len(pending) > 1 else None
        # simulated restart: fresh state + fresh handler objects
        bot2 = FakeBot()
        ctx2 = FakeContext(bot2, {"state": BotState()})
        if t_first is not None:
            await press(bot2, ctx2, ctx2.bot_data["state"], chat_a, f"tk:ok:{t_first}")
        s_first = await db_task_status(t_first) if t_first else None
        saved_task_edit = any(
            e["text"] == "Saved ✅\nSend Arun the PDF, due 2026-10-25 09:00 IST (Asia/Kolkata)"
            for e in bot2.edits
        )
        if t_second is not None:
            await press(bot2, ctx2, ctx2.bot_data["state"], chat_a, f"tk:no:{t_second}")
        s_second = await db_task_status(t_second) if t_second else None
        # user B on A's new pending task
        t_third = await db_add_pending_task(uid_a, "A private task")
        await press(bot2, ctx2, ctx2.bot_data["state"], chat_b, f"tk:ok:{t_third}")
        s_third = await db_task_status(t_third)
        b_edit = bot2.edits[-1]["text"] if bot2.edits else ""

    ok = (
        len(pending) == 2 and len(promise_msgs) == 2
        and s_first == "open" and s_second == "MISSING"
        and s_third == "pending" and b_edit == "That item is no longer available"
        and saved_task_edit
    )
    due_msg = next((m["text"] for m in promise_msgs if "Send Arun" in m["text"]), "")
    return row("B7", "task stage -> confirm/discard (restart)",
               "2 pending rows+messages; ok->open (Saved keeps title+due); no->deleted; B blocked",
               f"pending={len(pending)}, msgs={len(promise_msgs)}, first={s_first}, second={s_second}, "
               f"third={s_third}, B_edit={b_edit!r}",
               ok, f"due_msg={due_msg!r}, saved_task_edit={saved_task_edit}")


# ---------------------------------------------------------------------------
# B8
# ---------------------------------------------------------------------------
async def b8() -> dict:
    chat_a = next_chat()
    uid_a = await db_add_user(chat_a)
    chat_c = next_chat()
    uid_c = await db_add_user(chat_c)
    recs = [await db_add_record(uid_a, f"batch record {i}") for i in range(4)]
    rec_c = await db_add_record(uid_c, "not yours")
    bot, ctx, st = make_env()

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        return ReplyResult(turn_id="t", session_id=session_id, reply="ok", memory_used=[])

    async def postprocess(user_id, reply, *, now=None):
        return PostResult(pending_records=[
            PendingRecord(id=r, content=f"batch record {i}", kind="knowledge",
                          certainty="tentative", source="text")
            for i, r in enumerate(recs)
        ])

    with mock_orchestrator(respond, postprocess):
        await send_text(bot, ctx, st, chat_a, "four notes")
        await drain(st)
        batch_msgs = [m for m in bot.sent if (m["text"] or "").startswith("4 memories to review:")]
        single_msgs = [m for m in bot.sent if (m["text"] or "").startswith("Save this memory?")]
        batch_text = batch_msgs[0]["text"] if batch_msgs else ""
        # the batch message lists each item, showing the first 80 chars of its content
        lists_all = bool(batch_text) and all(f"batch record {i}" in batch_text for i in range(4))
        # press Save all, tampered csv to include another user's id (ownership must hold)
        csv = ",".join(str(r) for r in recs + [rec_c])
        await press(bot, ctx, st, chat_a, f"rec:allok:{csv}")
        statuses = [await db_record_status(r) for r in recs]
        status_c = await db_record_status(rec_c)
        saved_edit = any(e["text"] == "Saved 4 memories ✅" for e in bot.edits)

    cbs = all_callbacks(bot)
    max_cb = max((len(c.encode("utf-8")) for c in cbs), default=0)
    ok = (
        len(batch_msgs) == 1 and len(single_msgs) == 0 and lists_all
        and all(s == "confirmed" for s in statuses) and status_c == "pending" and saved_edit
        and max_cb <= 64
    )
    return row("B8", "batch records + owned-only + cb<=64",
               "1 batch msg listing all 4; 4 confirmed; C untouched; all cb<=64 bytes",
               f"batch_msgs={len(batch_msgs)}, single={len(single_msgs)}, lists_all={lists_all}, "
               f"statuses={statuses}, C={status_c}, max_cb_bytes={max_cb}",
               ok, f"saved_edit={saved_edit}, batch_text={batch_text!r}")


# ---------------------------------------------------------------------------
# B9
# ---------------------------------------------------------------------------
async def b9() -> dict:
    chat_a = next_chat()
    uid_a = await db_add_user(chat_a)
    chat_c = next_chat()
    uid_c = await db_add_user(chat_c)
    own = await db_add_record(uid_a, "SECRET-ALPHA-CONTENT", status="confirmed", kind="decision")
    other = await db_add_record(uid_c, "OTHER-USER-CONTENT", status="confirmed")
    bot, ctx, st = make_env()

    await press(bot, ctx, st, chat_a, f"src:{own}")
    owned_msgs = [m["text"] for m in bot.sent if m["chat_id"] == chat_a]
    owned_shown = any("SECRET-ALPHA-CONTENT" in t and "kind: decision" in t for t in owned_msgs)

    bot2 = FakeBot()
    ctx2 = FakeContext(bot2, {"state": BotState()})
    await press(bot2, ctx2, ctx2.bot_data["state"], chat_a, f"src:{other}")
    leaked = any("OTHER-USER-CONTENT" in m["text"] for m in bot2.sent)
    not_avail = any(m["text"] == "That item is no longer available" for m in bot2.sent)
    missing_id = 999999999
    await press(bot2, ctx2, ctx2.bot_data["state"], chat_a, f"src:{missing_id}")
    missing_ok = any(m["text"] == "That item is no longer available" for m in bot2.sent)

    ok = owned_shown and not_avail and (not leaked) and missing_ok
    return row("B9", "Show source ownership",
               "owned shows content+kind; other's/missing = not available, no leak",
               f"owned_shown={owned_shown}, other_not_avail={not_avail}, leaked={leaked}, missing_ok={missing_ok}",
               ok, "")


# ---------------------------------------------------------------------------
# B10 (a-d)
# ---------------------------------------------------------------------------
async def b10a() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()

    async def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    with mock_orchestrator(boom, make_postprocess()):
        try:
            await send_text(bot, ctx, st, chat, "please fail")
            raised = False
        except Exception:
            raised = True
    friendly = any(m["text"] == handlers.GENERIC_ERROR for m in bot.sent)
    ok = (not raised) and friendly
    return row("B10a", "respond raises -> friendly, no raise",
               "no exception; generic error message sent",
               f"raised={raised}, friendly_sent={friendly}", ok, "")


async def b10b() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    with mock_orchestrator(make_respond(reply_text=""), make_postprocess()):
        await send_text(bot, ctx, st, chat, "empty reply please")
        await drain(st)
    texts = [m["text"] for m in bot.sent if m["chat_id"] == chat]
    ok = any(orchestrator.FALLBACK_REPLY in t for t in texts)
    return row("B10b", "empty reply -> FALLBACK",
               f"reply contains {orchestrator.FALLBACK_REPLY!r}",
               f"texts={texts}", ok, "")


async def b10c() -> dict:
    chat = next_chat()
    uid = await db_add_user(chat)
    conf = await db_add_record(uid, "stays confirmed", status="confirmed")
    bot, ctx, st = make_env()
    with mock_orchestrator(make_respond(), make_postprocess()):
        await send_text(bot, ctx, st, chat, "first turn")
        await drain(st)
        st.chat(chat).history.append({"role": "user", "content": "leftover"})
        old_sid = st.chat(chat).session_id
        await handlers.on_newsession(FakeUpdate(chat, text="/newsession"), ctx, state=st)
        new_sid = st.chat(chat).session_id
        history_len = len(st.chat(chat).history)
        conf_status = await db_record_status(conf)
    ok = new_sid is not None and new_sid != old_sid and history_len == 0 and conf_status == "confirmed"
    return row("B10c", "/newsession resets session",
               "new session id; history cleared; confirmed record intact",
               f"old={old_sid}, new={new_sid}, history_len={history_len}, conf={conf_status}", ok, "")


async def b10d() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    captured = {"source": None, "text": None}

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        captured["source"] = source
        captured["text"] = text
        return ReplyResult(
            turn_id="turn-voice", session_id=session_id, reply="mocked reply", memory_used=[]
        )

    old_transcribe = handlers.audio_service.transcribe_bytes
    handlers.audio_service.transcribe_bytes = lambda data: "hello from voice"
    try:
        with mock_orchestrator(respond, make_postprocess()):
            await handlers.on_voice(FakeUpdate(chat, voice=True), ctx, state=st)
            await drain(st)
    finally:
        handlers.audio_service.transcribe_bytes = old_transcribe

    texts = [m["text"] for m in bot.sent if m["chat_id"] == chat]
    heard_first = bool(texts) and texts[0] == "Heard: hello from voice"
    replied = any(m["text"] == "mocked reply" for m in bot.sent if m["chat_id"] == chat)
    ok = (
        heard_first
        and replied
        and captured["source"] == "voice"
        and captured["text"] == "hello from voice"
    )
    return row("B10d", "voice -> transcribe then reply",
               'transcribed; "Heard: ..." first, then reply with source="voice"',
               f"texts={texts}, source={captured['source']}, text={captured['text']!r}", ok, "")


# ---------------------------------------------------------------------------
# B13: empty / whitespace-only text makes zero orchestrator calls
# ---------------------------------------------------------------------------
async def b13() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    counter = [0]
    with mock_orchestrator(make_respond(counter=counter), make_postprocess()):
        for text in ("", "   ", "\n\t  \n"):
            await send_text(bot, ctx, st, chat, text)
        await drain(st)
    sent_texts = [m["text"] for m in bot.sent if m["chat_id"] == chat]
    ok = counter[0] == 0 and sent_texts == []
    return row("B13", "empty/whitespace text -> no LLM call",
               "0 orchestrator calls, nothing sent",
               f"orchestrator_calls={counter[0]}, sent={sent_texts}",
               ok, "")


# ---------------------------------------------------------------------------
# B14: two updates from two chats are processed concurrently
# ---------------------------------------------------------------------------
async def b14() -> dict:
    chat_a = next_chat()
    chat_b = next_chat()
    uid_a = await db_add_user(chat_a)
    uid_b = await db_add_user(chat_b)
    bot, ctx, st = make_env()

    tag_by_uid = {uid_a: "a", uid_b: "b"}
    events: dict[str, dict] = {"a": {}, "b": {}}
    order: list[str] = []

    async def slow_respond(user_id, text, *, session_id, history, source="text", now=None):
        tag = tag_by_uid.get(user_id, "?")
        events[tag]["start"] = time.perf_counter()
        order.append(f"{tag}:start")
        await asyncio.sleep(0.5)
        events[tag]["end"] = time.perf_counter()
        order.append(f"{tag}:end")
        return ReplyResult(turn_id=f"turn-{tag}", session_id=session_id, reply="ok", memory_used=[])

    t0 = time.perf_counter()
    with mock_orchestrator(slow_respond, make_postprocess()):
        await asyncio.gather(
            send_text(bot, ctx, st, chat_a, "hi from A"),
            send_text(bot, ctx, st, chat_b, "hi from B"),
        )
        await drain(st)
    elapsed = time.perf_counter() - t0

    # The second respond must START before the first respond ENDS. If the two chats were
    # serialized the order would be a:start, a:end, b:start, b:end.
    concurrent = order[:2] in (["a:start", "b:start"], ["b:start", "a:start"])
    serial_elapsed = 1.0  # two 0.5s sleeps back-to-back (measured separately from DB setup)
    ok = concurrent
    return row("B14", "two chats processed concurrently",
               "both respond calls start before either finishes",
               f"concurrent={concurrent}, order={order}, elapsed={elapsed:.2f}s (serial sleeps would be ~{serial_elapsed:.2f}s)",
               ok, "")


# ---------------------------------------------------------------------------
# B15 (LaTeX stripping, reached through strip_markdown -> the _send_reply hook)
# ---------------------------------------------------------------------------
async def b15() -> dict:
    checks: list[tuple[str, bool, str]] = []

    def check(label: str, got: str, want: str) -> None:
        checks.append((label, got == want, f"got={got!r} want={want!r}"))

    # 1
    check("1 x^2", textfmt.strip_markdown(r"\(x^2\)"), "x^2")
    # 2
    check("2 frac", textfmt.strip_markdown(r"\(\frac{a}{b}\)"), "(a)/(b)")
    # 3
    check("3 sqrt", textfmt.strip_markdown(r"\(\sqrt{x}\)"), "sqrt(x)")

    # 4
    case4 = r"Formally \(f'(x)=\displaystyle\lim_{h\to0}\frac{f(x+h)-f(x)}{h}\)."
    out4 = textfmt.strip_markdown(case4)
    ok4 = (
        "\\" not in out4
        and "\\(" not in out4
        and "lim(h→0)" in out4
        and "/(h)" in out4
    )
    checks.append(("4 formula", ok4, f"out={out4!r}"))

    # 5
    case5 = r"Path C:\Users\x costs $5"
    check("5 path/price", textfmt.strip_markdown(case5), case5)

    # 6
    check("6 plain", textfmt.strip_markdown("plain text, no math"), "plain text, no math")

    # 7
    case7 = r"broken \(x^2 with no close"
    try:
        out7 = textfmt.strip_markdown(case7)
        checks.append(("7 unclosed", out7 == case7, f"out={out7!r}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("7 unclosed", False, f"raised {exc!r}"))

    # 8
    once = textfmt.strip_markdown(case4)
    twice = textfmt.strip_markdown(once)
    checks.append(("8 idempotent", once == twice, f"once={once!r} twice={twice!r}"))

    ok = all(good for _, good, _ in checks)
    failed = [label for label, good, _ in checks if not good]
    details = " | ".join(f"{label}: {'ok' if good else 'FAIL'} ({info})" for label, good, info in checks)
    return row("B15", "LaTeX math stripped in replies",
               "all 8 latex cases match exact expected output",
               f"failed={failed if failed else 'none'}",
               ok, details)


# ---------------------------------------------------------------------------
# B11 (REAL end-to-end)
# ---------------------------------------------------------------------------
async def b11() -> list[dict]:
    chat = next_chat()
    bot, ctx, st = make_env()
    captured: dict = {}
    real_respond = orchestrator.respond
    real_postprocess = orchestrator.postprocess

    async def recording_respond(*args, **kwargs):
        replied = await real_respond(*args, **kwargs)
        captured["reply"] = replied
        return replied

    async def recording_postprocess(*args, **kwargs):
        post = await real_postprocess(*args, **kwargs)
        captured["post"] = post
        return post

    rows: list[dict] = []
    with patched_settings(INVITE_CODES="TESTCODE", BOT_MAX_USERS=10_000_000):
        await handlers.cmd_start(FakeUpdate(chat, text="/start TESTCODE"), ctx, state=st)
    uid = await db_user_id_for_chat(chat)

    with mock_orchestrator(recording_respond, recording_postprocess):
        await send_text(bot, ctx, st, chat, "help me understand mathematical functions")
        reply1 = next((m for m in bot.sent if m["chat_id"] == chat and m["t"] > 0 and "Save this" not in m["text"]
                       and m["text"] != handlers.WELCOME), None)
        reply1_t = reply1["t"] if reply1 else None
        await drain(st)

    confirm_msg = next((m for m in bot.sent if m["text"].startswith("Save this memory?")), None)
    delta = (confirm_msg["t"] - reply1_t) if (confirm_msg and reply1_t is not None) else None

    confirmed_id = None
    if confirm_msg is not None:
        cbs = callbacks_of(confirm_msg["reply_markup"])
        ok_cb = next((c for c in cbs if c.startswith("rec:ok:")), None)
        if ok_cb:
            await press(bot, ctx, st, chat, ok_cb)
            confirmed_id = int(ok_cb.split(":")[-1])
    status_after = await db_record_status(confirmed_id) if confirmed_id else None

    post1 = captured.get("post")
    rows.append(row(
        "B11a", "real: first turn -> record prompt",
        "reply sent, follow-up pending-record prompt appears",
        f"reply_sent={reply1 is not None}, confirm_prompt={'yes' if confirm_msg else 'no'}, "
        f"delta_after_reply={f'{delta:.2f}s' if delta is not None else 'n/a'}",
        bool(reply1) and confirm_msg is not None,
        f"pending_records={len(post1.pending_records) if post1 else 'NO_POST'}, "
        f"pending_tasks={len(post1.pending_tasks) if post1 else 'NO_POST'}, "
        f"intent={getattr(post1, 'intent', None)}, "
        f"extraction_failed={getattr(post1, 'extraction_failed', None)}, "
        f"skipped={getattr(getattr(post1, 'trace', None), 'skipped', None)}, "
        f"errors={getattr(getattr(post1, 'trace', None), 'errors', None)}, "
        f"extract_failed_msg_sent={any(m['text'] == handlers.EXTRACTION_FAILED for m in bot.sent)}",
    ))
    rows.append(row(
        "B11b", "real: confirm record",
        "pressing ✅ sets status confirmed",
        f"confirmed_id={confirmed_id}, status={status_after}",
        status_after == "confirmed",
        "",
    ))

    with mock_orchestrator(recording_respond, recording_postprocess):
        await handlers.on_newsession(FakeUpdate(chat, text="/newsession"), ctx, state=st)
        await send_text(bot, ctx, st, chat, "I'm stuck on a calculus problem")
        await drain(st)

    reply2 = captured.get("reply")
    mem = list(getattr(reply2, "memory_used", []) or [])
    mem_ids = [m.record_id for m in mem]
    mentions_function = bool(reply2 and "function" in (reply2.reply or "").lower())

    last_msgs = [m for m in bot.sent if m["chat_id"] == chat and m["text"] != handlers.NEW_SESSION]
    footer_present = any("relevance " in m["text"] for m in last_msgs)
    src_button = any(
        any(c.startswith("src:") for c in callbacks_of(m.get("reply_markup")))
        for m in last_msgs
    )
    mem_has_confirmed = confirmed_id is not None and confirmed_id in mem_ids

    rows.append(row(
        "B11c", "real: second turn recall",
        "footer + Show source present; memory_used has confirmed record",
        f"footer={footer_present}, src_button={src_button}, mem_ids={mem_ids}, "
        f"confirmed_id={confirmed_id}, in_memory={mem_has_confirmed}",
        footer_present and src_button and mem_has_confirmed,
        f"mentions_function={mentions_function} (MEASURE ONLY)",
    ))
    return rows


# ---------------------------------------------------------------------------
# B12
# ---------------------------------------------------------------------------
async def b12() -> dict:
    module = importlib.import_module("backend.bot.__main__")
    module.configure_logging()
    levels = {
        name: logging.getLogger(name).level
        for name in ("httpx", "httpx2", "httpcore", "telegram")
    }
    ok = all(lvl >= logging.WARNING for lvl in levels.values())
    return row("B12", "logging silences token-bearing libs",
               "httpx/httpx2/httpcore/telegram level >= WARNING",
               f"levels={levels}", ok, "")


# ---------------------------------------------------------------------------
# B16
# ---------------------------------------------------------------------------
async def b16() -> dict:
    """A non-ASCII invite code gets the normal refusal, never the generic error handler."""
    chat = next_chat()
    bot, ctx, st = make_env()
    escaped = None
    with patched_settings(INVITE_CODES="TESTCODE"):
        try:
            await handlers.cmd_start(FakeUpdate(chat, text="/start é"), ctx, state=st)
        except Exception as exc:  # noqa: BLE001 - this is what would reach on_error in production
            escaped = type(exc).__name__
        texts = [m["text"] for m in bot.sent if m["chat_id"] == chat]
    uid = await db_user_id_for_chat(chat)
    refusal_sent = texts == [handlers.REFUSAL]
    generic_sent = any(t == handlers.GENERIC_ERROR for t in texts)
    ok = escaped is None and refusal_sent and not generic_sent and uid is None
    return row("B16", "non-ASCII invite code",
               "refusal reply, no generic error, no user row",
               f"escaped={escaped}, refusal_sent={refusal_sent}, "
               f"generic_sent={generic_sent}, row={uid}",
               ok, "")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def main() -> int:
    settings.require("DATABASE_URL")
    print("=" * 140)
    print("PHASE 4b: TELEGRAM BOT HANDLERS EVALUATION (text + voice)")
    print("=" * 140)

    results: list[dict] = []
    simple_cases = [
        b1, b2, b3, b4, b5, b6, b7, b8, b9, b10a, b10b, b10c, b10d, b12, b13, b14, b15, b16,
    ]
    try:
        for case in simple_cases:
            print(f"--- Running {case.__name__.upper()} ---")
            try:
                res = await case()
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                res = row(case.__name__.upper(), case.__name__, "-", "", False, f"unhandled: {exc!r}")
            results.append(res)
            print(f"    [{'PASS' if res['ok'] else 'FAIL'}] {res['name']}: {res['actual']}")

        print("--- Running B11 (REAL end-to-end) ---")
        try:
            b11_rows = await b11()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            b11_rows = [row("B11", "real end-to-end", "-", "", False, f"unhandled: {exc!r}")]
        for res in b11_rows:
            results.append(res)
            print(f"    [{'PASS' if res['ok'] else 'FAIL'}] {res['name']}: {res['actual']}")
    finally:
        await cleanup_users()
        await engine.dispose()

    print("\n" + "=" * 140)
    print(f"{'ID':<6} | {'NAME':<42} | {'EXPECTED':<52} | {'ACTUAL':<52} | {'STATUS':<6} | DETAILS")
    print("-" * 140)
    all_passed = True
    for r in results:
        if not r["ok"]:
            all_passed = False
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{r['id']:<6} | {r['name']:<42} | {r['expected']:<52} | {r['actual']:<52} | {status:<6} | {r['details']}")
    print("=" * 140)
    passed = sum(1 for r in results if r["ok"])
    print(f"RESULT: {'ALL PASS' if all_passed else 'FAILURES PRESENT'} ({passed}/{len(results)})")
    print(f"B11 mention check (MEASURE ONLY, not a gate): {next((r['details'] for r in results if r['id'] == 'B11c'), '')}")
    return 0 if all_passed else 1


async def cleanup_users() -> None:
    if not CHAT_IDS:
        return
    try:
        async with AsyncSessionLocal() as session:
            for chat_id in list(CHAT_IDS):
                user = (
                    await session.execute(select(User).where(User.telegram_chat_id == chat_id))
                ).scalar_one_or_none()
                if user is not None:
                    await session.delete(user)
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"   [WARN] cleanup failed: {exc!r}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
