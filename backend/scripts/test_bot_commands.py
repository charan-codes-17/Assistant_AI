"""backend/scripts/test_bot_commands.py
Phase 4c-1 evaluation for the Telegram bot commands /help, /memory and /forget.

Fakes: a FakeBot recording send_message / edit_message_text / answer_callback_query /
send_chat_action, plus simple fake update/context/callback objects. No Telegram network is used.

The real dev DB is used with synthetic users whose telegram_chat_id is in 980000-989999; those
users are purged at the START and again in a finally block. Embeddings are faked (deterministic,
zero network) except where a test explicitly blocks them. No LLM call is ever allowed here.

C1-C12 assert exact expectations; the process exits non-zero on any failure and prints the exit
code on the final EXIT line (added by the caller).

Run:
  backend\\venv\\Scripts\\python.exe -u backend\\scripts\\test_bot_commands.py
"""

import asyncio
import inspect
import logging
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import traceback
import unittest.mock
import uuid

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import func, select

import backend.services.embeddings as embeddings_module
import backend.services.llm as llm_module
from backend.bot import handlers
from backend.bot.state import BotState
from backend.config.settings import settings
from backend.database import AsyncSessionLocal, engine
from backend.models.models import (
    ConversationSession,
    Goal,
    InspectorEvent,
    KnowledgeRecord,
    LinkCode,
    Milestone,
    ProfileSummary,
    Progress,
    Project,
    Task,
    User,
)
from backend.services import orchestrator

CHAT_MIN = 980000
CHAT_MAX = 989999

# Tables that carry a user_id (deleted directly by the user-row cascade).
USER_SCOPED_MODELS = [
    KnowledgeRecord,
    Task,
    Goal,
    Project,
    ConversationSession,
    ProfileSummary,
    LinkCode,
]
# Every table, for the C10 "zero DB writes" snapshot.
COUNT_MODELS = USER_SCOPED_MODELS + [User]

CHAT_IDS: set[int] = set()
_next_chat = [CHAT_MIN]


def next_chat() -> int:
    _next_chat[0] += 1
    if _next_chat[0] > CHAT_MAX:
        raise RuntimeError("synthetic chat id range exhausted")
    CHAT_IDS.add(_next_chat[0])
    return _next_chat[0]


# ---------------------------------------------------------------------------
# Log capture (privacy, C12)
# ---------------------------------------------------------------------------
LOG_RECORDS: list[logging.LogRecord] = []
# [{"case": "C1", "start": int, "end": int, "markers": [str, ...]}, ...]
PRIVACY: list[dict] = []


class ListHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        LOG_RECORDS.append(record)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeChat:
    def __init__(self, chat_id: int, chat_type: str = "private") -> None:
        self.id = chat_id
        self.type = chat_type


class FakeMessage:
    def __init__(self, chat_id: int, message_id: int = 500, text=None) -> None:
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
    def __init__(self, chat_id: int, text=None, callback=None) -> None:
        self.effective_chat = FakeChat(chat_id)
        self.effective_user = FakeChat(chat_id)
        self.message = FakeMessage(chat_id, text=text)
        self.callback_query = callback


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.answers: list[str] = []
        self.actions: list[dict] = []

    async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
        self.sent.append({"chat_id": chat_id, "text": text, "reply_markup": reply_markup})
        return None

    async def send_chat_action(self, chat_id, action, **kwargs):
        self.actions.append({"chat_id": chat_id, "action": action})

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


def _unit_vector() -> list[float]:
    """A fixed, non-zero embedding vector (cosine distance is undefined for zero vectors)."""
    vec = [0.0] * settings.EMBEDDING_DIM
    vec[0] = 1.0
    return vec


def callbacks_of(markup) -> list[str]:
    rows = getattr(markup, "inline_keyboard", None) or []
    out: list[str] = []
    for row in rows:
        for button in row:
            cb = getattr(button, "callback_data", None)
            if cb is not None:
                out.append(cb)
    return out


def memory_lines(bot: FakeBot, chat_id: int) -> list[str]:
    """The numbered "/memory" result lines sent to one chat, in send order."""
    out: list[str] = []
    for msg in bot.sent:
        if msg["chat_id"] != chat_id:
            continue
        text = msg["text"] or ""
        if re.match(r"^\d+\. ", text):
            out.append(text)
    return out


async def press(bot: FakeBot, ctx: FakeContext, st: BotState, chat_id: int, data: str) -> None:
    await handlers.on_callback(
        FakeUpdate(chat_id, callback=FakeCallbackQuery(chat_id, data)), ctx, state=st
    )


def row(rid, name, expected, actual, ok, details="") -> dict:
    return {
        "id": rid,
        "name": name,
        "expected": expected,
        "actual": actual,
        "ok": bool(ok),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------
@contextmanager
def fake_embeddings(counter: list[int]):
    """Replace embeddings.embed_texts with a deterministic fake that counts calls."""

    async def _fake(texts, timeout=None):
        counter[0] += 1
        return [_unit_vector() for _ in texts]

    old = embeddings_module.embed_texts
    embeddings_module.embed_texts = _fake
    try:
        yield
    finally:
        embeddings_module.embed_texts = old


@contextmanager
def block_llm():
    """Make every public coroutine of the llm module raise if it is ever called."""
    patchers = []
    for name in dir(llm_module):
        if name.startswith("_"):
            continue
        attr = getattr(llm_module, name)
        if inspect.iscoroutinefunction(attr):

            async def _blocked(*a, _n: str = name, **k):
                raise AssertionError(f"LLM call must not happen: {_n}")

            patchers.append(unittest.mock.patch.object(llm_module, name, _blocked))
    try:
        for p in patchers:
            p.start()
        yield
    finally:
        for p in patchers:
            p.stop()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
async def db_add_user(chat_id: int) -> int:
    async with AsyncSessionLocal() as session:
        user = User(
            username=f"cmd_{uuid.uuid4().hex[:8]}",
            telegram_chat_id=chat_id,
            timezone="Asia/Kolkata",
        )
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


async def db_add_record_full(
    user_id: int,
    content: str,
    *,
    status: str = "confirmed",
    kind: str = "knowledge",
    certainty: str = "tentative",
    source: str = "text",
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> int:
    async with AsyncSessionLocal() as session:
        kwargs = dict(
            user_id=user_id,
            kind=kind,
            content=content,
            certainty=certainty,
            source=source,
            status=status,
            expires_at=expires_at,
            embedding=_unit_vector(),
        )
        if created_at is not None:
            kwargs["created_at"] = created_at
        rec = KnowledgeRecord(**kwargs)
        session.add(rec)
        await session.commit()
        await session.refresh(rec)
        return rec.id


async def db_record_exists(record_id: int) -> bool:
    async with AsyncSessionLocal() as session:
        return (await session.get(KnowledgeRecord, record_id)) is not None


async def db_task_status(task_id: int) -> str | None:
    async with AsyncSessionLocal() as session:
        task = await session.get(Task, task_id)
        return task.status if task else "MISSING"


async def db_count_by_user(model, user_id: int) -> int:
    async with AsyncSessionLocal() as session:
        stmt = select(func.count()).select_from(model).where(model.user_id == user_id)
        return (await session.execute(stmt)).scalar_one()


async def db_count_where(model, expr) -> int:
    async with AsyncSessionLocal() as session:
        stmt = select(func.count()).select_from(model).where(expr)
        return (await session.execute(stmt)).scalar_one()


async def snapshot_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    async with AsyncSessionLocal() as session:
        for model in COUNT_MODELS:
            stmt = select(func.count()).select_from(model)
            out[model.__tablename__] = (await session.execute(stmt)).scalar_one()
    return out


async def seed_all(uid: int) -> dict:
    """Create one row in every user-scoped table plus the transitive children for /forget."""
    async with AsyncSessionLocal() as session:
        goal = Goal(user_id=uid, title="C7 goal")
        project = Project(user_id=uid, title="C7 project")
        task = Task(user_id=uid, title="C7 task", status="open", origin="manual")
        convo = ConversationSession(user_id=uid, channel="telegram")
        link = LinkCode(
            user_id=uid,
            code=f"C7{uuid.uuid4().hex[:10]}",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        prof = ProfileSummary(user_id=uid, blob="C7SECRET profile blob")
        session.add_all([goal, project, task, convo, link, prof])
        await session.commit()
        await session.refresh(project)
        await session.refresh(convo)

        milestone = Milestone(project_id=project.id, title="C7 milestone")
        progress = Progress(project_id=project.id, note="C7SECRET progress note")
        event = InspectorEvent(session_id=convo.id, turn_id=1, directive="none")
        session.add_all([milestone, progress, event])
        await session.commit()
        return {"project_id": project.id, "session_id": convo.id}


async def purge_synthetic() -> int:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(User).where(
                    (User.telegram_chat_id >= CHAT_MIN) & (User.telegram_chat_id <= CHAT_MAX)
                )
            )
        ).scalars().all()
        rows += (
            await session.execute(select(User).where(User.username.like("cmd_%")))
        ).scalars().all()
        for user in rows:
            await session.delete(user)
        await session.commit()
        return len(rows)


# ---------------------------------------------------------------------------
# C1: /memory <topic> -> only this user's confirmed, active records
# ---------------------------------------------------------------------------
async def c1() -> dict:
    start = len(LOG_RECORDS)
    chat = next_chat()
    chat_b = next_chat()
    uid = await db_add_user(chat)
    uid_b = await db_add_user(chat_b)
    await db_add_record_full(
        uid, "C1CONF-A knowledge alpha", status="confirmed", kind="knowledge",
        certainty="decided", source="text",
    )
    await db_add_record_full(
        uid, "C1CONF-B decision beta", status="confirmed", kind="decision",
        certainty="tentative", source="voice",
    )
    await db_add_record_full(uid, "C1PENDING-C hidden", status="pending")
    await db_add_record_full(uid, "C1REJECTED-D hidden", status="rejected")
    await db_add_record_full(
        uid, "C1EXPIRED-E hidden", status="confirmed",
        expires_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    await db_add_record_full(uid_b, "C1OTHER-F hidden", status="confirmed")

    bot, ctx, st = make_env()
    counter = [0]
    with fake_embeddings(counter):
        await handlers.cmd_memory(FakeUpdate(chat, text="/memory travel plans"), ctx, state=st)

    lines = memory_lines(bot, chat)
    blob = "\n".join((m["text"] or "") for m in bot.sent) + "\n" + "\n".join(
        (e["text"] or "") for e in bot.edits
    )
    date_re = re.compile(r"\d{1,2} [A-Z][a-z]{2} \d{4}")
    content_ok = (
        len(lines) == 2
        and any(
            "C1CONF-A" in ln and "knowledge" in ln and "decided" in ln and "text" in ln
            for ln in lines
        )
        and any(
            "C1CONF-B" in ln and "decision" in ln and "tentative" in ln and "voice" in ln
            for ln in lines
        )
        and all(date_re.search(ln) for ln in lines)
    )
    hidden = ("C1PENDING-C", "C1REJECTED-D", "C1EXPIRED-E", "C1OTHER-F")
    no_leak = all(h not in blob for h in hidden)

    PRIVACY.append(
        {
            "case": "C1",
            "start": start,
            "end": len(LOG_RECORDS),
            "markers": [
                "C1CONF-A knowledge alpha",
                "C1CONF-B decision beta",
                "C1PENDING-C hidden",
                "C1REJECTED-D hidden",
                "C1EXPIRED-E hidden",
                "C1OTHER-F hidden",
            ],
        }
    )
    ok = content_ok and no_leak
    return row(
        "C1", "/memory <topic>: only own confirmed records",
        "2 lines with kind/certainty/source/date; pending/rejected/expired/other hidden",
        f"lines={len(lines)}, content_ok={content_ok}, no_leak={no_leak}",
        ok, f"lines={lines}",
    )


# ---------------------------------------------------------------------------
# C2: /memory (no topic) -> 5 newest confirmed, newest first
# ---------------------------------------------------------------------------
async def c2() -> dict:
    chat = next_chat()
    uid = await db_add_user(chat)
    base = datetime.now(timezone.utc)
    for i in range(6):
        await db_add_record_full(
            uid, f"C2REC-{i}", status="confirmed",
            created_at=base - timedelta(minutes=i),
        )

    bot, ctx, st = make_env()
    counter = [0]
    with fake_embeddings(counter):
        await handlers.cmd_memory(FakeUpdate(chat, text="/memory"), ctx, state=st)

    lines = memory_lines(bot, chat)
    order = [re.search(r"C2REC-\d", ln).group(0) for ln in lines if re.search(r"C2REC-\d", ln)]
    expected = ["C2REC-0", "C2REC-1", "C2REC-2", "C2REC-3", "C2REC-4"]
    ok = order == expected and len(lines) == 5 and counter[0] == 0
    return row(
        "C2", "/memory (no topic): 5 newest, newest first",
        "exactly [C2REC-0..C2REC-4] in that order; no embedding call",
        f"order={order}, lines={len(lines)}, embed_calls={counter[0]}",
        ok, "",
    )


# ---------------------------------------------------------------------------
# C3: /memory with no matches -> exact "nothing saved" message
# ---------------------------------------------------------------------------
async def c3() -> dict:
    chat = next_chat()
    uid = await db_add_user(chat)
    await db_add_record_full(uid, "C3PENDING only", status="pending")  # no confirmed rows

    bot, ctx, st = make_env()
    counter = [0]
    with fake_embeddings(counter):
        await handlers.cmd_memory(FakeUpdate(chat, text="/memory anything at all"), ctx, state=st)

    texts = [m["text"] for m in bot.sent if m["chat_id"] == chat]
    ok = texts == [handlers.MEMORY_EMPTY]
    return row(
        "C3", "/memory no matches -> nothing-saved message",
        f"exactly one message: {handlers.MEMORY_EMPTY!r}",
        f"texts={texts}",
        ok, f"expected={handlers.MEMORY_EMPTY!r}",
    )


# ---------------------------------------------------------------------------
# C4: two-step delete of one memory
# ---------------------------------------------------------------------------
async def c4() -> dict:
    start = len(LOG_RECORDS)
    chat = next_chat()
    chat_b = next_chat()
    uid = await db_add_user(chat)
    uid_b = await db_add_user(chat_b)
    rec = await db_add_record_full(uid, "C4SECRET delete me", status="confirmed")
    rec_keep = await db_add_record_full(uid, "C4SECRET keep me", status="confirmed")
    rec_other = await db_add_record_full(uid_b, "C4OTHER not yours", status="confirmed")
    rec_long = await db_add_record_full(uid, "L" * 150, status="confirmed")
    bot, ctx, st = make_env()

    # mem:del edits to the confirm step and deletes nothing
    await press(bot, ctx, st, chat, f"mem:del:{rec}")
    del_edit = bot.edits[-1]["text"]
    del_cbs = callbacks_of(bot.edits[-1]["reply_markup"])
    del_ok = (
        del_edit.startswith("Delete this memory? ")
        and del_cbs == [f"mem:yes:{rec}", f"mem:no:{rec}"]
        and await db_record_exists(rec)
    )

    # the preview is truncated to 100 chars
    await press(bot, ctx, st, chat, f"mem:del:{rec_long}")
    trunc_ok = bot.edits[-1]["text"] == f"Delete this memory? {'L' * 100}"

    # mem:no keeps the record
    await press(bot, ctx, st, chat, f"mem:no:{rec}")
    no_ok = bot.edits[-1]["text"] == handlers.CANCELLED and await db_record_exists(rec)

    # mem:yes deletes it
    await press(bot, ctx, st, chat, f"mem:yes:{rec}")
    yes_ok = bot.edits[-1]["text"] == handlers.MEMORY_DELETED and not await db_record_exists(rec)

    # pressing yes twice does not raise
    raised = False
    try:
        await press(bot, ctx, st, chat, f"mem:yes:{rec}")
    except Exception:  # noqa: BLE001
        raised = True
    twice_ok = (not raised) and bot.edits[-1]["text"] == handlers.NOT_AVAILABLE

    # another user's id -> NOT_AVAILABLE and their record remains
    await press(bot, ctx, st, chat, f"mem:del:{rec_other}")
    other_del_ok = bot.edits[-1]["text"] == handlers.NOT_AVAILABLE and await db_record_exists(rec_other)
    await press(bot, ctx, st, chat, f"mem:yes:{rec_other}")
    other_yes_ok = bot.edits[-1]["text"] == handlers.NOT_AVAILABLE and await db_record_exists(rec_other)

    keep_ok = await db_record_exists(rec_keep)
    PRIVACY.append(
        {
            "case": "C4",
            "start": start,
            "end": len(LOG_RECORDS),
            "markers": ["C4SECRET delete me", "C4SECRET keep me", "C4OTHER not yours", "L" * 100],
        }
    )
    ok = (
        del_ok and trunc_ok and no_ok and yes_ok and twice_ok
        and other_del_ok and other_yes_ok and keep_ok
    )
    return row(
        "C4", "mem:del/mem:no/mem:yes + ownership",
        "del shows confirm & deletes nothing; no keeps; yes deletes; yes twice safe; other blocked",
        f"del_ok={del_ok}, trunc_ok={trunc_ok}, no_ok={no_ok}, yes_ok={yes_ok}, "
        f"twice_ok={twice_ok}, other_del={other_del_ok}, other_yes={other_yes_ok}, keep={keep_ok}",
        ok, "",
    )


# ---------------------------------------------------------------------------
# C5: /forget step 1 warns, deletes nothing, shows Continue/Cancel
# ---------------------------------------------------------------------------
async def c5() -> dict:
    chat = next_chat()
    uid = await db_add_user(chat)
    rec = await db_add_record_full(uid, "C5SECRET keep me", status="confirmed")
    bot, ctx, st = make_env()

    await handlers.cmd_forget(FakeUpdate(chat, text="/forget"), ctx, state=st)

    last = bot.sent[-1]
    cbs = callbacks_of(last["reply_markup"])
    user_ok = await db_user_id_for_chat(chat) is not None
    rec_ok = await db_record_exists(rec)
    ok = (
        last["text"] == handlers.FORGET_WARNING
        and cbs == ["fg:1", "fg:no"]
        and user_ok
        and rec_ok
        and len(bot.edits) == 0
    )
    return row(
        "C5", "/forget step 1: warning only",
        "exact warning text; buttons [fg:1, fg:no]; user + records intact",
        f"text_ok={last['text'] == handlers.FORGET_WARNING}, cbs={cbs}, "
        f"user_ok={user_ok}, rec_ok={rec_ok}",
        ok, "",
    )


# ---------------------------------------------------------------------------
# C6: cancel at step 1 and at step 2 keeps everything
# ---------------------------------------------------------------------------
async def c6() -> dict:
    chat1 = next_chat()
    chat2 = next_chat()
    uid1 = await db_add_user(chat1)
    uid2 = await db_add_user(chat2)
    rec1 = await db_add_record_full(uid1, "C6SECRET one", status="confirmed")
    rec2 = await db_add_record_full(uid2, "C6SECRET two", status="confirmed")
    bot, ctx, st = make_env()

    # cancel at step 1
    await handlers.cmd_forget(FakeUpdate(chat1, text="/forget"), ctx, state=st)
    await press(bot, ctx, st, chat1, "fg:no")

    # cancel at step 2
    await handlers.cmd_forget(FakeUpdate(chat2, text="/forget"), ctx, state=st)
    await press(bot, ctx, st, chat2, "fg:1")
    await press(bot, ctx, st, chat2, "fg:no")

    u1 = await db_user_id_for_chat(chat1) is not None
    u2 = await db_user_id_for_chat(chat2) is not None
    r1 = await db_record_exists(rec1)
    r2 = await db_record_exists(rec2)
    cancels = sum(1 for e in bot.edits if e["text"] == handlers.CANCELLED)
    ok = u1 and u2 and r1 and r2 and cancels == 2
    return row(
        "C6", "cancel at step 1 and step 2",
        "both users + records remain; 2 'Cancelled.' edits",
        f"u1={u1}, u2={u2}, r1={r1}, r2={r2}, cancels={cancels}",
        ok, "",
    )


# ---------------------------------------------------------------------------
# C7: full /forget deletes everything for that user only, clears chat state
# ---------------------------------------------------------------------------
async def c7() -> dict:
    start = len(LOG_RECORDS)
    chat = next_chat()
    chat_b = next_chat()
    uid = await db_add_user(chat)
    uid_b = await db_add_user(chat_b)
    await db_add_record_full(uid, "C7SECRET record", status="confirmed")
    seeds = await seed_all(uid)
    b_rec = await db_add_record_full(uid_b, "C7OTHER keep", status="confirmed")
    async with AsyncSessionLocal() as session:
        other_task = Task(user_id=uid_b, title="C7OTHER task", status="open", origin="manual")
        session.add(other_task)
        await session.commit()
        await session.refresh(other_task)
        other_task_id = other_task.id

    bot, ctx, st = make_env()
    st.chat(chat).history.append({"role": "user", "content": "leftover"})
    st.chat(chat).session_id = 4242

    await handlers.cmd_forget(FakeUpdate(chat, text="/forget"), ctx, state=st)
    await press(bot, ctx, st, chat, "fg:1")
    fg2 = next(c for c in callbacks_of(bot.edits[-1]["reply_markup"]) if c.startswith("fg:2:"))
    await press(bot, ctx, st, chat, fg2)

    remaining = {m.__tablename__: await db_count_by_user(m, uid) for m in USER_SCOPED_MODELS}
    transitive = {
        "milestones": await db_count_where(Milestone, Milestone.project_id == seeds["project_id"]),
        "progress": await db_count_where(Progress, Progress.project_id == seeds["project_id"]),
        "inspector_events": await db_count_where(
            InspectorEvent, InspectorEvent.session_id == seeds["session_id"]
        ),
    }
    all_gone = all(v == 0 for v in remaining.values()) and all(v == 0 for v in transitive.values())

    b_rec_ok = await db_record_exists(b_rec)
    b_task_ok = await db_task_status(other_task_id) == "open"

    chat_state = st.chat(chat)
    state_cleared = (
        chat_state.history == []
        and chat_state.session_id is None
        and chat_state.forget_nonce is None
        and chat_state.forget_expires_at is None
    )

    done_edit = any(e["text"] == handlers.FORGET_DONE for e in bot.edits)

    await handlers.on_text(FakeUpdate(chat, text="hello again"), ctx, state=st)
    signup = bot.sent[-1]["text"] == handlers.SIGNUP_PROMPT

    PRIVACY.append(
        {
            "case": "C7",
            "start": start,
            "end": len(LOG_RECORDS),
            "markers": [
                "C7SECRET record",
                "C7SECRET profile blob",
                "C7SECRET progress note",
                "C7OTHER keep",
                "C7OTHER task",
            ],
        }
    )
    ok = all_gone and b_rec_ok and b_task_ok and state_cleared and done_edit and signup
    return row(
        "C7", "confirm: full delete, other user untouched, state cleared",
        "0 rows in every table for this user; other user intact; state cleared; next msg = signup",
        f"remaining={remaining}, transitive={transitive}, b_rec={b_rec_ok}, b_task={b_task_ok}, "
        f"state_cleared={state_cleared}, done_edit={done_edit}, signup={signup}",
        ok, "",
    )


# ---------------------------------------------------------------------------
# C8: forged / missing-step / expired fg:2 -> nothing deleted
# ---------------------------------------------------------------------------
async def c8() -> dict:
    # (a) forged nonce after step 1
    chat1 = next_chat()
    uid1 = await db_add_user(chat1)
    rec1 = await db_add_record_full(uid1, "C8SECRET a", status="confirmed")
    bot1, ctx1, st1 = make_env()
    await handlers.cmd_forget(FakeUpdate(chat1, text="/forget"), ctx1, state=st1)
    await press(bot1, ctx1, st1, chat1, "fg:1")
    await press(bot1, ctx1, st1, chat1, "fg:2:deadbeefdeadbeefdeadbeef")
    a_ok = (
        bot1.edits[-1]["text"] == handlers.FORGET_EXPIRED
        and await db_user_id_for_chat(chat1) is not None
        and await db_record_exists(rec1)
    )

    # (b) fg:2 with no step 1
    chat2 = next_chat()
    uid2 = await db_add_user(chat2)
    rec2 = await db_add_record_full(uid2, "C8SECRET b", status="confirmed")
    bot2, ctx2, st2 = make_env()
    await press(bot2, ctx2, st2, chat2, "fg:2:whatever")
    b_ok = (
        bot2.edits[-1]["text"] == handlers.FORGET_EXPIRED
        and await db_user_id_for_chat(chat2) is not None
        and await db_record_exists(rec2)
    )

    # (c) expired nonce (clock patched forward)
    chat3 = next_chat()
    uid3 = await db_add_user(chat3)
    rec3 = await db_add_record_full(uid3, "C8SECRET c", status="confirmed")
    bot3, ctx3, st3 = make_env()
    await handlers.cmd_forget(FakeUpdate(chat3, text="/forget"), ctx3, state=st3)
    await press(bot3, ctx3, st3, chat3, "fg:1")
    fg2 = next(c for c in callbacks_of(bot3.edits[-1]["reply_markup"]) if c.startswith("fg:2:"))
    real_now = handlers._forget_now()
    future = real_now + handlers.FORGET_NONCE_TTL_SECONDS + 5
    with unittest.mock.patch.object(handlers, "_forget_now", return_value=future):
        await press(bot3, ctx3, st3, chat3, fg2)
    c_ok = (
        bot3.edits[-1]["text"] == handlers.FORGET_EXPIRED
        and await db_user_id_for_chat(chat3) is not None
        and await db_record_exists(rec3)
    )

    ok = a_ok and b_ok and c_ok
    return row(
        "C8", "forged / no-step / expired fg:2 -> expired reply, nothing deleted",
        "each replies expired; users + records intact",
        f"forged={a_ok}, no_step1={b_ok}, expired={c_ok}",
        ok, "",
    )


# ---------------------------------------------------------------------------
# C9: /help lists only existing commands
# ---------------------------------------------------------------------------
async def c9() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    await handlers.cmd_help(FakeUpdate(chat, text="/help"), ctx, state=st)
    text = bot.sent[-1]["text"]

    required = ["/start", "/newsession", "/quiet", "/memory", "/forget", "/help"]
    forbidden = ["/link", "edit", "change date"]
    missing = [c for c in required if c not in text]
    present_bad = [f for f in forbidden if f.lower() in text.lower()]

    unknown = next_chat()
    bot2, ctx2, st2 = make_env()
    await handlers.cmd_help(FakeUpdate(unknown, text="/help"), ctx2, state=st2)
    unknown_ok = bot2.sent[-1]["text"] == handlers.SIGNUP_PROMPT

    ok = not missing and not present_bad and unknown_ok
    return row(
        "C9", "/help: existing commands only",
        "all 6 present; none of /link, edit, change date; unknown chat -> signup",
        f"missing={missing}, present_bad={present_bad}, unknown_ok={unknown_ok}",
        ok, f"help={text!r}",
    )


# ---------------------------------------------------------------------------
# C10: unknown chat for all three commands -> signup, no writes, no LLM/embedding
# ---------------------------------------------------------------------------
async def c10() -> dict:
    unknown = next_chat()
    bot, ctx, st = make_env()
    before = await snapshot_counts()

    async def boom_embed(*a, **k):
        raise AssertionError("embedding call must not happen")

    async def boom_llm(*a, **k):
        raise AssertionError("LLM call must not happen")

    old_embed = embeddings_module.embed_texts
    old_respond = orchestrator.respond
    old_post = orchestrator.postprocess
    embeddings_module.embed_texts = boom_embed
    orchestrator.respond = boom_llm
    orchestrator.postprocess = boom_llm
    try:
        await handlers.cmd_help(FakeUpdate(unknown, text="/help"), ctx, state=st)
        await handlers.cmd_memory(FakeUpdate(unknown, text="/memory secrets"), ctx, state=st)
        await handlers.cmd_forget(FakeUpdate(unknown, text="/forget"), ctx, state=st)
    finally:
        embeddings_module.embed_texts = old_embed
        orchestrator.respond = old_respond
        orchestrator.postprocess = old_post

    after = await snapshot_counts()
    texts = [m["text"] for m in bot.sent]
    all_prompt = len(texts) == 3 and all(t == handlers.SIGNUP_PROMPT for t in texts)
    no_writes = before == after
    ok = all_prompt and no_writes
    return row(
        "C10", "unknown chat: 3 commands -> signup only",
        "3 signup prompts; zero DB writes; zero LLM/embedding calls",
        f"texts={texts}, no_writes={no_writes}",
        ok, f"before={before}",
    )


# ---------------------------------------------------------------------------
# C11: /memory makes at most one embedding call and never an LLM call
# ---------------------------------------------------------------------------
async def c11() -> dict:
    chat = next_chat()
    uid = await db_add_user(chat)
    await db_add_record_full(uid, "C11 record", status="confirmed")
    bot, ctx, st = make_env()
    counter = [0]
    with fake_embeddings(counter), block_llm():
        await handlers.cmd_memory(FakeUpdate(chat, text="/memory anything"), ctx, state=st)

    lines = memory_lines(bot, chat)
    generic = any(m["text"] == handlers.GENERIC_ERROR for m in bot.sent)
    ok = counter[0] == 1 and not generic and len(lines) == 1
    return row(
        "C11", "/memory: one embedding call, no LLM call",
        "exactly 1 embed_texts call; no LLM call; one result; no generic error",
        f"embed_calls={counter[0]}, lines={len(lines)}, generic={generic}",
        ok, "",
    )


# ---------------------------------------------------------------------------
# C12: privacy: no record content in any log record during C1, C4, C7
# ---------------------------------------------------------------------------
async def c12() -> dict:
    leaked: list[str] = []
    empty_slices: list[str] = []
    for entry in PRIVACY:
        records = LOG_RECORDS[entry["start"]:entry["end"]]
        if not records:
            empty_slices.append(entry["case"])
        blob = "\n".join(r.getMessage() for r in records)
        for marker in entry["markers"]:
            if marker in blob:
                leaked.append(f"{entry['case']}:{marker}")
    captured_any = any(LOG_RECORDS[e["start"]:e["end"]] for e in PRIVACY)
    ok = not leaked and captured_any and len(PRIVACY) >= 3
    return row(
        "C12", "privacy: no record content in logs",
        "no seeded content appears in any captured record; capture is non-empty",
        f"leaked={leaked}, cases={[e['case'] for e in PRIVACY]}, "
        f"empty_slices={empty_slices}, captured_any={captured_any}",
        ok, "",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def main() -> int:
    settings.require("DATABASE_URL")

    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.INFO)  # so the handlers' info logs are actually captured
    capture = ListHandler()
    root.addHandler(capture)

    print("=" * 140)
    print("PHASE 4c-1: /help, /memory, /forget EVALUATION")
    print("=" * 140)

    results: list[dict] = []
    cases = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]
    try:
        purged = await purge_synthetic()
        print(f"PURGE at start: removed {purged} synthetic user(s).\n")
        for case in cases:
            print(f"--- Running {case.__name__.upper()} ---")
            try:
                res = await case()
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                res = row(case.__name__.upper(), case.__name__, "-", "", False, f"unhandled: {exc!r}")
            results.append(res)
            print(f"    [{'PASS' if res['ok'] else 'FAIL'}] {res['name']}: {res['actual']}")
    finally:
        print("\nCleaning up synthetic users...")
        cleaned = await purge_synthetic()
        print(f"Cleanup removed {cleaned} synthetic user(s).")
        root.removeHandler(capture)
        root.setLevel(prev_level)
        await engine.dispose()

    print("\n" + "=" * 140)
    print(f"{'ID':<5} | {'NAME':<52} | {'EXPECTED':<58} | {'ACTUAL'}")
    print("-" * 140)
    all_passed = True
    for r in results:
        if not r["ok"]:
            all_passed = False
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{r['id']:<5} | {r['name']:<52} | {r['expected']:<58} | {status} :: {r['actual']}")
    print("=" * 140)
    passed = sum(1 for r in results if r["ok"])
    print(f"RESULT: {'ALL PASS' if all_passed else 'FAILURES PRESENT'} ({passed}/{len(results)})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
