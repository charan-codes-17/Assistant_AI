"""backend/scripts/test_bot_contradiction.py
Phase 4c-3 evaluation for contradiction notices in the Telegram bot.

D1-D10 are deterministic: the orchestrator (respond + postprocess) is stubbed, so ZERO LLM calls
are made and the intervention decision is exactly what each case sets. A fake Bot records
sends/edits. The real dev DB is used for user rows only; throwaway users live in 930000-939999 and
are purged at the START and again in a finally block.

The MEASURE-ONLY section at the end runs the REAL pipeline (real LLM) for 8 turns and prints the
numbers as measured. It makes no assertions and is not part of the exit code.

No Telegram network is used. No file in the repo is written.

Run:
  backend\\venv\\Scripts\\python.exe -u backend\\scripts\\test_bot_contradiction.py
"""

import asyncio
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

from sqlalchemy import select

from backend.bot import handlers
from backend.bot.state import BotState
from backend.config.settings import settings
from backend.database import AsyncSessionLocal, engine
from backend.models.models import KnowledgeRecord, User
from backend.services import embeddings as embeddings_service
from backend.services import orchestrator
from backend.services.intervention import ConflictRef, InterventionDecision
from backend.services.orchestrator import PendingRecord, PostResult, ReplyResult

IST = ZoneInfo("Asia/Kolkata")

CHAT_IDS: set[int] = set()
_next_chat = [930000]


def next_chat() -> int:
    _next_chat[0] += 1
    CHAT_IDS.add(_next_chat[0])
    return _next_chat[0]


# ---------------------------------------------------------------------------
# Fakes (same shape as test_bot_handlers.py)
# ---------------------------------------------------------------------------
class FakeChat:
    def __init__(self, chat_id: int, chat_type: str = "private") -> None:
        self.id = chat_id
        self.type = chat_type


class FakeMessage:
    def __init__(self, chat_id: int, message_id: int = 1, text=None) -> None:
        self.chat = FakeChat(chat_id)
        self.chat_id = chat_id
        self.message_id = message_id
        self.text = text


class FakeTelegramUser:
    def __init__(self, uid: int) -> None:
        self.id = uid


class FakeCallbackQuery:
    def __init__(self, chat_id: int, data: str, message_id: int = 500) -> None:
        self.id = f"cb-{uuid.uuid4().hex[:6]}"
        self.data = data
        self.message = FakeMessage(chat_id, message_id=message_id)


class FakeUpdate:
    def __init__(self, chat_id: int, text=None, callback=None) -> None:
        self.effective_chat = FakeChat(chat_id)
        self.effective_user = FakeTelegramUser(chat_id)
        self.message = FakeMessage(chat_id, text=text)
        self.callback_query = callback


class FakeBot:
    """Records sends. `raise_on_prefix` makes the matching send raise (for D6)."""

    def __init__(self, raise_on_prefix: str | None = None) -> None:
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.actions: list[dict] = []
        self.raise_on_prefix = raise_on_prefix
        self._t0 = time.perf_counter()
        self._mid = 100

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
        if self.raise_on_prefix is not None and (text or "").startswith(self.raise_on_prefix):
            raise RuntimeError("send failed (D6)")
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
        pass


class FakeContext:
    def __init__(self, bot: FakeBot, bot_data=None) -> None:
        self.bot = bot
        self.bot_data = bot_data or {}


def make_env(raise_on_prefix: str | None = None) -> tuple[FakeBot, FakeContext, BotState]:
    bot = FakeBot(raise_on_prefix=raise_on_prefix)
    st = BotState()
    ctx = FakeContext(bot, {"state": st})
    return bot, ctx, st


# ---------------------------------------------------------------------------
# Log capture (D7)
# ---------------------------------------------------------------------------
class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # noqa: BLE001 - logging must never break the turn
            self.messages.append("<unformattable record>")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
async def db_add_user(chat_id: int) -> int:
    async with AsyncSessionLocal() as session:
        user = User(
            username=f"ctr_{uuid.uuid4().hex[:8]}",
            telegram_chat_id=chat_id,
            timezone="Asia/Kolkata",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.id


async def db_add_confirmed_record(user_id: int, content: str, embedding=None) -> int:
    async with AsyncSessionLocal() as session:
        rec = KnowledgeRecord(
            user_id=user_id,
            kind="knowledge",
            content=content,
            certainty="confirmed",
            source="text",
            status="confirmed",
            embedding=embedding,
        )
        session.add(rec)
        await session.commit()
        await session.refresh(rec)
        return rec.id


# ---------------------------------------------------------------------------
# Mocked orchestrator
# ---------------------------------------------------------------------------
def make_respond(reply_text="mocked reply", counter=None):
    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        if counter is not None:
            counter[0] += 1
        return ReplyResult(
            turn_id=f"turn-{uuid.uuid4().hex[:6]}",
            session_id=session_id,
            reply=reply_text,
            memory_used=[],
        )

    return respond


def make_postprocess(decision: InterventionDecision, pending_records=None, counter=None):
    async def postprocess(user_id, reply, *, now=None):
        if counter is not None:
            counter[0] += 1
        return PostResult(
            intent="question",
            pending_records=list(pending_records or []),
            pending_tasks=[],
            intervention=decision,
        )

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


async def drain(st: BotState) -> None:
    while st.background:
        await asyncio.gather(*list(st.background), return_exceptions=True)


async def send_text(bot, ctx, st, chat_id, text):
    await handlers.on_text(FakeUpdate(chat_id, text=text), ctx, state=st)


def row(rid, name, expected, actual, ok, details=""):
    return {
        "id": rid,
        "name": name,
        "expected": expected,
        "actual": actual,
        "ok": bool(ok),
        "details": details,
    }


def notices(bot: FakeBot, chat_id: int | None = None) -> list[dict]:
    out = [m for m in bot.sent if (m["text"] or "").startswith("Heads up:")]
    if chat_id is not None:
        out = [m for m in out if m["chat_id"] == chat_id]
    return out


def confirm_prompts(bot: FakeBot) -> list[dict]:
    return [m for m in bot.sent if (m["text"] or "").startswith("Save this memory?")]


def is_plain(text: str) -> bool:
    return not any(marker in text for marker in ("**", "`", "#", "\\("))


# ---------------------------------------------------------------------------
# D1: contradiction=True + conflicts_with + question -> exactly one notice
# ---------------------------------------------------------------------------
async def d1() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    decision = InterventionDecision(
        action="suggest",
        contradiction=True,
        contradiction_detail="You said Postgres for the project; now you want SQLite.",
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question="Keep Postgres or switch to SQLite?",
        reason="contradiction",
    )
    with mock_orchestrator(make_respond(), make_postprocess(decision)):
        await send_text(bot, ctx, st, chat, "Let's switch the project to SQLite instead.")
        await drain(st)
    found = notices(bot, chat)
    text = found[0]["text"] if found else ""
    ok = (
        len(found) == 1
        and "conflict with something you told me earlier" in text
        and "Postgres for the project; now you want SQLite." in text
        and "Keep Postgres or switch to SQLite?" in text
        and is_plain(text)
    )
    return row(
        "D1",
        "contradiction + conflicts_with + question",
        "exactly one plain-text notice in the originating chat with both parts",
        f"notices={len(found)}, in_chat={len(found) == 1}, plain={is_plain(text) if text else 'n/a'}",
        ok,
        f"notice={text!r}",
    )


# ---------------------------------------------------------------------------
# D2: contradiction=False -> no notice
# ---------------------------------------------------------------------------
async def d2() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    decision = InterventionDecision(
        action="suggest",
        contradiction=False,
        contradiction_detail="should not be shown",
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question="should not be shown?",
        reason="no contradiction",
    )
    with mock_orchestrator(make_respond(), make_postprocess(decision)):
        await send_text(bot, ctx, st, chat, "hello there")
        await drain(st)
    found = notices(bot)
    ok = len(found) == 0
    return row(
        "D2",
        "contradiction=False",
        "no notice",
        f"notices={len(found)}",
        ok,
        f"sent={[m['text'] for m in bot.sent]!r}",
    )


# ---------------------------------------------------------------------------
# D3: action "none" + contradiction=False -> no notice
# ---------------------------------------------------------------------------
async def d3() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    decision = InterventionDecision(
        action="none",
        contradiction=False,
        contradiction_detail="should not be shown",
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question="should not be shown?",
        reason="no contradiction",
    )
    with mock_orchestrator(make_respond(), make_postprocess(decision)):
        await send_text(bot, ctx, st, chat, "hello there")
        await drain(st)
    found = notices(bot)
    ok = len(found) == 0
    return row(
        "D3",
        "action none + contradiction=False",
        "no notice",
        f"notices={len(found)}",
        ok,
        f"sent={[m['text'] for m in bot.sent]!r}",
    )


# ---------------------------------------------------------------------------
# D3b: contradiction=True + conflicts_with + action "none" -> exactly one notice
# (the real-world case: the engine flags a contradiction but proposes no directive)
# ---------------------------------------------------------------------------
async def d3b() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    decision = InterventionDecision(
        action="none",
        contradiction=True,
        contradiction_detail="You said Postgres; now SQLite.",
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question="Postgres or SQLite?",
        reason="contradiction but no directive",
    )
    with mock_orchestrator(make_respond(), make_postprocess(decision)):
        await send_text(bot, ctx, st, chat, "Let's switch to SQLite.")
        await drain(st)
    found = notices(bot, chat)
    text = found[0]["text"] if found else ""
    ok = (
        len(found) == 1
        and "conflict with something you told me earlier" in text
        and "You said Postgres; now SQLite." in text
        and "Postgres or SQLite?" in text
        and is_plain(text)
    )
    return row(
        "D3b",
        "action none + contradiction + conflicts_with",
        "exactly one plain-text notice in the originating chat with both parts",
        f"notices={len(found)}, in_chat={len(found) == 1}, plain={is_plain(text) if text else 'n/a'}",
        ok,
        f"notice={text!r}",
    )


# ---------------------------------------------------------------------------
# D4: contradiction=True but conflicts_with None -> no notice
# ---------------------------------------------------------------------------
async def d4() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    decision = InterventionDecision(
        action="suggest",
        contradiction=True,
        contradiction_detail="You said Postgres; now SQLite.",
        conflicts_with=None,
        clarifying_question="Postgres or SQLite?",
        reason="contradiction without target",
    )
    with mock_orchestrator(make_respond(), make_postprocess(decision)):
        await send_text(bot, ctx, st, chat, "Let's switch to SQLite.")
        await drain(st)
    found = notices(bot)
    ok = len(found) == 0
    return row(
        "D4",
        "contradiction=True but conflicts_with None",
        "no notice",
        f"notices={len(found)}",
        ok,
        f"sent={[m['text'] for m in bot.sent]!r}",
    )


# ---------------------------------------------------------------------------
# D5: over-long fields -> truncated to the limit
# ---------------------------------------------------------------------------
async def d5() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    decision = InterventionDecision(
        action="suggest",
        contradiction=True,
        contradiction_detail="X" * 500,
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question="Q" * 500,
        reason="long fields",
    )
    with mock_orchestrator(make_respond(), make_postprocess(decision)):
        await send_text(bot, ctx, st, chat, "let's switch")
        await drain(st)
    found = notices(bot)
    text = found[0]["text"] if found else ""
    lines = text.split("\n") if text else []
    ok = (
        len(found) == 1
        and ("X" * handlers.NOTICE_FIELD_CHARS) in text
        and ("X" * (handlers.NOTICE_FIELD_CHARS + 1)) not in text
        and ("Q" * handlers.NOTICE_FIELD_CHARS) in text
        and ("Q" * (handlers.NOTICE_FIELD_CHARS + 1)) not in text
    )
    return row(
        "D5",
        "over-long fields truncated",
        f"summary and question each cut to {handlers.NOTICE_FIELD_CHARS} chars",
        f"notices={len(found)}, line0_len={len(lines[0]) if lines else 0}, "
        f"line1_len={len(lines[1]) if len(lines) > 1 else 0}",
        ok,
        "",
    )


# ---------------------------------------------------------------------------
# D6: notice send raises -> turn completes, confirm prompt still delivered
# ---------------------------------------------------------------------------
async def d6() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env(raise_on_prefix="Heads up:")
    decision = InterventionDecision(
        action="suggest",
        contradiction=True,
        contradiction_detail="You said Postgres; now SQLite.",
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question="Postgres or SQLite?",
        reason="contradiction, notice send will fail",
    )
    pending = [PendingRecord(id=777, content="some pending memory", kind="knowledge",
                             certainty="tentative", source="text")]
    with mock_orchestrator(make_respond(), make_postprocess(decision, pending_records=pending)):
        try:
            await send_text(bot, ctx, st, chat, "let's switch")
            await drain(st)
            raised = False
        except Exception as exc:  # noqa: BLE001
            raised = True
            err = repr(exc)
        else:
            err = ""
    prompts = confirm_prompts(bot)
    ok = (not raised) and len(prompts) >= 1
    return row(
        "D6",
        "notice send raises -> turn survives",
        "no exception; confirm prompt still delivered",
        f"raised={raised}, confirm_prompts={len(prompts)}",
        ok,
        f"err={err}",
    )


# ---------------------------------------------------------------------------
# D7: the conflicting record's text is NOT in any log line
# ---------------------------------------------------------------------------
async def d7() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    secret = "ZqConflictSecretTextDoNotLog"
    handler = ListHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    prev_level = logging.getLogger("backend.bot.handlers").level
    logging.getLogger("backend.bot.handlers").setLevel(logging.INFO)
    decision = InterventionDecision(
        action="suggest",
        contradiction=True,
        contradiction_detail=f"You said {secret} for the project.",
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question=f"Keep {secret} or switch?",
        reason="contradiction",
    )
    try:
        with mock_orchestrator(make_respond(), make_postprocess(decision)):
            await send_text(bot, ctx, st, chat, "let's switch")
            await drain(st)
    finally:
        root.removeHandler(handler)
        logging.getLogger("backend.bot.handlers").setLevel(prev_level)

    leaked = [m for m in handler.messages if secret in m]
    sent_notice = notices(bot, chat)
    ok = len(leaked) == 0 and len(handler.messages) > 0 and len(sent_notice) == 1
    return row(
        "D7",
        "no memory content in logs",
        "secret text absent from every captured log line",
        f"leaked={len(leaked)}, captured={len(handler.messages)}, notices={len(sent_notice)}",
        ok,
        "",
    )


# ---------------------------------------------------------------------------
# D8: notice is sent before the confirm prompt
# ---------------------------------------------------------------------------
async def d8() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    decision = InterventionDecision(
        action="suggest",
        contradiction=True,
        contradiction_detail="You said Postgres; now SQLite.",
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question="Postgres or SQLite?",
        reason="contradiction",
    )
    pending = [PendingRecord(id=778, content="pending memory one", kind="knowledge",
                             certainty="tentative", source="text")]
    with mock_orchestrator(make_respond(), make_postprocess(decision, pending_records=pending)):
        await send_text(bot, ctx, st, chat, "let's switch")
        await drain(st)
    notice = notices(bot, chat)
    prompt = confirm_prompts(bot)
    notice_i = bot.sent.index(notice[0]) if notice else -1
    prompt_i = bot.sent.index(prompt[0]) if prompt else -1
    ok = notice_i != -1 and prompt_i != -1 and notice_i < prompt_i
    return row(
        "D8",
        "notice before confirm prompt",
        "notice index < confirm-prompt index",
        f"notice_i={notice_i}, prompt_i={prompt_i}",
        ok,
        f"order={[('notice' if (m['text'] or '').startswith('Heads up:') else 'prompt' if (m['text'] or '').startswith('Save this memory?') else m['text'][:20]) for m in bot.sent]!r}",
    )


# ---------------------------------------------------------------------------
# D9: two users in separate chats -> only the originating chat receives it
# ---------------------------------------------------------------------------
async def d9() -> dict:
    chat_a = next_chat()
    chat_b = next_chat()
    await db_add_user(chat_a)
    await db_add_user(chat_b)
    bot, ctx, st = make_env()
    decision = InterventionDecision(
        action="suggest",
        contradiction=True,
        contradiction_detail="conflict applies to both users",
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question="Which one did you mean?",
        reason="contradiction",
    )
    with mock_orchestrator(make_respond(), make_postprocess(decision)):
        await send_text(bot, ctx, st, chat_a, "A contradicts")
        await drain(st)
        await send_text(bot, ctx, st, chat_b, "B contradicts")
        await drain(st)
    in_a = notices(bot, chat_a)
    in_b = notices(bot, chat_b)
    total = notices(bot)
    ok = len(in_a) == 1 and len(in_b) == 1 and len(total) == 2
    return row(
        "D9",
        "only originating chat notified",
        "chat A gets 1, chat B gets 1, no third chat",
        f"in_a={len(in_a)}, in_b={len(in_b)}, total={len(total)}",
        ok,
        "",
    )


# ---------------------------------------------------------------------------
# D10: a command turn (/help) -> no notice path executed
# ---------------------------------------------------------------------------
async def d10() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    counter = [0]
    decision = InterventionDecision(
        action="suggest",
        contradiction=True,
        contradiction_detail="You said Postgres; now SQLite.",
        conflicts_with=ConflictRef(kind="record", id=4242),
        clarifying_question="Postgres or SQLite?",
        reason="should never run for a command",
    )
    with mock_orchestrator(make_respond(), make_postprocess(decision, counter=counter)):
        await handlers.cmd_help(FakeUpdate(chat, text="/help"), ctx, state=st)
        await drain(st)
    found = notices(bot)
    ok = len(found) == 0 and counter[0] == 0
    return row(
        "D10",
        "command turn -> no notice path",
        "no notice; postprocess never called",
        f"notices={len(found)}, postprocess_calls={counter[0]}",
        ok,
        f"sent={[m['text'][:30] for m in bot.sent]!r}",
    )


# ---------------------------------------------------------------------------
# MEASURE-ONLY: real pipeline, no assertions
# ---------------------------------------------------------------------------
CONTRADICTING = [
    "Let's switch the project to SQLite instead.",
    "I'm going with SQLite for the database after all.",
    "Drop Postgres, we use SQLite now.",
]
NEUTRAL = [
    "What's a good way to name database migrations?",
    "Remind me what an index is.",
    "I had pizza for dinner.",
    "Can you explain connection pooling?",
    "Thanks, that helps.",
]


async def measure_only() -> None:
    print("\n" + "=" * 140)
    print("MEASURE ONLY (real LLM, no assertions): contradiction notices measured, not tuned")
    print("=" * 140)
    chat = next_chat()
    uid = await db_add_user(chat)
    content = (
        "Decided to use Postgres instead of SQLite for the project, because of concurrency."
    )
    try:
        vectors = await embeddings_service.embed_texts([content])
        embedding = vectors[0]
    except Exception as exc:  # noqa: BLE001
        print(f"   [WARN] could not embed the seeded decision ({type(exc).__name__}); "
              f"contradiction retrieval will likely find nothing.")
        embedding = None
    await db_add_confirmed_record(uid, content, embedding=embedding)

    bot, ctx, st = make_env()
    captured: dict = {}
    real_respond = orchestrator.respond
    real_postprocess = orchestrator.postprocess

    async def recording_respond(*args, **kwargs):
        reply = await real_respond(*args, **kwargs)
        captured["reply"] = reply
        return reply

    async def recording_postprocess(*args, **kwargs):
        post = await real_postprocess(*args, **kwargs)
        captured["post"] = post
        return post

    turns = [("contradicting", t) for t in CONTRADICTING] + [("neutral", t) for t in NEUTRAL]
    c_flagged = 0
    n_flagged = 0
    would_send_total = 0
    print(f"{'#':<3} | {'kind':<13} | {'directive':<9} | {'contradiction':<13} | {'notice?':<7} | turn")
    print("-" * 140)
    with mock_orchestrator(recording_respond, recording_postprocess):
        for index, (kind, text) in enumerate(turns, start=1):
            await handlers.on_newsession(FakeUpdate(chat, text="/newsession"), ctx, state=st)
            await send_text(bot, ctx, st, chat, text)
            await drain(st)
            post = captured.get("post")
            decision = getattr(post, "intervention", None)
            directive = getattr(decision, "action", None)
            flagged = bool(getattr(decision, "contradiction", False))
            would_send = handlers.contradiction_notice(decision) is not None if decision else False
            if kind == "contradicting" and flagged:
                c_flagged += 1
            if kind == "neutral" and flagged:
                n_flagged += 1
            if would_send:
                would_send_total += 1
            print(
                f"{index:<3} | {kind:<13} | {str(directive):<9} | {str(flagged):<13} | "
                f"{str(would_send):<7} | {text}"
            )
    print("-" * 140)
    print(f"TOTALS: contradictions flagged on the 3 contradicting turns: {c_flagged}/3")
    print(f"TOTALS: false flags on the 5 neutral turns: {n_flagged}/5")
    print(f"TOTALS: turns that would send a notice (of 8): {would_send_total}/8")
    print("(measured as-is; no prompt was tuned)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def purge_synthetic() -> None:
    try:
        async with AsyncSessionLocal() as session:
            users = (
                await session.execute(
                    select(User).where(User.telegram_chat_id.between(930000, 939999))
                )
            ).scalars().all()
            for user in users:
                await session.delete(user)
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"   [WARN] purge failed: {exc!r}")


async def main() -> int:
    settings.require("DATABASE_URL")
    print("=" * 140)
    print("PHASE 4c-3: CONTRADICTION NOTICE EVALUATION")
    print("=" * 140)
    print("Purge at start: removing synthetic users in 930000-939999 ...")
    await purge_synthetic()

    results: list[dict] = []
    cases = [d1, d2, d3, d3b, d4, d5, d6, d7, d8, d9, d10]
    try:
        for case in cases:
            print(f"--- Running {case.__name__.upper()} ---")
            try:
                res = await case()
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                res = row(case.__name__.upper(), case.__name__, "-", "", False, f"unhandled: {exc!r}")
            results.append(res)
            print(f"    [{'PASS' if res['ok'] else 'FAIL'}] {res['name']}: {res['actual']}")

        await measure_only()
    finally:
        print("\nCleaning up synthetic users ...")
        await purge_synthetic()
        await engine.dispose()

    print("\n" + "=" * 140)
    print(f"{'ID':<6} | {'NAME':<38} | {'EXPECTED':<50} | {'ACTUAL':<48} | STATUS")
    print("-" * 140)
    all_passed = True
    for r in results:
        if not r["ok"]:
            all_passed = False
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{r['id']:<6} | {r['name']:<38} | {r['expected']:<50} | {r['actual']:<48} | {status}")
    print("=" * 140)
    passed = sum(1 for r in results if r["ok"])
    print(f"RESULT: {'ALL PASS' if all_passed else 'FAILURES PRESENT'} ({passed}/{len(results)})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
