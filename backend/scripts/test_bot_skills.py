"""backend/scripts/test_bot_skills.py
Evaluation for the Telegram bot skills commands: /skills and /skill (plus the sk:ok / sk:no
callback branches they produce).

The handlers are called IN-PROCESS (no Telegram network). Fakes record every send_message /
edit_message_text / answer_callback_query. The real dev DB is used with synthetic users whose
telegram_chat_id is in 970000-979999; those users are purged at the START and again in a finally
block (the user-row cascade removes their tasks and user_skills).

Determinism / no network:
- backend.services.skills.chat is replaced with a capture stub, so no model is ever called.
- backend.services.skills.retrieve_memory is replaced with an offline stub (simple DB read) so
  the semantic skills (topic-brief, goal-breakdown) never touch an embedding service.

Run:
  backend\\venv\\Scripts\\python.exe backend\\scripts\\test_bot_skills.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import traceback
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import func, select

from backend.bot import handlers
from backend.bot.state import BotState
from backend.config.settings import settings
from backend.database import AsyncSessionLocal, engine
from backend.models.models import Task, User, UserSkill
from backend.services.skills import BUILTIN_SKILLS, reset_rate_limits

CHAT_MIN = 970000
CHAT_MAX = 979999

BUILTIN_ORDER = [
    "weekly-review",
    "day-plan",
    "open-loops",
    "topic-brief",
    "explain-simply",
    "decision-recall",
    "goal-breakdown",
    "stress-test-idea",
]

CHAT_IDS: set[int] = set()
_next_chat = [CHAT_MIN]


def next_chat() -> int:
    _next_chat[0] += 1
    if _next_chat[0] > CHAT_MAX:
        raise RuntimeError("synthetic chat id range exhausted")
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
    def __init__(self, chat_id: int, message_id: int = 700, text=None) -> None:
        self.chat = FakeChat(chat_id)
        self.chat_id = chat_id
        self.message_id = message_id
        self.text = text


class FakeCallbackQuery:
    def __init__(self, chat_id: int, data: str, message_id: int = 700) -> None:
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
        self.sent: List[dict] = []
        self.edits: List[dict] = []
        self.answers: List[str] = []
        self.actions: List[dict] = []

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


def make_env():
    bot = FakeBot()
    st = BotState()
    ctx = FakeContext(bot, {"state": st})
    return bot, ctx, st


def callbacks_of(markup) -> List[str]:
    rows = getattr(markup, "inline_keyboard", None) or []
    out: List[str] = []
    for row in rows:
        for button in row:
            cb = getattr(button, "callback_data", None)
            if cb is not None:
                out.append(cb)
    return out


def texts_to(bot: FakeBot, chat_id: int) -> List[str]:
    return [m["text"] for m in bot.sent if m["chat_id"] == chat_id]


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
# LLM + retrieval stubs
# ---------------------------------------------------------------------------
class LLMCapture:
    """Stands in for services.skills.chat: records calls, returns canned values."""

    def __init__(self, *responses: Any, default: str = "DRAFT REPLY") -> None:
        self.responses: List[Any] = list(responses)
        self.default = default
        self.calls: List[Any] = []

    async def __call__(self, tier: str, messages, tools=None, json_mode=False):
        self.calls.append(messages)
        value = self.responses.pop(0) if self.responses else self.default
        if isinstance(value, BaseException):
            raise value
        return value


async def fake_retrieve(session, user, query: str, now=None, timeout=None, top_k=None, min_similarity=None):
    """Offline stand-in for services.retrieval.retrieve_memory: an empty result, no embedding."""
    from backend.services.retrieval import RetrievalResult

    return RetrievalResult(items=[], candidates=[], candidates_considered=0, timings={})


def skill_patches(capture: LLMCapture) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(patch("backend.services.skills.chat", new=capture))
    stack.enter_context(patch("backend.services.skills.retrieve_memory", new=fake_retrieve))
    return stack


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
async def db_add_user(chat_id: int) -> int:
    async with AsyncSessionLocal() as session:
        user = User(
            username=f"botskill_{uuid.uuid4().hex[:8]}",
            telegram_chat_id=chat_id,
            timezone="Asia/Kolkata",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.id


async def db_add_user_skill(user_id: int, name: str, description: str, instructions: str) -> int:
    async with AsyncSessionLocal() as session:
        skill = UserSkill(
            user_id=user_id, name=name, description=description, instructions=instructions
        )
        session.add(skill)
        await session.commit()
        await session.refresh(skill)
        return skill.id


async def db_count_tasks(user_id: int) -> int:
    async with AsyncSessionLocal() as session:
        stmt = select(func.count()).select_from(Task).where(Task.user_id == user_id)
        return int((await session.execute(stmt)).scalar_one())


async def db_task_by_title(user_id: int, title: str):
    async with AsyncSessionLocal() as session:
        stmt = select(Task).where(Task.user_id == user_id, Task.title == title)
        return (await session.execute(stmt)).scalars().first()


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
            await session.execute(select(User).where(User.username.like("botskill_%")))
        ).scalars().all()
        for user in rows:
            await session.delete(user)
        await session.commit()
        return len(rows)


# ---------------------------------------------------------------------------
# Goal-breakdown JSON used by the pending-task tests
# ---------------------------------------------------------------------------
GOOD_JSON = json.dumps(
    {
        "summary": "Break the goal down.",
        "tasks": [
            {"title": "First step", "due": "2099-01-01"},
            {"title": "Past step", "due": "2020-01-01"},
            {"title": "Unscheduled step", "due": None},
        ],
    }
)


# ---------------------------------------------------------------------------
# K1: /skills lists built-ins and the empty custom-skills state
# ---------------------------------------------------------------------------
async def k1() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()

    await handlers.cmd_skills(FakeUpdate(chat, text="/skills"), ctx, state=st)

    texts = texts_to(bot, chat)
    blob = "\n".join(texts)
    missing = [name for name in BUILTIN_ORDER if name not in blob]
    desc_ok = BUILTIN_SKILLS["weekly-review"].description in blob
    ok = (
        len(texts) == 1
        and not missing
        and handlers.SKILLS_BUILTIN_HEADER in blob
        and handlers.SKILLS_CUSTOM_HEADER in blob
        and handlers.SKILLS_NO_CUSTOM in blob
        and desc_ok
    )
    return row(
        "K1", "/skills: built-ins + empty custom state",
        "1 message: 8 built-ins with descriptions + 'no custom skills' line",
        f"messages={len(texts)}, missing={missing}, desc_ok={desc_ok}",
        ok,
    )


# ---------------------------------------------------------------------------
# K2: /skills with a custom skill present includes it (and hides other users')
# ---------------------------------------------------------------------------
async def k2() -> dict:
    chat = next_chat()
    other = next_chat()
    uid = await db_add_user(chat)
    uid_other = await db_add_user(other)
    await db_add_user_skill(uid, "my-notes", "My notes skill", "Summarise my notes.")
    await db_add_user_skill(uid_other, "not-mine", "Other user skill", "Do other.")

    bot, ctx, st = make_env()
    await handlers.cmd_skills(FakeUpdate(chat, text="/skills"), ctx, state=st)

    texts = texts_to(bot, chat)
    blob = "\n".join(texts)
    ok = (
        len(texts) == 1
        and "my-notes - My notes skill" in blob
        and handlers.SKILLS_NO_CUSTOM not in blob
        and "not-mine" not in blob
        and all(name in blob for name in BUILTIN_ORDER)
    )
    return row(
        "K2", "/skills with a custom skill",
        "own custom skill listed; no-custom line gone; other user's skill hidden",
        f"messages={len(texts)}, has_custom={'my-notes - My notes skill' in blob}, "
        f"leaks_other={'not-mine' in blob}",
        ok,
    )


# ---------------------------------------------------------------------------
# K3: /skill <valid> with no input works for a no-input-required skill
# ---------------------------------------------------------------------------
async def k3() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    capture = LLMCapture(default="DRAFT REPLY")

    with skill_patches(capture):
        await handlers.cmd_skill(FakeUpdate(chat, text="/skill weekly-review"), ctx, state=st)

    texts = texts_to(bot, chat)
    ok = (
        texts == ["DRAFT REPLY"]
        and handlers.GENERIC_ERROR not in texts
        and len(capture.calls) == 1
    )
    return row(
        "K3", "/skill weekly-review (no input)",
        "reply sent, one LLM call, no generic error",
        f"texts={texts}, llm_calls={len(capture.calls)}",
        ok,
    )


# ---------------------------------------------------------------------------
# K4: /skill <valid> missing required input gives a sensible error, not a 500
# ---------------------------------------------------------------------------
async def k4() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    capture = LLMCapture()

    with skill_patches(capture):
        await handlers.cmd_skill(FakeUpdate(chat, text="/skill topic-brief"), ctx, state=st)

    texts = texts_to(bot, chat)
    last = texts[-1] if texts else ""
    ok = (
        len(texts) == 1
        and "needs an input" in last
        and handlers.GENERIC_ERROR not in texts
        and len(capture.calls) == 0
    )
    return row(
        "K4", "/skill topic-brief (missing required input)",
        "one friendly 'needs an input' message; no 500 / generic error; no LLM call",
        f"texts={texts}, llm_calls={len(capture.calls)}",
        ok,
    )


# ---------------------------------------------------------------------------
# K5: /skill <unknown> -> not-found-style message
# ---------------------------------------------------------------------------
async def k5() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    capture = LLMCapture()

    with skill_patches(capture):
        await handlers.cmd_skill(FakeUpdate(chat, text="/skill no-such-skill"), ctx, state=st)

    texts = texts_to(bot, chat)
    ok = texts == [handlers.SKILL_NOT_FOUND_TEXT] and len(capture.calls) == 0
    return row(
        "K5", "/skill unknown name",
        f"exactly one '{handlers.SKILL_NOT_FOUND_TEXT}' message; no LLM call",
        f"texts={texts}",
        ok,
    )


# ---------------------------------------------------------------------------
# K6: /skill goal-breakdown with pending tasks -> sk:ok / sk:no buttons (never tk:)
# ---------------------------------------------------------------------------
async def k6() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    capture = LLMCapture(GOOD_JSON)

    with skill_patches(capture):
        await handlers.cmd_skill(FakeUpdate(chat, text="/skill goal-breakdown ship it"), ctx, state=st)

    cb_msgs = [
        m for m in bot.sent if m["chat_id"] == chat and m["reply_markup"] is not None
    ]
    cbs = [cb for m in cb_msgs for cb in callbacks_of(m["reply_markup"])]
    any_tk = [c for c in cbs if c.startswith("tk:")]
    ok_pat = re.compile(r"^sk:ok:[0-9a-f-]{36}:\d+$")
    no_pat = re.compile(r"^sk:no:[0-9a-f-]{36}:\d+$")
    oks = [c for c in cbs if ok_pat.match(c)]
    nos = [c for c in cbs if no_pat.match(c)]
    first_prompt = "Save this task? First step (due 2099-01-01)"
    prompts_ok = any(m["text"] == first_prompt for m in cb_msgs)

    ok = (
        len(cb_msgs) == 3
        and len(oks) == 3
        and len(nos) == 3
        and not any_tk
        and prompts_ok
    )
    return row(
        "K6", "/skill goal-breakdown: pending task buttons",
        "3 candidate messages with sk:ok/sk:no buttons; no tk: prefix",
        f"button_msgs={len(cb_msgs)}, ok={len(oks)}, no={len(nos)}, tk={any_tk}, "
        f"first_prompt_ok={prompts_ok}",
        ok,
    )


# ---------------------------------------------------------------------------
# K7: sk:ok confirms the right candidate through the followup mechanism
# ---------------------------------------------------------------------------
async def k7() -> dict:
    chat = next_chat()
    uid = await db_add_user(chat)
    bot, ctx, st = make_env()
    capture = LLMCapture(GOOD_JSON)

    with skill_patches(capture):
        await handlers.cmd_skill(FakeUpdate(chat, text="/skill goal-breakdown ship it"), ctx, state=st)

    cb_msgs = [m for m in bot.sent if m["chat_id"] == chat and m["reply_markup"] is not None]
    first_cb = callbacks_of(cb_msgs[0]["reply_markup"])[0]
    turn_id = first_cb.split(":")[2]

    before = await db_count_tasks(uid)
    await press(bot, ctx, st, chat, f"sk:ok:{turn_id}:0")
    after = await db_count_tasks(uid)
    task = await db_task_by_title(uid, "First step")
    last_edit = bot.edits[-1]["text"] if bot.edits else ""

    ok = (
        after == before + 1
        and task is not None
        and task.status == "open"
        and task.origin == "extracted"
        and last_edit.startswith("Saved")
    )
    return row(
        "K7", "sk:ok callback confirms the right candidate",
        "exactly one Task 'First step' created; message replaced with a Saved notice",
        f"before={before}, after={after}, title={getattr(task, 'title', None)}, "
        f"status={getattr(task, 'status', None)}, edit={last_edit[:20]!r}",
        ok,
    )


# ---------------------------------------------------------------------------
# K8: sk:no discards it (no DB write)
# ---------------------------------------------------------------------------
async def k8() -> dict:
    chat = next_chat()
    uid = await db_add_user(chat)
    bot, ctx, st = make_env()
    capture = LLMCapture(GOOD_JSON)

    with skill_patches(capture):
        await handlers.cmd_skill(FakeUpdate(chat, text="/skill goal-breakdown ship it"), ctx, state=st)

    cb_msgs = [m for m in bot.sent if m["chat_id"] == chat and m["reply_markup"] is not None]
    first_cb = callbacks_of(cb_msgs[0]["reply_markup"])[0]
    turn_id = first_cb.split(":")[2]

    before = await db_count_tasks(uid)
    await press(bot, ctx, st, chat, f"sk:no:{turn_id}:1")
    after = await db_count_tasks(uid)
    last_edit = bot.edits[-1]["text"] if bot.edits else ""

    ok = after == before and last_edit == handlers._DISCARDED
    return row(
        "K8", "sk:no callback discards it",
        "no Task rows written; message replaced with the discard notice",
        f"before={before}, after={after}, edit={last_edit!r}",
        ok,
    )


# ---------------------------------------------------------------------------
# K9: expired / missing turn_id on a callback does not crash
# ---------------------------------------------------------------------------
async def k9() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    missing_turn = "00000000-0000-0000-0000-000000000000"

    raised = False
    try:
        await press(bot, ctx, st, chat, f"sk:ok:{missing_turn}:0")
        ok_edit = bot.edits[-1]["text"] == handlers.NOT_AVAILABLE
        await press(bot, ctx, st, chat, f"sk:no:{missing_turn}:1")
        no_edit = bot.edits[-1]["text"] == handlers.NOT_AVAILABLE
        await press(bot, ctx, st, chat, "sk:ok:garbage")
        bad_edit = bot.edits[-1]["text"] == handlers.NOT_AVAILABLE
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raised = True
        ok_edit = no_edit = bad_edit = False

    ok = (not raised) and ok_edit and no_edit and bad_edit
    return row(
        "K9", "expired/missing turn_id on a callback",
        "no crash; all three replies are the friendly 'no longer available' message",
        f"raised={raised}, ok={ok_edit}, no={no_edit}, garbage={bad_edit}",
        ok,
    )


# ---------------------------------------------------------------------------
# K10: rate limit (10/600s) surfaces as a friendly message, not a raw error
# ---------------------------------------------------------------------------
async def k10() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    capture = LLMCapture()
    reset_rate_limits()
    try:
        with skill_patches(capture):
            for _ in range(11):
                await handlers.cmd_skill(
                    FakeUpdate(chat, text="/skill weekly-review"), ctx, state=st
                )
    finally:
        reset_rate_limits()

    texts = texts_to(bot, chat)
    last = texts[-1] if texts else ""
    raw_leak = any("too many skill runs" in t for t in texts)
    ok = (
        last == handlers.SKILL_RATE_LIMITED_TEXT
        and not raw_leak
        and handlers.GENERIC_ERROR not in texts
        and len(capture.calls) == 10
    )
    return row(
        "K10", "rate limit: 10 runs then a friendly message",
        "11th run -> friendly rate-limit sentence; no raw 429 body / generic error",
        f"llm_calls={len(capture.calls)}, last={last!r}, raw_leak={raw_leak}",
        ok,
    )


# ---------------------------------------------------------------------------
# K11: a reply longer than 4096 chars is split, not truncated
# ---------------------------------------------------------------------------
async def k11() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    long_reply = "A" * 5000
    capture = LLMCapture(long_reply)

    with skill_patches(capture):
        await handlers.cmd_skill(FakeUpdate(chat, text="/skill weekly-review"), ctx, state=st)

    texts = texts_to(bot, chat)
    ok = (
        len(texts) >= 2
        and all(len(t) <= 4096 for t in texts)
        and "".join(texts) == long_reply
    )
    return row(
        "K11", "reply > 4096 chars is split",
        ">=2 messages, each <= 4096 chars, joined equals the full reply",
        f"messages={len(texts)}, lengths={[len(t) for t in texts]}, rejoined_ok={''.join(texts) == long_reply}",
        ok,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
CASES = [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11]


async def main() -> int:
    settings.require("DATABASE_URL")

    print("=" * 130)
    print("BOT SKILLS COMMANDS: /skills and /skill EVALUATION")
    print("=" * 130)

    results: List[dict] = []
    try:
        purged = await purge_synthetic()
        print(f"PURGE at start: removed {purged} synthetic user(s).\n")
        for case in CASES:
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
        await engine.dispose()

    print("\n" + "=" * 130)
    print(f"{'ID':<5} | {'STATUS':<6} | {'NAME':<48} | {'ACTUAL'}")
    print("-" * 130)
    all_passed = True
    for r in results:
        if not r["ok"]:
            all_passed = False
        print(
            f"{r['id']:<5} | {'PASS' if r['ok'] else 'FAIL':<6} | {r['name']:<48} | {r['actual']}"
        )
    print("=" * 130)
    passed = sum(1 for r in results if r["ok"])
    print(f"RESULT: {'ALL PASS' if all_passed else 'FAILURES PRESENT'} ({passed}/{len(results)})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
