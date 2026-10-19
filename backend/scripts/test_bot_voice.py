"""backend/scripts/test_bot_voice.py
Phase 4b evaluation for the Telegram bot voice handler.

Fake Telegram objects (no network) and a fake transcriber everywhere except V11.
The transcribe seam is patched at the model level (audio.get_whisper_model /
audio.transcribe_samples), so the real temp-file + PyAV decode path in
audio.transcribe_bytes still runs for every case.

V11 is MEASURE ONLY: it loads the real model and prints load/transcribe time and RSS.

Throwaway users only; every row is deleted in a finally block.
"""

import asyncio
import logging
import os
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import time
import uuid

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select

from backend.bot import handlers
from backend.bot.state import BotState
from backend.config.settings import settings
from backend.database import AsyncSessionLocal, engine
from backend.models.models import User
from backend.services import audio as audio_service
from backend.services import orchestrator
from backend.services.orchestrator import PendingRecord, PostResult, ReplyResult

CHAT_IDS: set[int] = set()
_next_chat = [950000]


def next_chat() -> int:
    _next_chat[0] += 1
    CHAT_IDS.add(_next_chat[0])
    return _next_chat[0]


# ---------------------------------------------------------------------------
# Real OGG/Opus test audio (3 s synthetic tone, in memory)
# ---------------------------------------------------------------------------
def _synth_ogg_bytes(seconds: float = 3.0) -> bytes:
    import numpy as np

    fd, path = tempfile.mkstemp(suffix=".ogg")
    os.close(fd)
    try:
        n = int(seconds * 16000)
        t = np.arange(n, dtype=np.float64) / 16000.0
        samples = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        audio_service.encode_ogg_opus(samples, path)
        with open(path, "rb") as handle:
            return handle.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


OGG_BYTES = _synth_ogg_bytes()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeChat:
    def __init__(self, chat_id: int, chat_type: str = "private") -> None:
        self.id = chat_id
        self.type = chat_type


class FakeVoice:
    def __init__(self, duration: int = 3, file_id: str = "voice-file") -> None:
        self.duration = duration
        self.file_id = file_id


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


class FakeUpdate:
    def __init__(self, chat_id: int, text=None, voice=False, duration: int = 3) -> None:
        self.effective_chat = FakeChat(chat_id)
        self.effective_user = FakeTelegramUser(chat_id)
        self.message = FakeMessage(
            chat_id, text=text, voice=FakeVoice(duration=duration) if voice else None
        )
        self.callback_query = None


class FakeTgFile:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def download_as_bytearray(self) -> bytearray:
        return bytearray(self._data)


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.answers: list[str] = []
        self.actions: list[dict] = []
        self.download_calls = 0
        self.download_error: Exception | None = None
        self._t0 = time.perf_counter()
        self._mid = 200

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    async def get_file(self, file_id, **kwargs):
        self.download_calls += 1
        if self.download_error is not None:
            raise self.download_error
        return FakeTgFile(OGG_BYTES)

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


def make_env():
    bot = FakeBot()
    st = BotState()
    ctx = FakeContext(bot, {"state": st})
    return bot, ctx, st


# ---------------------------------------------------------------------------
# Patchers
# ---------------------------------------------------------------------------
@contextmanager
def patched_settings(**kwargs):
    old = {k: getattr(settings, k) for k in kwargs}
    for key, value in kwargs.items():
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(settings, key, value)


@contextmanager
def patched_transcriber(fn):
    """Patch the Whisper model, keeping audio.transcribe_bytes' temp-file/decode path real."""
    old_model = audio_service.get_whisper_model
    old_tx = audio_service.transcribe_samples
    calls = {"n": 0}

    def fake_model():
        return object()

    def fake_tx(model, samples, vad_filter=True):
        calls["n"] += 1
        return fn(samples)

    audio_service.get_whisper_model = fake_model
    audio_service.transcribe_samples = fake_tx
    try:
        yield calls
    finally:
        audio_service.get_whisper_model = old_model
        audio_service.transcribe_samples = old_tx


@contextmanager
def mock_orchestrator(respond_fn, postprocess_fn):
    old_r, old_p = orchestrator.respond, orchestrator.postprocess
    orchestrator.respond = respond_fn
    orchestrator.postprocess = postprocess_fn
    try:
        yield
    finally:
        orchestrator.respond = old_r
        orchestrator.postprocess = old_p


def make_respond(reply_text="mocked reply", post_record=None):
    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        return ReplyResult(turn_id="turn", session_id=session_id, reply=reply_text, memory_used=[])
    return respond


def make_postprocess(pending_records=None):
    async def postprocess(user_id, reply, *, now=None):
        records = pending_records or []
        return PostResult(pending_records=records)
    return postprocess


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
async def db_add_user(chat_id: int) -> int:
    async with AsyncSessionLocal() as session:
        user = User(
            username=f"voice_{uuid.uuid4().hex[:8]}",
            telegram_chat_id=chat_id,
            timezone="Asia/Kolkata",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.id


async def warm_db() -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(select(User).limit(1))


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


async def drain(st) -> None:
    while st.background:
        pending = list(st.background)
        await asyncio.gather(*pending, return_exceptions=True)


def row(case_id, name, expected, actual, ok, details="") -> dict:
    return {"id": case_id, "name": name, "expected": expected, "actual": actual, "ok": ok, "details": details}


async def send_voice(chat_id: int, ctx, st, duration: int = 3) -> None:
    await handlers.on_voice(FakeUpdate(chat_id, voice=True, duration=duration), ctx, state=st)


def texts_for(bot: FakeBot, chat_id: int) -> list[str]:
    return [m["text"] for m in bot.sent if m["chat_id"] == chat_id]


# ---------------------------------------------------------------------------
# V1: unknown chat
# ---------------------------------------------------------------------------
async def v1() -> dict:
    chat = next_chat()  # no user row
    bot, ctx, st = make_env()
    with patched_transcriber(lambda samples: "should not run") as calls:
        await send_voice(chat, ctx, st)
    texts = texts_for(bot, chat)
    ok = (
        texts == [handlers.SIGNUP_PROMPT]
        and bot.download_calls == 0
        and calls["n"] == 0
    )
    return row("V1", "unknown chat voice -> signup",
               "signup prompt; 0 downloads; 0 transcriptions",
               f"texts={texts}, downloads={bot.download_calls}, transcriptions={calls['n']}", ok)


# ---------------------------------------------------------------------------
# V2: duration over the cap
# ---------------------------------------------------------------------------
async def v2() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    with patched_settings(VOICE_MAX_SECONDS=60):
        with patched_transcriber(lambda samples: "should not run") as calls:
            await send_voice(chat, ctx, st, duration=61)
    texts = texts_for(bot, chat)
    expected_msg = handlers.VOICE_TOO_LONG.format(60)
    ok = (
        texts == [expected_msg]
        and bot.download_calls == 0
        and calls["n"] == 0
    )
    return row("V2", "duration over cap -> refused",
               "too-long message; 0 downloads; 0 transcriptions",
               f"texts={texts}, downloads={bot.download_calls}, transcriptions={calls['n']}", ok)


# ---------------------------------------------------------------------------
# V3: happy path
# ---------------------------------------------------------------------------
async def v3() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    captured = {"n": 0, "text": None, "source": None}

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        captured["n"] += 1
        captured["text"] = text
        captured["source"] = source
        return ReplyResult(turn_id="t", session_id=session_id, reply="mocked reply", memory_used=[])

    record = PendingRecord(
        id=424242, content="User likes tea", kind="knowledge", certainty="tentative",
        source="voice", status="pending",
    )
    with patched_transcriber(lambda samples: "hello world"):
        with mock_orchestrator(respond, make_postprocess([record])):
            await send_voice(chat, ctx, st)
            await drain(st)

    texts = texts_for(bot, chat)
    heard_first = bool(texts) and texts[0] == "Heard: hello world"
    confirm_present = any(t.startswith("Save this memory?") for t in texts)
    confirm_after_reply = False
    if confirm_present:
        reply_idx = next(i for i, t in enumerate(texts) if t == "mocked reply")
        confirm_idx = next(i for i, t in enumerate(texts) if t.startswith("Save this memory?"))
        confirm_after_reply = confirm_idx > reply_idx
    ok = (
        heard_first
        and confirm_after_reply
        and captured["n"] == 1
        and captured["text"] == "hello world"
        and captured["source"] == "voice"
    )
    return row("V3", "voice happy path",
               '"Heard: ..." then reply; respond once with transcript + source=voice; confirm after',
               f"texts={texts}, respond_calls={captured['n']}, text={captured['text']!r}, "
               f"source={captured['source']}, confirm_after_reply={confirm_after_reply}", ok)


# ---------------------------------------------------------------------------
# V4: empty / whitespace transcripts
# ---------------------------------------------------------------------------
async def v4() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    captured = {"n": 0}

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        captured["n"] += 1
        return ReplyResult(turn_id="t", session_id=session_id, reply="mocked reply", memory_used=[])

    results = []
    for transcript in ("", "   "):
        bot.sent.clear()
        with patched_transcriber(lambda samples, tx=transcript: tx):
            with mock_orchestrator(respond, make_postprocess()):
                await send_voice(chat, ctx, st)
                await drain(st)
        results.append(texts_for(bot, chat) == [handlers.VOICE_NO_SPEECH])
    ok = all(results) and captured["n"] == 0
    return row("V4", "empty/whitespace transcript",
               "no-speech message; respond not called",
               f"per_case_ok={results}, respond_calls={captured['n']}", ok)


# ---------------------------------------------------------------------------
# V5: transcriber raises
# ---------------------------------------------------------------------------
async def v5() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    captured = {"n": 0}

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        captured["n"] += 1
        return ReplyResult(turn_id="t", session_id=session_id, reply="mocked reply", memory_used=[])

    def boom(samples):
        raise RuntimeError("transcriber exploded")

    raised = False
    try:
        with patched_transcriber(boom):
            with mock_orchestrator(respond, make_postprocess()):
                await send_voice(chat, ctx, st)
                await drain(st)
    except Exception:  # noqa: BLE001
        raised = True

    texts = texts_for(bot, chat)
    ok = (not raised) and texts == [handlers.VOICE_ERROR] and captured["n"] == 0
    return row("V5", "transcriber raises -> friendly",
               "friendly error; no raise; respond not called",
               f"raised={raised}, texts={texts}, respond_calls={captured['n']}", ok)


# ---------------------------------------------------------------------------
# V6: download raises
# ---------------------------------------------------------------------------
async def v6() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()
    bot.download_error = RuntimeError("download exploded")
    captured = {"n": 0}

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        captured["n"] += 1
        return ReplyResult(turn_id="t", session_id=session_id, reply="mocked reply", memory_used=[])

    raised = False
    try:
        with patched_transcriber(lambda samples: "should not run") as calls:
            with mock_orchestrator(respond, make_postprocess()):
                await send_voice(chat, ctx, st)
                await drain(st)
    except Exception:  # noqa: BLE001
        raised = True

    texts = texts_for(bot, chat)
    ok = (not raised) and texts == [handlers.VOICE_ERROR] and calls["n"] == 0
    return row("V6", "download raises -> friendly",
               "friendly error; no raise; transcribe not called",
               f"raised={raised}, texts={texts}, transcriptions={calls['n']}", ok)


# ---------------------------------------------------------------------------
# V7: non-blocking transcription
# ---------------------------------------------------------------------------
async def v7() -> dict:
    chat_a = next_chat()
    chat_b = next_chat()
    await db_add_user(chat_a)
    await db_add_user(chat_b)
    bot, ctx, st = make_env()

    tx = {}

    def slow_transcribe(samples):
        tx["start"] = time.perf_counter()
        time.sleep(0.6)
        tx["end"] = time.perf_counter()
        return "voice text"

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        return ReplyResult(turn_id="t", session_id=session_id, reply="ok", memory_used=[])

    # Prime chat B's conversation session so the comparison is about transcription, not DB setup.
    with patched_transcriber(lambda samples: "warm"):
        with mock_orchestrator(respond, make_postprocess()):
            await handlers.on_text(FakeUpdate(chat_b, text="warmup"), ctx, state=st)
            await drain(st)
    bot.sent.clear()
    bot.actions.clear()

    with patched_transcriber(slow_transcribe):
        with mock_orchestrator(respond, make_postprocess()):
            await asyncio.gather(
                handlers.on_voice(FakeUpdate(chat_a, voice=True), ctx, state=st),
                handlers.on_text(FakeUpdate(chat_b, text="a text message"), ctx, state=st),
            )
            await drain(st)

    b_msgs = [m for m in bot.sent if m["chat_id"] == chat_b]
    # Same clock: the fake bot stores times relative to bot._t0, the transcriber stores
    # absolute perf_counter values. Convert the bot time to absolute before comparing.
    b_reply_abs = (bot._t0 + b_msgs[0]["t"]) if b_msgs else None
    tx_end = tx.get("end")
    ok = b_reply_abs is not None and tx_end is not None and b_reply_abs < tx_end
    return row("V7", "transcription does not block the loop",
               "chat B reply (abs) < transcription end (abs)",
               f"b_reply_abs={b_reply_abs}, tx_end={tx_end}, "
               f"b_before_tx_end={ok}", ok)


# ---------------------------------------------------------------------------
# V8: timeout
# ---------------------------------------------------------------------------
async def v8() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        return ReplyResult(turn_id="t", session_id=session_id, reply="mocked reply", memory_used=[])

    def sleepy(samples):
        time.sleep(1.0)
        return "too late"

    raised = False
    try:
        with patched_settings(VOICE_TRANSCRIBE_TIMEOUT_SECONDS=0.3):
            with patched_transcriber(sleepy):
                with mock_orchestrator(respond, make_postprocess()):
                    await send_voice(chat, ctx, st)
                    await drain(st)
    except Exception:  # noqa: BLE001
        raised = True

    texts = texts_for(bot, chat)
    ok = (not raised) and texts == [handlers.VOICE_ERROR]
    return row("V8", "transcription timeout -> friendly",
               "friendly error; no raise",
               f"raised={raised}, texts={texts}", ok)


# ---------------------------------------------------------------------------
# V9: no audio persisted
# ---------------------------------------------------------------------------
def _temp_listing() -> set[str]:
    # Only ".ogg" names (the suffix transcribe_bytes uses); the OS temp dir is shared and
    # volatile, so unrelated files appearing there must not fail the check.
    try:
        return {name for name in os.listdir(tempfile.gettempdir()) if name.endswith(".ogg")}
    except OSError:
        return set()


def _repo_audio_files() -> set[str]:
    found: set[str] = set()
    skip = {"venv", ".git", "__pycache__", "node_modules", "whisper_models"}
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            if name.lower().endswith((".ogg", ".opus", ".wav", ".mp3", ".m4a")):
                found.add(str(Path(root) / name))
    return found


async def v9() -> dict:
    chat = next_chat()
    await db_add_user(chat)
    bot, ctx, st = make_env()

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        return ReplyResult(turn_id="t", session_id=session_id, reply="mocked reply", memory_used=[])

    tmp_before = _temp_listing()
    repo_before = _repo_audio_files()

    with patched_transcriber(lambda samples: "cleanup check"):
        with mock_orchestrator(respond, make_postprocess()):
            await send_voice(chat, ctx, st)
            await drain(st)

    tmp_after = _temp_listing()
    repo_after = _repo_audio_files()

    tmp_same = tmp_before == tmp_after
    new_repo = repo_after - repo_before
    ok = tmp_same and not new_repo
    return row("V9", "no audio persisted",
               "temp dir listing unchanged; no new audio file under the repo",
               f"tmp_same={tmp_same}, new_repo_audio={sorted(new_repo)}", ok)


# ---------------------------------------------------------------------------
# V10: transcription serialization
# ---------------------------------------------------------------------------
async def v10() -> dict:
    chat_a = next_chat()
    chat_b = next_chat()
    await db_add_user(chat_a)
    await db_add_user(chat_b)
    bot, ctx, st = make_env()

    active = {"now": 0, "max": 0}

    def tracked(samples):
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        try:
            time.sleep(0.2)
            return "voice text"
        finally:
            active["now"] -= 1

    async def respond(user_id, text, *, session_id, history, source="text", now=None):
        return ReplyResult(turn_id="t", session_id=session_id, reply="ok", memory_used=[])

    with patched_transcriber(tracked):
        with mock_orchestrator(respond, make_postprocess()):
            await asyncio.gather(
                handlers.on_voice(FakeUpdate(chat_a, voice=True), ctx, state=st),
                handlers.on_voice(FakeUpdate(chat_b, voice=True), ctx, state=st),
            )
            await drain(st)

    a_ok = any(t.startswith("Heard: ") for t in texts_for(bot, chat_a))
    b_ok = any(t.startswith("Heard: ") for t in texts_for(bot, chat_b))
    ok = active["max"] == 1 and a_ok and b_ok
    return row("V10", "two voice notes serialize",
               "max concurrent transcriptions == 1; both replied",
               f"max_concurrency={active['max']}, a_replied={a_ok}, b_replied={b_ok}", ok)


# ---------------------------------------------------------------------------
# V11: real model, MEASURE ONLY
# ---------------------------------------------------------------------------
def peak_rss_mb():
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


async def run_v11() -> None:
    print("--- Running V11 (REAL model, MEASURE ONLY) ---")
    model_dir = settings.whisper_model_dir
    print(f"    model name      : {settings.WHISPER_MODEL_NAME}")
    print(f"    model dir       : {model_dir}")
    print(f"    dir exists      : {model_dir.is_dir()}")
    if not model_dir.is_dir():
        print("    [MEASURE] SKIPPED: model directory missing")
        return

    audio_service._model = None  # force a real load so the timing is honest
    try:
        started = time.perf_counter()
        await asyncio.to_thread(audio_service.warmup)
        load_s = time.perf_counter() - started

        started = time.perf_counter()
        text = await asyncio.to_thread(audio_service.transcribe_bytes, OGG_BYTES)
        tx_s = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        print(f"    [MEASURE] SKIPPED: {type(exc).__name__}: {exc}")
        return

    rss = peak_rss_mb()
    print(f"    load time       : {load_s:.2f}s")
    print(f"    transcribe time : {tx_s:.2f}s")
    print(f"    peak RSS        : {'n/a' if rss is None else f'{rss:.1f} MB'}")
    print(f"    transcript chars: {len(text)} (a tone is not speech; numbers only prove the path runs)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def main() -> int:
    settings.require("DATABASE_URL")
    print("=" * 140)
    print("PHASE 4b: TELEGRAM BOT VOICE EVALUATION")
    print("=" * 140)

    results: list[dict] = []
    cases = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
    # Optional filter: `test_bot_voice.py V7` runs only the named cases (case-insensitive).
    wanted = {arg.upper() for arg in sys.argv[1:]}
    if wanted:
        cases = [case for case in cases if case.__name__.upper() in wanted]
    try:
        await warm_db()
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

        if not wanted or "V11" in wanted:
            await run_v11()
    finally:
        await cleanup_users()
        await engine.dispose()

    print("\n" + "=" * 140)
    print(f"{'ID':<5} | {'NAME':<38} | {'EXPECTED':<58} | {'ACTUAL':<58} | STATUS")
    print("-" * 140)
    all_passed = True
    for r in results:
        if not r["ok"]:
            all_passed = False
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{r['id']:<5} | {r['name']:<38} | {r['expected']:<58} | {r['actual']:<58} | {status}")
    print("=" * 140)
    passed = sum(1 for r in results if r["ok"])
    print(f"RESULT: {'ALL PASS' if all_passed else 'FAILURES PRESENT'} ({passed}/{len(results)})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
