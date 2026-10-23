"""backend/scripts/test_single_process.py
S1-S7 for single-process mode: the API lifespan hosting the Telegram poller, the reminder
scheduler and the Whisper warm-up.

No network, no real Telegram, no uvicorn and no bot polling: the Application, the scheduler
loop and the Whisper warm-up are replaced with fakes. /health is exercised in-process with
httpx.ASGITransport (its DB readiness check is the real one).

Usage:
  backend\\venv\\Scripts\\python.exe backend\\scripts\\test_single_process.py
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

from backend.api import main as main_module
from backend.config.settings import settings

RESULTS: list[tuple[str, str, bool, str]] = []

# Used only when the environment has no token at all, so the enabled-path tests can run.
# Never printed, never written anywhere.
FAKE_TOKEN = "000000000:single-process-test-placeholder"


def record(case_id: str, name: str, ok: bool, detail: str) -> None:
    RESULTS.append((case_id, name, ok, detail))
    print(f"[{case_id}] {name}: {'PASS' if ok else 'FAIL'} - {detail}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeUpdater:
    def __init__(self, calls: list, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail
        self.polling_kwargs: dict = {}

    async def start_polling(self, **kwargs):
        self.calls.append("start_polling")
        if self.fail:
            raise RuntimeError("start_polling failed (fake)")
        self.polling_kwargs = kwargs

    async def stop(self):
        self.calls.append("updater.stop")


class FakeApplication:
    """Records the lifecycle calls in order; the bot attribute satisfies the scheduler."""

    def __init__(self, calls: list, fail_start_polling: bool = False) -> None:
        self.calls = calls
        self.bot = object()
        self.updater = FakeUpdater(calls, fail=fail_start_polling)

    async def initialize(self):
        self.calls.append("initialize")

    async def start(self):
        self.calls.append("start")

    async def stop(self):
        self.calls.append("application.stop")

    async def shutdown(self):
        self.calls.append("application.shutdown")


def fake_scheduler(calls: list):
    async def scheduler_loop(bot):
        calls.append("scheduler")
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            calls.append("scheduler.cancelled")
            raise

    return scheduler_loop


def fake_warmup(calls: list, seconds: float = 0.0):
    def warmup():
        calls.append("warmup")
        if seconds:
            time.sleep(seconds)

    return warmup


async def health_status() -> int:
    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        return response.status_code


def enabled(on: bool) -> tuple[bool, str]:
    """Turn BOT_ENABLED on/off, guaranteeing a non-empty token. Returns the old values."""
    old_enabled, old_token = settings.BOT_ENABLED, settings.TELEGRAM_BOT_TOKEN
    settings.BOT_ENABLED = on
    if on and not settings.TELEGRAM_BOT_TOKEN.strip():
        settings.TELEGRAM_BOT_TOKEN = FAKE_TOKEN
    return old_enabled, old_token


def restore(old_enabled: bool, old_token: str) -> None:
    settings.BOT_ENABLED = old_enabled
    settings.TELEGRAM_BOT_TOKEN = old_token


# ---------------------------------------------------------------------------
# S1-S7
# ---------------------------------------------------------------------------
async def s1() -> None:
    """Default BOT_ENABLED=false: lifespan runs, no Application constructed, /health 200."""
    default_flag = settings.BOT_ENABLED
    old_enabled, old_token = enabled(False)
    built = -1
    status = 0
    try:
        with patch(
            "backend.bot.app.build_application",
            side_effect=AssertionError("build_application must not run by default"),
        ) as builder:
            async with main_module.lifespan(main_module.app):
                status = await health_status()
        built = builder.call_count
    finally:
        restore(old_enabled, old_token)
    ok = default_flag is False and built == 0 and status == 200
    record(
        "S1",
        "default BOT_ENABLED=false, no Application",
        ok,
        f"default={default_flag}, build_calls={built}, /health={status}",
    )


async def s2() -> None:
    """Enabled: initialize -> start -> start_polling -> scheduler; shutdown in reverse."""
    calls: list = []
    app = FakeApplication(calls)
    old_enabled, old_token = enabled(True)
    status = 0
    try:
        with patch("backend.bot.app.build_application", return_value=app), patch(
            "backend.bot.scheduler.scheduler_loop", fake_scheduler(calls)
        ), patch("backend.services.audio.warmup", fake_warmup(calls)):
            async with main_module.lifespan(main_module.app):
                await asyncio.sleep(0.05)
                status = await health_status()
            await asyncio.sleep(0.05)
    finally:
        restore(old_enabled, old_token)

    startup = [c for c in calls if c in ("initialize", "start", "start_polling", "scheduler")]
    startup_ok = startup[:4] == ["initialize", "start", "start_polling", "scheduler"]
    tail = calls[calls.index("scheduler.cancelled"):] if "scheduler.cancelled" in calls else []
    shutdown_ok = tail[:4] == [
        "scheduler.cancelled",
        "updater.stop",
        "application.stop",
        "application.shutdown",
    ]
    ok = startup_ok and shutdown_ok and status == 200
    record(
        "S2",
        "lifecycle order + reverse shutdown",
        ok,
        f"startup={startup}, shutdown={tail}, /health={status}, polling_kwargs={app.updater.polling_kwargs}",
    )


async def s3() -> None:
    """start_polling raises: nothing escapes, /health 200, cleanup ran, no scheduler."""
    calls: list = []
    app = FakeApplication(calls, fail_start_polling=True)
    old_enabled, old_token = enabled(True)
    escaped = None
    status = 0
    try:
        with patch("backend.bot.app.build_application", return_value=app), patch(
            "backend.bot.scheduler.scheduler_loop", fake_scheduler(calls)
        ), patch("backend.services.audio.warmup", fake_warmup(calls)):
            try:
                async with main_module.lifespan(main_module.app):
                    status = await health_status()
            except BaseException as exc:  # noqa: BLE001 - the whole point of S3
                escaped = repr(exc)
    finally:
        restore(old_enabled, old_token)
    cleanup_ok = all(
        name in calls for name in ("updater.stop", "application.stop", "application.shutdown")
    )
    ok = escaped is None and status == 200 and cleanup_ok and "scheduler" not in calls
    record(
        "S3",
        "start_polling failure is contained",
        ok,
        f"escaped={escaped}, /health={status}, calls={calls}",
    )


async def s4() -> None:
    """Empty token: the bot is skipped, /health 200."""
    old_enabled, old_token = enabled(True)
    settings.TELEGRAM_BOT_TOKEN = ""
    built = -1
    status = 0
    try:
        with patch(
            "backend.bot.app.build_application",
            side_effect=AssertionError("build_application must not run without a token"),
        ) as builder:
            async with main_module.lifespan(main_module.app):
                status = await health_status()
        built = builder.call_count
    finally:
        restore(old_enabled, old_token)
    ok = built == 0 and status == 200
    record("S4", "empty token skips the bot", ok, f"build_calls={built}, /health={status}")


async def s5() -> None:
    """Slow Warm-up (3 s): startup finishes in < 1 s and /health answers while it runs."""
    calls: list = []
    app = FakeApplication(calls)
    old_enabled, old_token = enabled(True)
    startup_s = -1.0
    total_s = -1.0
    status = 0
    try:
        with patch("backend.bot.app.build_application", return_value=app), patch(
            "backend.bot.scheduler.scheduler_loop", fake_scheduler(calls)
        ), patch("backend.services.audio.warmup", fake_warmup(calls, seconds=3.0)):
            started = time.perf_counter()
            async with main_module.lifespan(main_module.app):
                startup_s = time.perf_counter() - started
                status = await health_status()
            total_s = time.perf_counter() - started
    finally:
        restore(old_enabled, old_token)
    ok = startup_s < 1.0 and status == 200 and "warmup" in calls
    record(
        "S5",
        "warm-up never blocks startup",
        ok,
        f"startup={startup_s:.3f}s, total_in_lifespan={total_s:.3f}s, /health={status}, warmup_started={'warmup' in calls}",
    )


async def s6() -> None:
    """Importing backend.bot.__main__ does not start polling."""
    calls: list = []
    with patch(
        "backend.bot.app.build_application",
        side_effect=AssertionError("importing __main__ must not build an Application"),
    ) as builder:
        module = importlib.import_module("backend.bot.__main__")
        importlib.reload(module)
    ok = builder.call_count == 0 and callable(getattr(module, "main", None))
    record(
        "S6",
        "__main__ import has no side effects",
        ok,
        f"build_calls={builder.call_count}, main_callable={callable(getattr(module, 'main', None))}",
    )


async def s7() -> None:
    """Scheduler creation raises after a good start: contained, cleanup in order, no leftovers."""
    calls: list = []
    app = FakeApplication(calls)
    old_enabled, old_token = enabled(True)
    escaped = None
    status = 0

    def boom_scheduler(bot):  # noqa: ARG001 - mirrors scheduler_loop(bot)
        raise RuntimeError("scheduler creation failed (fake)")

    before = set(asyncio.all_tasks())
    try:
        with patch("backend.bot.app.build_application", return_value=app), patch(
            "backend.bot.scheduler.scheduler_loop", boom_scheduler
        ), patch("backend.services.audio.warmup", fake_warmup(calls)):
            try:
                async with main_module.lifespan(main_module.app):
                    status = await health_status()
            except BaseException as exc:  # noqa: BLE001 - the whole point of S7
                escaped = repr(exc)
    finally:
        restore(old_enabled, old_token)

    await asyncio.sleep(0.05)
    leftover = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and t not in before and not t.done()
    ]
    cleanup = [c for c in calls if c in ("updater.stop", "application.stop", "application.shutdown")]
    cleanup_ok = cleanup == ["updater.stop", "application.stop", "application.shutdown"]
    ok = escaped is None and status == 200 and cleanup_ok and not leftover
    record(
        "S7",
        "background-task failure is contained",
        ok,
        f"escaped={escaped}, /health={status}, cleanup={cleanup}, leftover_tasks={len(leftover)}, calls={calls}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def main() -> int:
    print("=" * 100)
    print("SINGLE-PROCESS MODE TESTS (S1-S7)")
    print("=" * 100)
    print(f"BOT_ENABLED as loaded (settings default / .env) = {settings.BOT_ENABLED}")

    for case in (s1, s2, s3, s4, s5, s6, s7):
        try:
            await case()
        except BaseException as exc:  # noqa: BLE001 - reported as a failed case
            traceback.print_exc()
            record(case.__name__.upper(), case.__name__, False, f"unhandled: {exc!r}")

    passed = sum(1 for _, _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("-" * 100)
    print(f"TOTAL: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
