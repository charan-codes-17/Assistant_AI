"""backend/scripts/test_embeddings_hedge.py
Unit evaluation of the strategy-B query-embedding hedge in backend/services/embeddings.py.

No network and no database are used: embeddings._single_request is patched with deterministic
fakes, and the retrieval fail-open path is driven with a fake embed_texts that raises a real
openai RateLimitError (429). Nothing here prints vectors, keys or memory text.

Cases (small budgets so the script runs fast):
  1. first request succeeds immediately      -> exactly 1 request fired
  2. first request hangs, second succeeds    -> success at ~interval, hung request cancelled
  3. every request hangs                     -> TimeoutError at ~budget, no leaked tasks
  4. first request raises 429, later succeeds-> the success is returned
  5. every request raises                    -> clear failure (the error), not a hang
  6. in-flight count never exceeds 4         -> recorded max <= 4
  7. retrieval path: embedding raises        -> no memory, embedding_failed, no exception

Usage (from the repo root):
  backend\\venv\\Scripts\\python.exe -u backend\\scripts\\test_embeddings_hedge.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Tuple
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx2

import backend.services.embeddings as emb_module
import backend.services.retrieval as retrieval_module
from backend.config.settings import settings

DIM = settings.EMBEDDING_DIM
MAX_IN_FLIGHT_DEFAULT = 4


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def fake_response(count: int) -> Any:
    """Minimal stand-in for an OpenAI embeddings response (only data/index/embedding used)."""
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=i, embedding=[0.001 * (i + 1)] * DIM)
            for i in range(count)
        ]
    )


def build_429() -> BaseException:
    """A real openai RateLimitError (HTTP 429) with no network involved."""
    import openai

    request = httpx2.Request("POST", "https://embedding.invalid/v1/embeddings")
    response = httpx2.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


class ScriptedEmbedder:
    """Deterministic fake for embeddings._single_request.

    Behaviour per call (the last one is reused if more requests are fired than scripted):
      ("ok", delay)                 -> sleep(delay) then return a fake response
      ("hang", None)                -> block until cancelled
      ("raise", delay, factory)     -> sleep(delay) then raise factory()
    Tracks calls, max concurrent in-flight, and cancellations.
    """

    def __init__(self, behaviours: List[Tuple[Any, ...]]) -> None:
        self.behaviours = behaviours
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.cancelled = 0

    async def __call__(self, texts: List[str]) -> Any:
        index = self.calls
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            behaviour = (
                self.behaviours[index]
                if index < len(self.behaviours)
                else self.behaviours[-1]
            )
            kind = behaviour[0]
            if kind == "hang":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled += 1
                    raise
                return fake_response(len(texts))
            delay = behaviour[1] or 0.0
            if delay:
                await asyncio.sleep(delay)
            if kind == "raise":
                raise behaviour[2]()
            return fake_response(len(texts))
        finally:
            self.in_flight -= 1


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def call_hedged(
    embedder: Callable[..., Any],
    *,
    interval: float,
    budget: float,
    max_in_flight: Optional[int] = None,
) -> Tuple[Any, Optional[BaseException], float, List[str], List[str], List[str]]:
    """Run embed_texts on the query path with patched settings and capture warnings.

    Returns (vectors, error, elapsed, logs, unretrieved, destroyed).
    """
    loop = asyncio.get_running_loop()
    unretrieved: List[str] = []
    destroyed: List[str] = []
    capture = LogCapture()
    emb_logger = logging.getLogger("backend.services.embeddings")
    emb_logger.addHandler(capture)

    def _handler(_loop: Any, context: dict) -> None:
        message = str(context.get("message", ""))
        if "never retrieved" in message:
            unretrieved.append(message)
        if "destroyed" in message or "pending" in message:
            destroyed.append(message)

    prev_handler = loop.get_exception_handler()
    loop.set_exception_handler(_handler)

    vectors: Any = None
    error: Optional[BaseException] = None
    t0 = loop.time()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(emb_module, "_single_request", embedder))
            stack.enter_context(
                patch.object(settings, "EMBEDDING_HEDGE_SECONDS", interval)
            )
            stack.enter_context(
                patch.object(settings, "EMBEDDING_TIMEOUT_SECONDS", budget)
            )
            if max_in_flight is not None:
                stack.enter_context(
                    patch.object(settings, "EMBEDDING_HEDGE_MAX_IN_FLIGHT", max_in_flight)
                )
            try:
                vectors = await emb_module.embed_texts(["hedge probe text"], timeout=budget)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - reported per case
                error = exc
    finally:
        elapsed = loop.time() - t0
        emb_logger.removeHandler(capture)
        loop.set_exception_handler(prev_handler)

    return vectors, error, elapsed, list(capture.messages), unretrieved, destroyed


def hedge_logged(logs: List[str]) -> bool:
    return any("hedge" in message.lower() for message in logs)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

async def case_1() -> Tuple[bool, str]:
    embedder = ScriptedEmbedder([("ok", 0.0)])
    vectors, error, elapsed, logs, unretrieved, destroyed = await call_hedged(
        embedder, interval=0.05, budget=0.5
    )
    passed = (
        error is None
        and vectors is not None
        and len(vectors) == 1
        and len(vectors[0]) == DIM
        and embedder.calls == 1
        and not hedge_logged(logs)
        and unretrieved == []
        and destroyed == []
    )
    return passed, (
        f"error={error!r} calls={embedder.calls} vectors={0 if vectors is None else len(vectors)} "
        f"elapsed={elapsed:.3f}s hedge_log={hedge_logged(logs)}"
    )


async def case_2() -> Tuple[bool, str]:
    embedder = ScriptedEmbedder([("hang", None), ("ok", 0.0)])
    vectors, error, elapsed, logs, unretrieved, destroyed = await call_hedged(
        embedder, interval=0.2, budget=1.0
    )
    passed = (
        error is None
        and vectors is not None
        and len(vectors) == 1
        and embedder.calls == 2
        and embedder.cancelled == 1
        and 0.15 <= elapsed < 0.6
        and hedge_logged(logs)
        and unretrieved == []
        and destroyed == []
    )
    return passed, (
        f"error={error!r} calls={embedder.calls} cancelled={embedder.cancelled} "
        f"elapsed={elapsed:.3f}s hedge_log={hedge_logged(logs)} "
        f"unretrieved={len(unretrieved)} destroyed={len(destroyed)}"
    )


async def case_3() -> Tuple[bool, str]:
    embedder = ScriptedEmbedder([("hang", None)])
    vectors, error, elapsed, logs, unretrieved, destroyed = await call_hedged(
        embedder, interval=0.05, budget=0.3, max_in_flight=MAX_IN_FLIGHT_DEFAULT
    )
    passed = (
        isinstance(error, asyncio.TimeoutError)
        and vectors is None
        and embedder.calls == MAX_IN_FLIGHT_DEFAULT
        and embedder.cancelled == MAX_IN_FLIGHT_DEFAULT
        and 0.25 <= elapsed < 0.7
        and unretrieved == []
        and destroyed == []
    )
    return passed, (
        f"error={type(error).__name__} calls={embedder.calls} cancelled={embedder.cancelled} "
        f"elapsed={elapsed:.3f}s unretrieved={len(unretrieved)} destroyed={len(destroyed)}"
    )


async def case_4() -> Tuple[bool, str]:
    embedder = ScriptedEmbedder([("raise", 0.0, build_429), ("ok", 0.0)])
    vectors, error, elapsed, logs, unretrieved, destroyed = await call_hedged(
        embedder, interval=0.05, budget=0.6
    )
    passed = (
        error is None
        and vectors is not None
        and len(vectors) == 1
        and embedder.calls == 2
    )
    return passed, (
        f"error={error!r} calls={embedder.calls} "
        f"vectors={0 if vectors is None else len(vectors)} elapsed={elapsed:.3f}s"
    )


async def case_5() -> Tuple[bool, str]:
    import openai

    embedder = ScriptedEmbedder([("raise", 0.0, build_429)])
    vectors, error, elapsed, logs, unretrieved, destroyed = await call_hedged(
        embedder, interval=0.02, budget=2.0, max_in_flight=MAX_IN_FLIGHT_DEFAULT
    )
    passed = (
        isinstance(error, openai.APIError)
        and vectors is None
        and embedder.calls == MAX_IN_FLIGHT_DEFAULT
        and elapsed < 0.5
    )
    return passed, (
        f"error={type(error).__name__} calls={embedder.calls} elapsed={elapsed:.3f}s "
        f"(budget=2.0s, must not hang)"
    )


async def case_6() -> Tuple[bool, str]:
    embedder = ScriptedEmbedder([("hang", None)])
    vectors, error, elapsed, logs, unretrieved, destroyed = await call_hedged(
        embedder, interval=0.05, budget=0.35
    )
    passed = (
        isinstance(error, asyncio.TimeoutError)
        and embedder.max_in_flight <= MAX_IN_FLIGHT_DEFAULT
        and embedder.max_in_flight == MAX_IN_FLIGHT_DEFAULT
        and embedder.calls == MAX_IN_FLIGHT_DEFAULT
    )
    return passed, (
        f"error={type(error).__name__} max_in_flight={embedder.max_in_flight} "
        f"calls={embedder.calls} limit={MAX_IN_FLIGHT_DEFAULT}"
    )


async def case_7() -> Tuple[bool, str]:
    async def failing_embed_texts(texts: List[str], timeout: Optional[float] = None) -> Any:
        raise build_429()

    exc: Optional[BaseException] = None
    result: Any = None
    with patch.object(retrieval_module, "embed_texts", failing_embed_texts):
        try:
            result = await retrieval_module.retrieve_memory(
                session=None,
                user=SimpleNamespace(id=42, timezone="Asia/Kolkata"),
                query="I am stuck on a calculus problem",
            )
        except BaseException as error:  # noqa: BLE001 - the case asserts no exception
            exc = error

    passed = (
        exc is None
        and result is not None
        and result.embedding_failed is True
        and result.timed_out is False
        and len(result.items) == 0
        and result.candidates_considered == 0
    )
    return passed, (
        f"raised={exc!r} embedding_failed="
        f"{None if result is None else result.embedding_failed} "
        f"timed_out={None if result is None else result.timed_out} "
        f"items={None if result is None else len(result.items)}"
    )


CASES: List[Tuple[str, str, Callable[[], Any]]] = [
    ("1", "first request succeeds immediately -> exactly 1 request fired", case_1),
    ("2", "first hangs, second succeeds -> success ~interval, hung task cancelled", case_2),
    ("3", "every request hangs -> fails ~budget, no leaked tasks", case_3),
    ("4", "first raises 429, later succeeds -> the success is returned", case_4),
    ("5", "every request raises -> clear failure, not a hang", case_5),
    ("6", "in-flight count never exceeds 4", case_6),
    ("7", "retrieval: embedding raises -> no memory, embedding_failed, no exception", case_7),
]


async def run() -> int:
    print("Query-embedding hedge (strategy B) unit tests")
    print(f"  embedding model : {settings.EMBEDDING_MODEL}")
    print(f"  embedding dim   : {DIM}")
    print(f"  settings default: interval={settings.EMBEDDING_HEDGE_SECONDS}s "
          f"max_in_flight={settings.EMBEDDING_HEDGE_MAX_IN_FLIGHT} "
          f"budget={settings.EMBEDDING_TIMEOUT_SECONDS}s")
    print()

    passed_count = 0
    for case_id, name, case in CASES:
        try:
            passed, detail = await case()
        except BaseException as error:  # noqa: BLE001 - a crashing case is a FAIL
            passed, detail = False, f"case raised {type(error).__name__}: {error}"
        passed_count += int(passed)
        print(f"[{case_id}] {'PASS' if passed else 'FAIL'}  {name}")
        print(f"       {detail}")

    failed = len(CASES) - passed_count
    print()
    print(f"TOTAL: {len(CASES)}  PASS: {passed_count}  FAIL: {failed}")
    return 0 if failed == 0 else 1


def main() -> int:
    try:
        exit_code = asyncio.run(run())
    except Exception:
        traceback.print_exc()
        print("EXIT: 1")
        return 1
    print(f"EXIT: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
