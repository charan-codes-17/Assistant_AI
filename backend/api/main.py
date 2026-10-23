"""backend/api/main.py
FastAPI application for Phase 3 (PRD §8).

Design notes:
- There is NO CORS middleware. The web client is served through a same-origin proxy, so
  cross-origin access is deliberately unsupported here.
- A single global exception handler converts any unhandled error into a generic
  {"detail": "internal error"} body plus a request id; the traceback is logged server-side
  and never returned to the client.
- Interactive docs (/docs, /redoc) are enabled ONLY when settings.APP_ENV == "development".
- The in-memory follow-up store is closed on application shutdown (its background
  postprocess tasks are cancelled; see followups.py).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.api.followups import store
from backend.api.routes_auth import router as auth_router
from backend.api.routes_chat import router as chat_router
from backend.api.routes_me import router as me_router
from backend.api.routes_memory import router as memory_router
from backend.api.routes_onboarding import router as onboarding_router
from backend.api.routes_profile import router as profile_router
from backend.api.routes_session import router as session_router
from backend.api.schemas import HealthResponse
from backend.config.settings import settings
from backend.database import AsyncSessionLocal
from backend.services.auth import auth_configured

logger = logging.getLogger(__name__)

# Docs are a development-only convenience (the seam itself only works in development).
_DOCS_ENABLED = settings.APP_ENV == "development"


async def _shutdown_application(application: Any) -> None:
    """Stop and close an Application that may only be partially started. Never raises."""
    updater = getattr(application, "updater", None)
    steps: list[tuple[str, Any]] = []
    if updater is not None:
        steps.append(("updater.stop", updater.stop))
    steps.append(("application.stop", application.stop))
    steps.append(("application.shutdown", application.shutdown))
    for name, stop in steps:
        try:
            await stop()
        except Exception:  # noqa: BLE001 - cleanup must not mask the original failure
            logger.exception("bot: %s failed during cleanup", name)


async def _start_bot(app: FastAPI, tasks: list[asyncio.Task]) -> Optional[Any]:
    """Start the Telegram poller, the reminder scheduler and the Whisper warm-up in-process.

    Single-process mode (settings.BOT_ENABLED). Drop-in replacement for the standalone
    worker's run_polling, which must never be used here: it owns its own event loop.

    Never raises. A missing token, a build failure, or any startup failure is logged
    without secrets and leaves the API serving HTTP.
    """
    if not settings.TELEGRAM_BOT_TOKEN.strip():
        logger.error(
            "BOT_ENABLED is set but TELEGRAM_BOT_TOKEN is empty: "
            "the bot is not started, the API keeps serving."
        )
        return None

    try:
        from backend.bot.app import build_application

        application = build_application()
    except Exception:  # noqa: BLE001
        logger.exception("bot: could not build the Application; the API keeps serving.")
        return None

    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"],
        )
    except Exception:  # noqa: BLE001
        logger.exception("bot: startup failed; the API keeps serving.")
        await _shutdown_application(application)
        return None

    try:
        app.state.bot_application = application
        logger.info("bot: long polling started")

        from backend.bot.scheduler import scheduler_loop

        tasks.append(
            asyncio.create_task(scheduler_loop(application.bot), name="reminder-scheduler")
        )
        logger.info("bot: reminder scheduler started")

        async def _warmup() -> None:
            """Load Whisper off the event loop. Never awaited during startup."""
            try:
                from backend.services import audio as audio_service

                await asyncio.to_thread(audio_service.warmup)
                logger.info("Whisper model warm-up complete.")
            except Exception:  # noqa: BLE001 - voice notes fail friendly until it succeeds
                logger.warning(
                    "Whisper warm-up failed; voice notes will return an error until it succeeds."
                )

        tasks.append(asyncio.create_task(_warmup(), name="whisper-warmup"))
    except Exception:  # noqa: BLE001 - a started poller must not be left behind
        logger.exception(
            "bot: could not start the background tasks; the API keeps serving."
        )
        app.state.bot_application = None
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - cleanup must not raise
                logger.exception("bot: a background task failed while being cancelled")
        tasks.clear()
        await _shutdown_application(application)
        return None

    return application


async def _stop_bot(application: Optional[Any], tasks: list[asyncio.Task]) -> None:
    """Reverse of _start_bot: cancel the background tasks, then stop the Application."""
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("bot: a background task failed during shutdown")
    tasks.clear()
    if application is not None:
        await _shutdown_application(application)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan. Startup logs auth readiness; shutdown cancels postprocess tasks.

    When settings.BOT_ENABLED is true, the Telegram poller, the reminder scheduler and the
    Whisper warm-up are started in this same process; all of them are stopped in reverse
    order here. Startup never waits for the Whisper model.
    """
    if not auth_configured():
        logger.error(
            "AUTH DISABLED: APP_ENV=production but JWT_SECRET is missing or shorter than 32 "
            "characters. /auth/login will return 503 and every session cookie will be rejected."
        )

    bot_tasks: list[asyncio.Task] = []
    bot_application = None
    if settings.BOT_ENABLED:
        bot_application = await _start_bot(app, bot_tasks)

    try:
        yield
    finally:
        await _stop_bot(bot_application, bot_tasks)
        await store.aclose()


app = FastAPI(
    title="Persistent-Memory Personal AI Assistant API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global fallback: never leak a stack trace or exception text to the client.

    A unique request id is returned so a user report can be matched to the server-side
    traceback log (logged here with logger.exception, which includes the traceback).
    """
    request_id = str(uuid.uuid4())
    logger.exception(
        "Unhandled error request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "request_id": request_id},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse | JSONResponse:
    """Liveness + DB/pgvector readiness. No auth. Never returns secrets or exception text."""
    now = datetime.now(timezone.utc).isoformat()
    pgvector_ok = False
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            result = await session.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1")
            )
            pgvector_ok = result.first() is not None
    except Exception:
        logger.exception("health: DB check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "db": False, "pgvector": False, "time": now},
        )
    return HealthResponse(status="ok", db=True, pgvector=pgvector_ok, time=now)


app.include_router(auth_router)
app.include_router(session_router)
app.include_router(me_router)
app.include_router(chat_router)
app.include_router(memory_router)
app.include_router(profile_router)
app.include_router(onboarding_router)
