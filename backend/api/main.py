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

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.api.followups import store
from backend.api.routes_chat import router as chat_router
from backend.api.routes_me import router as me_router
from backend.api.routes_memory import router as memory_router
from backend.api.routes_onboarding import router as onboarding_router
from backend.api.routes_profile import router as profile_router
from backend.api.routes_session import router as session_router
from backend.api.schemas import HealthResponse
from backend.config.settings import settings
from backend.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

# Docs are a development-only convenience (the seam itself only works in development).
_DOCS_ENABLED = settings.APP_ENV == "development"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan. Startup is a no-op; shutdown cancels postprocess tasks."""
    yield
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


app.include_router(session_router)
app.include_router(me_router)
app.include_router(chat_router)
app.include_router(memory_router)
app.include_router(profile_router)
app.include_router(onboarding_router)
