"""backend/scripts/deploy_smoke.py
Infrastructure smoke test for the first cloud deploy (Railway container + hosted Neon).

This is a SKELETON deploy check: it proves the runtime (Python, deps, Whisper aside),
DB reachability + latency, the pgvector extension, that migrations are at head, Nebius
reachability for embeddings and chat, and one end-to-end orchestrator turn.

It runs IDENTICALLY locally and inside the container. It makes NO HTTP requests to its own
API (no uvicorn, no httpx) -- it imports the services directly. It creates one throwaway
user and deletes it (and all its rows, via ON DELETE CASCADE) before exiting.

It NEVER prints secret values: only booleans, lengths, counts and setting NAMES.

Usage:
  backend\\venv\\Scripts\\python.exe -u backend\\scripts\\deploy_smoke.py
  backend\\venv\\Scripts\\python.exe -u backend\\scripts\\deploy_smoke.py --only ENV,DB
  backend\\venv\\Scripts\\python.exe -u backend\\scripts\\deploy_smoke.py --only DB,EMBED,CHAT

Exit code is non-zero if any selected section fails.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import platform
import statistics
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text

from backend.config.settings import settings
from backend.database import AsyncSessionLocal, engine
from backend.models.models import ConversationSession, KnowledgeRecord, User
from backend.services.embeddings import embed_texts
from backend.services.llm import chat
from backend.services.orchestrator import respond

# Settings with no default in Settings() OR with an insecure empty default that the services
# call settings.require() on. TELEGRAM_BOT_TOKEN is intentionally NOT here: the API service
# must start without it (only the (future) bot worker needs it).
REQUIRED_SETTINGS = [
    "NEBIUS_API_KEY",
    "CHAT_MODEL_FAST",
    "CHAT_MODEL_REASONING",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "DATABASE_URL",
]

SECTIONS = ["ENV", "DB", "EMBED", "CHAT", "E2E"]


def _row(rid: str, name: str, ok: bool, detail: str) -> Dict[str, Any]:
    return {"id": rid, "name": name, "status": "PASS" if ok else "FAIL", "details": detail}


# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------

def section_env() -> Dict[str, Any]:
    present = {name: bool(getattr(settings, name, None)) for name in REQUIRED_SETTINGS}
    all_present = all(present.values())
    is_development = settings.APP_ENV == "development"
    missing = [n for n, ok in present.items() if not ok]

    print(f"   python        : {platform.python_version()} ({platform.python_implementation()})")
    print(f"   platform      : {platform.platform()}")
    print(f"   APP_ENV       : {settings.APP_ENV!r}")
    print(f"   APP_ENV == 'development' (must be False in cloud): {is_development}")
    for name in REQUIRED_SETTINGS:
        print(f"   setting {name:<24}: present={present[name]}")
    print(f"   setting TELEGRAM_BOT_TOKEN present: {bool(settings.TELEGRAM_BOT_TOKEN)} "
          f"(not required for the API)")

    ok = all_present and not is_development
    detail = (
        f"py={platform.python_version()}, APP_ENV={settings.APP_ENV!r}, "
        f"development={is_development}, missing={missing or 'none'}"
    )
    return _row("ENV", "Runtime + required settings present", ok, detail)


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def _alembic_head() -> Optional[str]:
    """Head revision from the alembic scripts (absolute script_location, cwd-independent)."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "backend" / "alembic"))
    cfg.set_main_option(
        "version_locations", str(REPO_ROOT / "backend" / "alembic" / "versions")
    )
    return ScriptDirectory.from_config(cfg).get_current_head()


async def section_db() -> Dict[str, Any]:
    lats: List[float] = []
    first_latency: Optional[float] = None
    current_rev: Optional[str] = None
    vector_present = False
    tables: List[str] = []
    vector_version = None

    async with AsyncSessionLocal() as session:
        # 5 x SELECT 1. The first one also carries the pool connect / Neon wake-up.
        for i in range(5):
            t0 = time.perf_counter()
            await session.execute(text("SELECT 1"))
            lat = time.perf_counter() - t0
            lats.append(lat)
            if i == 0:
                first_latency = lat

        ext_rows = (
            await session.execute(
                text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
            )
        ).all()
        vector_present = len(ext_rows) > 0
        if vector_present:
            vector_version = ext_rows[0][1]

        rev_rows = (
            await session.execute(text("SELECT version_num FROM alembic_version"))
        ).scalars().all()
        current_rev = rev_rows[0] if rev_rows else None

        tables = list(
            (
                await session.execute(
                    text(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = 'public' ORDER BY tablename"
                    )
                )
            ).scalars().all()
        )

    head_rev = _alembic_head()

    print(f"   SELECT 1 latencies (s): {[round(x, 4) for x in lats]}")
    print(f"   first query latency   : {round(first_latency or 0.0, 4)} "
          f"(includes pool connect / Neon wake)")
    print(f"   min / median / max    : {round(min(lats), 4)} / "
          f"{round(statistics.median(lats), 4)} / {round(max(lats), 4)}")
    print(f"   pgvector extension    : present={vector_present} version={vector_version}")
    print(f"   alembic current       : {current_rev}")
    print(f"   alembic head (scripts): {head_rev}")
    print(f"   tables ({len(tables)}) : {tables}")

    ok = (
        len(lats) == 5
        and vector_present
        and current_rev is not None
        and head_rev is not None
        and current_rev == head_rev
        and len(tables) > 0
    )
    detail = (
        f"first={round(first_latency or 0.0, 4)}s, "
        f"min/med/max={round(min(lats), 4)}/{round(statistics.median(lats), 4)}/{round(max(lats), 4)}s, "
        f"vector={vector_present}, rev={current_rev}==head={head_rev}: {current_rev == head_rev}, "
        f"tables={len(tables)}"
    )
    return _row("DB", "Connect, latency, pgvector, migration head", ok, detail)


# ---------------------------------------------------------------------------
# EMBED
# ---------------------------------------------------------------------------

async def section_embed() -> Dict[str, Any]:
    t0 = time.perf_counter()
    vectors = await embed_texts(["deploy smoke embedding probe"])
    latency = time.perf_counter() - t0

    dim = len(vectors[0]) if vectors else 0
    ok = bool(vectors) and dim == settings.EMBEDDING_DIM

    print(f"   embedding latency (s): {round(latency, 4)}")
    print(f"   dimension            : {dim} (expected {settings.EMBEDDING_DIM})")

    detail = f"dim={dim}/{settings.EMBEDDING_DIM}, latency={round(latency, 4)}s"
    return _row("EMBED", "Nebius embedding reachable", ok, detail)


# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------

async def section_chat() -> Dict[str, Any]:
    t0 = time.perf_counter()
    msg = await chat(
        tier="fast",
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        tools=None,
    )
    latency = time.perf_counter() - t0

    content = str(msg).strip() if msg is not None else ""
    ok = len(content) > 0

    print(f"   chat latency (s)     : {round(latency, 4)}")
    print(f"   content length       : {len(content)}")
    print(f"   content              : {content[:80]!r}")

    detail = f"non_empty={ok}, len={len(content)}, latency={round(latency, 4)}s"
    return _row("CHAT", "Nebius fast-model chat reachable", ok, detail)


# ---------------------------------------------------------------------------
# E2E
# ---------------------------------------------------------------------------

async def section_e2e() -> Dict[str, Any]:
    seed_content = "User is learning mathematical functions"
    query = "I'm stuck on a calculus problem"

    user_id: Optional[int] = None
    record_id: Optional[int] = None
    try:
        async with AsyncSessionLocal() as session:
            user = User(
                username=f"deploy_smoke_{uuid.uuid4().hex[:8]}",
                timezone="Asia/Kolkata",
                quiet_mode=False,
                onboarding_state="testing",
            )
            session.add(user)
            await session.flush()

            conv = ConversationSession(user_id=user.id, channel="web")
            session.add(conv)
            await session.flush()

            # Real embedding path (same helper the production extractor uses).
            embedding = (await embed_texts([seed_content]))[0]
            record = KnowledgeRecord(
                user_id=user.id,
                kind="knowledge",
                content=seed_content,
                certainty="tentative",
                source="text",
                source_ref="deploy_smoke",
                embedding=embedding,
                status="confirmed",
            )
            session.add(record)
            await session.commit()
            await session.refresh(user)
            await session.refresh(conv)
            await session.refresh(record)

            user_id = user.id
            session_id = conv.id
            record_id = record.id

        result = await respond(user_id, query, session_id=session_id, history=[])

        memory_ids = [m.record_id for m in result.memory_used]
        reached = record_id in memory_ids
        timings = result.trace.timings
        retrieval_timings = result.retrieval.timings if result.retrieval else {}

        print(f"   reply                : {result.reply!r}")
        print(f"   reply mentions 'function': "
              f"{'yes' if 'function' in result.reply.lower() else 'no'}")
        print(f"   time_to_reply (s)    : {timings.get('time_to_reply')}")
        print(f"   pipeline timings     : {timings}")
        print(f"   retrieval timings    : {retrieval_timings}")
        print(f"   retrieval_timed_out  : {result.trace.retrieval_timed_out}")
        print(f"   embedding_failed     : {result.trace.retrieval_embedding_failed}")
        print(f"   candidates_considered: {result.trace.candidates_considered}")
        print(f"   seeded record in memory_used: {reached} (record_id={record_id}, "
              f"memory_used={memory_ids})")

        ok = (
            bool(result.reply.strip())
            and reached
            and not result.trace.retrieval_timed_out
            and not result.trace.retrieval_embedding_failed
        )
        detail = (
            f"reply_len={len(result.reply)}, memory_used={reached}, "
            f"time_to_reply={timings.get('time_to_reply')}s, "
            f"timed_out={result.trace.retrieval_timed_out}, "
            f"embedding_failed={result.trace.retrieval_embedding_failed}"
        )
        return _row("E2E", "Orchestrator turn recalls seeded memory", ok, detail)
    finally:
        # Full cleanup: deleting the user cascades to its session and knowledge records.
        if user_id is not None:
            try:
                async with AsyncSessionLocal() as session:
                    user = await session.get(User, user_id)
                    if user is not None:
                        await session.delete(user)
                        await session.commit()
                print(f"   cleanup: deleted throwaway user_id={user_id} (cascade)")
            except Exception as exc:  # noqa: BLE001
                print(f"   cleanup WARN: failed to delete user_id={user_id}: {exc}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

FUNCS = {
    "ENV": section_env,
    "DB": section_db,
    "EMBED": section_embed,
    "CHAT": section_chat,
    "E2E": section_e2e,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud deploy infrastructure smoke test")
    parser.add_argument(
        "--only",
        default=None,
        help="Run only sections, e.g. ENV or DB,EMBED,CHAT,E2E",
    )
    args = parser.parse_args()

    only: Optional[List[str]] = None
    if args.only:
        only = [x.strip().upper() for x in args.only.split(",") if x.strip()]
        unknown = [x for x in only if x not in FUNCS]
        if unknown:
            print(f"Unknown section(s): {unknown}. Valid: {SECTIONS}")
            sys.exit(2)

    selected = [s for s in SECTIONS if only is None or s in only]

    print("=" * 120)
    print("DEPLOY SMOKE TEST (skeleton: infra only, no product features)")
    print(f"APP_ENV={settings.APP_ENV!r}; sections: {selected}")
    print("=" * 120)

    results: List[Dict[str, Any]] = []
    for sid in selected:
        print(f"\n--- {sid} ---")
        try:
            if inspect.iscoroutinefunction(FUNCS[sid]):
                res = await FUNCS[sid]()
            else:
                res = FUNCS[sid]()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            res = _row(sid, sid, False, f"unhandled exception: {exc!r}")
        results.append(res)

    try:
        await engine.dispose()
    except Exception:  # noqa: BLE001
        pass

    print("\n" + "=" * 120)
    print(f"{'ID':<6} | {'NAME':<42} | {'STATUS':<6} | {'DETAILS'}")
    print("-" * 120)
    all_ok = True
    for r in results:
        if r["status"] != "PASS":
            all_ok = False
        print(f"{r['id']:<6} | {r['name']:<42} | {r['status']:<6} | {r['details']}")
    print("=" * 120)

    if all_ok:
        print("\nDEPLOY SMOKE: ALL SELECTED SECTIONS PASSED.")
    else:
        print("\nDEPLOY SMOKE: ONE OR MORE SECTIONS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
