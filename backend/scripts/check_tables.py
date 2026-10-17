"""
backend/scripts/check_tables.py
Verifies the database schema and tables in Neon Postgres using SQLAlchemy 2.0 async.
Does not print DATABASE_URL or credentials.
"""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from backend.config.settings import settings


async def check_database() -> None:
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async with engine.connect() as conn:
        # Check pgvector extension
        ext_res = await conn.execute(
            text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
        )
        ext_row = ext_res.fetchone()
        print(f"pgvector Extension: {ext_row[0]} (version {ext_row[1]})" if ext_row else "pgvector: NOT FOUND")

        # List all public tables
        tables_res = await conn.execute(
            text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
        )
        tables = [r[0] for r in tables_res.fetchall()]
        print(f"\nDiscovered {len(tables)} table(s) in public schema:")
        for t in tables:
            # Count columns
            col_res = await conn.execute(
                text("SELECT count(*) FROM information_schema.columns WHERE table_name = :t"),
                {"t": t},
            )
            col_count = col_res.scalar()
            print(f"  - {t:<25} ({col_count} columns)")

        # Verify knowledge_records embedding column type and dimension
        emb_res = await conn.execute(
            text("""
                SELECT udt_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'knowledge_records' AND column_name = 'embedding';
            """)
        )
        emb_row = emb_res.fetchone()
        if emb_row:
            print(f"\nKnowledgeRecord embedding column: {emb_row[0]} ({emb_row[1]})")

        # Check indexes on knowledge_records (specifically HNSW)
        idx_res = await conn.execute(
            text("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'knowledge_records' AND indexname = 'ix_knowledge_records_embedding_hnsw';
            """)
        )
        idx_row = idx_res.fetchone()
        if idx_row:
            print(f"HNSW Index verified: {idx_row[0]}")
            print(f"  Definition: {idx_row[1]}")

        # Check alembic_version table
        ver_res = await conn.execute(text("SELECT version_num FROM alembic_version"))
        ver_row = ver_res.fetchone()
        print(f"\nCurrent Alembic Migration Version: {ver_row[0] if ver_row else 'None'}\n")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(check_database())

