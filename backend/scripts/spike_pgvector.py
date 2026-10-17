"""
backend/scripts/spike_pgvector.py
Phase 1 Task 1.4 (second half) pgvector retrieval spike script.
Uses the real embedding provider from settings to embed test records and a query.
Creates a throwaway user and inserts 6 KnowledgeRecords:
  1. "mathematical functions (learning)" [confirmed]
  2. "Arun needs the PDF" [confirmed]
  3. "likes dark mode" [confirmed]
  4. "calculus limits notes" [pending]
  5. "wifi code 4821" [confirmed, expires_at in past]
  6. "favorite ice cream flavor is pistachio" [confirmed, unrelated]

Executes a pgvector cosine-distance query (<=> operator) for:
  "I'm stuck on a calculus problem"
Restricted to:
  status = 'confirmed' AND (expires_at IS NULL OR expires_at > now())

Prints ranked results with similarity scores, asserts pending and expired records are excluded,
and cleans up the throwaway user and records via cascading delete.
Throwaway script: not part of services/.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config.settings import settings
from backend.models import KnowledgeRecord, User


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using the configured provider in settings."""
    model_name = settings.EMBEDDING_MODEL
    print(f"Generating embeddings using model '{model_name}' (dim={settings.EMBEDDING_DIM})...")

    # Nebius / OpenAI-compatible endpoint
    if not model_name.startswith("sentence-transformers"):
        settings.require("NEBIUS_API_KEY")
        client = OpenAI(
            base_url=settings.NEBIUS_BASE_URL,
            api_key=settings.NEBIUS_API_KEY,
        )
        resp = client.embeddings.create(
            model=model_name,
            input=texts,
            dimensions=settings.EMBEDDING_DIM,
        )
        embeddings = [item.embedding for item in resp.data]
        return embeddings

    # Local sentence-transformers fallback
    from sentence_transformers import SentenceTransformer  # type: ignore
    local_model = SentenceTransformer(model_name)
    raw_embeddings = local_model.encode(texts, convert_to_numpy=True)
    return [vec.tolist() for vec in raw_embeddings]


async def run_spike() -> None:
    print("=" * 80)
    print("PHASE 1 TASK 1.4: PGVECTOR COSINE-DISTANCE RETRIEVAL SPIKE")
    print("=" * 80)

    # Database engine without printing credentials
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    now_utc = datetime.now(timezone.utc)
    expired_time = now_utc - timedelta(days=2)

    # 6 test record definitions
    records_data = [
        {
            "content": "mathematical functions (learning)",
            "kind": "knowledge",
            "certainty": "decided",
            "status": "confirmed",
            "expires_at": None,
        },
        {
            "content": "Arun needs the PDF",
            "kind": "knowledge",
            "certainty": "decided",
            "status": "confirmed",
            "expires_at": None,
        },
        {
            "content": "likes dark mode",
            "kind": "preference",
            "certainty": "decided",
            "status": "confirmed",
            "expires_at": None,
        },
        {
            "content": "calculus limits notes",
            "kind": "knowledge",
            "certainty": "tentative",
            "status": "pending",
            "expires_at": None,
        },
        {
            "content": "wifi code 4821",
            "kind": "ephemeral",
            "certainty": "decided",
            "status": "confirmed",
            "expires_at": expired_time,
        },
        {
            "content": "favorite ice cream flavor is pistachio",
            "kind": "preference",
            "certainty": "decided",
            "status": "confirmed",
            "expires_at": None,
        },
    ]

    query_text = "I'm stuck on a calculus problem"

    # Batch embed the 6 records + 1 query text
    all_texts = [r["content"] for r in records_data] + [query_text]
    all_embeddings = get_embeddings(all_texts)

    record_embeddings = all_embeddings[: len(records_data)]
    query_embedding = all_embeddings[-1]

    throwaway_username = f"spike_user_{uuid.uuid4().hex[:8]}"

    async with async_session() as session:
        throwaway_user = None
        try:
            # 1. Create throwaway user
            print(f"\n1. Creating throwaway user '{throwaway_username}'...")
            throwaway_user = User(
                username=throwaway_username,
                timezone="Asia/Kolkata",
                quiet_mode=False,
                onboarding_state="testing",
            )
            session.add(throwaway_user)
            await session.commit()
            await session.refresh(throwaway_user)
            user_id = throwaway_user.id
            print(f"   Created user ID: {user_id}")

            # 2. Insert 6 KnowledgeRecords with embeddings
            print(f"\n2. Inserting 6 test KnowledgeRecords for user ID {user_id}:")
            db_records: List[KnowledgeRecord] = []
            for idx, r_data in enumerate(records_data, start=1):
                rec = KnowledgeRecord(
                    user_id=user_id,
                    content=r_data["content"],
                    kind=r_data["kind"],
                    certainty=r_data["certainty"],
                    source="text",
                    status=r_data["status"],
                    expires_at=r_data["expires_at"],
                    embedding=record_embeddings[idx - 1],
                )
                session.add(rec)
                db_records.append(rec)
                expiry_str = (
                    rec.expires_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                    if rec.expires_at
                    else "None"
                )
                print(
                    f"   [{idx}] '{rec.content}' | status: {rec.status} | expires_at: {expiry_str}"
                )

            await session.commit()
            print("   All 6 records persisted to database.")

            # 3. Execute cosine-distance query using pgvector <=> operator
            print(f"\n3. Executing pgvector cosine-distance query (<=> operator):")
            print(f"   Query: {repr(query_text)}")
            print("   Filter: user_id = :uid AND status = 'confirmed' AND (expires_at IS NULL OR expires_at > now())")

            # KnowledgeRecord.embedding.cosine_distance(query_embedding) maps directly to
            # `knowledge_records.embedding <=> :query_embedding` in pgvector
            cosine_dist_expr = KnowledgeRecord.embedding.cosine_distance(query_embedding).label("distance")

            stmt = (
                select(KnowledgeRecord, cosine_dist_expr)
                .where(
                    KnowledgeRecord.user_id == user_id,
                    KnowledgeRecord.status == "confirmed",
                    or_(
                        KnowledgeRecord.expires_at.is_(None),
                        KnowledgeRecord.expires_at > func.now(),
                    ),
                )
                .order_by(cosine_dist_expr.asc())
            )

            result = await session.execute(stmt)
            ranked_matches: List[Tuple[KnowledgeRecord, float]] = result.all()

            # 4. Print ranked results with similarity scores
            print(f"\n4. Ranked Query Results ({len(ranked_matches)} matched):")
            print("-" * 90)
            print(f"{'Rank':<6} | {'Similarity':<12} | {'Distance':<10} | {'Status':<10} | {'Content'}")
            print("-" * 90)

            returned_contents: List[str] = []
            for rank, (rec, dist) in enumerate(ranked_matches, start=1):
                similarity = 1.0 - dist
                returned_contents.append(rec.content)
                print(
                    f"{rank:<6} | {similarity:+.4f}     | {dist:.4f}     | {rec.status:<10} | {rec.content}"
                )
            print("-" * 90)

            # 5. Assertions
            print("\n5. Verifying gating & filtering assertions:")

            # Assert pending record was NOT returned
            assert "calculus limits notes" not in returned_contents, (
                "Assertion failed: Pending record 'calculus limits notes' was returned!"
            )
            print("   [PASS] Pending record 'calculus limits notes' was NOT returned.")

            # Assert expired record was NOT returned
            assert "wifi code 4821" not in returned_contents, (
                "Assertion failed: Expired record 'wifi code 4821' was returned!"
            )
            print("   [PASS] Expired record 'wifi code 4821' was NOT returned.")

            # Assert all returned records are confirmed and active
            for rec, _ in ranked_matches:
                assert rec.status == "confirmed", f"Unexpected status: {rec.status}"
                if rec.expires_at:
                    assert rec.expires_at > now_utc, f"Unexpected expired record: {rec.content}"
            print("   [PASS] All returned records are confirmed and active.")

            # Assert exactly 4 records returned
            assert len(ranked_matches) == 4, f"Expected 4 active records, got {len(ranked_matches)}"
            print(f"   [PASS] Exactly 4 valid records returned (out of 6 inserted).")

            # Assert top match is the mathematical functions record
            top_rec, top_dist = ranked_matches[0]
            assert top_rec.content == "mathematical functions (learning)", (
                f"Expected top match 'mathematical functions (learning)', got '{top_rec.content}'"
            )
            top_similarity = 1.0 - top_dist
            print(f"   [PASS] Top match is '{top_rec.content}' with highest similarity ({top_similarity:.4f}).")

        finally:
            # 6. Clean up throwaway data via cascading delete
            if throwaway_user:
                print(f"\n6. Cleaning up throwaway user (ID: {throwaway_user.id})...")
                # Deleting user cascades to knowledge_records automatically via foreign key
                await session.delete(throwaway_user)
                await session.commit()

                # Verify clean up
                verify_stmt = select(func.count(KnowledgeRecord.id)).where(KnowledgeRecord.user_id == throwaway_user.id)
                remaining_count = await session.scalar(verify_stmt)
                assert remaining_count == 0, f"Cleanup incomplete: {remaining_count} records remain!"
                print(f"   Cleanup verified: 0 remaining records for user ID {throwaway_user.id}.")

    await engine.dispose()
    print("\n" + "=" * 80)
    print("SPIKE COMPLETED SUCCESSFULLY: All pgvector assertions passed.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_spike())

