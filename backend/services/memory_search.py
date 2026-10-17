from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.models import KnowledgeRecord


async def search_confirmed(
    session: AsyncSession,
    user_id: int,
    query_embedding: list[float],
    k: int = 5,
    min_similarity: Optional[float] = None,
) -> List[Tuple[KnowledgeRecord, float]]:
    """Retrieve confirmed, active knowledge records for a user ordered by cosine similarity.

    This is the single centralized function that enforces the retrieval filter:
      status = 'confirmed'
      AND (expires_at IS NULL OR expires_at > now())
      AND superseded_by IS NULL
      AND user_id = :uid

    Returns a list of (KnowledgeRecord, similarity) tuples sorted by descending similarity.
    """
    cosine_dist_expr = KnowledgeRecord.embedding.cosine_distance(query_embedding).label("distance")

    stmt = select(KnowledgeRecord, cosine_dist_expr).where(
        KnowledgeRecord.user_id == user_id,
        KnowledgeRecord.status == "confirmed",
        KnowledgeRecord.superseded_by.is_(None),
        or_(
            KnowledgeRecord.expires_at.is_(None),
            KnowledgeRecord.expires_at > func.now(),
        ),
    )

    if min_similarity is not None:
        # cosine_similarity = 1.0 - distance >= min_similarity => distance <= 1.0 - min_similarity
        max_distance = 1.0 - min_similarity
        stmt = stmt.where(cosine_dist_expr <= max_distance)

    stmt = stmt.order_by(cosine_dist_expr.asc()).limit(k)

    result = await session.execute(stmt)
    rows = result.all()

    matches: List[Tuple[KnowledgeRecord, float]] = []
    for rec, dist in rows:
        similarity = 1.0 - float(dist)
        matches.append((rec, similarity))

    return matches

