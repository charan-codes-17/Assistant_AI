import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI
from backend.config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

# One retry on timeout: total embedding budget is the per-attempt timeout * 2
# (query: EMBEDDING_TIMEOUT_SECONDS; batch default: EMBEDDING_BATCH_TIMEOUT_SECONDS).
EMBEDDING_MAX_ATTEMPTS = 2


def get_embeddings_client() -> AsyncOpenAI:
    """Return a shared AsyncOpenAI client configured for Nebius Token Factory."""
    global _client
    if _client is None:
        settings.require("NEBIUS_API_KEY", "EMBEDDING_MODEL", "EMBEDDING_DIM")
        _client = AsyncOpenAI(
            base_url=settings.NEBIUS_BASE_URL,
            api_key=settings.NEBIUS_API_KEY,
            timeout=30.0,
        )
    return _client


async def embed_texts(
    texts: list[str], timeout: float | None = None
) -> list[list[float]]:
    """Embed a list of text strings in a single batched call.

    Uses settings.EMBEDDING_MODEL and ALWAYS passes dimensions=settings.EMBEDDING_DIM.
    Each attempt is bounded by `timeout` when given; otherwise by
    settings.EMBEDDING_BATCH_TIMEOUT_SECONDS (the batch default). Retrieval passes the shorter
    settings.EMBEDDING_TIMEOUT_SECONDS for the query embedding. On timeout the call is retried
    once (WARNING logged on every retry). Total budget = per-attempt timeout * 2; if the retry
    also times out the timeout is re-raised so retrieval's fail-open applies.
    Raises ValueError if any returned vector dimension does not match settings.EMBEDDING_DIM.
    """
    if not texts:
        return []

    settings.require("EMBEDDING_MODEL", "EMBEDDING_DIM")
    client = get_embeddings_client()
    per_attempt_timeout = (
        timeout if timeout is not None else settings.EMBEDDING_BATCH_TIMEOUT_SECONDS
    )

    response: Any = None
    for attempt in range(1, EMBEDDING_MAX_ATTEMPTS + 1):
        try:
            response = await asyncio.wait_for(
                client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=texts,
                    dimensions=settings.EMBEDDING_DIM,
                ),
                timeout=per_attempt_timeout,
            )
            break
        except asyncio.TimeoutError:
            if attempt >= EMBEDDING_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Embedding call timed out after %.2fs; retrying (attempt %d/%d)",
                per_attempt_timeout,
                attempt + 1,
                EMBEDDING_MAX_ATTEMPTS,
            )

    # Sort items by original index to ensure alignment with input texts
    sorted_items = sorted(response.data, key=lambda item: item.index)

    embeddings: list[list[float]] = []
    for item in sorted_items:
        vec = item.embedding
        if len(vec) != settings.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIM}, "
                f"got {len(vec)} for text index {item.index} from model {settings.EMBEDDING_MODEL}"
            )
        embeddings.append(vec)

    if len(embeddings) != len(texts):
        raise ValueError(
            f"Embedding count mismatch: sent {len(texts)} texts, received {len(embeddings)} vectors"
        )

    return embeddings

