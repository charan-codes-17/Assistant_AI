import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI
from backend.config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

# Batch path retry count: the batch budget is the per-attempt timeout * this many attempts
# (per-attempt timeout: EMBEDDING_BATCH_TIMEOUT_SECONDS).
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


async def _single_request(texts: list[str]) -> Any:
    """Issue one embeddings request. Callers apply the timeout."""
    client = get_embeddings_client()
    return await client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,
        dimensions=settings.EMBEDDING_DIM,
    )


async def _embed_batch(texts: list[str]) -> Any:
    """Batch path: sequential attempts, each bounded by EMBEDDING_BATCH_TIMEOUT_SECONDS.

    On timeout the call is retried once (WARNING logged on every retry). If the retry also
    times out the timeout is re-raised so the caller's existing behaviour applies.
    """
    per_attempt_timeout = settings.EMBEDDING_BATCH_TIMEOUT_SECONDS
    for attempt in range(1, EMBEDDING_MAX_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(
                _single_request(texts), timeout=per_attempt_timeout
            )
        except asyncio.TimeoutError:
            if attempt >= EMBEDDING_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Embedding call timed out after %.2fs; retrying (attempt %d/%d)",
                per_attempt_timeout,
                attempt + 1,
                EMBEDDING_MAX_ATTEMPTS,
            )


async def _embed_query_hedged(texts: list[str], budget: float) -> Any:
    """Query path: hedge with a second identical request if the first is slow.

    Starts one request; if it has produced no result after settings.EMBEDDING_HEDGE_SECONDS
    a second identical request is started concurrently and whichever finishes first wins; the
    loser is cancelled (its exception is always consumed so there are no "task exception never
    retrieved" warnings). The whole call is bounded by `budget` (EMBEDDING_TIMEOUT_SECONDS);
    if that budget is exhausted an asyncio.TimeoutError is raised so retrieval's fail-open
    applies. A WARNING is logged when the hedge fires and when the budget is exhausted.
    """
    loop = asyncio.get_running_loop()
    hedge = settings.EMBEDDING_HEDGE_SECONDS
    deadline = loop.time() + budget

    first = asyncio.ensure_future(_single_request(texts))
    tasks: list[asyncio.Future] = [first]
    try:
        first_wait = min(hedge, budget)
        done, _ = await asyncio.wait({first}, timeout=first_wait)
        if not done:
            if loop.time() >= deadline:
                logger.warning(
                    "Query embedding budget of %.2fs exhausted; giving up", budget
                )
                raise asyncio.TimeoutError()
            logger.warning(
                "Query embedding produced no result after %.2fs; firing hedge request",
                first_wait,
            )
            second = asyncio.ensure_future(_single_request(texts))
            tasks.append(second)
            remaining = deadline - loop.time()
            done, _ = await asyncio.wait(
                {first, second},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                logger.warning(
                    "Query embedding budget of %.2fs exhausted; giving up", budget
                )
                raise asyncio.TimeoutError()
        winner = next(iter(done))
        return winner.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        # Consume results/exceptions of every task (cancelled or not) to avoid warnings.
        await asyncio.gather(*tasks, return_exceptions=True)


async def embed_texts(
    texts: list[str], timeout: float | None = None
) -> list[list[float]]:
    """Embed a list of text strings in a single batched call.

    Uses settings.EMBEDDING_MODEL and ALWAYS passes dimensions=settings.EMBEDDING_DIM.
    When `timeout` is given (the retrieval query path passes EMBEDDING_TIMEOUT_SECONDS) the
    call is hedged: one request, plus a second identical request if the first is slow after
    EMBEDDING_HEDGE_SECONDS; the total budget is `timeout`. When `timeout` is None (batch
    path, e.g. extraction dedupe and edit_record) requests are sequential with one retry,
    each bounded by EMBEDDING_BATCH_TIMEOUT_SECONDS.
    Raises asyncio.TimeoutError when the budget is exhausted (callers apply fail-open) and
    ValueError if any returned vector dimension does not match settings.EMBEDDING_DIM.
    """
    if not texts:
        return []

    settings.require("EMBEDDING_MODEL", "EMBEDDING_DIM")

    if timeout is None:
        response: Any = await _embed_batch(texts)
    else:
        response = await _embed_query_hedged(texts, timeout)

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
