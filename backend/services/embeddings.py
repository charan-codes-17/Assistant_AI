import logging
from typing import Any

from openai import AsyncOpenAI
from backend.config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


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


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings in a single batched call.

    Uses settings.EMBEDDING_MODEL and ALWAYS passes dimensions=settings.EMBEDDING_DIM.
    Raises ValueError if any returned vector dimension does not match settings.EMBEDDING_DIM.
    """
    if not texts:
        return []

    settings.require("EMBEDDING_MODEL", "EMBEDDING_DIM")
    client = get_embeddings_client()

    response = await client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,
        dimensions=settings.EMBEDDING_DIM,
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

