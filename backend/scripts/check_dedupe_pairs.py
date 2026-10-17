"""backend/scripts/check_dedupe_pairs.py
Embeddings-only script to compute and print cosine similarity for candidate deduplication pairs.
No database connection needed.
"""

import asyncio
import math
from pathlib import Path
import sys
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config.settings import settings
from backend.services.embeddings import embed_texts


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


PAIRS: List[Tuple[str, str]] = [
    (
        "User decided to use Postgres because it supports pgvector",
        "User decided to use MySQL instead",
    ),
    (
        "I'll send Arun the PDF tomorrow",
        "I'll send Arun the PDF on Friday",
    ),
    (
        "User is learning mathematical functions",
        "User knows mathematical functions well",
    ),
    (
        "User prefers short answers",
        "User prefers detailed answers",
    ),
    (
        "User is learning mathematical functions",
        "User wants to learn about math functions",
    ),
    (
        "User prefers dark mode",
        "User likes dark mode",
    ),
]


async def main() -> None:
    settings.require("NEBIUS_API_KEY", "EMBEDDING_MODEL", "EMBEDDING_DIM")

    all_texts = [text for pair in PAIRS for text in pair]
    embeddings = await embed_texts(all_texts)

    print("=" * 90)
    print("COSINE SIMILARITY CHECK FOR DEDUPLICATION / MERGE PAIRS")
    print(f"Model: {settings.EMBEDDING_MODEL} (dim={settings.EMBEDDING_DIM})")
    print("=" * 90)

    for i, (text_a, text_b) in enumerate(PAIRS):
        vec_a = embeddings[i * 2]
        vec_b = embeddings[i * 2 + 1]
        sim = cosine_similarity(vec_a, vec_b)
        print(f"Pair {i + 1}:")
        print(f"  A: {text_a!r}")
        print(f"  B: {text_b!r}")
        print(f"  Cosine Similarity: {sim:.4f}")
        print("-" * 90)


if __name__ == "__main__":
    asyncio.run(main())

