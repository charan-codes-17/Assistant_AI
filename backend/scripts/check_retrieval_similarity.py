"""backend/scripts/check_retrieval_similarity.py
Cosine similarity inspection script (embeddings only).
Computes and prints cosine similarity between a set of queries and two target memories:
  Target A: "User is learning mathematical functions"
  Target B: "Arun needs the PDF"
"""

import asyncio
import math
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")
from typing import List

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


async def main() -> None:
    settings.require(
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
    )

    target_a = "User is learning mathematical functions"
    target_b = "Arun needs the PDF"

    queries = [
        "I'm stuck on a calculus problem",
        "can you help with derivatives",
        "I don't understand integrals",
        "limits are confusing me",
        "struggling with my math homework",
        "help me with differentiation",
        "what should I cook for dinner",
    ]

    all_texts = [target_a, target_b] + queries
    vectors = await embed_texts(all_texts)

    vec_target_a = vectors[0]
    vec_target_b = vectors[1]
    query_vectors = vectors[2:]

    print("=" * 105)
    print("RETRIEVAL COSINE SIMILARITY ANALYSIS (EMBEDDINGS ONLY)")
    print(f"Model: {settings.EMBEDDING_MODEL} (Dimension: {settings.EMBEDDING_DIM})")
    print(f"Target A: {target_a!r}")
    print(f"Target B: {target_b!r}")
    print("=" * 105)
    print(f"{'#':<3} | {'QUERY':<42} | {'SIM TO TARGET A (Math)':<24} | {'SIM TO TARGET B (Arun)':<24}")
    print("-" * 105)

    for idx, (q, q_vec) in enumerate(zip(queries, query_vectors), start=1):
        sim_a = cosine_similarity(q_vec, vec_target_a)
        sim_b = cosine_similarity(q_vec, vec_target_b)
        print(f"{idx:<3} | {q:<42} | {sim_a:<24.4f} | {sim_b:<24.4f}")

    print("=" * 105)


if __name__ == "__main__":
    asyncio.run(main())

