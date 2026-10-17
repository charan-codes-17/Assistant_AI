"""backend/scripts/check_retrieval_similarity.py
Cosine similarity inspection script (embeddings only).
Measures cosine similarities across a diverse labelled benchmark of (memory, query, relevant?) pairs
to empirically calibrate RETRIEVAL_MIN_SIMILARITY.
"""

import asyncio
import math
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")
from typing import List, NamedTuple, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config.settings import settings
from backend.services.embeddings import embed_texts


class PairBenchmark(NamedTuple):
    memory: str
    query: str
    relevant: bool


BENCHMARK_PAIRS: List[PairBenchmark] = [
    # Required relevant pairs
    PairBenchmark("User is learning Python", "help me with recursion", True),
    PairBenchmark("User is vegetarian", "what should I cook tonight", True),
    PairBenchmark("User is learning mathematical functions", "I'm stuck on a calculus problem", True),
    PairBenchmark("User prefers short answers", "explain how DNS works", True),
    # Additional diverse relevant pairs
    PairBenchmark("User decided to use Postgres because it supports pgvector", "why did I choose Postgres?", True),
    PairBenchmark("Arun needs the PDF", "what did I say about Arun?", True),
    PairBenchmark("User works as a backend engineer in Go", "can you review this goroutine pattern?", True),
    PairBenchmark("User is training for a marathon", "suggest a weekend long run workout", True),
    PairBenchmark("User prefers dark mode in all IDEs", "what theme looks best in VS Code?", True),
    PairBenchmark("User lives in Bangalore", "what is the traffic like to Whitefield?", True),

    # Required irrelevant pairs
    PairBenchmark("User is vegetarian", "how do I center a div", False),
    PairBenchmark("User is learning Python", "best way to cook rice", False),
    PairBenchmark("User is learning mathematical functions", "what's the capital of France", False),
    # Additional diverse irrelevant pairs
    PairBenchmark("Arun needs the PDF", "how do I bake sourdough bread", False),
    PairBenchmark("User decided to use Postgres because it supports pgvector", "recommend a good movie for tonight", False),
    PairBenchmark("User works as a backend engineer in Go", "how do I treat a bee sting", False),
    PairBenchmark("User is training for a marathon", "explain the difference between SQL and NoSQL", False),
    PairBenchmark("User lives in Bangalore", "how to write a binary search tree in C++", False),
    PairBenchmark("User prefers dark mode in all IDEs", "what are the symptoms of dehydration", False),
    PairBenchmark("User is vegetarian", "explain CSS grid vs flexbox", False),
]


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

    # Collect unique texts to embed in batch
    unique_texts = list(dict.fromkeys(
        [p.memory for p in BENCHMARK_PAIRS] + [p.query for p in BENCHMARK_PAIRS]
    ))
    vectors = await embed_texts(unique_texts)
    vec_map = {text: vec for text, vec in zip(unique_texts, vectors)}

    results = []
    for pair in BENCHMARK_PAIRS:
        sim = cosine_similarity(vec_map[pair.memory], vec_map[pair.query])
        results.append((sim, pair.relevant, pair.memory, pair.query))

    # Sort descending by similarity
    results.sort(key=lambda x: x[0], reverse=True)

    relevant_sims = [sim for sim, rel, _, _ in results if rel]
    irrelevant_sims = [sim for sim, rel, _, _ in results if not rel]

    min_rel = min(relevant_sims) if relevant_sims else 0.0
    max_rel = max(relevant_sims) if relevant_sims else 0.0
    min_irrel = min(irrelevant_sims) if irrelevant_sims else 0.0
    max_irrel = max(irrelevant_sims) if irrelevant_sims else 0.0

    print("=" * 120)
    print("RETRIEVAL COSINE SIMILARITY CALIBRATION BENCHMARK (EMBEDDINGS ONLY)")
    print(f"Model: {settings.EMBEDDING_MODEL} (Dimension: {settings.EMBEDDING_DIM})")
    print(f"Total pairs measured: {len(results)} ({len(relevant_sims)} relevant, {len(irrelevant_sims)} irrelevant)")
    print("=" * 120)
    print(f"{'#':<3} | {'SIM':<8} | {'LABEL':<10} | {'MEMORY':<45} | {'QUERY':<42}")
    print("-" * 120)

    for idx, (sim, rel, mem, q) in enumerate(results, start=1):
        label_str = "RELEVANT" if rel else "IRRELEVANT"
        mem_short = (mem[:42] + "...") if len(mem) > 45 else mem
        q_short = (q[:39] + "...") if len(q) > 42 else q
        print(f"{idx:<3} | {sim:<8.4f} | {label_str:<10} | {mem_short:<45} | {q_short:<42}")

    print("=" * 120)
    print("SUMMARY METRICS:")
    print(f"  Relevant Group   : count={len(relevant_sims)}, min={min_rel:.4f}, max={max_rel:.4f}")
    print(f"  Irrelevant Group : count={len(irrelevant_sims)}, min={min_irrel:.4f}, max={max_irrel:.4f}")
    print(f"  Gap (Min Relevant - Max Irrelevant): {min_rel - max_irrel:+.4f}")
    print("=" * 120)


if __name__ == "__main__":
    asyncio.run(main())
