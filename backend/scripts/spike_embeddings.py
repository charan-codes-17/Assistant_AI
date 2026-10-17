"""
backend/scripts/spike_embeddings.py
Phase 1 Task 1.4 embedding spike script for Nebius Token Factory via OpenAI SDK.
Evaluates embedding models passed on the command line (or discovered via Nebius API) across
three validation sentence pairs:
  - Pair A: "help me understand mathematical functions" vs "I'm stuck on a calculus problem"
  - Pair B: "help me understand mathematical functions" vs "what should I cook for dinner"
  - Pair C: "I'll send Arun the PDF tomorrow" vs "reminder about sending a document"

A good model scores A and C clearly higher than B.
Also tests passing dimensions=1024 to verify if the API supports MRL truncation and how scores change.
Includes a lazy-imported local sentence-transformers (all-MiniLM-L6-v2, 384 dims) baseline.
Does not write to the database. Does not hardcode model IDs.
Throwaway script: not part of services/.
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI
from backend.config.settings import settings

# Canonical evaluation pairs
TEXT_PAIRS = {
    "A": {
        "text1": "help me understand mathematical functions",
        "text2": "I'm stuck on a calculus problem",
        "description": "math vs calculus (expected: high similarity)",
    },
    "B": {
        "text1": "help me understand mathematical functions",
        "text2": "what should I cook for dinner",
        "description": "math vs cooking (expected: low similarity / baseline noise)",
    },
    "C": {
        "text1": "I'll send Arun the PDF tomorrow",
        "text2": "reminder about sending a document",
        "description": "promise vs document reminder (expected: high similarity)",
    },
}


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate exact cosine similarity between two 1D vectors."""
    if len(v1) != len(v2) or len(v1) == 0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


def discover_embedding_models(client: OpenAI) -> List[str]:
    """Discover available embedding models on Nebius via client.models.list()."""
    print("\n" + "=" * 80)
    print("STEP 0: Discovering embedding models via Nebius API")
    print("=" * 80)
    discovered_ids: List[str] = []
    try:
        models_page = client.models.list()
        for m in models_page.data:
            model_id = getattr(m, "id", "")
            if model_id and "embed" in model_id.lower():
                discovered_ids.append(model_id)
        discovered_ids.sort()
        print(f"Discovered {len(discovered_ids)} embedding model(s) on Nebius:")
        for mid in discovered_ids:
            print(f"  - {mid}")
    except Exception as exc:
        print(f"  [ERROR] Failed to discover models: {type(exc).__name__}: {exc}")
    return discovered_ids


def compute_pair_similarities(
    embeddings_map: Dict[str, List[float]]
) -> Dict[str, float]:
    """Compute cosine similarities for canonical evaluation pairs A, B, and C."""
    scores: Dict[str, float] = {}
    for pair_key, pair_info in TEXT_PAIRS.items():
        v1 = embeddings_map[pair_info["text1"]]
        v2 = embeddings_map[pair_info["text2"]]
        scores[pair_key] = round(cosine_similarity(v1, v2), 4)
    return scores


def evaluate_nebius_model(
    client: OpenAI,
    model_id: str,
    target_dim: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Evaluate a Nebius embedding model with default dimensions and with dimensions=1024.
    Returns summary records for comparison reporting.
    """
    print("\n" + "#" * 80)
    print(f"EVALUATING NEBIUS EMBEDDING MODEL: {model_id}")
    print("#" * 80)

    # Collect unique texts to embed
    unique_texts: List[str] = list(
        dict.fromkeys(
            [pair["text1"] for pair in TEXT_PAIRS.values()]
            + [pair["text2"] for pair in TEXT_PAIRS.values()]
        )
    )

    results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Default dimensions evaluation
    # -------------------------------------------------------------
    print("\n--- 1. Default Dimensions Evaluation ---")
    default_scores: Optional[Dict[str, float]] = None
    default_dim: Optional[int] = None
    t0 = time.perf_counter()

    try:
        resp = client.embeddings.create(
            model=model_id,
            input=unique_texts,
        )
        latency = round(time.perf_counter() - t0, 3)
        embeddings_map = {
            text: resp.data[idx].embedding for idx, text in enumerate(unique_texts)
        }
        first_emb = resp.data[0].embedding
        default_dim = len(first_emb)
        default_scores = compute_pair_similarities(embeddings_map)

        print(f"  - Vector dimension: {default_dim}")
        print(f"  - Request latency: {latency}s")
        print(f"  - Tokens used: {resp.usage.total_tokens if resp.usage else 'N/A'}")
        print("  - Pair Cosine Similarities:")
        for k, score in default_scores.items():
            print(f"      Pair {k} ({TEXT_PAIRS[k]['description']}): {score:.4f}")

        # Quality check: A and C clearly higher than B
        diff_a_b = round(default_scores["A"] - default_scores["B"], 4)
        diff_c_b = round(default_scores["C"] - default_scores["B"], 4)
        is_good = default_scores["A"] > default_scores["B"] and default_scores["C"] > default_scores["B"]

        print(f"  - Separation vs B (noise): A - B = {diff_a_b:+.4f}, C - B = {diff_c_b:+.4f}")
        if is_good and diff_a_b >= 0.15 and diff_c_b >= 0.15:
            print("  - Quality Verdict: [EXCELLENT] A and C scored clearly higher than B.")
        elif is_good:
            print("  - Quality Verdict: [ACCEPTABLE] A and C scored higher than B, but with moderate separation.")
        else:
            print("  - Quality Verdict: [POOR] Failed to clearly separate related pairs from noise.")

        # pgvector limit check (§16 Risk)
        if default_dim > 2000:
            print(f"  - [WARNING §16 Risk]: Vector dimension {default_dim} > 2,000 pgvector HNSW/IVFFlat index limit!")
        else:
            print(f"  - [pgvector Compatible]: Vector dimension {default_dim} <= 2,000 pgvector index limit.")

        results.append({
            "model": model_id,
            "config": "Default dims",
            "dimension": default_dim,
            "latency": latency,
            "pair_A": default_scores["A"],
            "pair_B": default_scores["B"],
            "pair_C": default_scores["C"],
            "discriminative": "YES" if is_good else "NO",
            "pgvector_ok": "YES" if default_dim <= 2000 else "NO (>2k)",
            "accepted": True,
        })
    except Exception as exc:
        print(f"  [ERROR] Default embedding request failed: {type(exc).__name__}: {exc}")
        results.append({
            "model": model_id,
            "config": "Default dims",
            "dimension": "ERROR",
            "latency": round(time.perf_counter() - t0, 3),
            "pair_A": 0.0,
            "pair_B": 0.0,
            "pair_C": 0.0,
            "discriminative": "ERROR",
            "pgvector_ok": "ERROR",
            "accepted": False,
        })

    # -------------------------------------------------------------
    # 2. dimensions=target_dim (1024) evaluation
    # -------------------------------------------------------------
    print(f"\n--- 2. Custom dimensions={target_dim} Test (MRL Truncation) ---")
    t0_dim = time.perf_counter()

    try:
        resp_dim = client.embeddings.create(
            model=model_id,
            input=unique_texts,
            dimensions=target_dim,
        )
        lat_dim = round(time.perf_counter() - t0_dim, 3)
        embs_dim_map = {
            text: resp_dim.data[idx].embedding for idx, text in enumerate(unique_texts)
        }
        dim_actual = len(resp_dim.data[0].embedding)
        scores_dim = compute_pair_similarities(embs_dim_map)

        print(f"  - API Status: ACCEPTED (dimensions={target_dim} supported)")
        print(f"  - Returned vector dimension: {dim_actual}")
        print(f"  - Request latency: {lat_dim}s")
        print("  - Pair Cosine Similarities with custom dimensions:")
        for k, score in scores_dim.items():
            print(f"      Pair {k} ({TEXT_PAIRS[k]['description']}): {score:.4f}")

        # Compare with default dimension scores
        if default_scores:
            print("  - Score Changes vs Default Dimensions:")
            for k in ["A", "B", "C"]:
                delta = scores_dim[k] - default_scores[k]
                print(f"      Pair {k}: {default_scores[k]:.4f} -> {scores_dim[k]:.4f} (delta: {delta:+.4f})")

        diff_a_b_dim = round(scores_dim["A"] - scores_dim["B"], 4)
        diff_c_b_dim = round(scores_dim["C"] - scores_dim["B"], 4)
        is_good_dim = scores_dim["A"] > scores_dim["B"] and scores_dim["C"] > scores_dim["B"]
        print(f"  - Separation vs B (noise): A - B = {diff_a_b_dim:+.4f}, C - B = {diff_c_b_dim:+.4f}")

        if is_good_dim:
            print(f"  - Quality Verdict at {dim_actual} dims: [PASSED] A and C remain clearly higher than B.")
        else:
            print(f"  - Quality Verdict at {dim_actual} dims: [DEGRADED] Lost clear separation.")

        print(f"  - [pgvector Benefit]: {dim_actual} dims fits cleanly within pgvector's 2,000-dim indexing limit!")

        results.append({
            "model": model_id,
            "config": f"dimensions={target_dim}",
            "dimension": dim_actual,
            "latency": lat_dim,
            "pair_A": scores_dim["A"],
            "pair_B": scores_dim["B"],
            "pair_C": scores_dim["C"],
            "discriminative": "YES" if is_good_dim else "NO",
            "pgvector_ok": "YES" if dim_actual <= 2000 else "NO (>2k)",
            "accepted": True,
        })
    except Exception as exc:
        lat_dim = round(time.perf_counter() - t0_dim, 3)
        print(f"  - API Status: REJECTED / FAILED: {type(exc).__name__}: {exc}")
        results.append({
            "model": model_id,
            "config": f"dimensions={target_dim}",
            "dimension": "REJECTED",
            "latency": lat_dim,
            "pair_A": 0.0,
            "pair_B": 0.0,
            "pair_C": 0.0,
            "discriminative": "N/A",
            "pgvector_ok": "N/A",
            "accepted": False,
        })

    return results


def evaluate_sentence_transformers_baseline() -> Optional[Dict[str, Any]]:
    """
    Run local baseline using sentence-transformers (all-MiniLM-L6-v2, 384 dims).
    Imported lazily so the script does not fail if sentence-transformers is not installed.
    """
    print("\n" + "#" * 80)
    print("EVALUATING LOCAL BASELINE: sentence-transformers/all-MiniLM-L6-v2 (384 dims)")
    print("#" * 80)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as err:
        print("  [SKIPPED] 'sentence-transformers' is not installed in the environment.")
        print(f"  Details: {type(err).__name__}: {err}")
        print("  (Per roadmap §8 & §16, sentence-transformers is an optional fallback dependency.)")
        return None
    except Exception as exc:
        print(f"  [ERROR] Failed to import sentence-transformers: {type(exc).__name__}: {exc}")
        return None

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    unique_texts: List[str] = list(
        dict.fromkeys(
            [pair["text1"] for pair in TEXT_PAIRS.values()]
            + [pair["text2"] for pair in TEXT_PAIRS.values()]
        )
    )

    try:
        print(f"  Loading local model '{model_name}'...")
        load_start = time.perf_counter()
        model = SentenceTransformer(model_name)
        load_time = round(time.perf_counter() - load_start, 3)
        print(f"  Model loaded in {load_time}s.")

        infer_start = time.perf_counter()
        raw_embeddings = model.encode(unique_texts, convert_to_numpy=True)
        infer_latency = round(time.perf_counter() - infer_start, 3)

        embs_map: Dict[str, List[float]] = {
            text: raw_embeddings[idx].tolist() for idx, text in enumerate(unique_texts)
        }
        actual_dim = len(next(iter(embs_map.values())))
        scores = compute_pair_similarities(embs_map)

        print(f"  - Vector dimension: {actual_dim}")
        print(f"  - Inference latency: {infer_latency}s")
        print("  - Pair Cosine Similarities:")
        for k, score in scores.items():
            print(f"      Pair {k} ({TEXT_PAIRS[k]['description']}): {score:.4f}")

        diff_a_b = round(scores["A"] - scores["B"], 4)
        diff_c_b = round(scores["C"] - scores["B"], 4)
        is_good = scores["A"] > scores["B"] and scores["C"] > scores["B"]
        print(f"  - Separation vs B (noise): A - B = {diff_a_b:+.4f}, C - B = {diff_c_b:+.4f}")
        print(f"  - Quality Verdict: {'[PASSED] A and C scored clearly higher than B.' if is_good else '[POOR]'}")
        print("  - [pgvector Compatible]: 384 dims easily fits pgvector limits.")

        return {
            "model": model_name,
            "config": "Local PyTorch",
            "dimension": actual_dim,
            "latency": infer_latency,
            "pair_A": scores["A"],
            "pair_B": scores["B"],
            "pair_C": scores["C"],
            "discriminative": "YES" if is_good else "NO",
            "pgvector_ok": "YES",
            "accepted": True,
        }
    except Exception as exc:
        print(f"  [ERROR] sentence-transformers execution failed: {type(exc).__name__}: {exc}")
        return None


def print_comparison_table(records: List[Dict[str, Any]]) -> None:
    """Print one clean, comprehensive comparison table comparing all evaluated configurations."""
    if not records:
        print("\nNo embedding results to display.")
        return

    # Determine Model column width
    max_model_len = max(len(r["model"]) for r in records)
    model_col_w = max(max_model_len, 30)

    c_model = model_col_w
    c_cfg = 17
    c_dim = 9
    c_lat = 9
    c_pa = 9
    c_pb = 9
    c_pc = 9
    c_disc = 12
    c_pg = 14

    header = (
        f"| {'Model ID':<{c_model}} "
        f"| {'Config':<{c_cfg}} "
        f"| {'Dim':<{c_dim}} "
        f"| {'Latency':<{c_lat}} "
        f"| {'Pair A':<{c_pa}} "
        f"| {'Pair B':<{c_pb}} "
        f"| {'Pair C':<{c_pc}} "
        f"| {'A,C > B?':<{c_disc}} "
        f"| {'pgvector <=2k':<{c_pg}} |"
    )

    sep_line = (
        f"|{'-' * (c_model + 2)}"
        f"|{'-' * (c_cfg + 2)}"
        f"|{'-' * (c_dim + 2)}"
        f"|{'-' * (c_lat + 2)}"
        f"|{'-' * (c_pa + 2)}"
        f"|{'-' * (c_pb + 2)}"
        f"|{'-' * (c_pc + 2)}"
        f"|{'-' * (c_disc + 2)}"
        f"|{'-' * (c_pg + 2)}|"
    )

    table_width = len(header)

    print("\n" + "=" * table_width)
    print("EMBEDDING MODELS EVALUATION & COMPARISON TABLE".center(table_width))
    print("=" * table_width)
    print(header)
    print(sep_line)

    for r in records:
        dim_str = str(r["dimension"])
        lat_str = f"{r['latency']:.3f}s" if isinstance(r["latency"], (int, float)) else str(r["latency"])
        pa_str = f"{r['pair_A']:.4f}" if isinstance(r["pair_A"], (int, float)) else str(r["pair_A"])
        pb_str = f"{r['pair_B']:.4f}" if isinstance(r["pair_B"], (int, float)) else str(r["pair_B"])
        pc_str = f"{r['pair_C']:.4f}" if isinstance(r["pair_C"], (int, float)) else str(r["pair_C"])

        row = (
            f"| {r['model']:<{c_model}} "
            f"| {r['config']:<{c_cfg}} "
            f"| {dim_str:>{c_dim}} "
            f"| {lat_str:>{c_lat}} "
            f"| {pa_str:>{c_pa}} "
            f"| {pb_str:>{c_pb}} "
            f"| {pc_str:>{c_pc}} "
            f"| {r['discriminative']:^{c_disc}} "
            f"| {r['pgvector_ok']:^{c_pg}} |"
        )
        print(row)

    print("=" * table_width + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Task 1.4 embedding spike script for Nebius Token Factory via OpenAI SDK."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="One or more model IDs to test against Nebius. If omitted, discovers embedding models dynamically or checks .env.",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--models",
        dest="flag_models",
        nargs="+",
        help="Alternative flag to specify model ID(s) to evaluate.",
    )
    parser.add_argument(
        "--dimensions",
        "-d",
        type=int,
        default=1024,
        help="Custom dimensions parameter to test with MRL truncation (default: 1024).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Collect CLI models
    cli_models: List[str] = []
    if args.models:
        cli_models.extend(args.models)
    if args.flag_models:
        cli_models.extend(args.flag_models)
    cli_models = list(dict.fromkeys(cli_models))

    # API key is mandatory
    settings.require("NEBIUS_API_KEY")

    print("Initializing OpenAI client with Nebius configuration:")
    print(f"  NEBIUS_BASE_URL: {settings.NEBIUS_BASE_URL}")

    client = OpenAI(
        base_url=settings.NEBIUS_BASE_URL,
        api_key=settings.NEBIUS_API_KEY,
    )

    # Resolve target models
    target_models: List[str] = []
    if cli_models:
        target_models = cli_models
        print(f"\nTarget embedding model(s) specified on command-line ({len(target_models)}):")
        for m in target_models:
            print(f"  - {m}")
    else:
        # Dynamically discover embedding models from Nebius
        discovered = discover_embedding_models(client)
        if discovered:
            target_models = discovered
            print(f"\nUsing {len(target_models)} discovered embedding model(s) from Nebius API.")
        elif settings.EMBEDDING_MODEL and not settings.EMBEDDING_MODEL.startswith("sentence-transformers"):
            target_models = [settings.EMBEDDING_MODEL]
            print(f"\nFalling back to EMBEDDING_MODEL from .env: {settings.EMBEDDING_MODEL}")
        else:
            print("\n[WARNING] No embedding model IDs provided on command line and none discovered on Nebius.")
            print("Please pass a model ID: python backend/scripts/spike_embeddings.py <model_id>")

    all_records: List[Dict[str, Any]] = []

    # Evaluate each Nebius model
    for model_id in target_models:
        model_records = evaluate_nebius_model(client, model_id, target_dim=args.dimensions)
        all_records.extend(model_records)

    # Evaluate local baseline (lazily imported)
    baseline_record = evaluate_sentence_transformers_baseline()
    if baseline_record:
        all_records.append(baseline_record)

    # Print comparison table at the end
    print_comparison_table(all_records)

    return 0


if __name__ == "__main__":
    sys.exit(main())

