"""
backend/scripts/spike_nebius_chat.py
Phase 1 Task 1.3 connectivity & evaluation spike script for Nebius Token Factory via OpenAI SDK.
Accepts model IDs on the command line or defaults to .env configuration.
Evaluates each model across:
  1. The existing 4 tests (Plain prompt, Empty-content check, Strict JSON, Tool calling).
  2. No-tool-needed test: "what is 2+2?" with retrieve_memory tool available (verifies no wrong tool call).
  3. Tool round-trip test: "I'm stuck on calculus, check my notes" -> tool result -> final response check.
  4. 3-run latency and completion tokens benchmarking on plain prompt.
Prints a unified comparison table at the end.
Throwaway script: not part of services/.
"""

import argparse
import json
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

# Dummy memory retrieval tool definition used across tool-testing scenarios
RETRIEVE_MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_memory",
            "description": "Retrieve relevant memory records for a search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in memory.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def extract_reasoning_content(message: Any) -> Optional[str]:
    """Extract reasoning_content from message object or extra model fields if present."""
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return str(message.reasoning_content)
    if hasattr(message, "model_extra") and message.model_extra:
        extra = message.model_extra
        if isinstance(extra, dict) and "reasoning_content" in extra and extra["reasoning_content"]:
            return str(extra["reasoning_content"])
    return None


def clean_and_strip_json(raw: str) -> str:
    """Strip whitespace and optional markdown code fences from a JSON string."""
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```")
        cleaned = cleaned.strip()
    return cleaned


def list_available_models(client: OpenAI) -> List[str]:
    """Discover available models on Nebius via client.models.list()."""
    print("\n" + "=" * 80)
    print("STEP 0: Discovering available models via Nebius API")
    print("=" * 80)
    discovered_ids: List[str] = []
    try:
        models_page = client.models.list()
        for m in models_page.data:
            model_id = getattr(m, "id", "")
            if model_id:
                discovered_ids.append(model_id)
        discovered_ids.sort()
        print(f"Discovered {len(discovered_ids)} model(s) available on Nebius:")
        for mid in discovered_ids:
            print(f"  - {mid}")
    except Exception as exc:
        print(f"  [ERROR] Failed to list models: {type(exc).__name__}: {exc}")
    return discovered_ids


def run_tests_for_model(
    client: OpenAI,
    model_id: str,
    tier_label: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run evaluation suite for a single model ID:
      1. Plain prompt test
      2. Empty-content check (Roadmap §16 Risk 1)
      3. Strict JSON test (with and without response_format)
      4. Tool-calling test (single-turn trigger)
      5. No-tool-needed test ("what is 2+2?" with retrieve_memory tool)
      6. Tool round-trip test (calculus query -> tool call -> tool result -> final non-empty response)
      7. 3-run benchmark (latency and completion tokens averaged over 3 runs of plain prompt)
    """
    display_label = f"[{tier_label}] {model_id}" if tier_label else model_id
    print("\n" + "#" * 80)
    print(f"EVALUATING MODEL: {display_label}")
    print("#" * 80)

    results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # Test 1: Plain prompt
    # -------------------------------------------------------------
    print("\n--- Test 1: Plain Prompt ('Reply with the single word: pong') ---")
    t1_pass = False
    t1_detail = ""
    test1_message: Any = None
    t1_start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            temperature=0.0,
        )
        t1_lat = round(time.perf_counter() - t1_start, 3)
        choice = response.choices[0]
        test1_message = choice.message
        content_stripped = (test1_message.content or "").strip()
        reasoning_content = extract_reasoning_content(test1_message)
        finish_reason = choice.finish_reason
        usage = response.usage

        usage_str = (
            f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"
            if usage
            else "None"
        )

        print(f"  - Raw content: {repr(test1_message.content)}")
        print(f"  - Stripped content: {repr(content_stripped)}")
        print(f"  - Reasoning content: {repr(reasoning_content) if reasoning_content is not None else 'None'}")
        print(f"  - Finish reason: {finish_reason}")
        print(f"  - Usage tokens: {usage_str}")
        print(f"  - Latency: {t1_lat}s")

        t1_pass = True
        t1_detail = f"Latency {t1_lat}s, tokens {usage.total_tokens if usage else 0}"
    except Exception as exc:
        t1_detail = f"{type(exc).__name__}: {exc}"
        print(f"  [ERROR] Test 1 failed with exception: {t1_detail}")

    results.append({
        "tier": tier_label,
        "model": model_id,
        "test": "1. Plain Prompt",
        "status": "PASS" if t1_pass else "FAIL",
        "details": t1_detail,
    })

    # -------------------------------------------------------------
    # Test 2: Empty-content check
    # -------------------------------------------------------------
    print("\n--- Test 2: Empty-content check (FAIL if message.content is empty/None) ---")
    t2_pass = False
    t2_detail = ""

    try:
        if test1_message is None:
            print("  Re-querying model for empty-content check...")
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Reply with the single word: pong"}],
                temperature=0.0,
            )
            test1_message = resp.choices[0].message

        content_val = (test1_message.content or "").strip()
        reasoning_val = extract_reasoning_content(test1_message)

        if content_val != "":
            t2_pass = True
            t2_detail = f"Non-empty content ({len(content_val)} chars): {repr(content_val[:60])}"
            print(f"  - {t2_detail}")
        else:
            t2_pass = False
            t2_detail = (
                f"Content is empty or None! (reasoning_content present: {bool(reasoning_val)})"
            )
            print(f"  - [RISK 1 TRIGGERED] {t2_detail}")
    except Exception as exc:
        t2_detail = f"{type(exc).__name__}: {exc}"
        print(f"  [ERROR] Test 2 check failed with exception: {t2_detail}")

    results.append({
        "tier": tier_label,
        "model": model_id,
        "test": "2. Empty Content Check",
        "status": "PASS" if t2_pass else "FAIL",
        "details": t2_detail,
    })

    # -------------------------------------------------------------
    # Test 3: JSON test (with and without response_format)
    # -------------------------------------------------------------
    print("\n--- Test 3: JSON test (strict JSON {'intent':'question|note'}) ---")
    json_messages = [
        {
            "role": "system",
            "content": (
                "You are an input router. You must respond with strict JSON in the exact format: "
                '{"intent": "question|note"}. Do not include markdown codeblocks or extra text.'
            ),
        },
        {"role": "user", "content": "help me understand functions"},
    ]

    t3_without_pass = False
    t3_without_detail = ""
    t3_with_pass = False
    t3_with_detail = ""

    # Part A: Without response_format
    print("  [3a] Without response_format:")
    try:
        resp_no_fmt = client.chat.completions.create(
            model=model_id,
            messages=json_messages,
            temperature=0.0,
        )
        raw_content = (resp_no_fmt.choices[0].message.content or "").strip()
        print(f"    Raw content: {repr(raw_content)}")
        cleaned_content = clean_and_strip_json(raw_content)
        parsed = json.loads(cleaned_content)
        if isinstance(parsed, dict) and "intent" in parsed:
            t3_without_pass = True
            t3_without_detail = f"Parsed JSON intent='{parsed.get('intent')}'"
            print(f"    json.loads: SUCCESS -> {parsed}")
        else:
            t3_without_detail = f"JSON missing 'intent' key: {parsed}"
            print(f"    json.loads: {t3_without_detail}")
    except Exception as exc:
        t3_without_detail = f"{type(exc).__name__}: {exc}"
        print(f"    [ERROR] Without response_format: {t3_without_detail}")

    # Part B: With response_format={"type": "json_object"}
    print("  [3b] With response_format={'type': 'json_object'}:")
    try:
        resp_fmt = client.chat.completions.create(
            model=model_id,
            messages=json_messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw_content = (resp_fmt.choices[0].message.content or "").strip()
        print(f"    Raw content: {repr(raw_content)}")
        cleaned_content = clean_and_strip_json(raw_content)
        parsed = json.loads(cleaned_content)
        if isinstance(parsed, dict) and "intent" in parsed:
            t3_with_pass = True
            t3_with_detail = f"Parsed JSON intent='{parsed.get('intent')}'"
            print(f"    json.loads: SUCCESS -> {parsed}")
        else:
            t3_with_detail = f"JSON missing 'intent' key: {parsed}"
            print(f"    json.loads: {t3_with_detail}")
    except Exception as exc:
        t3_with_detail = f"{type(exc).__name__}: {exc}"
        print(f"    [ERROR] With response_format: {t3_with_detail}")

    overall_t3_pass = t3_without_pass or t3_with_pass
    report_detail = (
        f"Without fmt: {'PASS' if t3_without_pass else 'FAIL'} ({t3_without_detail}) | "
        f"With fmt: {'PASS' if t3_with_pass else 'FAIL'} ({t3_with_detail})"
    )
    print(f"  Summary report: {report_detail}")

    results.append({
        "tier": tier_label,
        "model": model_id,
        "test": "3. Strict JSON",
        "status": "PASS" if overall_t3_pass else "FAIL",
        "details": report_detail,
    })

    # -------------------------------------------------------------
    # Test 4: Tool-calling test (single-turn trigger)
    # -------------------------------------------------------------
    print("\n--- Test 4: Tool-calling test (retrieve_memory dummy tool) ---")
    t4_pass = False
    t4_detail = ""
    try:
        resp_tool = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "I'm stuck on calculus, check my notes"}],
            tools=RETRIEVE_MEMORY_TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )
        tool_msg = resp_tool.choices[0].message
        tool_calls = tool_msg.tool_calls
        if tool_calls and len(tool_calls) > 0:
            t4_pass = True
            call_names = [getattr(tc.function, "name", "unknown") for tc in tool_calls]
            t4_detail = f"tool_calls populated ({len(tool_calls)} call(s): {', '.join(call_names)})"
            print(f"  - tool_calls POPULATED: {t4_detail}")
        else:
            t4_pass = False
            t4_detail = "message.tool_calls is empty/None"
            print(f"  - tool_calls NOT POPULATED: {t4_detail}")
            print(f"    message.content (stripped): {repr((tool_msg.content or '').strip())}")
    except Exception as exc:
        t4_pass = False
        t4_detail = f"{type(exc).__name__}: {exc}"
        print(f"  - [ERROR] Tool-calling failed: {t4_detail}")

    results.append({
        "tier": tier_label,
        "model": model_id,
        "test": "4. Tool Calling",
        "status": "PASS" if t4_pass else "FAIL",
        "details": t4_detail,
    })

    # -------------------------------------------------------------
    # Test 5: No-tool-needed test ("what is 2+2?")
    # -------------------------------------------------------------
    print("\n--- Test 5: No-tool-needed test ('what is 2+2?' with retrieve_memory tool) ---")
    t5_pass = False
    t5_detail = ""
    try:
        resp_no_tool = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "what is 2+2?"}],
            tools=RETRIEVE_MEMORY_TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )
        no_tool_msg = resp_no_tool.choices[0].message
        no_tool_calls = no_tool_msg.tool_calls
        no_tool_content = (no_tool_msg.content or "").strip()
        no_tool_reasoning = extract_reasoning_content(no_tool_msg)

        print(f"  - Raw content: {repr(no_tool_msg.content)}")
        print(f"  - Stripped content: {repr(no_tool_content)}")
        print(f"  - Tool calls: {no_tool_calls}")
        if no_tool_reasoning:
            print(f"  - Reasoning: {repr(no_tool_reasoning[:100])}...")

        if no_tool_calls and len(no_tool_calls) > 0:
            call_names = [getattr(tc.function, "name", "unknown") for tc in no_tool_calls]
            t5_pass = False
            t5_detail = f"WRONGLY called tool(s): {', '.join(call_names)}"
            print(f"  - [FAIL] {t5_detail}")
        else:
            t5_pass = True
            t5_detail = f"Did not call tool (correct); reply: {repr(no_tool_content[:40])}"
            print(f"  - [PASS] {t5_detail}")
    except Exception as exc:
        t5_pass = False
        t5_detail = f"{type(exc).__name__}: {exc}"
        print(f"  - [ERROR] No-tool-needed test failed with exception: {t5_detail}")

    results.append({
        "tier": tier_label,
        "model": model_id,
        "test": "5. No Tool Needed (2+2)",
        "status": "PASS" if t5_pass else "FAIL",
        "details": t5_detail,
    })

    # -------------------------------------------------------------
    # Test 6: Tool round-trip test
    # -------------------------------------------------------------
    print("\n--- Test 6: Tool round-trip test (calculus query -> tool call -> tool result -> final answer) ---")
    t6_pass = False
    t6_detail = ""
    try:
        print("  [Step 1] Sending: 'I\\'m stuck on calculus, check my notes' with retrieve_memory tool...")
        resp_round1 = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "I'm stuck on calculus, check my notes"}],
            tools=RETRIEVE_MEMORY_TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )
        msg1 = resp_round1.choices[0].message
        tool_calls1 = msg1.tool_calls

        if not tool_calls1 or len(tool_calls1) == 0:
            t6_pass = False
            t6_detail = "Step 1 did not trigger any tool_calls; cannot complete round-trip."
            print(f"  - [FAIL] {t6_detail}")
            print(f"    Raw content: {repr(msg1.content)}")
        else:
            print(f"  - Step 1 produced {len(tool_calls1)} tool call(s):")
            for tc in tool_calls1:
                print(f"      id={tc.id}, name={tc.function.name}, args={tc.function.arguments}")

            # Assemble round 2 conversation messages
            round2_messages: List[Dict[str, Any]] = [
                {"role": "user", "content": "I'm stuck on calculus, check my notes"},
                {
                    "role": "assistant",
                    "content": msg1.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": getattr(tc, "type", "function") or "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls1
                    ],
                },
            ]
            for tc in tool_calls1:
                round2_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "User learned: mathematical functions, 2 days ago",
                })

            print("  [Step 2] Appended tool result message ('User learned: mathematical functions, 2 days ago') and calling model again...")
            resp_round2 = client.chat.completions.create(
                model=model_id,
                messages=round2_messages,
                tools=RETRIEVE_MEMORY_TOOLS,
                temperature=0.0,
            )
            msg2 = resp_round2.choices[0].message
            final_raw = msg2.content
            final_content = (final_raw or "").strip()
            final_reasoning = extract_reasoning_content(msg2)
            is_non_empty = bool(final_content)

            print("  - Final content (stripped):")
            print(f"    {repr(final_content)}")
            print(f"  - Is final content non-empty: {is_non_empty}")
            if final_reasoning:
                print(f"  - Final reasoning content: {repr(final_reasoning[:120])}...")

            if is_non_empty:
                t6_pass = True
                t6_detail = f"Non-empty ({len(final_content)} chars): {repr(final_content[:50])}"
                print("  - [PASS] Tool round-trip succeeded with non-empty content.")
            else:
                t6_pass = False
                t6_detail = f"Final content is empty or None! (reasoning present: {bool(final_reasoning)})"
                print(f"  - [FAIL] [RISK 1 TRIGGERED] {t6_detail}")
    except Exception as exc:
        t6_pass = False
        t6_detail = f"{type(exc).__name__}: {exc}"
        print(f"  - [ERROR] Tool round-trip failed with exception: {t6_detail}")

    results.append({
        "tier": tier_label,
        "model": model_id,
        "test": "6. Tool Round-Trip",
        "status": "PASS" if t6_pass else "FAIL",
        "details": t6_detail,
    })

    # -------------------------------------------------------------
    # Test 7: Benchmark (Latency & Completion Tokens averaged over 3 runs)
    # -------------------------------------------------------------
    print("\n--- Test 7: Benchmark (Latency & Completion Tokens averaged over 3 runs of plain prompt) ---")
    t7_pass = False
    t7_detail = ""
    latencies: List[float] = []
    tokens: List[int] = []

    try:
        for run_idx in range(1, 4):
            run_start = time.perf_counter()
            bench_resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Reply with the single word: pong"}],
                temperature=0.0,
            )
            run_lat = time.perf_counter() - run_start
            latencies.append(run_lat)

            usage = bench_resp.usage
            c_tok = usage.completion_tokens if usage else 0
            tokens.append(c_tok)

            run_content = (bench_resp.choices[0].message.content or "").strip()
            print(f"  - Run {run_idx}/3: latency={run_lat:.3f}s, completion_tokens={c_tok}, reply={repr(run_content)}")

        avg_latency = round(sum(latencies) / len(latencies), 3)
        avg_tokens = round(sum(tokens) / len(tokens), 1)
        t7_pass = True
        t7_detail = f"Avg latency: {avg_latency:.3f}s, Avg completion tokens: {avg_tokens:.1f}"
        print(f"  - Benchmark summary: {t7_detail}")
    except Exception as exc:
        avg_latency = 0.0
        avg_tokens = 0.0
        t7_detail = f"{type(exc).__name__}: {exc}"
        print(f"  - [ERROR] Benchmark runs failed with exception: {t7_detail}")

    results.append({
        "tier": tier_label,
        "model": model_id,
        "test": "7. 3-Run Benchmark",
        "status": "PASS" if t7_pass else "FAIL",
        "details": t7_detail,
    })

    # Model summary for comparison table
    all_pass = (
        t1_pass
        and t2_pass
        and overall_t3_pass
        and t4_pass
        and t5_pass
        and t6_pass
        and t7_pass
    )

    model_summary = {
        "model": model_id,
        "tier": tier_label,
        "plain_prompt": "PASS" if t1_pass else "FAIL",
        "empty_check": "PASS" if t2_pass else "FAIL",
        "strict_json": "PASS" if overall_t3_pass else "FAIL",
        "tool_calling": "PASS" if t4_pass else "FAIL",
        "no_tool_needed": "PASS" if t5_pass else "FAIL",
        "tool_roundtrip": "PASS" if t6_pass else "FAIL",
        "avg_latency": avg_latency,
        "avg_tokens": avg_tokens,
        "overall": "PASS" if all_pass else "FAIL",
    }

    return results, model_summary


def print_comparison_table(summaries: List[Dict[str, Any]]) -> None:
    """Print one clean, comprehensive comparison table comparing all evaluated models."""
    if not summaries:
        print("\nNo models were evaluated.")
        return

    # Determine Model column width based on longest model name
    max_model_len = max(len(s["model"]) for s in summaries)
    model_col_w = max(max_model_len, 28)

    c_model = model_col_w
    c_plain = 7
    c_empty = 11
    c_json = 11
    c_tool = 11
    c_notool = 15
    c_rtrip = 12
    c_lat = 13
    c_tok = 13
    c_overall = 9

    header = (
        f"| {'Model ID':<{c_model}} "
        f"| {'Plain':<{c_plain}} "
        f"| {'Empty Check':<{c_empty}} "
        f"| {'Strict JSON':<{c_json}} "
        f"| {'Tool Call':<{c_tool}} "
        f"| {'No-Tool (2+2)':<{c_notool}} "
        f"| {'Round-Trip':<{c_rtrip}} "
        f"| {'Avg Lat (3x)':<{c_lat}} "
        f"| {'Avg Tok (3x)':<{c_tok}} "
        f"| {'Overall':<{c_overall}} |"
    )

    sep_line = (
        f"|{'-' * (c_model + 2)}"
        f"|{'-' * (c_plain + 2)}"
        f"|{'-' * (c_empty + 2)}"
        f"|{'-' * (c_json + 2)}"
        f"|{'-' * (c_tool + 2)}"
        f"|{'-' * (c_notool + 2)}"
        f"|{'-' * (c_rtrip + 2)}"
        f"|{'-' * (c_lat + 2)}"
        f"|{'-' * (c_tok + 2)}"
        f"|{'-' * (c_overall + 2)}|"
    )

    table_width = len(header)

    print("\n" + "=" * table_width)
    print("MODEL EVALUATION COMPARISON TABLE".center(table_width))
    print("=" * table_width)
    print(header)
    print(sep_line)

    for s in summaries:
        lat_str = f"{s['avg_latency']:.3f}s"
        tok_str = f"{s['avg_tokens']:.1f}"
        row = (
            f"| {s['model']:<{c_model}} "
            f"| {s['plain_prompt']:<{c_plain}} "
            f"| {s['empty_check']:<{c_empty}} "
            f"| {s['strict_json']:<{c_json}} "
            f"| {s['tool_calling']:<{c_tool}} "
            f"| {s['no_tool_needed']:<{c_notool}} "
            f"| {s['tool_roundtrip']:<{c_rtrip}} "
            f"| {lat_str:>{c_lat}} "
            f"| {tok_str:>{c_tok}} "
            f"| {s['overall']:<{c_overall}} |"
        )
        print(row)

    print("=" * table_width + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model IDs."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Task 1.3 connectivity & evaluation spike script for Nebius Token Factory."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="One or more model IDs to evaluate. If omitted, uses model IDs from .env.",
    )
    parser.add_argument(
        "-m",
        "--model",
        "--models",
        dest="flag_models",
        nargs="+",
        help="Alternative flag to specify model IDs to evaluate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Collect CLI models if supplied
    cli_models: List[str] = []
    if args.models:
        cli_models.extend(args.models)
    if args.flag_models:
        cli_models.extend(args.flag_models)
    cli_models = list(dict.fromkeys(cli_models))

    # Validate settings requirement: API key is mandatory
    if cli_models:
        settings.require("NEBIUS_API_KEY")
    else:
        settings.require("NEBIUS_API_KEY", "CHAT_MODEL_FAST", "CHAT_MODEL_REASONING")

    print("Initializing OpenAI client with Nebius configuration:")
    print(f"  NEBIUS_BASE_URL: {settings.NEBIUS_BASE_URL}")

    client = OpenAI(
        base_url=settings.NEBIUS_BASE_URL,
        api_key=settings.NEBIUS_API_KEY,
    )

    # Step 0: Discover and list models available on Nebius
    list_available_models(client)

    # Resolve target models to evaluate
    targets: List[Tuple[str, str]] = []
    if cli_models:
        print(f"\nTarget models specified via command-line ({len(cli_models)}):")
        for m in cli_models:
            print(f"  - {m}")
            targets.append(("", m))
    else:
        # Default to .env models, deduplicating if identical
        env_dict: Dict[str, List[str]] = {}
        for tier_name, model_val in [
            ("FAST", settings.CHAT_MODEL_FAST),
            ("REASONING", settings.CHAT_MODEL_REASONING),
            ("TOOLCALL", settings.CHAT_MODEL_TOOLCALL),
        ]:
            if model_val:
                env_dict.setdefault(model_val, []).append(tier_name)

        print("\nTarget models configured via .env:")
        for model_id, tiers in env_dict.items():
            tier_str = "/".join(tiers)
            print(f"  - [{tier_str}] {model_id}")
            targets.append((tier_str, model_id))

    all_summaries: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    for tier_label, model_id in targets:
        test_results, model_summary = run_tests_for_model(client, model_id, tier_label)
        all_results.extend(test_results)
        all_summaries.append(model_summary)

    # Print the single unified comparison table at the end
    print_comparison_table(all_summaries)

    # Return exit code: 0 if all tests passed for all models, 1 if any failed
    any_failed = any(r["status"] == "FAIL" for r in all_results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
