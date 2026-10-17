import asyncio
import json
import logging
from typing import Any, TypeVar

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from backend.config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

NETWORK_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    httpx.RequestError,
    httpx.TimeoutException,
    TimeoutError,
    asyncio.TimeoutError,
)

_client: AsyncOpenAI | None = None


class LLMError(Exception):
    """Base exception for LLM service failures."""
    pass


class LLMEmptyResponse(LLMError):
    """Raised when the LLM returns empty content with no tool calls."""
    pass


class StrippedMessage(str):
    """A string subclass representing the stripped message from the LLM.

    Acts as a string for callers expecting str, and exposes .content,
    .role, and .tool_calls for callers inspecting completion details.
    Never exposes or includes reasoning_content.
    """

    role: str
    content: str | None
    tool_calls: list[Any] | None
    function_call: Any | None

    def __new__(
        cls,
        content: str = "",
        role: str = "assistant",
        tool_calls: list[Any] | None = None,
        function_call: Any | None = None,
    ):
        instance = super().__new__(cls, content)
        instance.role = role
        instance.content = content if content != "" else None
        instance.tool_calls = tool_calls
        instance.function_call = function_call
        return instance


def get_llm_client() -> AsyncOpenAI:
    """Get or create the shared AsyncOpenAI client."""
    global _client
    if _client is None:
        settings.require("NEBIUS_API_KEY")
        _client = AsyncOpenAI(
            base_url=settings.NEBIUS_BASE_URL,
            api_key=settings.NEBIUS_API_KEY,
            timeout=30.0,
        )
    return _client


def set_llm_client(client: AsyncOpenAI | None) -> None:
    """Override the client (used for testing or mocking)."""
    global _client
    _client = client


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```) and leading/trailing whitespace."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def resolve_model(tier: str, tools: list[dict[str, Any]] | None = None) -> str:
    """Map tier and tool options to model IDs configured in settings."""
    if tools:
        toolcall_model = getattr(settings, "CHAT_MODEL_TOOLCALL", "").strip()
        if toolcall_model:
            return toolcall_model
        settings.require("CHAT_MODEL_FAST")
        return settings.CHAT_MODEL_FAST

    if tier == "fast":
        settings.require("CHAT_MODEL_FAST")
        return settings.CHAT_MODEL_FAST
    elif tier == "reasoning":
        settings.require("CHAT_MODEL_REASONING")
        return settings.CHAT_MODEL_REASONING
    else:
        raise ValueError(f"Unknown tier: {tier!r}. Must be 'fast' or 'reasoning'.")


async def chat(
    tier: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    json_mode: bool = False,
) -> StrippedMessage:
    """Execute an asynchronous chat completion with retry logic.

    - tier: 'fast' or 'reasoning' (tool calls use CHAT_MODEL_TOOLCALL if set, else fast).
    - If final content is empty (and no tool_calls), retries once; if still empty, raises LLMEmptyResponse.
    - Never returns or logs reasoning_content to callers.
    - Sets request timeout (30s) and retries once on network errors.
    """
    model = resolve_model(tier, tools)
    client = get_llm_client()

    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "timeout": 30.0,
    }
    if json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}
    if tools:
        call_kwargs["tools"] = tools

    async def _execute_single_call() -> Any:
        for net_attempt in range(2):
            try:
                return await client.chat.completions.create(**call_kwargs)
            except NETWORK_EXCEPTIONS as net_err:
                if net_attempt == 0:
                    logger.warning(
                        "Network error on LLM call (%s: %s). Retrying in 1s...",
                        type(net_err).__name__,
                        net_err,
                    )
                    await asyncio.sleep(1.0)
                    continue
                logger.error(
                    "Network error on LLM retry failed (%s: %s).",
                    type(net_err).__name__,
                    net_err,
                )
                raise

    for empty_retry in range(2):
        response = await _execute_single_call()
        choice = response.choices[0]
        raw_msg = choice.message
        content_str = (raw_msg.content or "").strip()
        tool_calls = getattr(raw_msg, "tool_calls", None)
        has_tool_calls = bool(tool_calls)

        if content_str or has_tool_calls:
            return StrippedMessage(
                content=content_str,
                role=getattr(raw_msg, "role", "assistant") or "assistant",
                tool_calls=tool_calls,
                function_call=getattr(raw_msg, "function_call", None),
            )

        if empty_retry == 0:
            logger.warning("LLM response content was empty with no tool calls. Retrying once...")
            await asyncio.sleep(0.5)
            continue

    raise LLMEmptyResponse("LLM returned empty content with no tool calls after retry.")


async def chat_json(
    tier: str,
    messages: list[dict[str, Any]],
    schema: type[T],
) -> T | None:
    """Execute a chat completion returning a validated Pydantic schema instance.

    Uses response_format={"type":"json_object"}, strips content and code fences,
    and validates with Pydantic.
    On invalid JSON: retries ONCE with a correction message.
    If still invalid: logs the failure and returns None (never crashes).
    """
    first_raw: str = ""
    try:
        first_msg = await chat(tier=tier, messages=messages, json_mode=True)
        first_raw = str(first_msg)
        cleaned = strip_code_fences(first_raw)
        data = json.loads(cleaned)
        return schema.model_validate(data)
    except Exception as exc:
        logger.warning(
            "Initial JSON parsing/validation failed for schema %s: %s. Retrying with correction turn...",
            schema.__name__,
            exc,
        )

    correction_messages = list(messages) + [
        {"role": "assistant", "content": first_raw},
        {
            "role": "user",
            "content": (
                f"Your previous response was not valid JSON conforming to the schema {schema.__name__}. "
                "Please output ONLY valid JSON conforming to the requested schema. "
                "Do not include markdown code fences or conversational text."
            ),
        },
    ]

    try:
        retry_msg = await chat(tier=tier, messages=correction_messages, json_mode=True)
        retry_raw = str(retry_msg)
        cleaned_retry = strip_code_fences(retry_raw)
        data_retry = json.loads(cleaned_retry)
        return schema.model_validate(data_retry)
    except Exception as retry_exc:
        logger.error(
            "Retry JSON parsing/validation failed for schema %s: %s. Returning None.",
            schema.__name__,
            retry_exc,
        )
        return None

