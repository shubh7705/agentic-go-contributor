"""
services/llm.py — OpenRouter LLM wrapper.

Provides a thin, reusable interface over the OpenAI-compatible endpoint
exposed by OpenRouter, targeting the moonshotai/kimi-k2-0905 model.

Features:
- Singleton LLM client
- Structured output helpers (JSON mode)
- Streaming support
- Automatic retries via tenacity
- LangChain-compatible ChatOpenAI instance
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator, Optional

from langchain_openai import ChatOpenAI
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw OpenAI-compatible client (for direct calls)
# ---------------------------------------------------------------------------

_raw_client: Optional[OpenAI] = None


def get_raw_client() -> OpenAI:
    """Return (or lazily create) the raw OpenAI-compatible OpenRouter client."""
    global _raw_client
    if _raw_client is None:
        _raw_client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/agentic-go-contributor",
                "X-Title": "Agentic Go Contributor",
            },
        )
        logger.debug("Raw OpenAI client initialised (OpenRouter)")
    return _raw_client


# ---------------------------------------------------------------------------
# LangChain-compatible client
# ---------------------------------------------------------------------------

_lc_client: Optional[ChatOpenAI] = None


def get_llm() -> ChatOpenAI:
    """Return (or lazily create) a LangChain ChatOpenAI pointing at OpenRouter."""
    global _lc_client
    if _lc_client is None:
        _lc_client = ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
            base_url=settings.openrouter_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            default_headers={
                "HTTP-Referer": "https://github.com/agentic-go-contributor",
                "X-Title": "Agentic Go Contributor",
            },
        )
        logger.debug("LangChain ChatOpenAI client initialised (OpenRouter / %s)", settings.openrouter_model)
    return _lc_client


# ---------------------------------------------------------------------------
# Retry-decorated completion helpers
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
) -> str:
    """
    Send a chat completion request and return the assistant message text.

    Args:
        messages:    List of {role, content} dicts.
        temperature: Override the default temperature.
        max_tokens:  Override the default max_tokens.
        json_mode:   If True, request JSON response format.

    Returns:
        The assistant's reply as a plain string.
    """
    client = get_raw_client()
    kwargs: dict[str, Any] = {
        "model": settings.openrouter_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.llm_temperature,
        "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    logger.debug("Sending chat completion request (%d messages)", len(messages))
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    logger.debug("Received response (%d chars)", len(content))
    return content


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def chat_completion_json(
    messages: list[dict[str, str]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """
    Send a chat completion request and parse the response as JSON.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If the response cannot be parsed as JSON.
    """
    raw = chat_completion(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Attempt to extract a JSON block from markdown fences
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"LLM response is not valid JSON:\n{raw}") from exc


def stream_completion(
    messages: list[dict[str, str]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """
    Stream a chat completion, yielding text deltas as they arrive.

    Args:
        messages:    List of {role, content} dicts.
        temperature: Override the default temperature.
        max_tokens:  Override the default max_tokens.

    Yields:
        Text deltas from the streaming response.
    """
    client = get_raw_client()
    stream = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=messages,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Convenience: build a system+user message pair
# ---------------------------------------------------------------------------

def build_messages(system: str, user: str) -> list[dict[str, str]]:
    """Shorthand for constructing a two-message conversation."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
