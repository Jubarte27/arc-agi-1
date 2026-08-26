"""
LLM Client Module for interacting with chat model endpoints (e.g., Gemma 31B IT).
"""

import asyncio
import logging
from typing import Any, Dict, List
import time
from . import config


logger = logging.getLogger(__name__)


_rate_limit_lock = asyncio.Lock()
_last_request_start: float | None = None
_adaptive_request_delay = config.REQUEST_DELAY


def _is_rate_limit_error(error: Exception) -> bool:
    """Recognize provider rate-limit errors across SDKs and HTTP clients."""
    error_text = str(error).lower()
    return (
        "out of rate" in error_text
        or "rate limit" in error_text
        or "too many requests" in error_text
        or "429" in error_text
    )


def _increase_request_delay() -> float:
    """Increase the delay after a rate-limit response, up to the configured cap."""
    global _adaptive_request_delay

    _adaptive_request_delay *= config.RATE_LIMIT_BACKOFF_FACTOR
    if _adaptive_request_delay > config.MAX_REQUEST_DELAY:
        raise SystemExit(
            "Maximum request delay exceeded: "
            f"{_adaptive_request_delay:.2f}s > "
            f"configured limit of {config.MAX_REQUEST_DELAY:.2f}s."
        )
    return _adaptive_request_delay


async def _wait_for_request_slot() -> None:
    """Ensure API request starts are separated by the configured interval."""
    global _last_request_start

    async with _rate_limit_lock:
        now = time.monotonic()
        if _last_request_start is not None:
            wait_seconds = _adaptive_request_delay - (now - _last_request_start)
            if wait_seconds > 0:
                logger.debug("rate limiter waiting %.2fs", wait_seconds)
                await asyncio.sleep(wait_seconds)
        _last_request_start = time.monotonic()


def _call_google(
    messages: List[Dict[str, str]], model: str, api_key: str
) -> str:
    from google import genai
    from google.genai import types

    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"{role.capitalize()}:\n{content}\n")
    full_prompt = "\n".join(prompt_parts) + "\nAssistant:\n"

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(attempts=0),
        ),
    )
    try:
        chat = client.chats.create(model=model)
        response = chat.send_message(
            full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=0.2,
            ),
        )
        return response.text or ""
    finally:
        client.close()


def _call_openai(
    messages: List[Dict[str, str]], model: str, api_key: str, api_base_url: str
) -> str:
    from openai import OpenAI

    client_kwargs: dict[str,Any] = {"api_key": api_key}
    if api_base_url:
        client_kwargs["base_url"] = api_base_url
    client = OpenAI(**client_kwargs)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            top_p=0.95,
            max_tokens=16384,
            extra_body={"chat_template_kwargs":{"enable_thinking":False},"reasoning_budget":16384},
            stream=False
        )
        return response.choices[0].message.content or ""
    finally:
        client.close()


def _call_http(
    messages: List[Dict[str, str]], model: str, api_key: str, api_base_url: str
) -> str:
    import requests

    base = api_base_url.rstrip("/") if api_base_url else "https://api.openai.com/v1"
    with requests.post(
        f"{base}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": messages, "temperature": 0.2},
        timeout=120,
    ) as res:
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]


async def call_llm(
    messages: List[Dict[str, str]],
    model: str = config.MODEL_NAME,
    api_key: str = config.API_KEY,
    api_base_url: str = config.API_BASE_URL,
) -> str:
    """
    Sends chat messages to the LLM API and returns the generated text response.
    Uses the provider selected by config.API_PROVIDER.
    """
    call_start = time.perf_counter()
    logger.debug(
        "request started: model=%s, messages=%d, chars=%d",
        model,
        len(messages),
        sum(len(msg.get("content", "")) for msg in messages),
    )

    await _wait_for_request_slot()
    provider = config.API_PROVIDER
    provider_start = time.perf_counter()
    try:
        logger.debug("selected provider: %s", provider)
        if provider == "google":
            provider_call = _call_google
            response_text = await asyncio.to_thread(provider_call, messages, model, api_key)
        elif provider == "openai":
            response_text = await asyncio.to_thread(
                _call_openai, messages, model, api_key, api_base_url
            )
        else:
            response_text = await asyncio.to_thread(
                _call_http, messages, model, api_key, api_base_url
            )
        logger.debug(
            "%s request completed in %.2fs; total=%.2fs",
            provider,
            time.perf_counter() - provider_start,
            time.perf_counter() - call_start,
        )
        return response_text
    except Exception as e:
        if _is_rate_limit_error(e):
            new_delay = _increase_request_delay()
            logger.warning(
                "Rate limit detected; increasing request delay to %.2fs",
                new_delay,
            )
        logger.error(
            "%s request failed after %.2fs: %s",
            provider,
            time.perf_counter() - provider_start,
            e,
        )
        raise RuntimeError(
            f"Failed to communicate with the configured {provider} API. "
            f"Please verify its key, model, and endpoint. Error: {e}"
        ) from e
