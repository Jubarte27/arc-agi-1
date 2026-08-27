"""
LLM Client Module for interacting strictly with the official Google Gemini API (google-genai SDK).
Includes proactive Rate Limiting (RPM), Daily Quota Guard (RPD),
and intelligent 429/RPM wait-and-retry protections.
"""

import random
import re
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import errors, types

from . import config


logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Raised when authentication with the Google Gemini API fails (e.g. 401, 403, invalid key)."""
    pass


class QuotaExceededError(Exception):
    """Raised when the session/daily request count reaches the Free Tier safety limit."""
    pass


# Global tracking for API calls and proactive rate limiting
_REQUEST_COUNT: int = 0
_LAST_REQUEST_TIME: float = 0.0
_CLIENT_CACHE: Dict[str, genai.Client] = {}
_REQUEST_LOCK = threading.Lock()


def get_request_count() -> int:
    """Returns the total number of API requests made in this session."""
    with _REQUEST_LOCK:
        return _REQUEST_COUNT


def reset_request_count(value: int = 0) -> None:
    """Resets or initializes the session request counter."""
    global _REQUEST_COUNT
    with _REQUEST_LOCK:
        _REQUEST_COUNT = value


def _reserve_request_slot() -> None:
    """Serializes rate-limit waiting and quota accounting for concurrent workers."""
    global _REQUEST_COUNT, _LAST_REQUEST_TIME

    with _REQUEST_LOCK:
        if _REQUEST_COUNT >= config.MAX_DAILY_REQUESTS:
            raise QuotaExceededError(
                f"Daily Free Tier request safety limit reached ({_REQUEST_COUNT}/{config.MAX_DAILY_REQUESTS} requests). "
                "Execution paused gracefully. Use checkpoint resume to continue when quota resets."
            )

        _wait_for_rate_limit()
        _LAST_REQUEST_TIME = time.time()
        _REQUEST_COUNT += 1


def _extract_retry_delay(error: Exception) -> Optional[float]:
    """
    Extracts retry delay in seconds from Gemini API errors if present.
    Checks structured error details, dictionaries, and regex matches in the error message.
    """
    # 1. Check structured details or response dictionary if available
    details = getattr(error, "details", None) or getattr(error, "errors", None)
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                # Check for RetryInfo (e.g. {'@type': '...RetryInfo', 'retryDelay': '24s'})
                delay_val = item.get("retryDelay") or item.get("retry_delay")
                if delay_val:
                    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(delay_val))
                    if m:
                        try:
                            return float(m.group(1))
                        except ValueError:
                            pass

    # 2. Check regex patterns on string representation
    err_str = str(error)

    # Pattern: "Please retry in 24.367063794s" or "retry in 24s"
    m = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s", err_str, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Pattern: 'retryDelay': '24s' or "retry_delay": "24s"
    m = re.search(r"retry_?delay['\"]\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)\s*s?['\"]", err_str, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Pattern: "retry after 24" / "retry-after: 24"
    m = re.search(r"retry[-_\s]after[:\s]+([0-9]+(?:\.[0-9]+)?)", err_str, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return None


def _wait_for_rate_limit() -> None:
    """
    Enforces minimum delay between API calls to guarantee compliance with Free Tier RPM limits.
    """
    global _LAST_REQUEST_TIME
    if config.REQUEST_DELAY > 0 and _LAST_REQUEST_TIME > 0:
        elapsed = time.time() - _LAST_REQUEST_TIME
        if elapsed < config.REQUEST_DELAY:
            sleep_needed = config.REQUEST_DELAY - elapsed
            time.sleep(sleep_needed)


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initializes and caches a Google GenAI Client instance to prevent repeated
    initialization warnings and overhead on every request.
    """
    global _CLIENT_CACHE
    key = api_key or config.get_api_key()
    if not key:
        raise AuthError(
            "Google Gemini API Key is missing. "
            "Please export GEMINI_API_KEY or GOOGLE_API_KEY in your environment."
        )
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = genai.Client(api_key=key)
    return _CLIENT_CACHE[key]


def format_messages_for_gemini(
    messages: List[Dict[str, str]]
) -> tuple[Optional[str], List[types.Content]]:
    """
    Separates system instructions and formats conversation history into Gemini Content objects.
    """
    system_parts: List[str] = []
    contents: List[types.Content] = []

    for msg in messages:
        role = msg.get("role", "user").lower()
        content_text = msg.get("content", "")

        if role == "system":
            if content_text:
                system_parts.append(content_text)
        elif role in ("assistant", "model"):
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=content_text)]
                )
            )
        else:  # "user" or any other role
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content_text)]
                )
            )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def call_llm(
    messages: List[Dict[str, str]],
    model: str = config.MODEL_NAME,
    api_key: Optional[str] = None,
    max_retries: int = 5,
) -> str:
    """
    Sends chat messages to the Google Gemini API with proactive rate limiting,
    daily quota guard, and intelligent 429/RPM wait-and-retry.
    
    Args:
        messages: List of message dicts with 'role' and 'content'.
        model: Target Gemini model identifier (e.g. 'gemini-3.1-flash-lite').
        api_key: Optional Gemini API key override.
        max_retries: Maximum retry attempts for transient errors (429, 503, timeouts).
        
    Returns:
        Generated text response from the model.
        
    Raises:
        QuotaExceededError: If the daily request ceiling (e.g. 1450) is reached or hard quota lock.
        AuthError: If authentication fails (401/403 or missing API key).
        RuntimeError: If all retries are exhausted.
    """
    global _REQUEST_COUNT, _LAST_REQUEST_TIME

    client = get_gemini_client(api_key=api_key)
    system_instruction, contents = format_messages_for_gemini(messages)

    gen_config = types.GenerateContentConfig(
        temperature=config.TEMPERATURE,
        system_instruction=system_instruction,
    )

    if not contents:
        raise ValueError("At least one user message is required")

    chat = client.chats.create(
        model=model,
        config=gen_config,
        history=contents[:-1],
    )
    latest_message = contents[-1].parts

    base_delay = 4.0

    for attempt in range(1, max_retries + 1):
        try:
            # Reserve the quota and rate-limit slot before making the request.
            _reserve_request_slot()
            logger.debug(
                "Starting API request: model=%s attempt=%s request=%s/%s",
                model,
                attempt,
                get_request_count(),
                config.MAX_DAILY_REQUESTS,
            )

            response = chat.send_message(latest_message)
            return response.text or ""

        except errors.APIError as e:
            status_code = getattr(e, "code", None)
            err_msg = str(e).lower()

            # Immediate fail-fast for authentication errors
            if status_code in (401, 403) or any(
                term in err_msg for term in ["unauthenticated", "permission_denied", "api key not valid", "invalid api key"]
            ):
                raise AuthError(f"Fatal Gemini Authentication Error ({status_code}): {e}") from e

            # Immediate fail-fast for 404 (Model not found / deprecated)
            if status_code == 404 or "not_found" in err_msg or "no longer available" in err_msg:
                raise RuntimeError(
                    f"Fatal Gemini Model Error (404 NOT_FOUND): Model '{model}' was not found or is no longer available.\n{e}"
                ) from e

            # Handle 429 / Rate Limiting / Resource Exhausted
            if status_code == 429 or "resource_exhausted" in err_msg or "429" in err_msg:
                retry_delay = _extract_retry_delay(e)

                # Hard stop only if the API explicitly demands a multi-hour wait (true daily reset)
                if retry_delay is not None and retry_delay > 1800:
                    raise QuotaExceededError(
                        f"Google AI Studio Daily Quota Exceeded (Reset in {retry_delay:.0f}s): {e}"
                    ) from e

                if attempt == max_retries:
                    raise QuotaExceededError(
                        f"Google AI Studio Quota/Rate Limit Exceeded after {max_retries} retries: {e}"
                    ) from e

                # Determine sleep duration: use the API requested delay + buffer, or default 35s RPM buffer
                if retry_delay is not None:
                    sleep_duration = retry_delay + 3.0
                else:
                    sleep_duration = max(35.0, base_delay * (2 ** (attempt - 1))) + random.uniform(1.0, 3.0)

                time.sleep(sleep_duration)
                continue

            # Retry other transient errors (500, 503, etc.)
            if attempt == max_retries:
                raise RuntimeError(
                    f"Gemini API call failed after {max_retries} attempts (Status: {status_code}): {e}"
                ) from e

            sleep_duration = base_delay * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5)
            time.sleep(sleep_duration)

        except Exception as e:
            err_msg = str(e).lower()
            if any(term in err_msg for term in ["401", "403", "unauthorized", "invalid api key", "unauthenticated"]):
                raise AuthError(f"Fatal Gemini Authentication Error: {e}") from e

            if "404" in err_msg or "not_found" in err_msg or "no longer available" in err_msg:
                raise RuntimeError(
                    f"Fatal Gemini Model Error (404 NOT_FOUND): Model '{model}' was not found or is no longer available.\n{e}"
                ) from e

            if "429" in err_msg or "resource_exhausted" in err_msg:
                retry_delay = _extract_retry_delay(e)
                if retry_delay is not None and retry_delay > 1800:
                    raise QuotaExceededError(
                        f"Google AI Studio Daily Quota Exceeded (Reset in {retry_delay:.0f}s): {e}"
                    ) from e

                if attempt == max_retries:
                    raise QuotaExceededError(
                        f"Google AI Studio Quota/Rate Limit Exceeded after {max_retries} retries: {e}"
                    ) from e

                sleep_duration = (retry_delay + 3.0) if retry_delay is not None else (max(35.0, base_delay * (2 ** (attempt - 1))) + random.uniform(1.0, 3.0))
                time.sleep(sleep_duration)
                continue

            if attempt == max_retries:
                raise RuntimeError(
                    f"Gemini API call failed after {max_retries} attempts: {e}"
                ) from e

            sleep_duration = base_delay * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5)
            time.sleep(sleep_duration)

    return ""


