"""
Global configuration settings for the ARC-CEGIS experiment.
"""

import os
import re
import uuid

from dotenv import dotenv_values, find_dotenv


def _dotenv_paths_from_env() -> list[str]:
    raw = os.getenv("DOTENV", "").strip()
    if not raw:
        return [find_dotenv()]

    return [p.strip() for p in raw.split(re.escape(os.pathsep)) if p.strip()]


def _load_dotenv_values():
    merged: dict[str, str | None] = {}
    for dotenv_path in _dotenv_paths_from_env():
        merged.update(dotenv_values(dotenv_path))

    for key, value in merged.items():
        if value is not None and key not in os.environ:
            os.environ[key] = value


_load_dotenv_values()
# Model Definition (Default identifier compatible with Google AI Studio)
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-4-26b-a4b-it")

# Custom Endpoint (Leave empty for official Google GenAI SDK)
API_BASE_URL = os.getenv("API_BASE_URL", "")

# Select exactly one LLM transport: google, openai, or http.
# A configured custom endpoint defaults to the OpenAI-compatible SDK.
API_PROVIDER = os.getenv(
    "API_PROVIDER",
    "http" if API_BASE_URL else "google",
).lower()
if API_PROVIDER not in {"google", "openai", "http"}:
    raise ValueError(
        f"Unsupported API_PROVIDER={API_PROVIDER!r}; use 'google', 'openai', or 'http'."
    )

# API Key Resolution and Validation
API_KEY = (
    os.getenv("GEMMA_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
    if API_PROVIDER == "google" else

    os.getenv("OPENAI_API_KEY")
    or os.getenv("NVIDIA_API_KEY")
    or ""
    if API_PROVIDER == "openai" else
    
    os.getenv("API_KEY")
    or ""
)

# Execution Parameters & Reproducibility
MAX_CEGIS_ITERS = int(os.getenv("MAX_CEGIS_ITERS", "5"))
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "2.0"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.42"))

# Rate Limit Prevention (Free Tier: ~10-15 RPM)
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2.0"))  # Delay in seconds between API calls
RATE_LIMIT_BACKOFF_FACTOR = float(os.getenv("RATE_LIMIT_BACKOFF_FACTOR", "2.0"))
MAX_REQUEST_DELAY = float(os.getenv("MAX_REQUEST_DELAY", "130.0"))
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))

LOG_FILE = os.getenv("LOG_FILE", f"logs/arc_cegis{uuid.uuid4().hex[:8]}.log")