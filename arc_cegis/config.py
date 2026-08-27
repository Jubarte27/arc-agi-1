"""
Global configuration settings for the ARC-CEGIS experiment.
"""

import os

from dotenv import dotenv_values, find_dotenv


def _dotenv_paths_from_env() -> list[str]:
    raw = os.getenv("DOTENV", "").strip()
    if not raw:
        return [find_dotenv()]

    return [path.strip() for path in raw.split(os.pathsep) if path.strip()]


def _load_dotenv_values() -> None:
    merged: dict[str, str | None] = {}
    for dotenv_path in _dotenv_paths_from_env():
        if dotenv_path:
            merged.update(dotenv_values(dotenv_path))

    for key, value in merged.items():
        if value is not None and key not in os.environ:
            os.environ[key] = value


_load_dotenv_values()

# Model Definition (Official Google AI Studio / Gemini SDK)
# Default: "gemini-3.1-flash-lite" (High quota Free Tier: 1500 RPD / 250K TPM / 15 RPM)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
API_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1/",
    "nvidia": "https://integrate.api.nvidia.com/v1"
}
API_BASE_URL = os.getenv("API_BASE_URL") or API_BASE_URLS.get(LLM_PROVIDER)
API_KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY", "OPENAI_API_KEY"),
    "nvidia": ("NVIDIA_API_KEY", "OPENAI_API_KEY"),
}


def get_api_base_url(provider: str | None = None) -> str | None:
    """Returns an explicit API base URL or the default for the selected provider."""
    selected_provider = (provider or LLM_PROVIDER).strip().lower()
    return os.getenv("API_BASE_URL") or API_BASE_URLS.get(selected_provider)


def get_api_key(provider: str | None = None) -> str:
    """
    Returns the resolved API key for the selected provider.
    """
    selected_provider = (provider or LLM_PROVIDER).strip().lower()
    key_names = API_KEYS.get(selected_provider, ())
    return next((key for name in key_names if (key:=os.getenv(name))), "")


# Backward-compatible API_KEY resolution
API_KEY = get_api_key()

# Execution Parameters & Reproducibility
MAX_CEGIS_ITERS = int(os.getenv("MAX_CEGIS_ITERS", "5"))
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "2.0"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))  # 0.0 ensures deterministic output

# Rate Limit Prevention (Google AI Studio Free Tier: 15 RPM limit)
# 4.2s delay ensures <= 14.3 RPM to strictly avoid 429 Resource Exhausted
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "4.2"))

# Daily Quota Guard (Google AI Studio Free Tier: 1500 RPD limit / 250K TPM)
# Hard safety ceiling of 1450 requests before gracefully pausing
MAX_DAILY_REQUESTS = int(os.getenv("MAX_DAILY_REQUESTS", "1450"))

# Maximum number of ARC tasks evaluated concurrently.
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "8"))

# Log file is truncated whenever a new experiment process starts.
LOG_FILE = os.getenv("LOG_FILE", "experiment.log")