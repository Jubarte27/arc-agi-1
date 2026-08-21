"""
Global configuration settings for the ARC-CEGIS experiment.
"""

import os

# Model Definition (Official Google AI Studio / Gemini SDK)
# Default: "gemini-3.1-flash-lite" (High quota Free Tier: 1500 RPD / 250K TPM / 15 RPM)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")


def get_api_key() -> str:
    """
    Returns the resolved Gemini API key from environment variables silently,
    checking GEMINI_API_KEY then GOOGLE_API_KEY without redundant console warnings.
    """
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


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