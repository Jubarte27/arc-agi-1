"""
Global configuration settings for the ARC-CEGIS experiment.
"""

import os
import sys

# Model Definition (Default identifier compatible with Google AI Studio)
# Valid options: "gemma-2-27b-it", "gemma-2-9b-it", "gemini-2.0-flash"
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-2-27b-it")

# API Key Resolution and Validation
API_KEY = (
    os.getenv("GEMMA_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)

if not API_KEY:
    print(
        "\n[CRITICAL ERROR] No API key found!\n"
        "Set it in your terminal before running:\n"
        "  export GOOGLE_API_KEY='your_api_key_here'\n",
        file=sys.stderr,
    )

# Custom Endpoint (Leave empty for official Google GenAI SDK)
API_BASE_URL = os.getenv("API_BASE_URL", "")

# Execution Parameters & Reproducibility
MAX_CEGIS_ITERS = int(os.getenv("MAX_CEGIS_ITERS", "5"))
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "2.0"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))  # 0.0 ensures deterministic/low-variance output

# Rate Limit Prevention (Free Tier: ~10-15 RPM)
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2.0"))  # Delay in seconds between API calls