"""
ARC-AGI-1 Comparative Experiment Package.
"""

from . import config
from .config import (
    API_BASE_URL,
    API_BASE_URLS,
    API_KEYS,
    LLM_PROVIDER,
    MODEL_NAME,
    MAX_CEGIS_ITERS,
    TIMEOUT_SECONDS,
    REQUEST_DELAY,
    MAX_DAILY_REQUESTS,
    MAX_CONCURRENT_TASKS,
    LLMConfig,
    LLM_POOL,
    get_next_llm,
    get_api_key,
    get_api_base_url,
)
from .data_loader import load_tasks
from .experiment import evaluate_on_pairs, run_baseline, run_cegis
from .llm import (
    AuthError,
    QuotaExceededError,
    call_llm,
    get_request_count,
    reset_request_count,
)
from .prompts import build_counterexample_feedback, build_initial_prompt
from .sandbox import extract_python_code, run_transform
from . import local
from .local import (
    OllamaClient,
    OllamaServer,
    ensure_server_running,
    setup_local_ollama,
    get_local_openai_client,
)

__all__ = [
    "config",
    "local",
    "API_BASE_URL",
    "API_BASE_URLS",
    "API_KEYS",
    "LLM_PROVIDER",
    "MODEL_NAME",
    "MAX_CEGIS_ITERS",
    "TIMEOUT_SECONDS",
    "REQUEST_DELAY",
    "MAX_DAILY_REQUESTS",
    "MAX_CONCURRENT_TASKS",
    "LLMConfig",
    "LLM_POOL",
    "get_next_llm",
    "get_api_key",
    "get_api_base_url",
    "load_tasks",
    "evaluate_on_pairs",
    "run_baseline",
    "run_cegis",
    "call_llm",
    "AuthError",
    "QuotaExceededError",
    "get_request_count",
    "reset_request_count",
    "build_counterexample_feedback",
    "build_initial_prompt",
    "extract_python_code",
    "run_transform",
    "OllamaClient",
    "OllamaServer",
    "ensure_server_running",
    "setup_local_ollama",
    "get_local_openai_client",
]
