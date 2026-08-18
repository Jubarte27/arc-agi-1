"""
ARC-AGI-1 Comparative Experiment Package.
"""

from .config import MODEL_NAME, MAX_CEGIS_ITERS, TIMEOUT_SECONDS
from .data_loader import load_tasks
from .experiment import evaluate_on_test, run_baseline, run_cegis
from .llm import call_llm
from .prompts import build_counterexample_feedback, build_initial_prompt
from .sandbox import extract_python_code, run_transform

__all__ = [
    "MODEL_NAME",
    "MAX_CEGIS_ITERS",
    "TIMEOUT_SECONDS",
    "load_tasks",
    "evaluate_on_test",
    "run_baseline",
    "run_cegis",
    "call_llm",
    "build_counterexample_feedback",
    "build_initial_prompt",
    "extract_python_code",
    "run_transform",
]
