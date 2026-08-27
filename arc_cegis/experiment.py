"""
Experiment workflows: Baseline (1-shot) vs CEGIS (Counterexample-Guided Inductive Synthesis).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .llm import AuthError, QuotaExceededError, call_llm
from .prompts import build_counterexample_feedback, build_initial_prompt
from .sandbox import extract_python_code, run_transform


def evaluate_on_test(
    code_str: str,
    test_pairs: List[Dict[str, List[List[int]]]]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Evaluates the generated transform function against all test pairs.
    Returns (all_passed: bool, test_results: list).
    """
    test_results = []
    all_passed = True
    for i, pair in enumerate(test_pairs):
        inp = pair["input"]
        expected = pair["output"]
        success, actual, error_msg = run_transform(code_str, inp)
        is_match = (success and actual == expected)
        if not is_match:
            all_passed = False
        test_results.append({
            "test_idx": i,
            "success": success,
            "is_correct": is_match,
            "error_msg": error_msg,
            "actual_output": actual,
            "expected_output": expected,
        })
    return all_passed, test_results


def run_baseline(task: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    """
    Baseline Strategy: Single-turn LLM generation and evaluation on test pairs.
    """
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    messages = [
        {"role": "system", "content": "You are an expert AI solving ARC-AGI puzzles by writing Python code."},
        {"role": "user", "content": build_initial_prompt(train_pairs)},
    ]

    start_time = time.time()
    try:
        response_text = call_llm(messages, model=model) if model else call_llm(messages)
    except (AuthError, QuotaExceededError):
        raise
    except Exception as e:
        return {
            "strategy": "baseline",
            "success": False,
            "api_error": True,
            "error": f"LLM Call Failed: {str(e)}",
            "latency": time.time() - start_time,
            "generated_code": "",
            "test_results": [],
        }

    code_str = extract_python_code(response_text)
    all_test_passed, test_results = evaluate_on_test(code_str, test_pairs)
    latency = time.time() - start_time

    return {
        "strategy": "baseline",
        "success": all_test_passed,
        "api_error": False,
        "latency": latency,
        "generated_code": code_str,
        "test_results": test_results,
    }


def run_cegis(
    task: Dict[str, Any],
    max_iters: int = config.MAX_CEGIS_ITERS,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    CEGIS Strategy: Counterexample-Guided Inductive Synthesis with Semantic Feedback loop.
    Iteratively refines code using failed training examples as counterexamples.
    """
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    messages = [
        {"role": "system", "content": "You are an expert AI solving ARC-AGI puzzles by writing Python code."},
        {"role": "user", "content": build_initial_prompt(train_pairs)},
    ]

    start_time = time.time()
    iteration_history = []
    converged_train = False
    current_code = ""
    had_api_error = False

    for iteration in range(1, max_iters + 1):
        try:
            response_text = call_llm(messages, model=model) if model else call_llm(messages)
        except (AuthError, QuotaExceededError):
            raise
        except Exception as e:
            had_api_error = True
            iteration_history.append({"iteration": iteration, "error": f"LLM Call Failed: {str(e)}"})
            break

        current_code = extract_python_code(response_text)
        messages.append({"role": "assistant", "content": response_text})

        # Validate against all training examples
        failed_example: Optional[Tuple[int, List[List[int]], List[List[int]], Optional[List[List[int]]], str]] = None
        for idx, pair in enumerate(train_pairs):
            inp = pair["input"]
            expected = pair["output"]
            success, actual, error_msg = run_transform(current_code, inp)
            
            if not success or actual != expected:
                failed_example = (idx, inp, expected, actual, error_msg)
                break

        if failed_example is None:
            # Successfully passed 100% of training demonstrations
            converged_train = True
            iteration_history.append({
                "iteration": iteration,
                "status": "passed_all_train",
                "code": current_code
            })
            break
        else:
            idx, inp, expected, actual, error_msg = failed_example
            iteration_history.append({
                "iteration": iteration,
                "status": "failed_train_example",
                "failed_index": idx,
                "error_msg": error_msg,
                "code": current_code
            })
            # Generate semantic counterexample feedback
            feedback_msg = build_counterexample_feedback(idx, inp, expected, actual, error_msg)
            messages.append({"role": "user", "content": feedback_msg})

    # Evaluate final candidate code on test pairs
    all_test_passed, test_results = evaluate_on_test(current_code, test_pairs) if current_code else (False, [])
    latency = time.time() - start_time

    return {
        "strategy": "cegis",
        "success": all_test_passed,
        "api_error": had_api_error,
        "converged_train": converged_train,
        "iterations_used": len(iteration_history),
        "latency": latency,
        "generated_code": current_code,
        "iteration_history": iteration_history,
        "test_results": test_results,
    }
