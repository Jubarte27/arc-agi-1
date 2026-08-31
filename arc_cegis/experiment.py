"""
Experiment workflows: Baseline (1-shot) vs CEGIS (Counterexample-Guided Inductive Synthesis).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .llm import AuthError, QuotaExceededError, call_llm
from .prompts import build_counterexample_feedback, build_initial_prompt
from .sandbox import extract_python_code, run_transform


def cegis_temperature(iteration: int, temperature_step: float):
    return min(1.0, round(config.TEMPERATURE + (iteration - 1) * temperature_step, 4))

def evaluate_on_pairs(
    code_str: str,
    pairs: List[Dict[str, List[List[int]]]]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Evaluates the generated transform function against all task pairs (input, output).
    Returns (all_passed: bool, results: list).
    """
    results = []
    all_passed = True
    for i, pair in enumerate(pairs):
        inp = pair["input"]
        expected = pair["output"]
        success, actual, error_msg = run_transform(code_str, inp)
        is_match = (success and actual == expected)
        if not is_match:
            all_passed = False
        results.append({
            "test_idx": i,
            "success": success,
            "is_correct": is_match,
            "error_msg": error_msg,
            "actual_output": actual,
            "expected_output": expected,
        })
    return all_passed, results


def run_baseline(
    task: Dict[str, Any],
    model: Optional[str] = None,
    initial_response: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Baseline Strategy: Single-turn LLM generation and evaluation.

    If *initial_response* is provided it is used directly instead of making a
    new LLM call, allowing the first turn to be shared with the CEGIS runs.
    """
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    start_time = time.time()
    if initial_response is not None:
        response_text = initial_response
    else:
        messages = [
            {"role": "system", "content": "You are an expert AI solving ARC-AGI puzzles by writing Python code."},
            {"role": "user", "content": build_initial_prompt(train_pairs)},
        ]
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
                "train_success": False,
                "train_results": [],
                "test_success": False,
                "test_results": [],
            }

    code_str = extract_python_code(response_text)
    all_train_passed, train_results = evaluate_on_pairs(code_str, train_pairs)
    all_test_passed, test_results = evaluate_on_pairs(code_str, test_pairs)
    latency = time.time() - start_time

    return {
        "strategy": "baseline",
        "success": all_train_passed and all_test_passed,
        "api_error": False,
        "train_success": all_train_passed,
        "train_results": train_results,
        "test_success": all_test_passed,
        "latency": latency,
        "generated_code": code_str,
        "test_results": test_results,
    }


def run_cegis(
    task: Dict[str, Any],
    anti_cheat: bool = False,
    max_iters: int = config.MAX_CEGIS_ITERS,
    model: Optional[str] = None,
    temperature_step: float = 0,
    initial_response: Optional[str] = None,
) -> Dict[str, Any]:
    """
    CEGIS Strategy: Counterexample-Guided Inductive Synthesis with Semantic Feedback loop.
    Iteratively refines code using failed training examples as counterexamples.
    Passes `iteration` to `call_llm` to compute escalating temperature based on config.TEMPERATURE.

    If *initial_response* is provided it is used as the model's reply to the
    first prompt (iteration 1), avoiding a redundant API call when multiple
    strategies share the same opening turn.
    """
    train_pairs = task.get("train", [])

    messages = [
        {"role": "system", "content": "You are an expert AI solving ARC-AGI puzzles by writing Python code."},
        {"role": "user", "content": build_initial_prompt(
            train_pairs,
        )},
    ]

    start_time = time.time()
    iteration_history = []
    converged_train = False
    current_code = ""
    had_api_error = False

    for iteration in range(1, max_iters + 1):
        current_temp = cegis_temperature(iteration, temperature_step)
        if iteration == 1 and initial_response is not None:
            response_text = initial_response
        else:
            try:
                response_text = call_llm(messages, model=model, temperature=current_temp)
            except (AuthError, QuotaExceededError):
                raise
            except Exception as e:
                had_api_error = True
                iteration_history.append({
                    "iteration": iteration,
                    "error": f"LLM Call Failed: {str(e)}",
                    "temperature": current_temp,
                })
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
                "code": current_code,
                "temperature": current_temp,
            })
            break
        else:
            idx, inp, expected, actual, error_msg = failed_example
            iteration_history.append({
                "iteration": iteration,
                "status": "failed_train_example",
                "failed_index": idx,
                "error_msg": error_msg,
                "code": current_code,
                "temperature": current_temp,
            })
            # Generate semantic counterexample feedback
            feedback_msg = build_counterexample_feedback(
                idx,
                inp,
                expected,
                actual,
                error_msg,
                anti_cheat=anti_cheat,
            )
            messages.append({"role": "user", "content": feedback_msg})

    test_pairs = task.get("test", [])
    if had_api_error and not current_code:
        all_train_passed, train_results = False, []
        all_test_passed, test_results = False, []
    else:
        all_train_passed, train_results = evaluate_on_pairs(current_code, train_pairs)
        all_test_passed, test_results = evaluate_on_pairs(current_code, test_pairs)
    latency = time.time() - start_time

    if anti_cheat:
        strategy = "cegis_anticheat"
    else:
        strategy = "cegis"

    return {
        "strategy": strategy,
        "success": all_train_passed and all_test_passed,
        "api_error": had_api_error,
        "converged_train": converged_train,
        "train_success": all_train_passed,
        "train_results": train_results,
        "test_success": all_test_passed,
        "iterations_used": len(iteration_history),
        "latency": latency,
        "generated_code": current_code,
        "iteration_history": iteration_history,
        "test_results": test_results,
    }
