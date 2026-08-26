"""
Experiment workflows: Baseline (1-shot) vs CEGIS (Counterexample-Guided Inductive Synthesis).
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .llm import call_llm
from .prompts import build_counterexample_feedback, build_initial_prompt
from .sandbox import extract_python_code, run_transform


logger = logging.getLogger(__name__)


def evaluate_on_test(
    code_str: str,
    test_pairs: List[Dict[str, List[List[int]]]]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Evaluates the generated transform function against all test pairs.
    Returns (all_passed: bool, test_results: list).
    """
    evaluation_start = time.perf_counter()
    test_results = []
    all_passed = True
    for i, pair in enumerate(test_pairs):
        inp = pair["input"]
        expected = pair["output"]
        example_start = time.perf_counter()
        success, actual, error_msg = run_transform(code_str, inp)
        is_match = (success and actual == expected)
        logger.debug(
            "test example %d/%d: %s in %.3fs%s",
            i + 1,
            len(test_pairs),
            "PASS" if is_match else "FAIL",
            time.perf_counter() - example_start,
            f" ({error_msg})" if error_msg else "",
        )
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
    logger.debug("test evaluation completed in %.2fs", time.perf_counter() - evaluation_start)
    return all_passed, test_results


async def run_baseline(task: Dict[str, Any], model: str = config.MODEL_NAME) -> Dict[str, Any]:
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
    logger.info("baseline started: train_examples=%d, test_examples=%d", len(train_pairs), len(test_pairs))
    try:
        response_text = await call_llm(messages, model=model)
    except Exception as e:
        return {
            "strategy": "baseline",
            "success": False,
            "error": f"LLM Call Failed: {str(e)}",
            "latency": time.time() - start_time,
            "generated_code": "",
            "test_results": [],
        }

    code_str = extract_python_code(response_text)
    logger.debug("baseline response received: chars=%d, code_chars=%d", len(response_text), len(code_str))
    all_test_passed, test_results = evaluate_on_test(code_str, test_pairs)
    latency = time.time() - start_time

    return {
        "strategy": "baseline",
        "success": all_test_passed,
        "latency": latency,
        "generated_code": code_str,
        "test_results": test_results,
    }


async def run_cegis(
    task: Dict[str, Any],
    max_iters: int = config.MAX_CEGIS_ITERS,
    model: str = config.MODEL_NAME
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

    for iteration in range(1, max_iters + 1):
        iteration_start = time.perf_counter()
        logger.info("CEGIS iteration %d/%d started", iteration, max_iters)
        try:
            response_text = await call_llm(messages, model=model)
        except Exception as e:
            logger.error("CEGIS iteration %d LLM failed after %.2fs: %s", iteration, time.perf_counter() - iteration_start, e)
            iteration_history.append({"iteration": iteration, "error": f"LLM Call Failed: {str(e)}"})
            break

        current_code = extract_python_code(response_text)
        logger.debug("CEGIS iteration %d response received: chars=%d, code_chars=%d", iteration, len(response_text), len(current_code))
        messages.append({"role": "assistant", "content": response_text})

        # Validate against all training examples
        failed_example: Optional[Tuple[int, List[List[int]], List[List[int]], Optional[List[List[int]]], str]] = None
        training_eval_start = time.perf_counter()
        for idx, pair in enumerate(train_pairs):
            inp = pair["input"]
            expected = pair["output"]
            example_start = time.perf_counter()
            success, actual, error_msg = run_transform(current_code, inp)
            logger.debug(
                "train example %d/%d: %s in %.3fs%s",
                idx + 1,
                len(train_pairs),
                "PASS" if success and actual == expected else "FAIL",
                time.perf_counter() - example_start,
                f" ({error_msg})" if error_msg else "",
            )
            
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
            logger.info(
                "CEGIS iteration %d completed in %.2fs; training evaluation=%.2fs",
                iteration,
                time.perf_counter() - iteration_start,
                time.perf_counter() - training_eval_start,
            )
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

        logger.info(
            "CEGIS iteration %d completed in %.2fs; training evaluation=%.2fs",
            iteration,
            time.perf_counter() - iteration_start,
            time.perf_counter() - training_eval_start,
        )

    # Evaluate final candidate code on test pairs
    all_test_passed, test_results = evaluate_on_test(current_code, test_pairs) if current_code else (False, [])
    latency = time.time() - start_time

    return {
        "strategy": "cegis",
        "success": all_test_passed,
        "converged_train": converged_train,
        "iterations_used": len(iteration_history),
        "latency": latency,
        "generated_code": current_code,
        "iteration_history": iteration_history,
        "test_results": test_results,
    }
