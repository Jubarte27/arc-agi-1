"""
Main entry point for running comparative ARC-AGI-1 experiments (Baseline vs CEGIS).
Supports Google Gemini and OpenAI-compatible providers such as Groq.
Includes Proactive Rate Limiter (RPM), Daily Quota Guard (RPD), and Checkpoint / Resume.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from typing import Any, Dict, List, Set

from arc_cegis import (
    AuthError,
    QuotaExceededError,
    config,
    get_request_count,
    load_tasks,
    run_baseline,
    run_cegis,
    call_llm,
)


logger = logging.getLogger(__name__)


_EMERGENCY_STATE: Dict[str, Any] = {
    "output_path": "results_experiment.json",
    "model": config.MODEL_NAME,
    "max_iters": config.MAX_CEGIS_ITERS,
    "total_tasks": 0,
    "detailed_results": [],
    "baseline_correct": 0,
    "cegis_correct": 0,
    "faulty_task_ids": set(),
}


def perform_health_check(model: str) -> None:
    """
    Executes a fast test query to validate API key, connectivity, and model availability.
    Terminates execution immediately if the check fails.
    """
    logger.info("Performing initial %s API health check...", config.LLM_PROVIDER)
    test_messages = [{"role": "user", "content": "Responda apenas 'OK'"}]
    try:
        response = call_llm(test_messages, model=model, max_retries=3)
        logger.info("Health check PASSED! Model '%s' responded successfully: %s", model, response.strip()[:30])
    except AuthError as auth_err:
        logger.critical("FATAL AUTHENTICATION ERROR: %s", auth_err)
        logger.critical("Please verify the API key for provider '%s' is correctly set.", config.LLM_PROVIDER)
        raise SystemExit(1)
    except QuotaExceededError as q_err:
        logger.critical("FATAL QUOTA ERROR: %s", q_err)
        raise SystemExit(1)
    except Exception as err:
        logger.critical("FATAL HEALTH CHECK ERROR: Failed to connect to %s API: %s", config.LLM_PROVIDER, err)
        logger.critical("Verify internet connectivity and model name '%s'.", model)
        raise SystemExit(1)


def save_checkpoint(
    output_path: str,
    model: str,
    max_iters: int,
    total_tasks: int,
    detailed_results: List[Dict[str, Any]],
    baseline_correct: int,
    cegis_correct: int,
    faulty_task_ids: Set[str],
    checkpoint_error: str = "",
) -> None:
    """
    Safely writes current experiment state and results to JSON disk.
    """
    completed_count = len(detailed_results)
    base_acc = (baseline_correct / completed_count) * 100 if completed_count > 0 else 0.0
    cegis_acc = (cegis_correct / completed_count) * 100 if completed_count > 0 else 0.0

    output_payload = {
        "config": {
            "model": model,
            "max_cegis_iters": max_iters,
            "timeout_seconds": config.TIMEOUT_SECONDS,
            "request_delay_seconds": config.REQUEST_DELAY,
            "max_daily_requests": config.MAX_DAILY_REQUESTS,
            "total_tasks_in_dataset": total_tasks,
            "completed_tasks": completed_count,
            "faulty_tasks": len(faulty_task_ids),
            "total_requests_used": get_request_count(),
        },
        "summary": {
            "baseline_accuracy": base_acc,
            "cegis_accuracy": cegis_acc,
            "baseline_correct": baseline_correct,
            "cegis_correct": cegis_correct,
        },
        "results": detailed_results,
        "faulty_task_ids": sorted(faulty_task_ids),
    }
    if checkpoint_error:
        output_payload["emergency_error"] = checkpoint_error

    temp_path = f"{output_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    os.replace(temp_path, output_path)


def load_checkpoint(output_path: str) -> tuple[List[Dict[str, Any]], int, int, Set[str], Set[str]]:
    """
    Loads completed task results from an existing checkpoint file.
    Returns (detailed_results, baseline_correct, cegis_correct, completed_task_ids, faulty_task_ids).
    """
    if not os.path.exists(output_path):
        return [], 0, 0, set(), set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        detailed_results = data.get("results", [])
        baseline_correct = 0
        cegis_correct = 0
        completed_task_ids = set()
        faulty_task_ids = set(data.get("faulty_task_ids", []))

        for item in detailed_results:
            task_id = item.get("task_id")
            if task_id:
                completed_task_ids.add(task_id)
            if item.get("baseline", {}).get("success"):
                baseline_correct += 1
            if item.get("cegis", {}).get("success"):
                cegis_correct += 1

        return detailed_results, baseline_correct, cegis_correct, completed_task_ids, faulty_task_ids
    except Exception as e:
        logger.warning("Failed to read checkpoint from '%s': %s. Starting fresh.", output_path, e)
        return [], 0, 0, set(), set()


def save_emergency_checkpoint(error: BaseException) -> None:
    """Best-effort save to a separate file when main exits unexpectedly."""
    state = _EMERGENCY_STATE
    emergency_path = f"{state['output_path']}.emergency.json"
    error_message = f"{type(error).__name__}: {error}"
    try:
        save_checkpoint(
            output_path=emergency_path,
            model=state["model"],
            max_iters=state["max_iters"],
            total_tasks=state["total_tasks"],
            detailed_results=state["detailed_results"],
            baseline_correct=state["baseline_correct"],
            cegis_correct=state["cegis_correct"],
            faulty_task_ids=state["faulty_task_ids"],
            checkpoint_error=error_message,
        )
        logger.critical("Emergency checkpoint saved to '%s'.", emergency_path)
    except Exception:
        logger.exception("Emergency checkpoint failed; attempting raw fallback.")
        try:
            with open(f"{emergency_path}.raw", "w", encoding="utf-8") as file:
                json.dump({"error": error_message, "state": state}, file, default=str, indent=2)
        except Exception:
            logger.exception("Raw emergency checkpoint also failed.")

def _run_main() -> None:
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    file_handler = logging.FileHandler(config.LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler], force=True)
    parser = argparse.ArgumentParser(
        description="ARC-AGI-1 Comparative Experiment: Baseline (1-shot) vs CEGIS with Free Tier Protections"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="./data",
        help="Path to ARC task JSON file or directory containing task JSON files (default: ./data)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=config.MAX_CEGIS_ITERS,
        help=f"Maximum CEGIS refinement iterations per task (default: {config.MAX_CEGIS_ITERS})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=("gemini", "groq"),
        default=config.LLM_PROVIDER,
        help=f"LLM provider (default: {config.LLM_PROVIDER})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        help=f"Model name / endpoint (default: {config.MODEL_NAME})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_experiment.json",
        help="Output JSON path to save experiment results (default: results_experiment.json)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Automatically resume from existing output checkpoint file (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Overwrite existing output file and restart from scratch",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the initial provider API connectivity check (not recommended)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=config.REQUEST_DELAY,
        help=f"Minimum delay between requests in seconds (default: {config.REQUEST_DELAY}s)",
    )
    parser.add_argument(
        "--max-daily-requests",
        type=int,
        default=config.MAX_DAILY_REQUESTS,
        help=f"Maximum daily requests ceiling before safe pause (default: {config.MAX_DAILY_REQUESTS})",
    )
    args = parser.parse_args()

    _EMERGENCY_STATE.update({
        "output_path": args.output,
        "model": args.model,
        "max_iters": args.max_iters,
    })

    # Update dynamic config overrides
    config.REQUEST_DELAY = args.request_delay
    config.MAX_DAILY_REQUESTS = args.max_daily_requests
    config.LLM_PROVIDER = args.provider
    config.API_BASE_URL = config.get_api_base_url(args.provider)

    rpm_effective = int(60.0 / config.REQUEST_DELAY) if config.REQUEST_DELAY > 0 else 0

    logger.info("ARC-AGI-1 Comparative Experiment: Baseline vs CEGIS (%s)", config.LLM_PROVIDER)
    logger.info("Model: %s | Max CEGIS Iters: %s | Timeout: %ss", args.model, args.max_iters, config.TIMEOUT_SECONDS)
    logger.info("Rate Delay: %ss (<= %s RPM) | Daily Quota Guard: %s RPD", config.REQUEST_DELAY, rpm_effective, config.MAX_DAILY_REQUESTS)

    # 1. Health check
    if not args.skip_health_check:
        perform_health_check(args.model)

    # 2. Load tasks
    tasks_dict = load_tasks(args.tasks)
    if args.max_tasks:
        tasks_dict = dict(list(tasks_dict.items())[:args.max_tasks])

    total_tasks = len(tasks_dict)
    logger.info("Loaded %s task(s) in evaluation set.", total_tasks)
    _EMERGENCY_STATE["total_tasks"] = total_tasks

    # 3. Checkpoint / Resume recovery
    detailed_results: List[Dict[str, Any]] = []
    baseline_correct = 0
    cegis_correct = 0
    completed_task_ids: Set[str] = set()
    faulty_task_ids: Set[str] = set()
    _EMERGENCY_STATE.update({
        "detailed_results": detailed_results,
        "faulty_task_ids": faulty_task_ids,
    })

    if args.resume and os.path.exists(args.output):
        detailed_results, baseline_correct, cegis_correct, completed_task_ids, faulty_task_ids = load_checkpoint(args.output)
        _EMERGENCY_STATE.update({
            "detailed_results": detailed_results,
            "baseline_correct": baseline_correct,
            "cegis_correct": cegis_correct,
            "faulty_task_ids": faulty_task_ids,
        })
        if completed_task_ids or faulty_task_ids:
            logger.info(
                "Checkpoint found. Resuming from '%s': %s completed, %s faulty, %s/%s tasks accounted for "
                "(Baseline: %s, CEGIS: %s).",
                args.output, len(completed_task_ids), len(faulty_task_ids),
                len(completed_task_ids | faulty_task_ids), total_tasks, baseline_correct, cegis_correct,
            )

    consecutive_api_failures = 0
    CIRCUIT_BREAKER_THRESHOLD = 3
    quota_exhausted = False
    auth_failed = False

    def evaluate_task(task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run both strategies for one task in a worker thread."""
        base_res = run_baseline(task_data, model=args.model)
        cegis_res = run_cegis(task_data, max_iters=args.max_iters, model=args.model)
        return {
            "task_id": task_id,
            "baseline": base_res,
            "cegis": cegis_res,
        }

    # 4. Run evaluations with Protections and Incremental Checkpointing
    current_task_id: str | None = None
    try:
        pending_tasks = [
            (task_id, task_data)
            for task_id, task_data in tasks_dict.items()
            if task_id not in completed_task_ids and task_id not in faulty_task_ids
        ]
        worker_count = max(1, config.MAX_CONCURRENT_TASKS)
        logger.info("Running %s task(s) with %s concurrent worker(s).", len(pending_tasks), worker_count)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {
                executor.submit(evaluate_task, task_id, task_data): task_id
                for task_id, task_data in pending_tasks
            }

            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                current_task_id = task_id
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                except AuthError as auth_err:
                    auth_failed = True
                    logger.error("AUTH ERROR: %s", auth_err)
                    for pending in future_to_task:
                        pending.cancel()
                    continue
                except QuotaExceededError as quota_err:
                    quota_exhausted = True
                    logger.error("SAFE PAUSE / DAILY QUOTA GUARD: %s", quota_err)
                    for pending in future_to_task:
                        pending.cancel()
                    continue
                except Exception as err:
                    faulty_task_ids.add(task_id)
                    logger.exception("TASK ERROR: %s failed and will not be added to results: %s", task_id, err)
                    continue

                base_res = result["baseline"]
                cegis_res = result["cegis"]
                task_api_error = base_res.get("api_error", False) or cegis_res.get("api_error", False)
                if base_res["success"]:
                    baseline_correct += 1
                if cegis_res["success"]:
                    cegis_correct += 1
                if task_api_error:
                    consecutive_api_failures += 1
                else:
                    consecutive_api_failures = 0

                detailed_results.append(result)
                completed_task_ids.add(task_id)
                _EMERGENCY_STATE.update({
                    "baseline_correct": baseline_correct,
                    "cegis_correct": cegis_correct,
                })
                logger.info(
                    "[%s/%s] %s: Baseline=%s, CEGIS=%s (API Requests: %s/%s)",
                    len(detailed_results), total_tasks, task_id,
                    "PASSED" if base_res["success"] else "FAILED",
                    "PASSED" if cegis_res["success"] else "FAILED",
                    get_request_count(), config.MAX_DAILY_REQUESTS,
                )
                save_checkpoint(
                    output_path=args.output,
                    model=args.model,
                    max_iters=args.max_iters,
                    total_tasks=total_tasks,
                    detailed_results=detailed_results,
                    baseline_correct=baseline_correct,
                    cegis_correct=cegis_correct,
                    faulty_task_ids=faulty_task_ids,
                )

                if consecutive_api_failures >= CIRCUIT_BREAKER_THRESHOLD:
                    logger.error("CIRCUIT BREAKER: %s consecutive tasks had API errors.", CIRCUIT_BREAKER_THRESHOLD)
                    for pending in future_to_task:
                        pending.cancel()
                    break

        if auth_failed:
            logger.error("Experiment stopped because API authentication failed.")
        elif quota_exhausted:
            logger.error("Progress safely saved to '%s'; rerun to resume after quota reset.", args.output)

    except KeyboardInterrupt:
        logger.warning("Interrupted. Saving current checkpoint...")
        raise SystemExit(0)
    except Exception as err:
        if current_task_id is not None:
            faulty_task_ids.add(current_task_id)
        logger.exception("EXPERIMENT ERROR: %s", err)
        raise
    finally:
        try:
            save_checkpoint(
                output_path=args.output,
                model=args.model,
                max_iters=args.max_iters,
                total_tasks=total_tasks,
                detailed_results=detailed_results,
                baseline_correct=baseline_correct,
                cegis_correct=cegis_correct,
                faulty_task_ids=faulty_task_ids,
            )
            logger.info("Checkpoint safely saved to '%s'. You can resume at any time.", args.output)
        except Exception:
            logger.exception("Failed to save checkpoint to '%s'.", args.output)

    # 5. Print Summary
    evaluated_count = len(detailed_results)
    base_acc = (baseline_correct / evaluated_count) * 100 if evaluated_count > 0 else 0.0
    cegis_acc = (cegis_correct / evaluated_count) * 100 if evaluated_count > 0 else 0.0

    logger.info("EXPERIMENT SUMMARY%s", " (PAUSED - DAILY QUOTA REACHED)" if quota_exhausted else "")
    logger.info("Total Tasks in Dataset: %s", total_tasks)
    if total_tasks:
        logger.info("Tasks Completed: %s/%s (%.1f%%)", evaluated_count, total_tasks, evaluated_count / total_tasks * 100)
    logger.info("Total API Calls Used: %s/%s", get_request_count(), config.MAX_DAILY_REQUESTS)
    logger.info("Baseline Accuracy: %s", f"{baseline_correct}/{evaluated_count} ({base_acc:.2f}%)" if evaluated_count else "N/A")
    logger.info("CEGIS Accuracy: %s", f"{cegis_correct}/{evaluated_count} ({cegis_acc:.2f}%)" if evaluated_count else "N/A")
    if evaluated_count > 0:
        logger.info("Absolute Gain: %+.2f%%", cegis_acc - base_acc)
    logger.info("Results safely saved to: %s", args.output)


def main() -> None:
    """Run the experiment and preserve state if any failure escapes handling."""
    try:
        _run_main()
    except BaseException as error:
        save_emergency_checkpoint(error)
        raise


if __name__ == "__main__":
    main()
