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
from util.parsing import compact_long_numeric_lists

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
PROGRESS_CHECKPOINT_INTERVAL = 5


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


def extract_first_n_runs(input_path: str, output_path: str, n: int) -> None:
    """Write a result file containing only its first ``n`` runs."""
    if n < 0:
        raise ValueError("n must be non-negative")

    with open(input_path, "r", encoding="utf-8") as results_file:
        payload = json.load(results_file)

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("The result file must contain a 'results' list.")

    selected_results = results[:n]
    baseline_correct = sum(
        bool(item.get("baseline", {}).get("success"))
        for item in selected_results
        if isinstance(item, dict)
    )
    cegis_correct = sum(
        bool(item.get("cegis", {}).get("success"))
        for item in selected_results
        if isinstance(item, dict)
    )
    selected_count = len(selected_results)

    output_payload = dict(payload)
    output_payload["results"] = selected_results
    output_payload["summary"] = {
        "baseline_accuracy": baseline_correct / selected_count * 100 if selected_count else 0.0,
        "cegis_accuracy": cegis_correct / selected_count * 100 if selected_count else 0.0,
        "baseline_correct": baseline_correct,
        "cegis_correct": cegis_correct,
    }
    if isinstance(output_payload.get("config"), dict):
        output_payload["config"] = dict(output_payload["config"])
        output_payload["config"]["completed_tasks"] = selected_count
        request_count = 0
        has_iteration_history = False
        for item in selected_results:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("baseline"), dict) and item["baseline"]:
                request_count += 1
            cegis_result = item.get("cegis")
            if isinstance(cegis_result, dict) and isinstance(
                cegis_result.get("iteration_history"), list
            ):
                has_iteration_history = True
                request_count += len(cegis_result["iteration_history"])
        if has_iteration_history:
            output_payload["config"]["total_requests_used"] = request_count
    if isinstance(output_payload.get("faulty_task_ids"), list):
        selected_ids = {
            item.get("task_id") for item in selected_results if isinstance(item, dict)
        }
        output_payload["faulty_task_ids"] = [
            task_id
            for task_id in output_payload["faulty_task_ids"]
            if task_id in selected_ids
        ]

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as results_file:
        serialized_payload = json.dumps(output_payload, indent=2)
        results_file.write(compact_long_numeric_lists(serialized_payload))


def perform_health_check(
    model: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
) -> None:
    """
    Executes a fast test query to validate API key, connectivity, and model availability.
    Terminates execution immediately if the check fails.
    """
    target_provider = provider or config.LLM_PROVIDER
    target_model = model or config.MODEL_NAME
    logger.info("Performing initial %s API health check...", target_provider)
    test_messages = [{"role": "user", "content": "Responda apenas 'OK'"}]
    try:
        response = call_llm(
            test_messages,
            model=target_model,
            provider=target_provider,
            api_key=api_key,
            max_retries=3,
        )
        logger.info(
            "Health check PASSED! Model '%s' (Provider: %s) responded successfully: %s",
            target_model,
            target_provider,
            response.strip()[:30],
        )
    except AuthError as auth_err:
        logger.critical("FATAL AUTHENTICATION ERROR: %s", auth_err)
        logger.critical("Please verify the API key for provider '%s' is correctly set.", target_provider)
        raise SystemExit(1)
    except QuotaExceededError as q_err:
        logger.critical("FATAL QUOTA ERROR: %s", q_err)
        raise SystemExit(1)
    except Exception as err:
        logger.critical("FATAL HEALTH CHECK ERROR: Failed to connect to %s API: %s", target_provider, err)
        logger.critical("Verify internet connectivity and model name '%s'.", target_model)
        raise SystemExit(1)


def perform_pool_health_check(pool: list[config.LLMConfig] | None = None) -> None:
    """
    Executes health checks across all LLM configurations in the pool.
    Terminates execution immediately if any pool entry fails.
    """
    target_pool = pool if pool is not None else config.LLM_POOL
    if not target_pool:
        logger.warning("LLM pool is empty; running single provider health check.")
        perform_health_check()
        return

    logger.info("Performing initial LLM pool health check across %s entries...", len(target_pool))
    for idx, llm_cfg in enumerate(target_pool, start=1):
        logger.info(
            "Checking pool entry [%s/%s]: provider=%s, model=%s (pool_index=%s)...",
            idx,
            len(target_pool),
            llm_cfg.provider,
            llm_cfg.model,
            llm_cfg.pool_index,
        )
        try:
            perform_health_check(
                model=llm_cfg.model,
                provider=llm_cfg.provider,
                api_key=llm_cfg.api_key,
            )
        except SystemExit:
            logger.critical(
                "Pool health check FAILED on entry [%s/%s]: provider=%s, model=%s.",
                idx,
                len(target_pool),
                llm_cfg.provider,
                llm_cfg.model,
            )
            raise

    logger.info("Pool health check PASSED for all %s entries!", len(target_pool))


perform_pool_heath_check = perform_pool_health_check


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
        serialized_payload = json.dumps(output_payload, indent=2)
        f.write(compact_long_numeric_lists(serialized_payload))
    os.replace(temp_path, output_path)


def progress_checkpoint_path(output_path: str, completed_count: int, total_tasks: int) -> str:
    """Build a progress-labelled checkpoint path beside the main output file."""
    output_root, extension = os.path.splitext(output_path)
    return f"{output_root}_{completed_count}_{total_tasks}{extension}"


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
    if config.LLM_POOL:
        parsed_pool = [
            {
                "provider": llm_config.provider,
                "model": llm_config.model,
                "request_delay": llm_config.request_delay,
                "max_daily_requests": llm_config.max_daily_requests,
                "max_concurrent_tasks": llm_config.max_concurrent_tasks,
                "pool_index": llm_config.pool_index,
            }
            for llm_config in config.LLM_POOL
        ]
        logger.info("Parsed LLM pool: %s", parsed_pool)
    else:
        logger.info("Parsed LLM pool: empty; using global configuration.")
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
        "--extract-first-n",
        type=int,
        default=None,
        help="Extract the first N runs from --extract-input and write them to --extract-output",
    )
    parser.add_argument(
        "--extract-input",
        type=str,
        default=None,
        help="Input results JSON for --extract-first-n",
    )
    parser.add_argument(
        "--extract-output",
        type=str,
        default=None,
        help="Output results JSON for --extract-first-n",
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

    if args.extract_first_n is not None:
        if not args.extract_input or not args.extract_output:
            parser.error("--extract-first-n requires --extract-input and --extract-output")
        extract_first_n_runs(args.extract_input, args.extract_output, args.extract_first_n)
        logger.info("Extracted first %s run(s) to '%s'.", args.extract_first_n, args.extract_output)
        return

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
    
    model = config.LLM_POOL[0].model if config.LLM_POOL else args.model

    logger.info("ARC-AGI-1 Comparative Experiment: Baseline vs CEGIS (%s)", config.LLM_PROVIDER)
    logger.info("Model: %s | Max CEGIS Iters: %s | Timeout: %ss", model, args.max_iters, config.TIMEOUT_SECONDS)
    logger.info("Rate Delay: %ss (<= %s RPM) | Daily Quota Guard: %s RPD", config.REQUEST_DELAY, rpm_effective, config.MAX_DAILY_REQUESTS)

    # 1. Health check
    if not args.skip_health_check:
        if config.LLM_POOL:
            perform_pool_health_check()
        else:
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
        selected_model = None if config.LLM_POOL else args.model
        base_res = run_baseline(task_data, model=selected_model)
        cegis_res = run_cegis(task_data, max_iters=args.max_iters, model=selected_model)
        return {
            "task_id": task_id,
            "baseline": base_res,
            "cegis": cegis_res,
        }

    # 4. Run evaluations with Protections and Incremental Checkpointing
    current_task_id: str | None = None
    previous_progress_path: str | None = None
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
                    break
                except QuotaExceededError as quota_err:
                    quota_exhausted = True
                    logger.error("SAFE PAUSE / DAILY QUOTA GUARD: %s", quota_err)
                    for pending in future_to_task:
                        pending.cancel()
                    break
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
                completed_count = len(detailed_results)
                if completed_count % PROGRESS_CHECKPOINT_INTERVAL == 0:
                    progress_path = progress_checkpoint_path(args.output, completed_count, total_tasks)
                    save_checkpoint(
                        output_path=progress_path,
                        model=args.model,
                        max_iters=args.max_iters,
                        total_tasks=total_tasks,
                        detailed_results=detailed_results,
                        baseline_correct=baseline_correct,
                        cegis_correct=cegis_correct,
                        faulty_task_ids=faulty_task_ids,
                    )
                    if previous_progress_path and os.path.exists(previous_progress_path):
                        os.remove(previous_progress_path)
                    previous_progress_path = progress_path
                    logger.info("Progress checkpoint saved to '%s'.", progress_path)

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
