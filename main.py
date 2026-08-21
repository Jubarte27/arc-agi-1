"""
Main entry point for running comparative ARC-AGI-1 experiments (Baseline vs CEGIS).
Strictly integrated with the official Google Gemini API (google-genai SDK).
Includes Proactive Rate Limiter (RPM), Daily Quota Guard (RPD), and Checkpoint / Resume.
"""

import argparse
import json
import os
import sys
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


def perform_health_check(model: str) -> None:
    """
    Executes a fast test query to validate API key, connectivity, and model availability.
    Terminates execution immediately if the check fails.
    """
    print("Performing initial Gemini API health check...")
    test_messages = [{"role": "user", "content": "Responda apenas 'OK'"}]
    try:
        response = call_llm(test_messages, model=model, max_retries=3)
        print(f"Health check PASSED! Model '{model}' responded successfully: {response.strip()[:30]}\n")
    except AuthError as auth_err:
        print("\n" + "=" * 70, file=sys.stderr)
        print(f" [FATAL AUTHENTICATION ERROR] {auth_err}", file=sys.stderr)
        print(" Please verify that GEMINI_API_KEY or GOOGLE_API_KEY is correctly set.", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)
    except QuotaExceededError as q_err:
        print("\n" + "=" * 70, file=sys.stderr)
        print(f" [FATAL QUOTA ERROR] {q_err}", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print("\n" + "=" * 70, file=sys.stderr)
        print(f" [FATAL HEALTH CHECK ERROR] Failed to connect to Gemini API: {err}", file=sys.stderr)
        print(f" Verify internet connectivity and model name '{model}'.", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)


def save_checkpoint(
    output_path: str,
    model: str,
    max_iters: int,
    total_tasks: int,
    detailed_results: List[Dict[str, Any]],
    baseline_correct: int,
    cegis_correct: int,
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
            "total_requests_used": get_request_count(),
        },
        "summary": {
            "baseline_accuracy": base_acc,
            "cegis_accuracy": cegis_acc,
            "baseline_correct": baseline_correct,
            "cegis_correct": cegis_correct,
        },
        "results": detailed_results,
    }

    temp_path = f"{output_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    os.replace(temp_path, output_path)


def load_checkpoint(output_path: str) -> tuple[List[Dict[str, Any]], int, int, Set[str]]:
    """
    Loads completed task results from an existing checkpoint file.
    Returns (detailed_results, baseline_correct, cegis_correct, completed_task_ids).
    """
    if not os.path.exists(output_path):
        return [], 0, 0, set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        detailed_results = data.get("results", [])
        baseline_correct = 0
        cegis_correct = 0
        completed_task_ids = set()

        for item in detailed_results:
            task_id = item.get("task_id")
            if task_id:
                completed_task_ids.add(task_id)
            if item.get("baseline", {}).get("success"):
                baseline_correct += 1
            if item.get("cegis", {}).get("success"):
                cegis_correct += 1

        return detailed_results, baseline_correct, cegis_correct, completed_task_ids
    except Exception as e:
        print(f"[Warning] Failed to read checkpoint from '{output_path}': {e}. Starting fresh.")
        return [], 0, 0, set()


def main() -> None:
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
        help="Skip the initial Gemini API connectivity check (not recommended)",
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

    # Update dynamic config overrides
    config.REQUEST_DELAY = args.request_delay
    config.MAX_DAILY_REQUESTS = args.max_daily_requests

    rpm_effective = int(60.0 / config.REQUEST_DELAY) if config.REQUEST_DELAY > 0 else 0

    print("=" * 70)
    print(" ARC-AGI-1 Comparative Experiment: Baseline vs CEGIS (Google AI Studio / GenAI)")
    print(f" Model: {args.model} | Max CEGIS Iters: {args.max_iters} | Timeout: {config.TIMEOUT_SECONDS}s")
    print(f" Rate Delay: {config.REQUEST_DELAY}s (<= {rpm_effective} RPM) | Daily Quota Guard: {config.MAX_DAILY_REQUESTS} RPD")
    print("=" * 70)

    # 1. Health check
    if not args.skip_health_check:
        perform_health_check(args.model)

    # 2. Load tasks
    tasks_dict = load_tasks(args.tasks)
    if args.max_tasks:
        tasks_dict = dict(list(tasks_dict.items())[:args.max_tasks])

    total_tasks = len(tasks_dict)
    print(f"Loaded {total_tasks} task(s) in evaluation set.")

    # 3. Checkpoint / Resume recovery
    detailed_results: List[Dict[str, Any]] = []
    baseline_correct = 0
    cegis_correct = 0
    completed_task_ids: Set[str] = set()

    if args.resume and os.path.exists(args.output):
        detailed_results, baseline_correct, cegis_correct, completed_task_ids = load_checkpoint(args.output)
        if completed_task_ids:
            print(
                f"[Checkpoint Found] Resuming experiment from '{args.output}'.\n"
                f" -> {len(completed_task_ids)}/{total_tasks} tasks already completed "
                f"(Baseline: {baseline_correct}, CEGIS: {cegis_correct}).\n"
            )

    consecutive_api_failures = 0
    CIRCUIT_BREAKER_THRESHOLD = 3
    quota_exhausted = False

    # 4. Run evaluations with Protections and Incremental Checkpointing
    try:
        for i, (task_id, task_data) in enumerate(tasks_dict.items(), start=1):
            if task_id in completed_task_ids:
                continue

            print(f"[{i}/{total_tasks}] Task: {task_id} (API Requests: {get_request_count()}/{config.MAX_DAILY_REQUESTS})")
            task_api_error = False

            try:
                # Run Baseline
                base_res = run_baseline(task_data, model=args.model)
                if base_res.get("api_error"):
                    task_api_error = True
                if base_res["success"]:
                    baseline_correct += 1
                print(f"   - Baseline: {'PASSED' if base_res['success'] else 'FAILED'} ({base_res['latency']:.2f}s)")

                # Run CEGIS
                cegis_res = run_cegis(task_data, max_iters=args.max_iters, model=args.model)
                if cegis_res.get("api_error"):
                    task_api_error = True
                if cegis_res["success"]:
                    cegis_correct += 1
                print(
                    f"   - CEGIS:    {'PASSED' if cegis_res['success'] else 'FAILED'} "
                    f"(Iters: {cegis_res.get('iterations_used', 0)}, "
                    f"Train Converged: {cegis_res.get('converged_train', False)}, "
                    f"{cegis_res['latency']:.2f}s)"
                )

            except AuthError as auth_err:
                print("\n" + "!" * 70, file=sys.stderr)
                print(f" [CIRCUIT BREAKER / AUTH ERROR] {auth_err}", file=sys.stderr)
                print(" Aborting experiment immediately to protect API credentials.", file=sys.stderr)
                print("!" * 70 + "\n", file=sys.stderr)
                break

            except QuotaExceededError as quota_err:
                quota_exhausted = True
                print("\n" + "!" * 70, file=sys.stderr)
                print(f" [SAFE PAUSE / DAILY QUOTA GUARD] {quota_err}", file=sys.stderr)
                print(f" Progress safely saved to '{args.output}'.", file=sys.stderr)
                print(" When your daily quota resets, simply re-run the same command to resume seamlessly:", file=sys.stderr)
                print(f"   python3 main.py --tasks {args.tasks} --model {args.model} --output {args.output}", file=sys.stderr)
                print("!" * 70 + "\n", file=sys.stderr)
                break

            if task_api_error:
                consecutive_api_failures += 1
                print(f"   [Warning] API error occurred ({consecutive_api_failures}/{CIRCUIT_BREAKER_THRESHOLD} consecutive).")
                if consecutive_api_failures >= CIRCUIT_BREAKER_THRESHOLD:
                    print("\n" + "!" * 70, file=sys.stderr)
                    print(f" [CIRCUIT BREAKER TRIGGERED] {CIRCUIT_BREAKER_THRESHOLD} consecutive tasks failed with API errors.", file=sys.stderr)
                    print(" Aborting experiment execution to prevent wasted calls.", file=sys.stderr)
                    print("!" * 70 + "\n", file=sys.stderr)
                    break
            else:
                consecutive_api_failures = 0

            detailed_results.append({
                "task_id": task_id,
                "baseline": base_res,
                "cegis": cegis_res,
            })
            completed_task_ids.add(task_id)

            # Incremental checkpoint save after every completed task
            save_checkpoint(
                output_path=args.output,
                model=args.model,
                max_iters=args.max_iters,
                total_tasks=total_tasks,
                detailed_results=detailed_results,
                baseline_correct=baseline_correct,
                cegis_correct=cegis_correct,
            )

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Execution stopped by user. Saving current checkpoint...")
        save_checkpoint(
            output_path=args.output,
            model=args.model,
            max_iters=args.max_iters,
            total_tasks=total_tasks,
            detailed_results=detailed_results,
            baseline_correct=baseline_correct,
            cegis_correct=cegis_correct,
        )
        print(f"Checkpoint safely saved to '{args.output}'. You can resume at any time.")
        sys.exit(0)

    # 5. Print Summary
    evaluated_count = len(detailed_results)
    base_acc = (baseline_correct / evaluated_count) * 100 if evaluated_count > 0 else 0.0
    cegis_acc = (cegis_correct / evaluated_count) * 100 if evaluated_count > 0 else 0.0

    print("\n" + "=" * 70)
    print(" EXPERIMENT SUMMARY" + (" (PAUSED - DAILY QUOTA REACHED)" if quota_exhausted else ""))
    print("=" * 70)
    print(f"Total Tasks in Dataset : {total_tasks}")
    print(f"Tasks Completed        : {evaluated_count}/{total_tasks} ({(evaluated_count / total_tasks * 100):.1f}%)" if total_tasks else "")
    print(f"Total API Calls Used   : {get_request_count()}/{config.MAX_DAILY_REQUESTS}")
    print(f"Baseline Accuracy      : {baseline_correct}/{evaluated_count} ({base_acc:.2f}%)" if evaluated_count else "Baseline Accuracy: N/A")
    print(f"CEGIS Accuracy         : {cegis_correct}/{evaluated_count} ({cegis_acc:.2f}%)" if evaluated_count else "CEGIS Accuracy: N/A")
    if evaluated_count > 0:
        print(f"Absolute Gain          : {cegis_acc - base_acc:+.2f}%")
    print("=" * 70)
    print(f"Results safely saved to: {args.output}\n")


if __name__ == "__main__":
    main()
