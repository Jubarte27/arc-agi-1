#!/usr/bin/env python3
"""
Main entry point for running comparative ARC-AGI-1 experiments (Baseline vs CEGIS).
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

from arc_cegis import config, load_tasks, run_baseline, run_cegis


logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()
    args = parse_args()
    if not config.API_KEY:
        logger.error(
            "No API key found. Set GOOGLE_API_KEY, GEMMA_API_KEY, "
            "OPENAI_API_KEY, NVIDIA_API_KEY, or API_KEY before running."
        )

    logger.info("=" * 70)
    logger.info(" ARC-AGI-1 Comparative Experiment: Baseline vs CEGIS")
    logger.info(
        " Model: %s | Max CEGIS Iters: %d | Timeout: %ss | Concurrent Tasks: %d | API Interval: %ss",
        args.model, args.max_iters, config.TIMEOUT_SECONDS, args.max_concurrent_tasks, config.REQUEST_DELAY,
    )
    logger.info("=" * 70)

    # 1. Load tasks
    tasks_dict = load_tasks(args.tasks)
    if args.max_tasks:
        tasks_dict = dict(list(tasks_dict.items())[:args.max_tasks])

    total_tasks = len(tasks_dict)
    logger.info("Loaded %d task(s) to evaluate.", total_tasks)

    async def run_task(
        task_number: int, task_id: str, task_data: Dict[str, Any], semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        async with semaphore:
            task_start = time.perf_counter()
            logger.info("[%d/%d] Task started: %s", task_number, total_tasks, task_id)
            base_res, cegis_res = await asyncio.gather(
                run_baseline(task_data, model=args.model),
                run_cegis(task_data, max_iters=args.max_iters, model=args.model),
            )
            logger.info(
                "[%d/%d] Task completed: %s | Baseline=%s (%.2fs) | CEGIS=%s (%.2fs) | wall=%.2fs",
                task_number, total_tasks, task_id,
                "PASSED" if base_res["success"] else "FAILED", base_res["latency"],
                "PASSED" if cegis_res["success"] else "FAILED", cegis_res["latency"],
                time.perf_counter() - task_start,
            )
            return {
                "task_id": task_id,
                "baseline": base_res,
                "cegis": cegis_res,
            }

    semaphore = asyncio.Semaphore(max(1, args.max_concurrent_tasks))
    task_results = await asyncio.gather(*(
        run_task(task_number, task_id, task_data, semaphore)
        for task_number, (task_id, task_data) in enumerate(tasks_dict.items(), start=1)
    ))

    detailed_results: List[Dict[str, Any]] = list(task_results)
    baseline_correct = sum(result["baseline"]["success"] for result in detailed_results)
    cegis_correct = sum(result["cegis"]["success"] for result in detailed_results)

    # 3. Print Summary
    base_acc = (baseline_correct / total_tasks) * 100 if total_tasks > 0 else 0.0
    cegis_acc = (cegis_correct / total_tasks) * 100 if total_tasks > 0 else 0.0

    logger.info("=" * 70)
    logger.info(" EXPERIMENT SUMMARY")
    logger.info("Total Tasks Evaluated : %d", total_tasks)
    logger.info("Baseline Accuracy     : %d/%d (%.2f%%)", baseline_correct, total_tasks, base_acc)
    logger.info("CEGIS Accuracy        : %d/%d (%.2f%%)", cegis_correct, total_tasks, cegis_acc)
    logger.info("Absolute Gain         : %+.2f%%", cegis_acc - base_acc)
    logger.info("=" * 70)

    # 4. Save results to JSON
    output_payload = {
        "config": {
            "model": args.model,
            "max_cegis_iters": args.max_iters,
            "timeout_seconds": config.TIMEOUT_SECONDS,
            "request_delay": config.REQUEST_DELAY,
            "max_concurrent_tasks": args.max_concurrent_tasks,
            "total_tasks": total_tasks,
        },
        "summary": {
            "baseline_accuracy": base_acc,
            "cegis_accuracy": cegis_acc,
            "baseline_correct": baseline_correct,
            "cegis_correct": cegis_correct,
        },
        "results": detailed_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("Detailed experiment results saved to: %s", args.output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ARC-AGI-1 Comparative Experiment: Baseline (1-shot) vs CEGIS (Semantic Counterexample Feedback)"
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
        "--max-concurrent-tasks",
        type=int,
        default=config.MAX_CONCURRENT_TASKS,
        help=f"Maximum tasks to run concurrently (default: {config.MAX_CONCURRENT_TASKS})",
    )

    return parser.parse_args()

def configure_logging(log_file: str | None = None) -> None:
    """Send all logs to a file and significant messages to stdout."""
    root_logger = logging.getLogger()

    log_path = Path(log_file or config.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stdout_handler)


if __name__ == "__main__":
    asyncio.run(main())
