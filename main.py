#!/usr/bin/env python3
"""
Main entry point for running comparative ARC-AGI-1 experiments (Baseline vs CEGIS).
"""

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List

from arc_cegis import config, load_tasks, run_baseline, run_cegis


async def main() -> None:
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

    args = parser.parse_args()

    print("=" * 70)
    print(" ARC-AGI-1 Comparative Experiment: Baseline vs CEGIS")
    print(
        f" Model: {args.model} | Max CEGIS Iters: {args.max_iters} | "
        f"Timeout: {config.TIMEOUT_SECONDS}s | Concurrent Tasks: {args.max_concurrent_tasks} | "
        f"API Interval: {config.REQUEST_DELAY}s"
    )
    print("=" * 70)

    # 1. Load tasks
    tasks_dict = load_tasks(args.tasks)
    if args.max_tasks:
        tasks_dict = dict(list(tasks_dict.items())[:args.max_tasks])

    total_tasks = len(tasks_dict)
    print(f"Loaded {total_tasks} task(s) to evaluate.\n")

    async def run_task(
        task_number: int, task_id: str, task_data: Dict[str, Any], semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        async with semaphore:
            task_start = time.perf_counter()
            print(f"[{task_number}/{total_tasks}] Task started: {task_id}", flush=True)
            base_res, cegis_res = await asyncio.gather(
                run_baseline(task_data, model=args.model),
                run_cegis(task_data, max_iters=args.max_iters, model=args.model),
            )
            print(
                f"[{task_number}/{total_tasks}] Task completed: {task_id} | "
                f"Baseline={'PASSED' if base_res['success'] else 'FAILED'} "
                f"({base_res['latency']:.2f}s) | "
                f"CEGIS={'PASSED' if cegis_res['success'] else 'FAILED'} "
                f"({cegis_res['latency']:.2f}s) | "
                f"wall={time.perf_counter() - task_start:.2f}s",
                flush=True,
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

    print("\n" + "=" * 70)
    print(" EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total Tasks Evaluated : {total_tasks}")
    print(f"Baseline Accuracy     : {baseline_correct}/{total_tasks} ({base_acc:.2f}%)")
    print(f"CEGIS Accuracy        : {cegis_correct}/{total_tasks} ({cegis_acc:.2f}%)")
    print(f"Absolute Gain         : {cegis_acc - base_acc:+.2f}%")
    print("=" * 70)

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

    print(f"\nDetailed experiment results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
