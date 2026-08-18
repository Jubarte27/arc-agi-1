#!/usr/bin/env python3
"""
Main entry point for running comparative ARC-AGI-1 experiments (Baseline vs CEGIS).
"""

import argparse
import json
from typing import Any, Dict, List

from arc_cegis import config, load_tasks, run_baseline, run_cegis


def main() -> None:
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

    args = parser.parse_args()

    print("=" * 70)
    print(" ARC-AGI-1 Comparative Experiment: Baseline vs CEGIS")
    print(f" Model: {args.model} | Max CEGIS Iters: {args.max_iters} | Timeout: {config.TIMEOUT_SECONDS}s")
    print("=" * 70)

    # 1. Load tasks
    tasks_dict = load_tasks(args.tasks)
    if args.max_tasks:
        tasks_dict = dict(list(tasks_dict.items())[:args.max_tasks])

    total_tasks = len(tasks_dict)
    print(f"Loaded {total_tasks} task(s) to evaluate.\n")

    detailed_results: List[Dict[str, Any]] = []
    baseline_correct = 0
    cegis_correct = 0

    # 2. Run evaluations
    for i, (task_id, task_data) in enumerate(tasks_dict.items(), start=1):
        print(f"[{i}/{total_tasks}] Task: {task_id}")
        
        # Run Baseline
        base_res = run_baseline(task_data, model=args.model)
        if base_res["success"]:
            baseline_correct += 1
        print(f"   - Baseline: {'PASSED' if base_res['success'] else 'FAILED'} ({base_res['latency']:.2f}s)")

        # Run CEGIS
        cegis_res = run_cegis(task_data, max_iters=args.max_iters, model=args.model)
        if cegis_res["success"]:
            cegis_correct += 1
        print(
            f"   - CEGIS:    {'PASSED' if cegis_res['success'] else 'FAILED'} "
            f"(Iters: {cegis_res.get('iterations_used', 0)}, "
            f"Train Converged: {cegis_res.get('converged_train', False)}, "
            f"{cegis_res['latency']:.2f}s)"
        )

        detailed_results.append({
            "task_id": task_id,
            "baseline": base_res,
            "cegis": cegis_res,
        })

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
    main()
