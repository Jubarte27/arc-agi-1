#!/usr/bin/env python3
"""Task Topology Analysis: Universally Solved vs Universally Unsolved Tasks.

Computes task difficulty metrics, consensus rates, solve count distributions,
and detects inductive trap tasks (where models overfit on training demonstrations).
Independent of specific model names or hardcoded benchmark results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def analyze_task_topology(
    df: pd.DataFrame,
    strategy_col: str = "cegis_success",
    train_conv_col: str = "cegis_converged_train",
    test_succ_col: str = "cegis_test_success",
) -> dict[str, Any]:
    """Analyze task solvability topology across all models in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark dataframe containing 'task_id', 'model', and strategy outcome columns.
    strategy_col : str
        Boolean column indicating end-to-end task success (default: 'cegis_success').
    train_conv_col : str
        Boolean column indicating convergence on training examples (default: 'cegis_converged_train').
    test_succ_col : str
        Boolean column indicating test set generalization (default: 'cegis_test_success').

    Returns
    -------
    dict[str, Any]
        Dictionary containing summary statistics, universally solved/unsolved task lists,
        solve count distribution, and inductive trap hotspots.
    """
    model_col = "friendly_name" if "friendly_name" in df.columns else "model"
    total_models = df[model_col].nunique()
    total_tasks = df["task_id"].nunique()

    # Per-task aggregation
    task_agg = df.groupby("task_id").agg(
        solved_count=(strategy_col, "sum"),
        total_evals=(strategy_col, "count"),
        train_conv_count=(train_conv_col, "sum") if train_conv_col in df.columns else (strategy_col, "count"),
        test_succ_count=(test_succ_col, "sum") if test_succ_col in df.columns else (strategy_col, "sum"),
    ).reset_index()

    task_agg["solve_rate"] = task_agg["solved_count"] / task_agg["total_evals"]
    task_agg["train_conv_rate"] = task_agg["train_conv_count"] / task_agg["total_evals"]

    # Universally solved (all models succeeded)
    universally_solved_df = task_agg[task_agg["solved_count"] == total_models]
    universally_solved_tasks = sorted(universally_solved_df["task_id"].tolist())

    # Universally unsolved (0 models succeeded)
    universally_unsolved_df = task_agg[task_agg["solved_count"] == 0]
    universally_unsolved_tasks = sorted(universally_unsolved_df["task_id"].tolist())

    # Solve count distribution (0 to total_models)
    dist_counts = task_agg["solved_count"].value_counts().reindex(range(total_models + 1), fill_value=0)
    distribution = [
        {
            "models_solving": int(k),
            "task_count": int(dist_counts[k]),
            "percentage": float(dist_counts[k] / total_tasks * 100),
        }
        for k in range(total_models + 1)
    ]

    # Inductive trap tasks:
    # High train convergence (>= 50% of models) but zero or very low test success (<= 1 model)
    inductive_traps_df = task_agg[
        (task_agg["train_conv_rate"] >= 0.50) & (task_agg["solved_count"] <= 1)
    ].sort_values("train_conv_rate", ascending=False)

    inductive_traps = [
        {
            "task_id": str(row["task_id"]),
            "train_converged_models": int(row["train_conv_count"]),
            "solved_models": int(row["solved_count"]),
            "train_convergence_rate": float(row["train_conv_rate"]),
            "test_success_rate": float(row["solve_rate"]),
        }
        for _, row in inductive_traps_df.iterrows()
    ]

    # Task difficulty tiers
    easy_tasks = task_agg[task_agg["solve_rate"] >= 0.70]["task_id"].tolist()
    moderate_tasks = task_agg[(task_agg["solve_rate"] >= 0.30) & (task_agg["solve_rate"] < 0.70)]["task_id"].tolist()
    hard_tasks = task_agg[(task_agg["solve_rate"] > 0.0) & (task_agg["solve_rate"] < 0.30)]["task_id"].tolist()

    return {
        "strategy": strategy_col,
        "total_models": total_models,
        "total_tasks": total_tasks,
        "universally_solved_count": len(universally_solved_tasks),
        "universally_solved_percentage": float(len(universally_solved_tasks) / total_tasks * 100),
        "universally_solved_tasks": universally_solved_tasks,
        "universally_unsolved_count": len(universally_unsolved_tasks),
        "universally_unsolved_percentage": float(len(universally_unsolved_tasks) / total_tasks * 100),
        "universally_unsolved_tasks": universally_unsolved_tasks,
        "solve_count_distribution": distribution,
        "inductive_traps_count": len(inductive_traps),
        "inductive_traps": inductive_traps,
        "difficulty_tiers": {
            "easy_ge_70pct": len(easy_tasks),
            "moderate_30_to_70pct": len(moderate_tasks),
            "hard_lt_30pct": len(hard_tasks),
            "unsolved_0pct": len(universally_unsolved_tasks),
        },
        "per_task_table": task_agg.to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze ARC task topology and solvability consensus.")
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="Path to benchmark CSV.")
    parser.add_argument("--strategy", type=str, default="cegis_success", help="Strategy success column.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"task_id": str})
    results = analyze_task_topology(df, strategy_col=args.strategy)

    print("=" * 60)
    print(f"TASK TOPOLOGY ANALYSIS (Strategy: {results['strategy']})")
    print("=" * 60)
    print(f"Total tasks: {results['total_tasks']} | Total models: {results['total_models']}")
    print(f"Universally Solved (All {results['total_models']} models): {results['universally_solved_count']} ({results['universally_solved_percentage']:.1f}%)")
    print(f"  Tasks: {results['universally_solved_tasks']}")
    print(f"Universally Unsolved (0 models): {results['universally_unsolved_count']} ({results['universally_unsolved_percentage']:.1f}%)")
    print(f"  Tasks: {results['universally_unsolved_tasks']}")
    print("\nSolve Count Distribution:")
    for d in results["solve_count_distribution"]:
        bar = "#" * int(d["task_count"])
        print(f"  {d['models_solving']:2d} models: {d['task_count']:3d} tasks ({d['percentage']:5.1f}%) | {bar}")

    print(f"\nInductive Trap Hotspots (Train converged >= 50%, Solved <= 1 model): {results['inductive_traps_count']}")
    for trap in results["inductive_traps"][:10]:
        print(f"  Task {trap['task_id']}: {trap['train_converged_models']}/{results['total_models']} models converged on train, but only {trap['solved_models']} passed test!")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON results to {out_p}")


if __name__ == "__main__":
    main()

