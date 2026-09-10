#!/usr/bin/env python3
"""Iteration Dynamics & Horizon Economics Analysis.

Measures the convergence distribution across CEGIS repair iterations,
quantifies marginal gains per additional counterexample, detects diminishing returns,
and computes the optimal iteration horizon cutoff.
Independent of specific model names or hardcoded benchmark results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def analyze_iteration_dynamics(
    df: pd.DataFrame,
    strategy: str = "cegis",
) -> dict[str, Any]:
    """Analyze iteration convergence distributions and diminishing returns.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark dataframe.
    strategy : str
        Strategy name (default: 'cegis').

    Returns
    -------
    dict[str, Any]
        Dictionary of iteration dynamics, marginal gains, and cutoff recommendations.
    """
    success_col = f"{strategy}_success"
    iter_col = f"{strategy}_iterations_used"
    lat_col = f"{strategy}_latency_sec"

    if success_col not in df.columns or iter_col not in df.columns:
        raise ValueError(f"Required columns '{success_col}' or '{iter_col}' not found in dataframe.")

    solved_df = df[df[success_col] == True]
    all_models = sorted(df["model"].unique().tolist())
    max_iter = int(df[iter_col].max())

    # Overall solved distribution by iteration
    iter_counts = solved_df[iter_col].value_counts().reindex(range(1, max_iter + 1), fill_value=0)
    total_solved = len(solved_df)
    baseline_solved = int(iter_counts.get(1, 0))
    total_recovered = total_solved - baseline_solved

    iteration_table = []
    cum_recovered = 0

    for i in range(1, max_iter + 1):
        count_i = int(iter_counts[i])
        pct_of_solved = float(count_i / total_solved * 100) if total_solved > 0 else 0.0

        if i == 1:
            rec_i = 0
            pct_of_rec = 0.0
            cum_pct_rec = 0.0
        else:
            rec_i = count_i
            cum_recovered += rec_i
            pct_of_rec = float(rec_i / total_recovered * 100) if total_recovered > 0 else 0.0
            cum_pct_rec = float(cum_recovered / total_recovered * 100) if total_recovered > 0 else 0.0

        # Mean latency of tasks solved at iteration i
        tasks_at_i = solved_df[solved_df[iter_col] == i]
        mean_lat_i = float(tasks_at_i[lat_col].mean()) if (lat_col in tasks_at_i.columns and len(tasks_at_i) > 0) else None

        iteration_table.append({
            "iteration": i,
            "tasks_solved": count_i,
            "pct_of_all_solved": pct_of_solved,
            "recovered_tasks": rec_i,
            "pct_of_recoveries": pct_of_rec,
            "cumulative_recovered": cum_recovered,
            "cumulative_pct_of_recoveries": cum_pct_rec,
            "mean_latency_sec": mean_lat_i,
        })

    # Per-model breakdown
    model_crosstab = (
        pd.crosstab(solved_df["model"], solved_df[iter_col])
        .reindex(index=all_models, columns=range(1, max_iter + 1), fill_value=0)
    )

    per_model = {}
    for m in all_models:
        row = model_crosstab.loc[m]
        m_solved = int(row.sum())
        m_rec = int(row.iloc[1:].sum())
        per_model[m] = {
            "total_solved": m_solved,
            "recovered_count": m_rec,
            "iteration_distribution": {int(k): int(v) for k, v in row.items()},
        }

    # Optimal cutoff recommendation: first iteration reaching >= 80% cumulative recoveries
    recommended_cutoff = max_iter
    for entry in iteration_table:
        if entry["iteration"] > 1 and entry["cumulative_pct_of_recoveries"] >= 80.0:
            recommended_cutoff = entry["iteration"]
            break

    cutoff_recovered_share = (
        iteration_table[recommended_cutoff - 1]["cumulative_pct_of_recoveries"]
        if recommended_cutoff <= max_iter else 100.0
    )

    return {
        "strategy": strategy,
        "total_evaluations": len(df),
        "total_solved": total_solved,
        "baseline_solved_iter_1": baseline_solved,
        "total_recovered_iter_ge_2": total_recovered,
        "max_iterations_allowed": max_iter,
        "recommended_iteration_cap": recommended_cutoff,
        "recommended_cap_recovery_retention_pct": cutoff_recovered_share,
        "iteration_table": iteration_table,
        "per_model_breakdown": per_model,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze CEGIS iteration dynamics and diminishing returns.")
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="Path to benchmark CSV.")
    parser.add_argument("--strategy", type=str, default="cegis", help="Strategy prefix to analyze.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"task_id": str})
    results = analyze_iteration_dynamics(df, strategy=args.strategy)

    print("=" * 70)
    print(f"ITERATION DYNAMICS & DIMINISHING RETURNS (Strategy: {results['strategy']})")
    print("=" * 70)
    print(f"Total Solved: {results['total_solved']} | Baseline (Iter 1): {results['baseline_solved_iter_1']} | Recovered: {results['total_recovered_iter_ge_2']}")
    print(f"Recommended Iteration Cap: {results['recommended_iteration_cap']} iterations (Retains {results['recommended_cap_recovery_retention_pct']:.1f}% of recoveries)\n")

    print(f"{'Iter':4s} | {'Solved':6s} | {'% All':6s} | {'Recovered':9s} | {'% Recov':7s} | {'Cumul % Recov':13s} | {'Mean Latency':12s}")
    print("-" * 75)
    for r in results["iteration_table"]:
        lat_str = f"{r['mean_latency_sec']:8.1f}s" if r["mean_latency_sec"] is not None else "     N/A"
        print(f"{r['iteration']:4d} | {r['tasks_solved']:6d} | {r['pct_of_all_solved']:5.1f}% | {r['recovered_tasks']:9d} | {r['pct_of_recoveries']:6.1f}% | {r['cumulative_pct_of_recoveries']:12.1f}% | {lat_str}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON results to {out_p}")


if __name__ == "__main__":
    main()

