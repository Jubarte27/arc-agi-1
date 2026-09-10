#!/usr/bin/env python3
"""Guttman Subsumption Hierarchy & Scalability Analysis.

Computes the empirical task subsumption matrix P(M_A | M_B) across models,
consecutive subsumption chains, and formal Guttman scalability metrics
(Coefficient of Reproducibility CR, Minimum Marginal Reproducibility MMR,
Coefficient of Scalability CS).
Independent of specific model names or hardcoded benchmark results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def analyze_guttman_hierarchy(
    df: pd.DataFrame,
    strategy_col: str = "cegis_success",
) -> dict[str, Any]:
    """Calculate the Guttman Subsumption Hierarchy and Scalability metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'task_id', 'model', and strategy success column.
    strategy_col : str
        Column name for success flag (default: 'cegis_success').

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - ordered_models: list of models ranked by accuracy descending
        - model_accuracies: dictionary mapping model to accuracy
        - pairwise_subsumption: nested dict of P(M_row | M_col)
        - consecutive_subsumption: list of P(M_{i} | M_{i+1})
        - guttman_metrics: CR, MMR, CS, error count, total evaluations
    """
    # Create pivot table: tasks x models
    pivot = df.pivot(index="task_id", columns="model", values=strategy_col).astype(int)

    # Rank models by accuracy descending
    model_acc = pivot.mean(axis=0).sort_values(ascending=False)
    ordered_models = model_acc.index.tolist()
    pivot = pivot[ordered_models]

    n_tasks, n_models = pivot.shape

    # Pairwise Subsumption Matrix: S[i, j] = P(M_i solves t | M_j solves t)
    subsumption_matrix: dict[str, dict[str, float]] = {}
    for i, m_a in enumerate(ordered_models):
        subsumption_matrix[m_a] = {}
        for j, m_b in enumerate(ordered_models):
            b_solved = pivot[m_b] == 1
            if b_solved.sum() > 0:
                prob = float((pivot.loc[b_solved, m_a] == 1).mean())
            else:
                prob = 1.0
            subsumption_matrix[m_a][m_b] = prob

    # Consecutive Subsumption Chain: P(M_{i} | M_{i+1})
    consecutive_chain = []
    for i in range(len(ordered_models) - 1):
        m_high = ordered_models[i]
        m_low = ordered_models[i + 1]
        prob = subsumption_matrix[m_high][m_low]
        low_count = int(pivot[m_low].sum())
        consecutive_chain.append({
            "stronger_model": m_high,
            "weaker_model": m_low,
            "subsumption_probability": prob,
            "subsumption_percentage": prob * 100,
            "weaker_model_solved_count": low_count,
        })

    # Formal Guttman Scale metrics (Goodenough-Edwards error technique)
    X = pivot.values
    ideal_X = np.zeros_like(X)
    for t in range(n_tasks):
        c_t = int(X[t, :].sum())
        ideal_X[t, :c_t] = 1

    # Total scale errors: discrepancies from ideal triangular pattern
    raw_discrepancies = int(np.abs(X - ideal_X).sum())
    scale_errors = raw_discrepancies / 2.0

    total_evals = n_tasks * n_models
    cr = 1.0 - (scale_errors / total_evals)

    # Minimum Marginal Reproducibility (MMR)
    p_j = pivot.mean(axis=0).values
    mmr = float(np.mean([max(p, 1.0 - p) for p in p_j]))

    # Coefficient of Scalability (CS)
    cs = float((cr - mmr) / (1.0 - mmr)) if mmr < 1.0 else 1.0

    # Interpret scalability
    is_valid_scale = bool(cr >= 0.90 and cs >= 0.60)

    return {
        "strategy": strategy_col,
        "n_models": n_models,
        "n_tasks": n_tasks,
        "ordered_models": ordered_models,
        "model_accuracies": {m: float(acc) for m, acc in model_acc.items()},
        "pairwise_subsumption": subsumption_matrix,
        "consecutive_subsumption": consecutive_chain,
        "guttman_metrics": {
            "scale_errors": scale_errors,
            "total_evaluations": total_evals,
            "coefficient_of_reproducibility_cr": float(cr),
            "minimum_marginal_reproducibility_mmr": float(mmr),
            "coefficient_of_scalability_cs": float(cs),
            "is_valid_guttman_scale": is_valid_scale,
            "cr_threshold": 0.90,
            "cs_threshold": 0.60,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Guttman Subsumption Hierarchy for ARC models.")
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="Path to benchmark CSV.")
    parser.add_argument("--strategy", type=str, default="cegis_success", help="Strategy success column.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional output CSV for subsumption matrix.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"task_id": str})
    results = analyze_guttman_hierarchy(df, strategy_col=args.strategy)

    print("=" * 70)
    print(f"GUTTMAN SUBSUMPTION HIERARCHY (Strategy: {results['strategy']})")
    print("=" * 70)
    print("Model Ranking by Accuracy:")
    for rank, (m, acc) in enumerate(results["model_accuracies"].items(), 1):
        print(f"  {rank:2d}. {m:35s}: {acc * 100:5.1f}%")

    print("\nConsecutive Subsumption Chain P(M_{i} | M_{i+1}):")
    for link in results["consecutive_subsumption"]:
        print(
            f"  {link['stronger_model'][:25]:25s} subsumes {link['weaker_model'][:25]:25s}: "
            f"{link['subsumption_percentage']:5.1f}% "
            f"(over {link['weaker_model_solved_count']} tasks)"
        )

    gm = results["guttman_metrics"]
    print("\nFormal Guttman Scalogram Metrics:")
    print(f"  Scale Errors: {gm['scale_errors']:.1f} / {gm['total_evaluations']} evaluations")
    print(f"  Coefficient of Reproducibility (CR): {gm['coefficient_of_reproducibility_cr']:.4f} (Threshold >= 0.90)")
    print(f"  Minimum Marginal Reproducibility (MMR): {gm['minimum_marginal_reproducibility_mmr']:.4f}")
    print(f"  Coefficient of Scalability (CS): {gm['coefficient_of_scalability_cs']:.4f} (Threshold >= 0.60)")
    print(f"  Valid 1-Dimensional Scale: {'YES' if gm['is_valid_guttman_scale'] else 'NO'}")

    if args.output_csv:
        sub_df = pd.DataFrame(results["pairwise_subsumption"])
        sub_df.to_csv(args.output_csv)
        print(f"\nSaved Subsumption Matrix CSV to {args.output_csv}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON results to {out_p}")


if __name__ == "__main__":
    main()

