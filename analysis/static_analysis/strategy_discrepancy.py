#!/usr/bin/env python3
"""Strategy Discrepancy & Complementarity Analysis (e.g. CEGIS vs. AntiCheat).

Evaluates the agreement, divergence, and complementary coverage between
two evaluation strategies, computing Cohen's Kappa, Jaccard similarity,
and the theoretical Oracle Ensemble performance upper bound.
Independent of specific model names or hardcoded benchmark results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def analyze_strategy_discrepancy(
    df: pd.DataFrame,
    strat1_col: str = "cegis_success",
    strat2_col: str = "anticheat_success",
    strat1_name: str = "CEGIS",
    strat2_name: str = "AntiCheat",
) -> dict[str, Any]:
    """Analyze discrepancies and complementarity between two strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark dataframe.
    strat1_col : str
        First strategy success column (default: 'cegis_success').
    strat2_col : str
        Second strategy success column (default: 'anticheat_success').
    strat1_name : str
        Display name for strategy 1.
    strat2_name : str
        Display name for strategy 2.

    Returns
    -------
    dict[str, Any]
        Contingency statistics, Jaccard similarity, Cohen's Kappa,
        and Oracle ensemble performance.
    """
    if strat1_col not in df.columns or strat2_col not in df.columns:
        raise ValueError(f"Columns '{strat1_col}' or '{strat2_col}' missing from dataframe.")

    s1 = df[strat1_col].astype(bool).values
    s2 = df[strat2_col].astype(bool).values
    n = len(df)

    both_succ = int(((s1 == True) & (s2 == True)).sum())
    s1_only = int(((s1 == True) & (s2 == False)).sum())
    s2_only = int(((s1 == False) & (s2 == True)).sum())
    both_fail = int(((s1 == False) & (s2 == False)).sum())

    total_agreements = both_succ + both_fail
    total_discrepancies = s1_only + s2_only

    agreement_rate = total_agreements / n
    discrepancy_rate = total_discrepancies / n

    # Jaccard similarity of solved sets
    union_solved = both_succ + s1_only + s2_only
    jaccard_similarity = both_succ / union_solved if union_solved > 0 else 1.0

    # Cohen's Kappa
    p_observed = agreement_rate
    p_e1 = (s1.mean() * s2.mean()) + ((1.0 - s1.mean()) * (1.0 - s2.mean()))
    kappa = (p_observed - p_e1) / (1.0 - p_e1) if p_e1 < 1.0 else 1.0

    # Oracle ensemble (union of successes)
    oracle_accuracy = union_solved / n
    s1_accuracy = s1.mean()
    s2_accuracy = s2.mean()
    oracle_lift_over_s1 = oracle_accuracy - s1_accuracy
    oracle_lift_over_s2 = oracle_accuracy - s2_accuracy

    # Per-model breakdown
    per_model = {}
    model_col = "friendly_name" if "friendly_name" in df.columns else "model"
    for m, grp in df.groupby(model_col):
        m_s1 = grp[strat1_col].astype(bool).values
        m_s2 = grp[strat2_col].astype(bool).values
        m_n = len(grp)

        m_both = int(((m_s1 == True) & (m_s2 == True)).sum())
        m_s1_only = int(((m_s1 == True) & (m_s2 == False)).sum())
        m_s2_only = int(((m_s1 == False) & (m_s2 == True)).sum())
        m_union = m_both + m_s1_only + m_s2_only

        per_model[str(m)] = {
            f"{strat1_name}_solved": int(m_s1.sum()),
            f"{strat2_name}_solved": int(m_s2.sum()),
            "both_solved": m_both,
            f"{strat1_name}_only_solved": m_s1_only,
            f"{strat2_name}_only_solved": m_s2_only,
            "oracle_ensemble_solved": m_union,
            "oracle_lift": float((m_union / m_n) - m_s1.mean()),
            "jaccard_similarity": float(m_both / m_union) if m_union > 0 else 1.0,
        }

    return {
        "strategy_1": strat1_name,
        "strategy_2": strat2_name,
        "total_evaluations": n,
        "contingency_table": {
            "both_succeeded": both_succ,
            f"{strat1_name}_only_succeeded": s1_only,
            f"{strat2_name}_only_succeeded": s2_only,
            "both_failed": both_fail,
        },
        "agreement_rate": float(agreement_rate),
        "discrepancy_rate": float(discrepancy_rate),
        "jaccard_similarity": float(jaccard_similarity),
        "cohens_kappa": float(kappa),
        "oracle_ensemble_metrics": {
            f"{strat1_name}_accuracy": float(s1_accuracy),
            f"{strat2_name}_accuracy": float(s2_accuracy),
            "oracle_ensemble_accuracy": float(oracle_accuracy),
            f"oracle_lift_over_{strat1_name}": float(oracle_lift_over_s1),
            f"oracle_lift_over_{strat2_name}": float(oracle_lift_over_s2),
        },
        "per_model_breakdown": per_model,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze discrepancy and complementarity between two strategies.")
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="Path to benchmark CSV.")
    parser.add_argument("--strat1", type=str, default="cegis_success", help="Strategy 1 column.")
    parser.add_argument("--strat2", type=str, default="anticheat_success", help="Strategy 2 column.")
    parser.add_argument("--name1", type=str, default="CEGIS", help="Strategy 1 label.")
    parser.add_argument("--name2", type=str, default="AntiCheat", help="Strategy 2 label.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"task_id": str})
    results = analyze_strategy_discrepancy(
        df,
        strat1_col=args.strat1,
        strat2_col=args.strat2,
        strat1_name=args.name1,
        strat2_name=args.name2,
    )

    print("=" * 70)
    print(f"STRATEGY COMPLEMENTARITY ANALYSIS ({results['strategy_1']} vs. {results['strategy_2']})")
    print("=" * 70)
    ct = results["contingency_table"]
    s1_name = results["strategy_1"]
    s2_name = results["strategy_2"]
    s1_only_key = f"{s1_name}_only_succeeded"
    s2_only_key = f"{s2_name}_only_succeeded"
    tot = results["total_evaluations"]

    print("Contingency Breakdown:")
    print(f"  Both Succeeded:                {ct['both_succeeded']:4d} ({ct['both_succeeded']/tot*100:5.1f}%)")
    print(f"  {s1_name} Only Succeeded:         {ct[s1_only_key]:4d} ({ct[s1_only_key]/tot*100:5.1f}%)")
    print(f"  {s2_name} Only Succeeded:     {ct[s2_only_key]:4d} ({ct[s2_only_key]/tot*100:5.1f}%)")
    print(f"  Both Failed:                   {ct['both_failed']:4d} ({ct['both_failed']/tot*100:5.1f}%)")

    print("\nAgreement & Complementarity Metrics:")
    print(f"  Agreement Rate:                {results['agreement_rate']*100:5.1f}%")
    print(f"  Discrepancy Rate:              {results['discrepancy_rate']*100:5.1f}%")
    print(f"  Jaccard Similarity:            {results['jaccard_similarity']:.4f}")
    print(f"  Cohen's Kappa:                 {results['cohens_kappa']:.4f}")

    om = results["oracle_ensemble_metrics"]
    s1_acc_key = f"{s1_name}_accuracy"
    s2_acc_key = f"{s2_name}_accuracy"
    s1_lift_key = f"oracle_lift_over_{s1_name}"

    print("\nOracle Ensemble Headroom:")
    print(f"  {s1_name} Standalone Accuracy:      {om[s1_acc_key]*100:5.1f}%")
    print(f"  {s2_name} Standalone Accuracy:  {om[s2_acc_key]*100:5.1f}%")
    print(f"  Oracle Ensemble Accuracy:      {om['oracle_ensemble_accuracy']*100:5.1f}%")
    print(f"  Ensemble Lift over {s1_name}:       +{om[s1_lift_key]*100:4.1f}%")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON results to {out_p}")


if __name__ == "__main__":
    main()
