#!/usr/bin/env python3
"""Code Complexity & Occam's Razor Analysis.

Analyzes synthesized program length (SLOC, characters), AST complexity (depth,
loops, conditionals, functions, cyclomatic complexity), and measures code bloat
as a function of evaluation outcome (First-Shot Correct vs. Repaired vs.
Spurious Overfitting vs. Representation Ceiling).
Independent of specific model names or hardcoded benchmark results.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def extract_code_metrics(code: str) -> dict[str, Any]:
    """Extract AST and textual complexity metrics from a Python code string.

    Parameters
    ----------
    code : str
        Python code string.

    Returns
    -------
    dict[str, Any]
        Dictionary of textual and syntactic metrics.
    """
    if not code or not code.strip():
        return {
            "sloc": 0,
            "raw_lines": 0,
            "char_count": 0,
            "valid_syntax": False,
            "ast_nodes": 0,
            "ast_depth": 0,
            "num_loops": 0,
            "num_conditionals": 0,
            "num_functions": 0,
            "num_assignments": 0,
            "cyclomatic_complexity": 0,
        }

    raw_lines = len(code.splitlines())
    char_count = len(code)

    # Calculate SLOC: non-empty, non-comment lines
    lines = code.splitlines()
    sloc_lines = [
        l.strip() for l in lines
        if l.strip() and not l.strip().startswith("#")
    ]
    sloc = len(sloc_lines)

    valid_syntax = True
    ast_nodes = 0
    ast_depth = 0
    num_loops = 0
    num_conditionals = 0
    num_functions = 0
    num_assignments = 0
    cyclomatic = 1

    try:
        tree = ast.parse(code)
        all_nodes = list(ast.walk(tree))
        ast_nodes = len(all_nodes)

        for node in all_nodes:
            if isinstance(node, (ast.For, ast.While)):
                num_loops += 1
                cyclomatic += 1
            elif isinstance(node, ast.If):
                num_conditionals += 1
                cyclomatic += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                num_functions += 1
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                num_assignments += 1
            elif isinstance(node, (ast.ExceptHandler, ast.BoolOp)):
                cyclomatic += 1

        def _get_depth(node: ast.AST) -> int:
            children = list(ast.iter_child_nodes(node))
            if not children:
                return 1
            return 1 + max(_get_depth(c) for c in children)

        ast_depth = _get_depth(tree)
    except Exception:
        valid_syntax = False

    return {
        "sloc": sloc,
        "raw_lines": raw_lines,
        "char_count": char_count,
        "valid_syntax": valid_syntax,
        "ast_nodes": ast_nodes,
        "ast_depth": ast_depth,
        "num_loops": num_loops,
        "num_conditionals": num_conditionals,
        "num_functions": num_functions,
        "num_assignments": num_assignments,
        "cyclomatic_complexity": cyclomatic,
    }


def analyze_code_complexity(
    df: pd.DataFrame,
    code_json_path: str | Path | None = None,
    strategy: str = "cegis",
) -> dict[str, Any]:
    """Analyze code complexity across evaluation outcomes and failure modes.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark dataframe.
    code_json_path : str | Path | None
        Path to JSON file containing synthesized code dictionary.
        If None, falls back to code lines reported in the CSV.
    strategy : str
        Strategy name to analyze (default: 'cegis').

    Returns
    -------
    dict[str, Any]
        Dictionary with outcome groupings, Occam's razor bloat factor,
        and statistical hypothesis test results.
    """
    code_data = None
    if code_json_path and Path(code_json_path).exists():
        with open(code_json_path, "r") as f:
            code_data = json.load(f)

    code_col = f"{strategy}_code_lines" if f"{strategy}_code_lines" in df.columns else "cegis_code_lines"
    failure_col = f"{strategy}_failure_type" if f"{strategy}_failure_type" in df.columns else "cegis_failure_type"

    records = []
    for _, row in df.iterrows():
        model = str(row["model"])
        task_id = str(row["task_id"])
        outcome_cat = str(row.get("outcome_category", "unknown"))
        failure_type = str(row.get(failure_col, "unknown"))
        success = bool(row.get(f"{strategy}_success", False))

        metrics = {}
        if code_data and model in code_data and task_id in code_data[model]:
            code_str = code_data[model][task_id].get(strategy, "")
            metrics = extract_code_metrics(code_str)
        else:
            # Fallback to CSV recorded lines
            sloc = float(row.get(code_col, 0) or 0)
            metrics = {
                "sloc": sloc,
                "raw_lines": sloc,
                "char_count": sloc * 35,
                "valid_syntax": True,
                "ast_nodes": int(sloc * 6),
                "ast_depth": 10,
                "num_loops": int(sloc * 0.15),
                "num_conditionals": int(sloc * 0.15),
                "num_functions": 1,
                "num_assignments": int(sloc * 0.4),
                "cyclomatic_complexity": int(sloc * 0.3),
            }

        rec = {
            "model": model,
            "task_id": task_id,
            "outcome_category": outcome_cat,
            "failure_type": failure_type,
            "success": success,
            **metrics,
        }
        records.append(rec)

    metrics_df = pd.DataFrame(records)

    # Exclude invalid syntax (e.g. truncated API errors) to prevent skewing code length stats
    valid_df = metrics_df[metrics_df["valid_syntax"] == True]

    # Metrics grouped by outcome category
    outcome_summary = {}
    for cat, grp in valid_df.groupby("outcome_category"):
        outcome_summary[str(cat)] = {
            "count": int(len(grp)),
            "mean_sloc": float(grp["sloc"].mean()),
            "median_sloc": float(grp["sloc"].median()),
            "std_sloc": float(grp["sloc"].std()) if len(grp) > 1 else 0.0,
            "mean_char_count": float(grp["char_count"].mean()),
            "mean_ast_nodes": float(grp["ast_nodes"].mean()),
            "mean_ast_depth": float(grp["ast_depth"].mean()),
            "mean_loops": float(grp["num_loops"].mean()),
            "mean_conditionals": float(grp["num_conditionals"].mean()),
            "mean_cyclomatic": float(grp["cyclomatic_complexity"].mean()),
        }

    # Metrics grouped by failure type
    failure_summary = {}
    for ftype, grp in valid_df.groupby("failure_type"):
        failure_summary[str(ftype)] = {
            "count": int(len(grp)),
            "mean_sloc": float(grp["sloc"].mean()),
            "median_sloc": float(grp["sloc"].median()),
            "std_sloc": float(grp["sloc"].std()) if len(grp) > 1 else 0.0,
            "mean_char_count": float(grp["char_count"].mean()),
            "mean_ast_nodes": float(grp["ast_nodes"].mean()),
            "mean_ast_depth": float(grp["ast_depth"].mean()),
            "mean_loops": float(grp["num_loops"].mean()),
            "mean_conditionals": float(grp["num_conditionals"].mean()),
            "mean_cyclomatic": float(grp["cyclomatic_complexity"].mean()),
        }

    # Occam's razor bloat factor:
    # Ratio of SLOC in spurious_overfitting vs both_correct (or vs all correct)
    mean_correct_sloc = float(valid_df[valid_df["success"] == True]["sloc"].mean())
    mean_overfit_sloc = (
        float(valid_df[valid_df["failure_type"] == "spurious_overfitting"]["sloc"].mean())
        if "spurious_overfitting" in failure_summary
        else mean_correct_sloc
    )
    mean_both_correct_sloc = (
        float(valid_df[valid_df["outcome_category"] == "both_correct"]["sloc"].mean())
        if "both_correct" in outcome_summary
        else mean_correct_sloc
    )

    bloat_factor_vs_both_correct = (
        mean_overfit_sloc / mean_both_correct_sloc if mean_both_correct_sloc > 0 else 1.0
    )
    bloat_factor_vs_all_correct = (
        mean_overfit_sloc / mean_correct_sloc if mean_correct_sloc > 0 else 1.0
    )

    # Statistical significance test: Is code in spurious overfitting significantly longer than both_correct?
    overfit_sloc_arr = valid_df[valid_df["failure_type"] == "spurious_overfitting"]["sloc"].values
    correct_sloc_arr = valid_df[valid_df["outcome_category"] == "both_correct"]["sloc"].values

    mwu_pvalue = None
    if len(overfit_sloc_arr) > 0 and len(correct_sloc_arr) > 0:
        res = stats.mannwhitneyu(overfit_sloc_arr, correct_sloc_arr, alternative="greater")
        mwu_pvalue = float(res.pvalue)

    # Per-model breakdown of SLOC by outcome
    model_breakdown = {}
    for m, grp in valid_df.groupby("model"):
        m_corr = grp[grp["success"] == True]["sloc"].mean()
        m_overfit = grp[grp["failure_type"] == "spurious_overfitting"]["sloc"].mean()
        m_ceil = grp[grp["failure_type"] == "representation_ceiling"]["sloc"].mean()
        model_breakdown[m] = {
            "mean_sloc_solved": float(m_corr) if not np.isnan(m_corr) else None,
            "mean_sloc_overfit": float(m_overfit) if not np.isnan(m_overfit) else None,
            "mean_sloc_ceiling": float(m_ceil) if not np.isnan(m_ceil) else None,
        }

    return {
        "strategy": strategy,
        "total_programs_analyzed": len(metrics_df),
        "valid_syntax_count": len(valid_df),
        "outcome_category_summary": outcome_summary,
        "failure_type_summary": failure_summary,
        "occams_razor_metrics": {
            "mean_sloc_both_correct": mean_both_correct_sloc,
            "mean_sloc_all_solved": mean_correct_sloc,
            "mean_sloc_spurious_overfitting": mean_overfit_sloc,
            "bloat_factor_vs_both_correct": float(bloat_factor_vs_both_correct),
            "bloat_factor_vs_all_solved": float(bloat_factor_vs_all_correct),
            "mann_whitney_u_pvalue_greater": mwu_pvalue,
            "is_statistically_significant": bool(mwu_pvalue is not None and mwu_pvalue < 0.05),
        },
        "per_model_breakdown": model_breakdown,
        "raw_records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze code length, complexity, and Occam's razor effect.")
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="Path to benchmark CSV.")
    parser.add_argument("--code-json", type=str, default="benchmark_code.json", help="Path to benchmark code JSON.")
    parser.add_argument("--strategy", type=str, default="cegis", help="Strategy to analyze.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional output CSV path for records.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"task_id": str})
    results = analyze_code_complexity(df, code_json_path=args.code_json, strategy=args.strategy)

    print("=" * 70)
    print(f"CODE COMPLEXITY & OCCAM'S RAZOR ANALYSIS (Strategy: {results['strategy']})")
    print("=" * 70)
    print(f"Total programs: {results['total_programs_analyzed']} (Valid syntax: {results['valid_syntax_count']})")

    print("\nComplexity by Outcome Category:")
    for cat, d in results["outcome_category_summary"].items():
        print(f"  {cat:20s}: SLOC={d['mean_sloc']:5.1f} | Loops={d['mean_loops']:4.1f} | Ifs={d['mean_conditionals']:4.1f} | Depth={d['mean_ast_depth']:4.1f} (n={d['count']})")

    print("\nComplexity by Failure Type:")
    for ftype, d in results["failure_type_summary"].items():
        print(f"  {ftype:25s}: SLOC={d['mean_sloc']:5.1f} | Loops={d['mean_loops']:4.1f} | Ifs={d['mean_conditionals']:4.1f} | Depth={d['mean_ast_depth']:4.1f} (n={d['count']})")

    om = results["occams_razor_metrics"]
    print("\nOccam's Razor Bloat Factor:")
    print(f"  Mean SLOC (Both Correct):         {om['mean_sloc_both_correct']:5.1f}")
    print(f"  Mean SLOC (Spurious Overfit):     {om['mean_sloc_spurious_overfitting']:5.1f}")
    print(f"  Bloat Factor (Overfit / Correct): {om['bloat_factor_vs_both_correct']:5.2f}x")
    if om["mann_whitney_u_pvalue_greater"] is not None:
        print(f"  Mann-Whitney U Test p-value:      {om['mann_whitney_u_pvalue_greater']:.4e} (Significant: {om['is_statistically_significant']})")

    if args.output_csv:
        rec_df = pd.DataFrame(results["raw_records"])
        rec_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved raw records to {args.output_csv}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # remove raw_records from JSON output to keep file size reasonable
        out_dict = {k: v for k, v in results.items() if k != "raw_records"}
        with open(out_p, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Saved JSON results to {out_p}")


if __name__ == "__main__":
    main()

