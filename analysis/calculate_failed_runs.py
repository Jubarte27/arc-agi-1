#!/usr/bin/env python3
"""Calculate failed runs and error breakdown per model on ARC-AGI benchmark.

Computes:
1. Task-level failure counts across strategies (Baseline, CEGIS, AntiCheat).
2. "Did Not Generate Python Code" error column: counts instances where models
   output empty responses, provider refusals, or conversational English without
   valid Python code.
3. Code execution crashes: actual runtime / syntax exceptions on attempted code.
4. Error taxonomy breakdown (Spurious Overfitting vs. Representation Ceiling).
5. Total internal attempt / iteration failures within the CEGIS loop.

Usage:
  python calculate_failed_runs.py
  python calculate_failed_runs.py --csv benchmark.csv --code-json benchmark_code.json
  python calculate_failed_runs.py --csv benchmark.csv --output-csv failed_runs_summary.csv
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


def is_did_not_generate_python(code_str: str | None, err_msg: str | None) -> bool:
    """Determine whether a run failed because it did not generate Python code.

    Triggers when:
    - Code string is empty or None.
    - Error message explicitly states 'Empty code string provided.'.
    - Model returned provider refusal (e.g. "I couldn't generate a complete response...").
    - Model output conversational English without defining a Python function (`def `).
    - Code failed to parse at line 1 due to plain natural language text.
    """
    err_str = str(err_msg) if pd.notna(err_msg) else ""

    if "Empty code string provided" in err_str:
        return True

    if not code_str or not code_str.strip():
        # Empty code with an error or failure
        if err_str:
            return True
        return False

    # Check for known refusal messages
    if "couldn't generate" in code_str or "cannot generate" in code_str:
        return True

    # Check if a Python function is defined
    if "def " not in code_str:
        return True

    # Try AST parse to verify if it contains a function definition
    try:
        tree = ast.parse(code_str)
        has_func = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
        if not has_func:
            return True
    except Exception:
        # Check if first non-empty line starts with 'def ' or valid Python construct
        non_empty = [l.strip() for l in code_str.splitlines() if l.strip()]
        if not non_empty or not any(l.startswith("def ") for l in non_empty):
            return True

    return False


def compute_failed_runs(
    df: pd.DataFrame,
    code_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute detailed failure statistics for each model in the benchmark.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark dataframe.
    code_data : dict[str, Any] | None
        Optional dictionary of synthesized code loaded from benchmark_code.json.

    Returns
    -------
    dict[str, Any]
        Structured dictionary with model-level and pooled failure statistics.
    """
    model_col = "friendly_name" if "friendly_name" in df.columns else "model"
    total_tasks = df["task_id"].nunique()
    models = sorted(df[model_col].unique().tolist())

    model_stats = []
    pooled = {
        "total_tasks": len(df),
        "baseline_failed": int((df["baseline_success"] == False).sum()),
        "baseline_did_not_generate_python": 0,
        "cegis_failed": int((df["cegis_success"] == False).sum()),
        "cegis_did_not_generate_python": 0,
        "anticheat_failed": int((df["anticheat_success"] == False).sum()) if "anticheat_success" in df.columns else 0,
        "anticheat_did_not_generate_python": 0,
        "spurious_overfitting": int((df["cegis_failure_type"] == "spurious_overfitting").sum()) if "cegis_failure_type" in df.columns else 0,
        "representation_ceiling": int((df["cegis_failure_type"] == "representation_ceiling").sum()) if "cegis_failure_type" in df.columns else 0,
        "baseline_crashes": 0,
        "cegis_crashes": 0,
        "anticheat_crashes": 0,
        "cegis_api_errors": int(df["cegis_api_error"].sum()) if "cegis_api_error" in df.columns else 0,
        "cegis_sub_attempt_failures": 0,
    }

    for m in models:
        grp = df[df[model_col] == m]
        n = len(grp)

        base_fails = int((grp["baseline_success"] == False).sum())
        cegis_fails = int((grp["cegis_success"] == False).sum())
        anti_fails = int((grp["anticheat_success"] == False).sum()) if "anticheat_success" in grp.columns else 0

        # Failure types in CEGIS
        spur_overfit = (
            int((grp["cegis_failure_type"] == "spurious_overfitting").sum())
            if "cegis_failure_type" in grp.columns
            else int(((grp["cegis_converged_train"] == True) & (grp["cegis_test_success"] == False)).sum())
        )
        rep_ceiling = (
            int((grp["cegis_failure_type"] == "representation_ceiling").sum())
            if "cegis_failure_type" in grp.columns
            else int((grp["cegis_converged_train"] == False).sum())
        )

        # API errors
        base_api_err = int(grp["baseline_api_error"].sum()) if "baseline_api_error" in grp.columns else 0
        cegis_api_err = int(grp["cegis_api_error"].sum()) if "cegis_api_error" in grp.columns else 0
        anti_api_err = int(grp["anticheat_api_error"].sum()) if "anticheat_api_error" in grp.columns else 0

        # Strategy-level categorization: Did Not Generate Python vs. Crashes vs. Pure Logic
        base_no_code, base_crashes = 0, 0
        cegis_no_code, cegis_crashes = 0, 0
        anti_no_code, anti_crashes = 0, 0

        for _, r in grp.iterrows():
            tid = str(r["task_id"])
            m_name = str(r[model_col])
            raw_m = str(r.get("raw_model", m_name))

            def _get_code(strat_key: str) -> str:
                if not code_data:
                    return ""
                if m_name in code_data and tid in code_data[m_name]:
                    return code_data[m_name][tid].get(strat_key, "")
                if raw_m in code_data and tid in code_data[raw_m]:
                    return code_data[raw_m][tid].get(strat_key, "")
                return ""

            # Baseline
            if r["baseline_success"] == False:
                c_base = _get_code("baseline")
                err_base = r.get("baseline_error")
                if is_did_not_generate_python(c_base, err_base):
                    base_no_code += 1
                elif pd.notna(err_base) and str(err_base).strip():
                    base_crashes += 1

            # CEGIS
            if r["cegis_success"] == False:
                c_cegis = _get_code("cegis")
                err_cegis = r.get("cegis_error")
                if is_did_not_generate_python(c_cegis, err_cegis):
                    cegis_no_code += 1
                elif pd.notna(err_cegis) and str(err_cegis).strip():
                    cegis_crashes += 1

            # AntiCheat
            if r.get("anticheat_success") == False:
                c_anti = _get_code("cegis_anticheat")
                err_anti = r.get("anticheat_error")
                if is_did_not_generate_python(c_anti, err_anti):
                    anti_no_code += 1
                elif pd.notna(err_anti) and str(err_anti).strip():
                    anti_crashes += 1

        # Pure logic failures: failures where valid python code ran without exception
        base_logic_fails = base_fails - base_no_code - base_crashes - base_api_err
        cegis_logic_fails = cegis_fails - cegis_no_code - cegis_crashes - cegis_api_err
        anti_logic_fails = anti_fails - anti_no_code - anti_crashes - anti_api_err

        # Total internal code generation failures in the CEGIS loop:
        cegis_sub_attempt_fails = 0
        if "cegis_iterations_used" in grp.columns:
            for _, r in grp.iterrows():
                iters = int(r.get("cegis_iterations_used", 1) or 1)
                succ = bool(r.get("cegis_success", False))
                cegis_sub_attempt_fails += iters if not succ else (iters - 1)

        pooled["baseline_did_not_generate_python"] += base_no_code
        pooled["cegis_did_not_generate_python"] += cegis_no_code
        pooled["anticheat_did_not_generate_python"] += anti_no_code
        pooled["baseline_crashes"] += base_crashes
        pooled["cegis_crashes"] += cegis_crashes
        pooled["anticheat_crashes"] += anti_crashes
        pooled["cegis_sub_attempt_failures"] += cegis_sub_attempt_fails

        model_stats.append({
            "model": m,
            "total_evaluated_tasks": n,
            "baseline_failed_tasks": base_fails,
            "baseline_failure_rate_pct": float(base_fails / n * 100),
            "baseline_did_not_generate_python": base_no_code,
            "baseline_code_crashes": base_crashes,
            "baseline_api_errors": base_api_err,
            "baseline_pure_logic_fails": base_logic_fails,
            "cegis_failed_tasks": cegis_fails,
            "cegis_failure_rate_pct": float(cegis_fails / n * 100),
            "cegis_did_not_generate_python": cegis_no_code,
            "cegis_spurious_overfitting": spur_overfit,
            "cegis_representation_ceiling": rep_ceiling,
            "cegis_code_crashes": cegis_crashes,
            "cegis_api_errors": cegis_api_err,
            "cegis_pure_logic_fails": cegis_logic_fails,
            "cegis_sub_attempt_failures": cegis_sub_attempt_fails,
            "anticheat_failed_tasks": anti_fails,
            "anticheat_failure_rate_pct": float(anti_fails / n * 100),
            "anticheat_did_not_generate_python": anti_no_code,
            "anticheat_code_crashes": anti_crashes,
            "anticheat_api_errors": anti_api_err,
            "anticheat_pure_logic_fails": anti_logic_fails,
        })

    # Sort models by CEGIS failed tasks ascending (best model first)
    model_stats.sort(key=lambda x: x["cegis_failed_tasks"])

    return {
        "models_count": len(models),
        "total_tasks_per_model": total_tasks,
        "pooled": pooled,
        "per_model": model_stats,
    }


def print_terminal_report(results: dict[str, Any]) -> None:
    """Print clean formatted tables to terminal."""
    print("=" * 108)
    print("FAILED RUNS BREAKDOWN PER MODEL (ARC-AGI-1 BENCHMARK)")
    print("=" * 108)

    # 1. Main failure summary across strategies
    print(f"\n1. TASK FAILURE COUNTS BY STRATEGY (Total Tasks = {results['total_tasks_per_model']}):")
    print(f"{'Model':26s} | {'Baseline Fails':14s} | {'CEGIS Fails':12s} | {'AntiCheat Fails':15s} | {'CEGIS Loop Fails':16s}")
    print("-" * 108)

    for m in results["per_model"]:
        base_str = f"{m['baseline_failed_tasks']:2d} ({m['baseline_failure_rate_pct']:5.1f}%)"
        cegis_str = f"{m['cegis_failed_tasks']:2d} ({m['cegis_failure_rate_pct']:5.1f}%)"
        anti_str = f"{m['anticheat_failed_tasks']:2d} ({m['anticheat_failure_rate_pct']:5.1f}%)"
        print(
            f"{m['model'][:26]:26s} | {base_str:14s} | {cegis_str:12s} | {anti_str:15s} | {m['cegis_sub_attempt_failures']:4d} attempts"
        )

    pl = results["pooled"]
    print("-" * 108)
    tot_evals = pl["total_tasks"]
    p_base_str = f"{pl['baseline_failed']:3d} ({pl['baseline_failed']/tot_evals*100:5.1f}%)"
    p_cegis_str = f"{pl['cegis_failed']:3d} ({pl['cegis_failed']/tot_evals*100:5.1f}%)"
    p_anti_str = f"{pl['anticheat_failed']:3d} ({pl['anticheat_failed']/tot_evals*100:5.1f}%)"
    print(
        f"{'TOTAL (POOLED)':26s} | {p_base_str:14s} | {p_cegis_str:12s} | {p_anti_str:15s} | {pl['cegis_sub_attempt_failures']:4d} attempts"
    )

    # 2. CEGIS Error Mechanism including "Did Not Generate Python Code"
    print("\n2. CEGIS ERROR BREAKDOWN (Including 'Did Not Generate Python Code'):")
    print(f"{'Model':26s} | {'Total Fails':11s} | {'Did Not Gen Code':16s} | {'Code Crashes':12s} | {'API Errs':8s} | {'Pure Logic':10s}")
    print("-" * 98)
    for m in results["per_model"]:
        print(
            f"{m['model'][:26]:26s} | {m['cegis_failed_tasks']:11d} | "
            f"{m['cegis_did_not_generate_python']:16d} | "
            f"{m['cegis_code_crashes']:12d} | "
            f"{m['cegis_api_errors']:8d} | "
            f"{m['cegis_pure_logic_fails']:10d}"
        )
    print("-" * 98)
    print(
        f"{'TOTAL (POOLED)':26s} | {pl['cegis_failed']:11d} | "
        f"{pl['cegis_did_not_generate_python']:16d} | "
        f"{pl['cegis_crashes']:12d} | "
        f"{pl['cegis_api_errors']:8d} | "
        f"{pl['cegis_failed'] - pl['cegis_did_not_generate_python'] - pl['cegis_crashes'] - pl['cegis_api_errors']:10d}"
    )

    # 3. CEGIS Taxonomy (Overfit vs Ceiling)
    print("\n3. CEGIS TAXONOMY BREAKDOWN (Overfitting vs. Representation Ceiling):")
    print(f"{'Model':26s} | {'Spurious Overfit':16s} | {'Representation Ceiling':22s}")
    print("-" * 72)
    for m in results["per_model"]:
        print(
            f"{m['model'][:26]:26s} | {m['cegis_spurious_overfitting']:16d} | {m['cegis_representation_ceiling']:22d}"
        )
    print("=" * 108)


def generate_markdown_summary(results: dict[str, Any], output_path: Path) -> None:
    """Generate Markdown report."""
    lines = []
    lines.append("# Failed Runs Summary per Model\n")
    lines.append("| Model | Baseline Fails | CEGIS Fails | AntiCheat Fails | Did Not Generate Python Code | Code Crashes | API Errors | Pure Logic Fails | Spurious Overfit | Representation Ceiling | Total Loop Attempts Failed |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in results["per_model"]:
        lines.append(
            f"| **{m['model']}** | {m['baseline_failed_tasks']} ({m['baseline_failure_rate_pct']:.1f}%) | "
            f"{m['cegis_failed_tasks']} ({m['cegis_failure_rate_pct']:.1f}%) | "
            f"{m['anticheat_failed_tasks']} ({m['anticheat_failure_rate_pct']:.1f}%) | "
            f"**{m['cegis_did_not_generate_python']}** | "
            f"{m['cegis_code_crashes']} | {m['cegis_api_errors']} | {m['cegis_pure_logic_fails']} | "
            f"{m['cegis_spurious_overfitting']} | {m['cegis_representation_ceiling']} | {m['cegis_sub_attempt_failures']} |"
        )

    pl = results["pooled"]
    tot = pl["total_tasks"]
    p_logic = pl["cegis_failed"] - pl["cegis_did_not_generate_python"] - pl["cegis_crashes"] - pl["cegis_api_errors"]
    lines.append(
        f"| **TOTAL (POOLED)** | **{pl['baseline_failed']}** ({pl['baseline_failed']/tot*100:.1f}%) | "
        f"**{pl['cegis_failed']}** ({pl['cegis_failed']/tot*100:.1f}%) | "
        f"**{pl['anticheat_failed']}** ({pl['anticheat_failed']/tot*100:.1f}%) | "
        f"**{pl['cegis_did_not_generate_python']}** | "
        f"**{pl['cegis_crashes']}** | **{pl['cegis_api_errors']}** | **{p_logic}** | "
        f"**{pl['spurious_overfitting']}** | **{pl['representation_ceiling']}** | **{pl['cegis_sub_attempt_failures']}** |"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Calculate failed runs and error taxonomy per model.")
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="Path to benchmark CSV.")
    parser.add_argument("--code-json", type=str, default="benchmark_code.json", help="Path to benchmark code JSON.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional output CSV path.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--output-markdown", type=str, default=None, help="Optional output Markdown path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"task_id": str})

    code_data = None
    code_path = Path(args.code_json) if args.code_json else None
    if code_path and code_path.exists():
        with open(code_path, "r") as f:
            code_data = json.load(f)

    results = compute_failed_runs(df, code_data=code_data)

    print_terminal_report(results)

    if args.output_csv:
        csv_df = pd.DataFrame(results["per_model"])
        csv_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved CSV summary to: {args.output_csv}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON summary to: {args.output_json}")

    if args.output_markdown:
        generate_markdown_summary(results, Path(args.output_markdown))
        print(f"Saved Markdown summary to: {args.output_markdown}")


if __name__ == "__main__":
    main()
