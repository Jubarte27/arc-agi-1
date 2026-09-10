#!/usr/bin/env python3
"""Compare CEGIS vs. AntiCheat Task-Level Outcomes.

Identifies and analyzes:
1. Which tasks were solved by CEGIS but NOT AntiCheat (CEGIS-only).
2. Which tasks were solved by AntiCheat but NOT CEGIS (AntiCheat-only).
3. Per-model breakdown and global task-level consensus.
4. Failure mode inspection (why the losing strategy failed: overfit vs ceiling vs crash).
5. Code-level inspection and structural comparisons between the two strategies.

Usage:
  python compare_cegis_anticheat_tasks.py
  python compare_cegis_anticheat_tasks.py --csv benchmark.csv --code-json benchmark_code.json
  python compare_cegis_anticheat_tasks.py --model "Gemini 3.5 Flash Lite"
  python compare_cegis_anticheat_tasks.py --task "0becf7df" --inspect
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def analyze_cegis_vs_anticheat_tasks(
    df: pd.DataFrame,
    code_data: dict[str, Any] | None = None,
    model_filter: str | None = None,
    task_filter: str | None = None,
) -> dict[str, Any]:
    """Identify and analyze all tasks with diverging outcomes between CEGIS and AntiCheat.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark dataframe.
    code_data : dict[str, Any] | None
        Dictionary of synthesized Python programs.
    model_filter : str | None
        Optional model name filter.
    task_filter : str | None
        Optional task ID filter.

    Returns
    -------
    dict[str, Any]
        Structured dictionary with model breakdowns, task-level summaries,
        and detailed divergent instance records.
    """
    model_col = "friendly_name" if "friendly_name" in df.columns else "model"

    if model_filter:
        df = df[df[model_col].str.lower() == model_filter.lower()]
    if task_filter:
        df = df[df["task_id"] == task_filter]

    models = sorted(df[model_col].unique().tolist())

    # 1. Per-Model Discrepancies
    per_model_summary = []
    divergent_instances = []

    for m in models:
        grp = df[df[model_col] == m]
        cegis_only = grp[(grp["cegis_success"] == True) & (grp["anticheat_success"] == False)]
        anti_only = grp[(grp["cegis_success"] == False) & (grp["anticheat_success"] == True)]
        both_succ = grp[(grp["cegis_success"] == True) & (grp["anticheat_success"] == True)]
        both_fail = grp[(grp["cegis_success"] == False) & (grp["anticheat_success"] == False)]

        c_tasks = sorted(cegis_only["task_id"].tolist())
        a_tasks = sorted(anti_only["task_id"].tolist())

        per_model_summary.append({
            "model": m,
            "total_tasks": len(grp),
            "both_succeeded_count": len(both_succ),
            "cegis_only_count": len(c_tasks),
            "cegis_only_tasks": c_tasks,
            "anticheat_only_count": len(a_tasks),
            "anticheat_only_tasks": a_tasks,
            "both_failed_count": len(both_fail),
            "net_difference": len(a_tasks) - len(c_tasks),
        })

        # Detailed record for each divergent task
        for _, row in cegis_only.iterrows():
            tid = str(row["task_id"])
            divergent_instances.append({
                "model": m,
                "task_id": tid,
                "winning_strategy": "CEGIS",
                "losing_strategy": "AntiCheat",
                "cegis_success": True,
                "anticheat_success": False,
                "cegis_iterations": int(row.get("cegis_iterations_used", 1)),
                "anticheat_iterations": int(row.get("anticheat_iterations_used", 1)),
                "anticheat_converged_train": bool(row.get("anticheat_converged_train", False)),
                "losing_failure_reason": (
                    "spurious_overfitting" if bool(row.get("anticheat_converged_train", False))
                    else ("code_crash" if pd.notna(row.get("anticheat_error")) else "representation_ceiling")
                ),
                "cegis_code_lines": int(row.get("cegis_code_lines", 0) or 0),
                "anticheat_code_lines": int(row.get("anticheat_code_lines", 0) or 0),
            })

        for _, row in anti_only.iterrows():
            tid = str(row["task_id"])
            divergent_instances.append({
                "model": m,
                "task_id": tid,
                "winning_strategy": "AntiCheat",
                "losing_strategy": "CEGIS",
                "cegis_success": False,
                "anticheat_success": True,
                "cegis_iterations": int(row.get("cegis_iterations_used", 1)),
                "anticheat_iterations": int(row.get("anticheat_iterations_used", 1)),
                "cegis_converged_train": bool(row.get("cegis_converged_train", False)),
                "losing_failure_reason": str(row.get("cegis_failure_type", "unknown")),
                "cegis_code_lines": int(row.get("cegis_code_lines", 0) or 0),
                "anticheat_code_lines": int(row.get("anticheat_code_lines", 0) or 0),
            })

    # 2. Global Task-Level Consensus
    all_divergent_tasks = sorted(list(set(d["task_id"] for d in divergent_instances)))
    task_level_summary = []

    for tid in all_divergent_tasks:
        task_grp = df[df["task_id"] == tid]
        c_solvers = task_grp[task_grp["cegis_success"] == True][model_col].tolist()
        a_solvers = task_grp[task_grp["anticheat_success"] == True][model_col].tolist()

        c_only_models = [m for m in c_solvers if m not in a_solvers]
        a_only_models = [m for m in a_solvers if m not in c_solvers]

        bias = "Tied"
        if len(c_only_models) > len(a_only_models):
            bias = "Favors CEGIS"
        elif len(a_only_models) > len(c_only_models):
            bias = "Favors AntiCheat"

        task_level_summary.append({
            "task_id": tid,
            "bias": bias,
            "cegis_total_solves": len(c_solvers),
            "anticheat_total_solves": len(a_solvers),
            "cegis_only_models": c_only_models,
            "anticheat_only_models": a_only_models,
            "net_lift": len(a_solvers) - len(c_solvers),
        })

    # Sort task summary by net lift
    task_level_summary.sort(key=lambda x: (x["net_lift"], x["task_id"]))

    total_cegis_only = sum(m["cegis_only_count"] for m in per_model_summary)
    total_anti_only = sum(m["anticheat_only_count"] for m in per_model_summary)

    return {
        "total_models": len(models),
        "total_divergent_tasks": len(all_divergent_tasks),
        "total_cegis_only_instances": total_cegis_only,
        "total_anticheat_only_instances": total_anti_only,
        "per_model_summary": per_model_summary,
        "task_level_summary": task_level_summary,
        "divergent_instances": divergent_instances,
    }


def print_terminal_report(results: dict[str, Any], show_details: bool = True) -> None:
    """Print clean formatted comparison tables to terminal."""
    print("=" * 95)
    print("CEGIS VS. ANTICHEAT: DIVERGENT TASK ANALYSIS")
    print("=" * 95)
    print(f"Total Divergent Tasks (at least 1 model differed): {results['total_divergent_tasks']}")
    print(f"Total CEGIS-only Wins across models:             {results['total_cegis_only_instances']}")
    print(f"Total AntiCheat-only Wins across models:         {results['total_anticheat_only_instances']}")
    print("=" * 95)

    print("\n1. MODEL-BY-MODEL DISCREPANCY BREAKDOWN:")
    print(f"{'Model':26s} | {'CEGIS Only':10s} | {'AntiCheat Only':14s} | {'Net Diff':8s} | Discrepant Task IDs")
    print("-" * 95)

    for m in results["per_model_summary"]:
        diff_str = f"{m['net_difference']:+2d}"
        c_cnt = f"{m['cegis_only_count']:2d} tasks"
        a_cnt = f"{m['anticheat_only_count']:2d} tasks"

        tasks_preview = []
        if m["cegis_only_tasks"]:
            tasks_preview.append(f"CEGIS: {', '.join(m['cegis_only_tasks'][:4])}{'...' if len(m['cegis_only_tasks']) > 4 else ''}")
        if m["anticheat_only_tasks"]:
            tasks_preview.append(f"Anti: {', '.join(m['anticheat_only_tasks'][:4])}{'...' if len(m['anticheat_only_tasks']) > 4 else ''}")

        preview_str = " | ".join(tasks_preview) if tasks_preview else "No discrepancies"
        print(f"{m['model'][:26]:26s} | {c_cnt:10s} | {a_cnt:14s} | {diff_str:8s} | {preview_str}")

    print("\n2. TOP TASKS WITH STRONGEST STRATEGY BIAS:")
    print(f"{'Task ID':10s} | {'Bias':18s} | {'CEGIS Solves':12s} | {'Anti Solves':12s} | Model Breakdown")
    print("-" * 95)

    for t in results["task_level_summary"]:
        if abs(t["net_lift"]) >= 1:
            c_desc = f"CEGIS won: {', '.join(t['cegis_only_models'])}" if t["cegis_only_models"] else ""
            a_desc = f"Anti won: {', '.join(t['anticheat_only_models'])}" if t["anticheat_only_models"] else ""
            desc = " ; ".join(filter(None, [c_desc, a_desc]))
            print(f"{t['task_id']:10s} | {t['bias']:18s} | {t['cegis_total_solves']:12d} | {t['anticheat_total_solves']:12d} | {desc}")

    if show_details and results["divergent_instances"]:
        print("\n3. SAMPLE DIVERGENT INSTANCES & REASON WHY LOSER FAILED:")
        print(f"{'Task ID':10s} | {'Model':24s} | {'Winner':9s} | {'Loser Failure Reason':25s} | Iter (Win vs Lose)")
        print("-" * 95)
        for inst in results["divergent_instances"][:12]:
            win_strat = inst["winning_strategy"]
            win_iter = inst["cegis_iterations"] if win_strat == "CEGIS" else inst["anticheat_iterations"]
            lose_iter = inst["anticheat_iterations"] if win_strat == "CEGIS" else inst["cegis_iterations"]
            print(
                f"{inst['task_id']:10s} | {inst['model'][:24]:24s} | {win_strat:9s} | "
                f"{inst['losing_failure_reason'][:25]:25s} | Iter {win_iter} vs {lose_iter}"
            )
    print("=" * 95)


def generate_markdown_report(results: dict[str, Any], output_path: Path) -> None:
    """Generate Markdown report detailing all divergent tasks."""
    lines = []
    lines.append("# CEGIS vs. AntiCheat: Task-Level Divergence Report\n")
    lines.append("> Detailed task-by-task breakdown of tasks solved by CEGIS but not AntiCheat and vice-versa.\n")

    lines.append("## 1. Executive Summary\n")
    lines.append(f"- **Total Divergent Tasks**: **{results['total_divergent_tasks']} tasks**")
    lines.append(f"- **CEGIS-Only Wins (CEGIS solved, AntiCheat failed)**: **{results['total_cegis_only_instances']} instances**")
    lines.append(f"- **AntiCheat-Only Wins (AntiCheat solved, CEGIS failed)**: **{results['total_anticheat_only_instances']} instances**\n")

    lines.append("## 2. Per-Model Task Breakdown\n")
    lines.append("| Model | CEGIS Only Solved | AntiCheat Only Solved | Net Diff | Tasks Solved by CEGIS Only | Tasks Solved by AntiCheat Only |")
    lines.append("| :--- | :---: | :---: | :---: | :--- | :--- |")

    for m in results["per_model_summary"]:
        c_list = ", ".join(f"`{t}`" for t in m["cegis_only_tasks"]) if m["cegis_only_tasks"] else "None"
        a_list = ", ".join(f"`{t}`" for t in m["anticheat_only_tasks"]) if m["anticheat_only_tasks"] else "None"
        lines.append(
            f"| **{m['model']}** | {m['cegis_only_count']} | {m['anticheat_only_count']} | {m['net_difference']:+d} | {c_list} | {a_list} |"
        )
    lines.append("")

    lines.append("## 3. Global Task-Level Consensus\n")
    lines.append("| Task ID | Bias | CEGIS Solves | AntiCheat Solves | CEGIS-Only Models | AntiCheat-Only Models |")
    lines.append("| :--- | :---: | :---: | :---: | :--- | :--- |")

    for t in results["task_level_summary"]:
        c_models = ", ".join(t["cegis_only_models"]) if t["cegis_only_models"] else "—"
        a_models = ", ".join(t["anticheat_only_models"]) if t["anticheat_only_models"] else "—"
        lines.append(
            f"| `{t['task_id']}` | **{t['bias']}** | {t['cegis_total_solves']} | {t['anticheat_total_solves']} | {c_models} | {a_models} |"
        )
    lines.append("")

    lines.append("## 4. Complete Divergent Instances Record\n")
    lines.append("| Task ID | Model | Winning Strategy | Losing Failure Reason | Winner Iters | Loser Iters | Code Lines (Win vs Lose) |")
    lines.append("| :--- | :--- | :---: | :--- | :---: | :---: | :---: |")

    for inst in results["divergent_instances"]:
        win = inst["winning_strategy"]
        win_iters = inst["cegis_iterations"] if win == "CEGIS" else inst["anticheat_iterations"]
        lose_iters = inst["anticheat_iterations"] if win == "CEGIS" else inst["cegis_iterations"]
        win_lines = inst["cegis_code_lines"] if win == "CEGIS" else inst["anticheat_code_lines"]
        lose_lines = inst["anticheat_code_lines"] if win == "CEGIS" else inst["cegis_code_lines"]
        lines.append(
            f"| `{inst['task_id']}` | {inst['model']} | **{win}** | `{inst['losing_failure_reason']}` | {win_iters} | {lose_iters} | {win_lines} vs {lose_lines} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def inspect_specific_task(
    task_id: str,
    df: pd.DataFrame,
    code_data: dict[str, Any] | None,
) -> None:
    """Print in-depth qualitative inspection of a specific task."""
    model_col = "friendly_name" if "friendly_name" in df.columns else "model"
    task_rows = df[df["task_id"] == task_id]

    if task_rows.empty:
        print(f"Task ID '{task_id}' not found in benchmark.")
        return

    print("=" * 80)
    print(f"DEEP DIVE INSPECTION: TASK {task_id}")
    print("=" * 80)

    for _, row in task_rows.iterrows():
        m = row[model_col]
        raw_m = row.get("raw_model", m)
        c_succ = row["cegis_success"]
        a_succ = row.get("anticheat_success")

        status = "AGREEMENT"
        if c_succ and not a_succ:
            status = ">>> CEGIS WON, ANTICHEAT FAILED <<<"
        elif not c_succ and a_succ:
            status = ">>> ANTICHEAT WON, CEGIS FAILED <<<"

        print(f"\nModel: {m} -> {status}")
        print(f"  Baseline:  {row['baseline_success']} (Iter 1)")
        print(f"  CEGIS:     {c_succ} (Iter {row.get('cegis_iterations_used')}) | Failure: {row.get('cegis_failure_type')}")
        print(f"  AntiCheat: {a_succ} (Iter {row.get('anticheat_iterations_used')}) | Error: {row.get('anticheat_error')}")

        if code_data:
            c_code = ""
            a_code = ""
            if m in code_data and task_id in code_data[m]:
                c_code = code_data[m][task_id].get("cegis", "")
                a_code = code_data[m][task_id].get("cegis_anticheat", "")
            elif raw_m in code_data and task_id in code_data[raw_m]:
                c_code = code_data[raw_m][task_id].get("cegis", "")
                a_code = code_data[raw_m][task_id].get("cegis_anticheat", "")

            if c_code:
                print("  [CEGIS Code Excerpt - First 12 lines]:")
                for line in c_code.splitlines()[:12]:
                    print(f"    {line}")
            if a_code:
                print("  [AntiCheat Code Excerpt - First 12 lines]:")
                for line in a_code.splitlines()[:12]:
                    print(f"    {line}")


def main():
    parser = argparse.ArgumentParser(description="Find tasks solved by CEGIS but not AntiCheat and vice-versa.")
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="Path to benchmark CSV.")
    parser.add_argument("--code-json", type=str, default="benchmark_code.json", help="Path to code JSON.")
    parser.add_argument("--model", type=str, default=None, help="Filter to specific model.")
    parser.add_argument("--task", type=str, default=None, help="Inspect specific task ID.")
    parser.add_argument("--inspect", action="store_true", help="Print code inspection for task.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional output CSV path.")
    parser.add_argument("--output-markdown", type=str, default=None, help="Optional output Markdown path.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"task_id": str})

    code_data = None
    code_path = Path(args.code_json) if args.code_json else None
    if code_path and code_path.exists():
        with open(code_path, "r") as f:
            code_data = json.load(f)

    if args.task and args.inspect:
        inspect_specific_task(args.task, df, code_data)
        return

    results = analyze_cegis_vs_anticheat_tasks(
        df,
        code_data=code_data,
        model_filter=args.model,
        task_filter=args.task,
    )

    print_terminal_report(results)

    if args.output_csv:
        csv_df = pd.DataFrame(results["divergent_instances"])
        csv_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved CSV of divergent instances to: {args.output_csv}")

    if args.output_markdown:
        generate_markdown_report(results, Path(args.output_markdown))
        print(f"Saved Markdown report to: {args.output_markdown}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON summary to: {args.output_json}")


if __name__ == "__main__":
    main()

