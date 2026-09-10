#!/usr/bin/env python3
"""Unified Static Analysis Suite for ARC-AGI Benchmark Data.

Executes all result-independent, model-agnostic, and task-agnostic structural analyses:
1. Task Topology: Universally Solved vs Universally Unsolved Tasks & Inductive Traps
2. Guttman Subsumption Hierarchy & 1D Psychometric Scalability (CR, MMR, CS)
3. Code Complexity & Occam's Razor Bloat by Evaluation Outcome
4. Iteration Dynamics & Convergence Horizon Optimization
5. Strategy Discrepancy, Complementarity & Oracle Ensemble Headroom
6. Coordinate Leakage & Hardcoding Static Scanner
7. Automatic Generation of Publication-Grade Markdown Reports & Figures

Usage:
  python run_static_analysis.py
  python run_static_analysis.py --csv benchmark.csv --code-json benchmark_code.json --output-dir ./static_analysis_output --plots
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Add parent directory to sys.path if running as script
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from static_analysis.task_topology import analyze_task_topology
from static_analysis.guttman_hierarchy import analyze_guttman_hierarchy
from static_analysis.code_complexity import analyze_code_complexity
from static_analysis.iteration_dynamics import analyze_iteration_dynamics
from static_analysis.strategy_discrepancy import analyze_strategy_discrepancy
from static_analysis.coordinate_leakage import analyze_coordinate_leakage
from static_analysis.plotting import (
    plot_task_solve_distribution,
    plot_guttman_hierarchy,
    plot_code_complexity_by_outcome,
    plot_iteration_diminishing_returns,
    plot_strategy_discrepancy,
)


def generate_markdown_report(
    topology: dict[str, Any],
    guttman: dict[str, Any],
    complexity: dict[str, Any],
    iterations: dict[str, Any],
    discrepancy: dict[str, Any],
    leakage: dict[str, Any] | None,
    output_path: Path,
    has_plots: bool = True,
) -> None:
    """Generate a comprehensive, structured Markdown report."""
    lines = []
    lines.append("# ARC-AGI Static Analysis Report: Structural & Result-Independent Dynamics\n")
    lines.append("> Auto-generated report synthesizing task topology, Guttman scaling, code complexity, iteration dynamics, and strategy complementarity.\n")

    # 1. Executive Summary Table
    lines.append("## 1. Structural Overview\n")
    lines.append(f"- **Total Benchmark Tasks**: {topology['total_tasks']}")
    lines.append(f"- **Total Evaluated Models**: {topology['total_models']}")
    lines.append(f"- **Universally Solved Tasks** (All models): **{topology['universally_solved_count']}** ({topology['universally_solved_percentage']:.1f}%)")
    lines.append(f"- **Universally Unsolved Tasks** (0 models): **{topology['universally_unsolved_count']}** ({topology['universally_unsolved_percentage']:.1f}%)")
    lines.append(f"- **Guttman Coefficient of Reproducibility (CR)**: **{guttman['guttman_metrics']['coefficient_of_reproducibility_cr']:.4f}** (Threshold $\\ge 0.90$)")
    lines.append(f"- **Guttman Coefficient of Scalability (CS)**: **{guttman['guttman_metrics']['coefficient_of_scalability_cs']:.4f}** (Threshold $\\ge 0.60$)")
    om = complexity["occams_razor_metrics"]
    lines.append(f"- **Occam's Razor Bloat Factor**: **{om['bloat_factor_vs_both_correct']:.2f}x** longer code under spurious overfitting ($p = {om['mann_whitney_u_pvalue_greater']:.2e}$)")
    lines.append(f"- **Recommended CEGIS Iteration Horizon**: **{iterations['recommended_iteration_cap']} iterations** (captures **{iterations['recommended_cap_recovery_retention_pct']:.1f}%** of all recoveries)")
    strat1 = discrepancy['strategy_1']
    strat2 = discrepancy['strategy_2']
    ens_acc = discrepancy['oracle_ensemble_metrics']['oracle_ensemble_accuracy'] * 100
    ens_lift = discrepancy['oracle_ensemble_metrics'][f'oracle_lift_over_{strat1}'] * 100
    lines.append(f"- **Oracle Ensemble Accuracy ({strat1} + {strat2})**: **{ens_acc:.1f}%** (+{ens_lift:.1f}% lift)\n")

    # 2. Task Topology
    lines.append("## 2. Task Topology: Universally Solved vs. Universally Unsolved\n")
    if has_plots:
        lines.append("![Task Solve Distribution](plot_task_solve_distribution.png)\n")
    lines.append(f"### Universally Solved Tasks (`{len(topology['universally_solved_tasks'])}` tasks)")
    lines.append(f"`{', '.join(topology['universally_solved_tasks']) if topology['universally_solved_tasks'] else 'None'}`\n")
    lines.append(f"### Universally Unsolved Tasks (`{len(topology['universally_unsolved_tasks'])}` tasks)")
    lines.append(f"`{', '.join(topology['universally_unsolved_tasks']) if topology['universally_unsolved_tasks'] else 'None'}`\n")

    lines.append("### Solve Count Frequency Distribution")
    lines.append("| Models Solving | Task Count | Percentage | Visual Bar |")
    lines.append("| :---: | :---: | :---: | :--- |")
    for d in topology["solve_count_distribution"]:
        bar = "█" * int(d["task_count"])
        lines.append(f"| {d['models_solving']} | {d['task_count']} | {d['percentage']:.1f}% | `{bar}` |")
    lines.append("")

    if topology["inductive_traps"]:
        lines.append("### Inductive Trap Hotspots (Train Converged $\\ge 50\\%$, Test Solved $\\le 1$ model)")
        lines.append("| Task ID | Models Converged on Train | Models Passing Test | Train Convergence Rate |")
        lines.append("| :---: | :---: | :---: | :---: |")
        for t in topology["inductive_traps"]:
            lines.append(f"| `{t['task_id']}` | {t['train_converged_models']}/{topology['total_models']} | {t['solved_models']}/{topology['total_models']} | {t['train_convergence_rate']*100:.1f}% |")
        lines.append("")

    # 3. Guttman Hierarchy
    lines.append("## 3. Guttman Subsumption Hierarchy & 1D Scalability\n")
    if has_plots:
        lines.append("![Guttman Subsumption Matrix](plot_guttman_subsumption.png)\n")
    lines.append("In an ideal Guttman scale, any task solved by a weaker model is monotonically subsumed by a stronger model.\n")
    lines.append("### Consecutive Subsumption Chain $P(M_{i} \\mid M_{i+1})$")
    lines.append("| Stronger Model ($M_i$) | Weaker Model ($M_{i+1}$) | Subsumption Rate | Weaker Solved Count |")
    lines.append("| :--- | :--- | :---: | :---: |")
    for link in guttman["consecutive_subsumption"]:
        lines.append(f"| `{link['stronger_model']}` | `{link['weaker_model']}` | **{link['subsumption_percentage']:.1f}%** | {link['weaker_model_solved_count']} tasks |")
    lines.append("")

    gm = guttman["guttman_metrics"]
    lines.append("### Psychometric Scalability Metrics")
    lines.append(f"- **Coefficient of Reproducibility (CR)**: **{gm['coefficient_of_reproducibility_cr']:.4f}** (Criterion $\\ge 0.90$)")
    lines.append(f"- **Minimum Marginal Reproducibility (MMR)**: **{gm['minimum_marginal_reproducibility_mmr']:.4f}**")
    lines.append(f"- **Coefficient of Scalability (CS)**: **{gm['coefficient_of_scalability_cs']:.4f}** (Criterion $\\ge 0.60$)")
    lines.append(f"- **Valid 1-Dimensional Scale**: **{'CONFIRMED (Strong Scaling)' if gm['is_valid_guttman_scale'] else 'REJECTED'}**\n")

    # 4. Code Complexity & Occam's Razor
    lines.append("## 4. Code Complexity & Occam's Razor Bloat Analysis\n")
    if has_plots:
        lines.append("![Code Complexity Boxplot](plot_code_complexity_by_outcome.png)\n")
    lines.append("### Complexity by Evaluation Outcome")
    lines.append("| Outcome Category | Sample Count | Mean SLOC | Mean Loops | Mean Ifs | Mean AST Depth |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for cat, d in complexity["outcome_category_summary"].items():
        lines.append(f"| `{cat}` | {d['count']} | **{d['mean_sloc']:.1f}** | {d['mean_loops']:.1f} | {d['mean_conditionals']:.1f} | {d['mean_ast_depth']:.1f} |")
    lines.append("")

    lines.append("### Failure Mode Breakdown")
    lines.append("| Failure Type | Sample Count | Mean SLOC | Mean Loops | Mean Ifs | Mean Cyclomatic |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for ftype, d in complexity["failure_type_summary"].items():
        lines.append(f"| `{ftype}` | {d['count']} | **{d['mean_sloc']:.1f}** | {d['mean_loops']:.1f} | {d['mean_conditionals']:.1f} | {d['mean_cyclomatic']:.1f} |")
    lines.append("")
    lines.append(f"> **Occam's Razor Law**: Programs that overfit the training demonstrations are **{om['bloat_factor_vs_both_correct']:.2f}x** longer than first-shot correct programs ($p = {om['mann_whitney_u_pvalue_greater']:.2e}$), adding ad-hoc branching conditionals to fit edge cases.\n")

    # 5. Iteration Dynamics
    lines.append("## 5. Iteration Convergence & Diminishing Returns\n")
    if has_plots:
        lines.append("![Iteration Diminishing Returns](plot_iteration_diminishing_returns.png)\n")
    lines.append("### Iteration Recovery Distribution")
    lines.append("| Iteration | Tasks Solved | % of Solved | Recovered Tasks | % of Recoveries | Cumulative % Recoveries | Mean Latency |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in iterations["iteration_table"]:
        lat_str = f"{r['mean_latency_sec']:.1f}s" if r["mean_latency_sec"] is not None else "N/A"
        lines.append(f"| {r['iteration']} | {r['tasks_solved']} | {r['pct_of_all_solved']:.1f}% | {r['recovered_tasks']} | {r['pct_of_recoveries']:.1f}% | **{r['cumulative_pct_of_recoveries']:.1f}%** | {lat_str} |")
    lines.append("")
    lines.append(f"> **Horizon Optimization Recommendation**: Setting the CEGIS horizon to **{iterations['recommended_iteration_cap']} iterations** captures **{iterations['recommended_cap_recovery_retention_pct']:.1f}%** of all possible semantic repairs while discarding the long-tail latency bottleneck.\n")

    # 6. Strategy Complementarity
    lines.append(f"## 6. Strategy Complementarity: {discrepancy['strategy_1']} vs. {discrepancy['strategy_2']}\n")
    if has_plots:
        lines.append("![Strategy Discrepancy](plot_strategy_discrepancy.png)\n")
    ct = discrepancy["contingency_table"]
    tot = discrepancy["total_evaluations"]
    s1 = discrepancy["strategy_1"]
    s2 = discrepancy["strategy_2"]
    lines.append("### Contingency Matrix")
    lines.append(f"- **Both Succeeded**: {ct['both_succeeded']} ({ct['both_succeeded']/tot*100:.1f}%)")
    lines.append(f"- **{s1} Only Succeeded**: {ct[f'{s1}_only_succeeded']} ({ct[f'{s1}_only_succeeded']/tot*100:.1f}%)")
    lines.append(f"- **{s2} Only Succeeded**: {ct[f'{s2}_only_succeeded']} ({ct[f'{s2}_only_succeeded']/tot*100:.1f}%)")
    lines.append(f"- **Both Failed**: {ct['both_failed']} ({ct['both_failed']/tot*100:.1f}%)")
    lines.append(f"- **Jaccard Index**: **{discrepancy['jaccard_similarity']:.4f}** | **Cohen's Kappa**: **{discrepancy['cohens_kappa']:.4f}**\n")

    om_ens = discrepancy["oracle_ensemble_metrics"]
    lines.append("### Oracle Ensemble Upper Bound")
    lines.append(f"- **{s1} Standalone Accuracy**: {om_ens[f'{s1}_accuracy']*100:.1f}%")
    lines.append(f"- **{s2} Standalone Accuracy**: {om_ens[f'{s2}_accuracy']*100:.1f}%")
    lines.append(f"- **Oracle Ensemble Accuracy**: **{om_ens['oracle_ensemble_accuracy']*100:.1f}%** (Lift: **+{om_ens[f'oracle_lift_over_{s1}']*100:.1f}%**)\n")

    # 7. Coordinate Leakage Scanner (if available)
    if leakage:
        lines.append("## 7. Static Code Scanner: Coordinate Leakage & Cheating\n")
        lines.append(f"- **Total Code Snippets Scanned**: {leakage['total_snippets_scanned']}")
        lines.append(f"- **Flagged Snippets with Hardcoded Coordinates**: **{leakage['flagged_snippets_count']}** ({leakage['leakage_rate_pct']:.2f}%)\n")
        lines.append("### Breakdown by Strategy")
        for strat, cnt in leakage["strategy_counts"].items():
            lines.append(f"- `{strat}`: {cnt} snippets")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def run_full_static_analysis(
    csv_path: str | Path = "benchmark.csv",
    code_json_path: str | Path | None = "benchmark_code.json",
    output_dir: str | Path = "static_analysis_output",
    strategy_col: str = "cegis_success",
    compare_col: str = "anticheat_success",
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Run all static analyses and generate artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading benchmark dataset from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={"task_id": str})

    print("[2/7] Analyzing Task Topology (Universally Solved vs. Unsolved & Traps)...")
    topology_res = analyze_task_topology(df, strategy_col=strategy_col)

    # Save task topology CSV
    task_df = pd.DataFrame(topology_res["per_task_table"])
    task_df.to_csv(out_dir / "task_topology.csv", index=False)

    print("[3/7] Analyzing Guttman Subsumption Hierarchy & Scalogram...")
    guttman_res = analyze_guttman_hierarchy(df, strategy_col=strategy_col)

    # Save Guttman CSV
    guttman_df = pd.DataFrame(guttman_res["pairwise_subsumption"])
    guttman_df.to_csv(out_dir / "guttman_subsumption_matrix.csv")

    print("[4/7] Analyzing Code Complexity & Occam's Razor Bloat...")
    complexity_res = analyze_code_complexity(
        df,
        code_json_path=code_json_path if (code_json_path and Path(code_json_path).exists()) else None,
        strategy=strategy_col.split("_")[0],
    )
    comp_df = pd.DataFrame(complexity_res["raw_records"])
    comp_df.to_csv(out_dir / "code_complexity_by_outcome.csv", index=False)

    print("[5/7] Analyzing Iteration Dynamics & Convergence Horizon...")
    iteration_res = analyze_iteration_dynamics(df, strategy=strategy_col.split("_")[0])
    iter_df = pd.DataFrame(iteration_res["iteration_table"])
    iter_df.to_csv(out_dir / "iteration_dynamics.csv", index=False)

    print(f"[6/7] Analyzing Strategy Discrepancy ({strategy_col} vs. {compare_col})...")
    discrepancy_res = analyze_strategy_discrepancy(
        df,
        strat1_col=strategy_col,
        strat2_col=compare_col,
        strat1_name=strategy_col.split("_")[0].upper(),
        strat2_name=compare_col.split("_")[0].capitalize(),
    )

    leakage_res = None
    if code_json_path and Path(code_json_path).exists():
        print(f"[6.5/7] Scanning code in {code_json_path} for coordinate leakage...")
        leakage_res = analyze_coordinate_leakage(code_json_path)

    has_plots = False
    if generate_plots:
        print("[7/7] Generating publication-grade figures...")
        try:
            plot_task_solve_distribution(topology_res, out_dir / "plot_task_solve_distribution.png")
            plot_guttman_hierarchy(guttman_res, out_dir / "plot_guttman_subsumption.png")
            plot_code_complexity_by_outcome(complexity_res, out_dir / "plot_code_complexity_by_outcome.png")
            plot_iteration_diminishing_returns(iteration_res, out_dir / "plot_iteration_diminishing_returns.png")
            plot_strategy_discrepancy(discrepancy_res, out_dir / "plot_strategy_discrepancy.png")
            has_plots = True
            print("  Figures successfully written to output directory.")
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

    # Generate Markdown Report
    report_path = out_dir / "static_analysis_report.md"
    generate_markdown_report(
        topology=topology_res,
        guttman=guttman_res,
        complexity=complexity_res,
        iterations=iteration_res,
        discrepancy=discrepancy_res,
        leakage=leakage_res,
        output_path=report_path,
        has_plots=has_plots,
    )
    print(f"\nStatic Analysis Report generated: {report_path}")

    # Generate complete JSON summary
    summary_json_path = out_dir / "static_analysis_summary.json"
    summary_data = {
        "task_topology": {k: v for k, v in topology_res.items() if k != "per_task_table"},
        "guttman_hierarchy": guttman_res,
        "code_complexity": {k: v for k, v in complexity_res.items() if k != "raw_records"},
        "iteration_dynamics": iteration_res,
        "strategy_discrepancy": discrepancy_res,
        "coordinate_leakage": {k: v for k, v in leakage_res.items() if k != "flagged_snippets"} if leakage_res else None,
    }
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Static Analysis JSON summary generated: {summary_json_path}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Run complete ARC-AGI static analysis suite.")
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="Path to benchmark CSV.")
    parser.add_argument("--code-json", type=str, default="benchmark_code.json", help="Path to benchmark code JSON.")
    parser.add_argument("--output-dir", type=str, default="static_analysis_output", help="Output directory path.")
    parser.add_argument("--strategy", type=str, default="cegis_success", help="Primary strategy success column.")
    parser.add_argument("--compare", type=str, default="anticheat_success", help="Comparison strategy success column.")
    parser.add_argument("--no-plots", action="store_true", help="Disable generating PNG plots.")
    args = parser.parse_args()

    run_full_static_analysis(
        csv_path=args.csv,
        code_json_path=args.code_json if Path(args.code_json).exists() else None,
        output_dir=args.output_dir,
        strategy_col=args.strategy,
        compare_col=args.compare,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
