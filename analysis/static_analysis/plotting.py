#!/usr/bin/env python3
"""Plotting engine for ARC Static Analysis.

Generates publication-quality visualizations for:
- Task Solve Distribution (Universally Solved vs Unsolved)
- Guttman Subsumption Matrix & Scalogram
- Code Complexity & Occam's Razor Bloat by Outcome
- Iteration Convergence & Diminishing Returns
- Strategy Complementarity & Oracle Ensemble Headroom
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_plot_style():
    """Configure clean publication style."""
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        plt.style.use("default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_task_solve_distribution(topology_results: dict[str, Any], output_path: str | Path) -> None:
    """Plot task solve count distribution highlighting universal extremes."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    dist = topology_results["solve_count_distribution"]
    x = [d["models_solving"] for d in dist]
    y = [d["task_count"] for d in dist]
    total_models = topology_results["total_models"]

    # Color code: 0 = red (unsolved), middle = blue/slate, N = green (universally solved)
    colors = []
    for val in x:
        if val == 0:
            colors.append("#d9534f")  # Red for unsolved
        elif val == total_models:
            colors.append("#5cb85c")  # Green for universally solved
        else:
            colors.append("#4a90e2")  # Blue for intermediate

    bars = ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.8, width=0.65)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(
                f"{int(h)}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel(f"Number of Models Solving the Task (out of {total_models})", fontsize=11, fontweight="bold")
    ax.set_ylabel("Task Count", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Task Solvability Distribution: Universally Solved ({topology_results['universally_solved_count']}) vs. Unsolved ({topology_results['universally_unsolved_count']})",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_ylim(0, max(y) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_guttman_hierarchy(guttman_results: dict[str, Any], output_path: str | Path) -> None:
    """Plot pairwise subsumption heatmap and consecutive subsumption rates."""
    setup_plot_style()
    models = guttman_results["ordered_models"]
    matrix = guttman_results["pairwise_subsumption"]
    K = len(models)

    # Short clean names
    short_names = [m.split("/")[-1] for m in models]

    # Convert to array
    data = np.zeros((K, K))
    for i, m_a in enumerate(models):
        for j, m_b in enumerate(models):
            data[i, j] = matrix[m_a][m_b]

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=1.0)

    # Annotate cells
    for i in range(K):
        for j in range(K):
            val = data[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.5, fontweight="bold")

    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)

    ax.set_xlabel("Conditioning Model B (Tasks Solved by B)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Subsuming Model A (Also Solved by A)", fontsize=11, fontweight="bold", labelpad=8)

    gm = guttman_results["guttman_metrics"]
    title_str = (
        f"Guttman Task Subsumption Matrix P(Model A | Model B)\n"
        f"Coefficient of Reproducibility (CR) = {gm['coefficient_of_reproducibility_cr']:.3f} | "
        f"Scalability (CS) = {gm['coefficient_of_scalability_cs']:.3f}"
    )
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Conditional Probability P(A | B)", fontsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_code_complexity_by_outcome(complexity_results: dict[str, Any], output_path: str | Path) -> None:
    """Plot boxplot of SLOC by outcome category showing the Occam's razor effect."""
    setup_plot_style()
    raw_records = complexity_results.get("raw_records", [])
    if not raw_records:
        return

    df = pd.DataFrame(raw_records)
    # Filter out empty or syntax errors
    df = df[df["valid_syntax"] == True]

    categories = [
        ("Both Correct\n(First-Shot)", df[df["outcome_category"] == "both_correct"]["sloc"].values),
        ("Semantic Recovery\n(Repaired)", df[df["outcome_category"] == "semantic_recovery"]["sloc"].values),
        ("Spurious Overfit\n(Failed Test)", df[df["failure_type"] == "spurious_overfitting"]["sloc"].values),
        ("Representation Ceiling\n(Failed Train)", df[df["failure_type"] == "representation_ceiling"]["sloc"].values),
    ]

    labels = [c[0] for c in categories]
    data = [c[1] for c in categories]

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=7),
        medianprops=dict(color="black", linewidth=1.5),
    )

    palette = ["#4C72B0", "#55A868", "#ff7f0e", "#d62728"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    om = complexity_results["occams_razor_metrics"]
    ax.set_ylabel("Source Lines of Code (SLOC)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Occam's Razor in Program Synthesis: Code Bloat by Evaluation Outcome\n"
        f"Bloat Factor = {om['bloat_factor_vs_both_correct']:.2f}x (p = {om['mann_whitney_u_pvalue_greater']:.2e})",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_iteration_diminishing_returns(iteration_results: dict[str, Any], output_path: str | Path) -> None:
    """Plot convergence count and cumulative recovery percentage across iterations."""
    setup_plot_style()
    table = iteration_results["iteration_table"]
    iters = [r["iteration"] for r in table]
    solved = [r["tasks_solved"] for r in table]
    cum_pct = [r["cumulative_pct_of_recoveries"] for r in table]

    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=300)

    # Bar chart for tasks solved
    color_bar = "#4a90e2"
    bars = ax1.bar(iters, solved, color=color_bar, alpha=0.85, width=0.55, edgecolor="black", label="Tasks Solved")
    ax1.set_xlabel("Iteration Number (1 = Baseline, 2-5 = Repairs)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Tasks Solved", color="#1c4b82", fontsize=11, fontweight="bold")
    ax1.set_xticks(iters)
    ax1.set_ylim(0, max(solved) * 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{int(h)}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Secondary line for cumulative recovery %
    ax2 = ax1.twinx()
    color_line = "#e65100"
    ax2.plot(iters[1:], cum_pct[1:], color=color_line, marker="s", linewidth=2.2, markersize=8, label="Cumulative Recoveries (%)")
    ax2.set_ylabel("Cumulative Recovery Share (%)", color=color_line, fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.grid(False)

    for x_i, y_i in zip(iters[1:], cum_pct[1:]):
        ax2.annotate(f"{y_i:.1f}%", (x_i, y_i), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=color_line)

    cap = iteration_results["recommended_iteration_cap"]
    ret = iteration_results["recommended_cap_recovery_retention_pct"]
    plt.title(
        f"Iteration Convergence & Diminishing Returns\nRecommended Cap: {cap} Iterations (Retains {ret:.1f}% of Recoveries)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_strategy_discrepancy(discrepancy_results: dict[str, Any], output_path: str | Path) -> None:
    """Plot strategy contingency breakdown and Oracle Ensemble Headroom."""
    setup_plot_style()
    ct = discrepancy_results["contingency_table"]
    s1_name = discrepancy_results["strategy_1"]
    s2_name = discrepancy_results["strategy_2"]

    labels = [
        f"Both\nSucceeded\n({ct['both_succeeded']})",
        f"{s1_name} Only\nSucceeded\n({ct[f'{s1_name}_only_succeeded']})",
        f"{s2_name} Only\nSucceeded\n({ct[f'{s2_name}_only_succeeded']})",
        f"Both\nFailed\n({ct['both_failed']})",
    ]
    vals = [
        ct["both_succeeded"],
        ct[f"{s1_name}_only_succeeded"],
        ct[f"{s2_name}_only_succeeded"],
        ct["both_failed"],
    ]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", width=0.55, alpha=0.85)

    tot = discrepancy_results["total_evaluations"]
    for bar in bars:
        h = bar.get_height()
        pct = (h / tot) * 100
        ax.annotate(f"{pct:.1f}%", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Evaluation Count", fontsize=11, fontweight="bold")
    om = discrepancy_results["oracle_ensemble_metrics"]
    ax.set_title(
        f"Strategy Discrepancy & Complementarity: {s1_name} vs. {s2_name}\n"
        f"Oracle Ensemble Accuracy = {om['oracle_ensemble_accuracy']*100:.1f}% "
        f"(+{om[f'oracle_lift_over_{s1_name}']*100:.1f}% over {s1_name})",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

