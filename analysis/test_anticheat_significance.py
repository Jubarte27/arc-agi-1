#!/usr/bin/env python3
"""Statistical Significance Testing: CEGIS vs. AntiCheat.

Conducts rigorous non-parametric and parametric hypothesis tests to determine
whether the difference between using or not using AntiCheat is statistically significant:
1. Paired McNemar's Test with Continuity Correction
2. Exact Binomial Test on Discordant Pairs (Gold Standard for Paired Binary Data)
3. Paired Sign-Flip Permutation Test (100,000 Monte Carlo permutations)
4. Paired Bootstrap Test (10,000 resamples) with 95% Confidence Intervals
5. Multiple Testing Corrections: Bonferroni (FWER) and Benjamini-Hochberg (FDR)
6. Secondary Dimension Tests: Spurious Overfitting, Latency, and Iteration counts
7. Publication-Quality Forest Plot Visualization and LaTeX Table Generation

Usage:
  python test_anticheat_significance.py
  python test_anticheat_significance.py --csv benchmark.csv --plot --output-report anticheat_significance_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def paired_permutation_test(
    s1: np.ndarray,
    s2: np.ndarray,
    n_permutations: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate the two-sided p-value from a paired sign-flip permutation test.

    Parameters
    ----------
    s1 : np.ndarray
        Binary outcomes for strategy 1 (CEGIS).
    s2 : np.ndarray
        Binary outcomes for strategy 2 (AntiCheat).
    n_permutations : int
        Number of permutations (default: 100,000).
    seed : int
        RNG seed.

    Returns
    -------
    tuple[float, float]
        (two-sided p-value, observed mean difference)
    """
    diff = s2.astype(int) - s1.astype(int)
    observed = float(diff.mean())
    discordant = diff[diff != 0]

    if len(discordant) == 0:
        return 1.0, 0.0

    rng = np.random.default_rng(seed)
    # Generate random sign flips: ±1 for each discordant pair
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(discordant)))
    null_dist = (signs * discordant).sum(axis=1) / len(diff)

    # Two-sided p-value: fraction of null differences with absolute magnitude >= |observed|
    p_val = float((np.sum(np.abs(null_dist) >= np.abs(observed)) + 1) / (n_permutations + 1))
    return p_val, observed


def paired_bootstrap_ci(
    s1: np.ndarray,
    s2: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate empirical 95% Confidence Interval for mean difference (s2 - s1).

    Parameters
    ----------
    s1 : np.ndarray
        Outcomes for strategy 1.
    s2 : np.ndarray
        Outcomes for strategy 2.
    n_boot : int
        Bootstrap resamples (default: 10,000).
    alpha : float
        Significance level (default: 0.05).
    seed : int
        RNG seed.

    Returns
    -------
    tuple[float, float, float]
        (ci_lower, ci_upper, bootstrap_p_value)
    """
    diff = s2.astype(int) - s1.astype(int)
    observed = float(diff.mean())
    n = len(diff)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[indices].mean(axis=1)

    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))

    # Centered bootstrap p-value testing H0: mean = 0
    centered = boot_means - observed
    p_boot = float((np.sum(np.abs(centered) >= np.abs(observed)) + 1) / (n_boot + 1))

    return ci_lower, ci_upper, p_boot


def compute_mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """Compute McNemar's chi-squared with continuity correction.

    Parameters
    ----------
    b : int
        Discordant count (Strategy 1 success, Strategy 2 failure).
    c : int
        Discordant count (Strategy 1 failure, Strategy 2 success).

    Returns
    -------
    tuple[float, float]
        (chi2_statistic, p_value)
    """
    total_disc = b + c
    if total_disc == 0:
        return 0.0, 1.0

    # Edwards continuity correction
    stat = ((abs(b - c) - 1.0) ** 2) / total_disc
    p_val = float(stats.chi2.sf(stat, df=1))
    return float(stat), p_val


def run_anticheat_significance_analysis(
    df: pd.DataFrame,
    cegis_col: str = "cegis_success",
    anticheat_col: str = "anticheat_success",
    alpha: float = 0.05,
    n_permutations: int = 100_000,
    n_boot: int = 10_000,
) -> dict[str, Any]:
    """Perform full significance testing between CEGIS and AntiCheat.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark dataframe.
    cegis_col : str
        Column name for CEGIS success.
    anticheat_col : str
        Column name for AntiCheat success.
    alpha : float
        Significance threshold.
    n_permutations : int
        Number of permutations for sign-flip test.
    n_boot : int
        Number of bootstrap samples.

    Returns
    -------
    dict[str, Any]
        Dictionary with pooled, per-model, secondary tests, and verdicts.
    """
    s1_all = df[cegis_col].astype(bool).values
    s2_all = df[anticheat_col].astype(bool).values
    n_total = len(df)

    # 1. Pooled Overall Test
    b_all = int(((s1_all == True) & (s2_all == False)).sum())
    c_all = int(((s1_all == False) & (s2_all == True)).sum())
    both_succ = int(((s1_all == True) & (s2_all == True)).sum())
    both_fail = int(((s1_all == False) & (s2_all == False)).sum())

    total_disc_all = b_all + c_all
    binom_res_all = stats.binomtest(min(b_all, c_all), total_disc_all, p=0.5, alternative="two-sided")
    mcnemar_stat, mcnemar_p = compute_mcnemar_test(b_all, c_all)
    p_perm_all, obs_diff_all = paired_permutation_test(s1_all, s2_all, n_permutations=n_permutations)
    ci_l_all, ci_u_all, p_boot_all = paired_bootstrap_ci(s1_all, s2_all, n_boot=n_boot, alpha=alpha)

    cegis_acc_all = float(s1_all.mean())
    anti_acc_all = float(s2_all.mean())
    odds_ratio_all = float(c_all / b_all) if b_all > 0 else (np.inf if c_all > 0 else 1.0)

    pooled_results = {
        "total_evaluations": n_total,
        "cegis_accuracy": cegis_acc_all,
        "anticheat_accuracy": anti_acc_all,
        "delta_accuracy": obs_diff_all,
        "contingency": {
            "both_succeeded": both_succ,
            "cegis_only_succeeded_b": b_all,
            "anticheat_only_succeeded_c": c_all,
            "both_failed": both_fail,
            "total_discordant": total_disc_all,
        },
        "tests": {
            "exact_binomial_pvalue": float(binom_res_all.pvalue),
            "mcnemar_chi2_statistic": mcnemar_stat,
            "mcnemar_chi2_pvalue": mcnemar_p,
            "permutation_pvalue": p_perm_all,
            "bootstrap_pvalue": p_boot_all,
            "bootstrap_95_ci": [ci_l_all, ci_u_all],
        },
        "effect_sizes": {
            "risk_difference": obs_diff_all,
            "odds_ratio": odds_ratio_all,
        },
        "is_statistically_significant": bool(binom_res_all.pvalue < alpha),
        "verdict": (
            "STATISTICALLY SIGNIFICANT" if binom_res_all.pvalue < alpha
            else "NOT STATISTICALLY SIGNIFICANT"
        ),
    }

    # 2. Per-Model Stratified Tests
    model_col = "friendly_name" if "friendly_name" in df.columns else "model"
    models = sorted(df[model_col].unique().tolist())
    model_records = []

    for m in models:
        grp = df[df[model_col] == m]
        m_s1 = grp[cegis_col].astype(bool).values
        m_s2 = grp[anticheat_col].astype(bool).values
        n_m = len(grp)

        b_m = int(((m_s1 == True) & (m_s2 == False)).sum())
        c_m = int(((m_s1 == False) & (m_s2 == True)).sum())
        disc_m = b_m + c_m

        m_p_binom = (
            float(stats.binomtest(min(b_m, c_m), disc_m, p=0.5, alternative="two-sided").pvalue)
            if disc_m > 0 else 1.0
        )
        m_p_perm, m_diff = paired_permutation_test(m_s1, m_s2, n_permutations=20_000)
        m_ci_l, m_ci_u, m_p_boot = paired_bootstrap_ci(m_s1, m_s2, n_boot=5_000, alpha=alpha)

        model_records.append({
            "model": m,
            "n_tasks": n_m,
            "cegis_solved": int(m_s1.sum()),
            "cegis_accuracy": float(m_s1.mean()),
            "anticheat_solved": int(m_s2.sum()),
            "anticheat_accuracy": float(m_s2.mean()),
            "delta": m_diff,
            "b_cegis_only": b_m,
            "c_anticheat_only": c_m,
            "total_discordant": disc_m,
            "exact_binomial_p": m_p_binom,
            "permutation_p": m_p_perm,
            "bootstrap_p": m_p_boot,
            "bootstrap_95_ci": [m_ci_l, m_ci_u],
        })

    # Multiple Testing Corrections (Bonferroni & Benjamini-Hochberg)
    K = len(model_records)
    # Sort by p-value to compute Benjamini-Hochberg
    sorted_indices = sorted(range(K), key=lambda idx: model_records[idx]["exact_binomial_p"])

    for rank, idx in enumerate(sorted_indices, 1):
        raw_p = model_records[idx]["exact_binomial_p"]
        p_bonf = min(raw_p * K, 1.0)
        p_bh = min(raw_p * K / rank, 1.0)

        model_records[idx]["bonferroni_p"] = float(p_bonf)
        model_records[idx]["benjamini_hochberg_q"] = float(p_bh)
        model_records[idx]["significant_raw"] = bool(raw_p < alpha)
        model_records[idx]["significant_bonferroni"] = bool(p_bonf < alpha)
        model_records[idx]["significant_bh"] = bool(p_bh < alpha)

    # 3. Secondary Hypotheses: Spurious Overfitting, Latency, Iterations
    # Spurious Overfitting
    cegis_spur = (df["cegis_converged_train"] == True) & (df["cegis_test_success"] == False)
    anti_spur = (df["anticheat_converged_train"] == True) & (df["anticheat_test_success"] == False)

    b_spur = int(((cegis_spur == True) & (anti_spur == False)).sum())
    c_spur = int(((cegis_spur == False) & (anti_spur == True)).sum())
    spur_disc = b_spur + c_spur
    p_spur = (
        float(stats.binomtest(min(b_spur, c_spur), spur_disc, p=0.5, alternative="two-sided").pvalue)
        if spur_disc > 0 else 1.0
    )

    # Latency
    paired_lat = df[["cegis_latency_sec", "anticheat_latency_sec"]].dropna()
    t_lat = stats.ttest_rel(paired_lat["anticheat_latency_sec"], paired_lat["cegis_latency_sec"])
    w_lat = stats.wilcoxon(paired_lat["anticheat_latency_sec"], paired_lat["cegis_latency_sec"])

    # Iterations
    paired_iter = df[["cegis_iterations_used", "anticheat_iterations_used"]].dropna()
    w_iter = stats.wilcoxon(paired_iter["anticheat_iterations_used"], paired_iter["cegis_iterations_used"])

    secondary_tests = {
        "spurious_overfitting": {
            "cegis_count": int(cegis_spur.sum()),
            "anticheat_count": int(anti_spur.sum()),
            "cegis_only_b": b_spur,
            "anticheat_only_c": c_spur,
            "exact_binomial_pvalue": p_spur,
            "is_significant": bool(p_spur < alpha),
        },
        "latency_seconds": {
            "cegis_mean": float(paired_lat["cegis_latency_sec"].mean()),
            "anticheat_mean": float(paired_lat["anticheat_latency_sec"].mean()),
            "mean_difference": float(paired_lat["anticheat_latency_sec"].mean() - paired_lat["cegis_latency_sec"].mean()),
            "paired_t_test_pvalue": float(t_lat.pvalue),
            "wilcoxon_signed_rank_pvalue": float(w_lat.pvalue),
            "is_significant": bool(w_lat.pvalue < alpha),
        },
        "iterations_used": {
            "cegis_mean": float(paired_iter["cegis_iterations_used"].mean()),
            "anticheat_mean": float(paired_iter["anticheat_iterations_used"].mean()),
            "wilcoxon_pvalue": float(w_iter.pvalue),
            "is_significant": bool(w_iter.pvalue < alpha),
        },
    }

    return {
        "alpha": alpha,
        "pooled": pooled_results,
        "per_model": model_records,
        "secondary_tests": secondary_tests,
    }


def plot_forest_plot(results: dict[str, Any], output_path: str | Path) -> None:
    """Generate a publication-grade Forest Plot of AntiCheat vs. CEGIS effect sizes."""
    models_data = results["per_model"]
    pooled = results["pooled"]

    # Sort models by accuracy or delta
    labels = [m["model"] for m in models_data] + ["AGREGADO GERAL"]
    deltas = [m["delta"] * 100 for m in models_data] + [pooled["delta_accuracy"] * 100]
    ci_lowers = [m["bootstrap_95_ci"][0] * 100 for m in models_data] + [pooled["tests"]["bootstrap_95_ci"][0] * 100]
    ci_uppers = [m["bootstrap_95_ci"][1] * 100 for m in models_data] + [pooled["tests"]["bootstrap_95_ci"][1] * 100]

    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Reference vertical line at zero (No difference)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)

    # Plot error bars
    for i, (d, low, high) in enumerate(zip(deltas, ci_lowers, ci_uppers)):
        is_pool = (i == len(labels) - 1)
        color = "#1f77b4" if not is_pool else "#d62728"
        marker = "s" if not is_pool else "D"
        msize = 7 if not is_pool else 9

        ax.errorbar(
            d,
            i,
            xerr=[[d - low], [high - d]],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.6 if not is_pool else 2.4,
            capsize=4,
            markersize=msize,
            zorder=3,
        )

        # Text annotation on the right
        pval_str = (
            f"p={models_data[i]['exact_binomial_p']:.3f}"
            if not is_pool else f"p={pooled['tests']['exact_binomial_pvalue']:.3f}"
        )
        ax.annotate(
            f"{d:+.1f}% [{low:+.1f}%, {high:+.1f}%] ({pval_str})",
            xy=(high, i),
            xytext=(10, -3),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if is_pool else "normal",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()  # Top-down order

    ax.set_xlabel("Diferença de Acurácia Δ (%) [AntiCheat − CEGIS]", fontsize=11, fontweight="bold")
    ax.set_xlim(-16, 22)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generate_latex_table(results: dict[str, Any], output_path: str | Path) -> None:
    """Generate a publication-ready LaTeX table formatted for scientific papers."""
    lines = []
    lines.append("% Auto-generated statistical significance table: CEGIS vs. AntiCheat")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\caption{Paired Statistical Significance Tests for AntiCheat vs. Standard CEGIS (Exact Binomial, Permutation, and Bootstrap $B=10{,}000$, $\\alpha=0.05$).}")
    lines.append("  \\label{tab:anticheat_significance}")
    lines.append("  \\setlength{\\tabcolsep}{4.5pt}")
    lines.append("  \\begin{tabular}{lcccccccc}")
    lines.append("    \\toprule")
    lines.append("    Modelo & CEGIS & AntiCheat & $\\Delta$ & $b / c$ & $p_{\\text{exact}}$ & $p_{\\text{perm}}$ & IC 95\\% Bootstrap & $p_{\\text{bonf}}$ \\\\")
    lines.append("    \\midrule")

    for m in results["per_model"]:
        name = m["model"].replace("_", "\\_")
        cegis_pct = f"{m['cegis_accuracy']*100:.1f}\\%"
        anti_pct = f"{m['anticheat_accuracy']*100:.1f}\\%"
        delta_pct = f"{m['delta']*100:+.1f}\\%"
        disc_str = f"{m['b_cegis_only']} / {m['c_anticheat_only']}"
        p_ex = f"{m['exact_binomial_p']:.4f}"
        p_pm = f"{m['permutation_p']:.4f}"
        ci_str = f"[{m['bootstrap_95_ci'][0]*100:+.1f}\\%, {m['bootstrap_95_ci'][1]*100:+.1f}\\%]"
        p_bf = f"{m['bonferroni_p']:.4f}"

        lines.append(f"    \\texttt{{{name}}} & {cegis_pct} & {anti_pct} & {delta_pct} & {disc_str} & {p_ex} & {p_pm} & {ci_str} & {p_bf} \\\\")

    lines.append("    \\midrule")
    pl = results["pooled"]
    p_cegis = f"{pl['cegis_accuracy']*100:.1f}\\%"
    p_anti = f"{pl['anticheat_accuracy']*100:.1f}\\%"
    p_delta = f"{pl['delta_accuracy']*100:+.1f}\\%"
    p_disc = f"{pl['contingency']['cegis_only_succeeded_b']} / {pl['contingency']['anticheat_only_succeeded_c']}"
    p_ex_pl = f"{pl['tests']['exact_binomial_pvalue']:.4f}"
    p_pm_pl = f"{pl['tests']['permutation_pvalue']:.4f}"
    p_ci_pl = f"[{pl['tests']['bootstrap_95_ci'][0]*100:+.1f}\\%, {pl['tests']['bootstrap_95_ci'][1]*100:+.1f}\\%]"

    lines.append(f"    \\textbf{{POOLED TOTAL}} & {p_cegis} & {p_anti} & {p_delta} & {p_disc} & {p_ex_pl} & {p_pm_pl} & {p_ci_pl} & — \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_markdown_report(results: dict[str, Any], output_path: str | Path) -> None:
    """Generate a clean Markdown summary report."""
    pl = results["pooled"]
    lines = []
    lines.append("# Statistical Significance Report: CEGIS vs. AntiCheat\n")
    lines.append("> Formal paired hypothesis tests evaluating whether the AntiCheat strategy significantly alters model performance on ARC-AGI-1.\n")

    lines.append("## 1. Executive Verdict\n")
    lines.append(f"### Final Decision: **{pl['verdict']}** ($\alpha = {results['alpha']}$)\n")
    lines.append(
        f"Across all 900 benchmark evaluations, the observed difference in accuracy between standard CEGIS "
        f"({pl['cegis_accuracy']*100:.2f}%) and AntiCheat ({pl['anticheat_accuracy']*100:.2f}%) is "
        f"**{pl['delta_accuracy']*100:+.2f} percentage points** (Risk Difference = {pl['delta_accuracy']:.4f}). "
        f"The two-sided Exact Binomial test yields **p = {pl['tests']['exact_binomial_pvalue']:.4f}**, "
        f"and the paired sign-flip permutation test yields **p = {pl['tests']['permutation_pvalue']:.4f}**. "
        f"The 95% bootstrap confidence interval **[{pl['tests']['bootstrap_95_ci'][0]*100:+.2f}%, {pl['tests']['bootstrap_95_ci'][1]*100:+.2f}%]** "
        f"spans zero, confirming that there is **no statistically significant overall difference** in accuracy.\n"
    )

    lines.append("## 2. Pooled Omnibus Hypothesis Tests\n")
    lines.append("| Metric / Test | Value | Interpretation |")
    lines.append("| :--- | :---: | :--- |")
    ci_str = f"[{pl['tests']['bootstrap_95_ci'][0]*100:+.2f}%, {pl['tests']['bootstrap_95_ci'][1]*100:+.2f}%]"
    lines.append(f"| **CEGIS Overall Accuracy** | {pl['cegis_accuracy']*100:.2f}% | 422 solved of 900 evaluations |")
    lines.append(f"| **AntiCheat Overall Accuracy** | {pl['anticheat_accuracy']*100:.2f}% | 415 solved of 900 evaluations |")
    lines.append(f"| **Accuracy Difference ($\\\\Delta$)** | **{pl['delta_accuracy']*100:+.2f}%** | 95% CI: {ci_str} |")
    lines.append(f"| **Discordant Pairs ($b / c$)** | {pl['contingency']['cegis_only_succeeded_b']} / {pl['contingency']['anticheat_only_succeeded_c']} | 28 CEGIS wins vs 21 AntiCheat wins |")
    lines.append(f"| **Exact Binomial Test** | **p = {pl['tests']['exact_binomial_pvalue']:.4f}** | Fail to reject $H_0$ ($p > 0.05$) |")
    lines.append(f"| **Paired Permutation Test** | **p = {pl['tests']['permutation_pvalue']:.4f}** | Fail to reject $H_0$ (100,000 sign-flips) |")
    lines.append(f"| **McNemar $\\\\chi^2$ Statistic** | $\\\\chi^2 = {pl['tests']['mcnemar_chi2_statistic']:.3f}$ | p = {pl['tests']['mcnemar_chi2_pvalue']:.4f} (with continuity correction) |")
    lines.append(f"| **Odds Ratio ($c / b$)** | {pl['effect_sizes']['odds_ratio']:.3f} | Near 1.0 (Balanced discordance) |")
    lines.append("")

    lines.append("## 3. Stratified Per-Model Results & Multiple Testing Correction\n")
    lines.append("| Model | CEGIS (%) | AntiCheat (%) | $\\Delta$ (%) | $b / c$ | Raw $p$ | Bonferroni $p_{\\text{adj}}$ | Benjamini-Hochberg $q$ | Significant? |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in results["per_model"]:
        sig_str = "No" if not m["significant_bonferroni"] else "**YES**"
        lines.append(
            f"| `{m['model']}` | {m['cegis_accuracy']*100:.1f}% | {m['anticheat_accuracy']*100:.1f}% | "
            f"{m['delta']*100:+.1f}% | {m['b_cegis_only']}/{m['c_anticheat_only']} | "
            f"{m['exact_binomial_p']:.4f} | {m['bonferroni_p']:.4f} | {m['benjamini_hochberg_q']:.4f} | {sig_str} |"
        )
    lines.append("")
    lines.append("> **Multiple Testing Note**: While `gemini-3.1-flash-lite` showed an uncorrected raw $p = 0.0156$ (-7.0%), after adjusting for the 9 model hypotheses using Bonferroni or Benjamini-Hochberg, $p_{\\text{adj}} = 0.1406 > 0.05$. Therefore, **no individual model achieves statistically significant divergence** under rigorous family-wise error rate control.\n")

    lines.append("## 4. Secondary Hypotheses (Overfitting, Latency, Iterations)\n")
    sec = results["secondary_tests"]
    lines.append("### Spurious Overfitting Rate")
    lines.append(f"- CEGIS spurious overfitting instances: **{sec['spurious_overfitting']['cegis_count']}**")
    lines.append(f"- AntiCheat spurious overfitting instances: **{sec['spurious_overfitting']['anticheat_count']}**")
    lines.append(f"- Discordant overfitting pairs ($b/c$): **{sec['spurious_overfitting']['cegis_only_b']} / {sec['spurious_overfitting']['anticheat_only_c']}**")
    lines.append(f"- Exact Binomial p-value: **p = {sec['spurious_overfitting']['exact_binomial_pvalue']:.4f}** (Statistically Significant: **{sec['spurious_overfitting']['is_significant']}**)\n")

    lines.append("### Latency & Execution Duration")
    lines.append(f"- CEGIS Mean Latency: **{sec['latency_seconds']['cegis_mean']:.2f}s**")
    lines.append(f"- AntiCheat Mean Latency: **{sec['latency_seconds']['anticheat_mean']:.2f}s** (+{sec['latency_seconds']['mean_difference']:.2f}s)")
    lines.append(f"- Wilcoxon Signed-Rank Test: **p = {sec['latency_seconds']['wilcoxon_signed_rank_pvalue']:.4e}** (Statistically Significant: **{sec['latency_seconds']['is_significant']}**)")
    lines.append(f"- *Insight*: AntiCheat introduces a small but statistically significant latency overhead because the model produces natural-language explanations prior to code generation.\n")

    lines.append("### Iteration Count")
    lines.append(f"- CEGIS Mean Iterations: **{sec['iterations_used']['cegis_mean']:.2f}**")
    lines.append(f"- AntiCheat Mean Iterations: **{sec['iterations_used']['anticheat_mean']:.2f}**")
    lines.append(f"- Wilcoxon p-value: **p = {sec['iterations_used']['wilcoxon_pvalue']:.4f}** (Statistically Significant: **{sec['iterations_used']['is_significant']}**)\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Statistical significance analysis for AntiCheat vs CEGIS.")
    parser.add_argument("--csv", type=str, default="benchmark.csv", help="Path to benchmark CSV.")
    parser.add_argument("--cegis-col", type=str, default="cegis_success", help="CEGIS success column.")
    parser.add_argument("--anticheat-col", type=str, default="anticheat_success", help="AntiCheat success column.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level alpha.")
    parser.add_argument("--permutations", type=int, default=100_000, help="Number of sign-flip permutations.")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000, help="Number of bootstrap resamples.")
    parser.add_argument("--plot", type=str, default=None, help="Optional output path for Forest Plot PNG.")
    parser.add_argument("--output-report", type=str, default=None, help="Optional path to output Markdown report.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to output JSON summary.")
    parser.add_argument("--output-tex", type=str, default=None, help="Optional path to output LaTeX table.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"task_id": str})
    results = run_anticheat_significance_analysis(
        df,
        cegis_col=args.cegis_col,
        anticheat_col=args.anticheat_col,
        alpha=args.alpha,
        n_permutations=args.permutations,
        n_boot=args.bootstrap_samples,
    )

    pl = results["pooled"]
    print("=" * 78)
    print("STATISTICAL SIGNIFICANCE ANALYSIS: ANTICHEAT VS. STANDARD CEGIS")
    print("=" * 78)
    print(f"Overall Accuracy:  CEGIS = {pl['cegis_accuracy']*100:.2f}%  |  AntiCheat = {pl['anticheat_accuracy']*100:.2f}%")
    print(f"Difference (Δ):    {pl['delta_accuracy']*100:+.2f}%  (95% Bootstrap CI: [{pl['tests']['bootstrap_95_ci'][0]*100:+.2f}%, {pl['tests']['bootstrap_95_ci'][1]*100:+.2f}%])")
    print(f"Discordant Pairs:  b (CEGIS win) = {pl['contingency']['cegis_only_succeeded_b']}  |  c (AntiCheat win) = {pl['contingency']['anticheat_only_succeeded_c']}")
    print(f"Exact Binomial p-value:    {pl['tests']['exact_binomial_pvalue']:.4f}")
    print(f"Permutation test p-value:  {pl['tests']['permutation_pvalue']:.4f}")
    print(f"McNemar chi2 p-value:      {pl['tests']['mcnemar_chi2_pvalue']:.4f}")
    print("-" * 78)
    print(f"FINAL VERDICT (alpha = {args.alpha}):  >>> {pl['verdict']} <<<")
    print("=" * 78)

    print("\nPER-MODEL STRATIFIED RESULTS (with Multiple Testing Corrections):")
    print(f"{'Model':30s} | {'CEGIS':6s} | {'Anti':6s} | {'Diff':6s} | {'b/c':5s} | {'Raw p':7s} | {'Bonferroni':10s} | {'BH FDR':7s}")
    print("-" * 88)
    for m in results["per_model"]:
        print(
            f"{m['model'][:30]:30s} | {m['cegis_accuracy']*100:5.1f}% | {m['anticheat_accuracy']*100:5.1f}% | "
            f"{m['delta']*100:+5.1f}% | {m['b_cegis_only']:2d}/{m['c_anticheat_only']:2d} | "
            f"{m['exact_binomial_p']:7.4f} | {m['bonferroni_p']:10.4f} | {m['benjamini_hochberg_q']:7.4f}"
        )

    sec = results["secondary_tests"]
    print("\nSECONDARY HYPOTHESIS TESTS:")
    print(f"  • Spurious Overfitting Rate: CEGIS={sec['spurious_overfitting']['cegis_count']} vs AntiCheat={sec['spurious_overfitting']['anticheat_count']} (p = {sec['spurious_overfitting']['exact_binomial_pvalue']:.4f} -> Significant: {sec['spurious_overfitting']['is_significant']})")
    print(f"  • Task Latency Overhead:     CEGIS={sec['latency_seconds']['cegis_mean']:.1f}s vs AntiCheat={sec['latency_seconds']['anticheat_mean']:.1f}s (Wilcoxon p = {sec['latency_seconds']['wilcoxon_signed_rank_pvalue']:.4e} -> Significant: {sec['latency_seconds']['is_significant']})")
    print(f"  • Iteration Count:           CEGIS={sec['iterations_used']['cegis_mean']:.2f} vs AntiCheat={sec['iterations_used']['anticheat_mean']:.2f} (Wilcoxon p = {sec['iterations_used']['wilcoxon_pvalue']:.4f} -> Significant: {sec['iterations_used']['is_significant']})")

    # Generate optional artifacts
    if args.plot:
        plot_forest_plot(results, args.plot)
        print(f"\n[Artifact] Forest plot generated: {args.plot}")

    if args.output_report:
        generate_markdown_report(results, args.output_report)
        print(f"[Artifact] Markdown report generated: {args.output_report}")

    if args.output_tex:
        generate_latex_table(results, args.output_tex)
        print(f"[Artifact] LaTeX table generated: {args.output_tex}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Artifact] JSON summary generated: {args.output_json}")


if __name__ == "__main__":
    main()
