"""Statistical analysis and plots for baseline versus CEGIS results."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__:
    from .load_results_experiment import load_results
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.load_results_experiment import load_results


def _exact_mcnemar_p_value(baseline_only: int, cegis_only: int) -> float:
    """Return the two-sided exact McNemar p-value for any difference."""
    discordant = baseline_only + cegis_only
    if discordant == 0:
        return 1.0

    lower_tail = sum(
        math.comb(discordant, successes)
        for successes in range(min(baseline_only, cegis_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def _validate_results(dataframe: pd.DataFrame, strategy: str = "cegis") -> pd.DataFrame:
    strategy_col = f"{strategy}.success"
    required = {"task_id", "baseline.success", strategy_col}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if dataframe["task_id"].duplicated().any():
        raise ValueError("Each task_id must occur at most once.")

    paired = dataframe[["task_id", "baseline.success", strategy_col]].copy()
    if strategy != "cegis":
        paired = paired.rename(columns={strategy_col: "cegis.success"})
    for column in ("baseline.success", "cegis.success"):
        values = paired[column]
        if values.isna().any() or not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"Column '{column}' must contain only boolean values.")
        paired[column] = values.astype(bool)
    return paired


def calculate_significance(
    dataframe: pd.DataFrame,
    alpha: float = 0.05,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    strategy: str = "cegis",
) -> dict[str, Any]:
    """Calculate a two-sided McNemar and a directional bootstrap test."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")

    paired = _validate_results(dataframe, strategy=strategy)
    if paired.empty:
        raise ValueError("The results file must contain at least one task.")
    baseline = paired["baseline.success"].to_numpy(dtype=bool)
    cegis = paired["cegis.success"].to_numpy(dtype=bool)
    differences = cegis.astype(int) - baseline.astype(int)
    baseline_only = int(((baseline == 1) & (cegis == 0)).sum())
    cegis_only = int(((baseline == 0) & (cegis == 1)).sum())
    sample_size = len(paired)

    rng = np.random.default_rng(seed)
    bootstrap = differences[
        rng.integers(0, sample_size, size=(bootstrap_samples, sample_size))
    ].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
    observed_difference = float(differences.mean())
    null_bootstrap = bootstrap - observed_difference
    bootstrap_p_value = float(
        (np.count_nonzero(null_bootstrap >= observed_difference) + 1)
        / (bootstrap_samples + 1)
    )
    mcnemar_p_value = _exact_mcnemar_p_value(baseline_only, cegis_only)

    return {
        "n": sample_size,
        "strategy": strategy,
        "baseline_correct": int(baseline.sum()),
        "cegis_correct": int(cegis.sum()),
        "baseline_accuracy": float(baseline.mean()),
        "cegis_accuracy": float(cegis.mean()),
        "accuracy_difference": float(differences.mean()),
        "baseline_only": baseline_only,
        "cegis_only": cegis_only,
        "same_outcome": int((baseline == cegis).sum()),
        "mcnemar_p_value": mcnemar_p_value,
        "bootstrap_better_p_value": bootstrap_p_value,
        "alpha": alpha,
        "reject_difference": mcnemar_p_value < alpha,
        "reject_cegis_better": observed_difference > 0 and bootstrap_p_value < alpha,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_samples": bootstrap_samples,
    }


def generate_plots(
    dataframe: pd.DataFrame, statistics: dict[str, Any], plots_dir: Path
) -> list[str]:
    """Generate accuracy and paired-outcome plots and return their filenames."""
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    strategy_label = "CEGIS AntiCheat" if statistics.get("strategy") == "cegis_anticheat" else "CEGIS"
    labels = ["Baseline", strategy_label]
    accuracies = [statistics["baseline_accuracy"], statistics["cegis_accuracy"]]

    fig, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(labels, accuracies, color=["#4267ac", "#d97736"], width=0.55)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy by method")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar, accuracy in zip(bars, accuracies):
        axis.text(bar.get_x() + bar.get_width() / 2, accuracy + 0.03, f"{accuracy:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_comparison.png", dpi=160)
    plt.close(fig)

    outcomes = [
        "Both correct",
        "Both wrong",
        "Baseline only",
        f"{strategy_label} only",
    ]
    counts = [
        int(((dataframe["baseline.success"]) & (dataframe["cegis.success"])).sum()),
        int((~dataframe["baseline.success"] & ~dataframe["cegis.success"]).sum()),
        statistics["baseline_only"],
        statistics["cegis_only"],
    ]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(outcomes, counts, color=["#4c956c", "#9aa0a6", "#4267ac", "#d97736"])
    axis.set_ylabel("Number of tasks")
    axis.set_title("Paired outcomes by task")
    axis.tick_params(axis="x", rotation=15)
    for index, count in enumerate(counts):
        axis.text(index, count + max(counts + [1]) * 0.02, str(count), ha="center")
    fig.tight_layout()
    fig.savefig(plots_dir / "paired_outcomes.png", dpi=160)
    plt.close(fig)
    return ["plots/accuracy_comparison.png", "plots/paired_outcomes.png"]


def write_report(statistics: dict[str, Any], plot_paths: list[str], output_dir: Path) -> None:
    """Write a concise Markdown report linking generated plots."""
    labels_map = {
        "cegis": "CEGIS",
        "cegis_anticheat": "CEGIS AntiCheat",
        "cegis_geometric_instructions": "CEGIS Geometric",
        "cegis_explain_yourself": "CEGIS Explain",
    }
    strategy_label = labels_map.get(statistics.get("strategy", ""), statistics.get("strategy", "CEGIS"))
    difference_result = "rejeitada" if statistics["reject_difference"] else "não rejeitada"
    better_result = "rejeitada" if statistics["reject_cegis_better"] else "não rejeitada"
    report = f"""# Baseline vs. {strategy_label}

Análise pareada de **{statistics['n']} tarefas**, com α = {statistics['alpha']:.2f}.

- Baseline: {statistics['baseline_accuracy']:.1%} ({statistics['baseline_correct']}/{statistics['n']})
- {strategy_label}: {statistics['cegis_accuracy']:.1%} ({statistics['cegis_correct']}/{statistics['n']})
- Diferença {strategy_label} − Baseline: {statistics['accuracy_difference']:.1%}
- Pares discordantes: baseline apenas = {statistics['baseline_only']}; {strategy_label} apenas = {statistics['cegis_only']}
- McNemar exato bilateral (diferença entre métodos): p = {statistics['mcnemar_p_value']:.4f}; H0 **{difference_result}**
- Bootstrap pareado unilateral ({strategy_label} melhor): p = {statistics['bootstrap_better_p_value']:.4f}; H0 **{better_result}**
- IC bootstrap de 95% para a diferença: [{statistics['bootstrap_ci_low']:.1%}, {statistics['bootstrap_ci_high']:.1%}]

## Plots

![Accuracy comparison]({plot_paths[0]})

![Paired outcomes]({plot_paths[1]})
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze baseline versus CEGIS results.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--strategy",
        type=str,
        default="cegis",
        choices=["cegis", "cegis_anticheat", "cegis_geometric_instructions", "cegis_explain_yourself"],
        help="Comparison strategy (default: cegis)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataframe = load_results(args.input)
    statistics = calculate_significance(dataframe, args.alpha, args.bootstrap_samples, args.seed, strategy=args.strategy)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_paths = generate_plots(
        _validate_results(dataframe, strategy=args.strategy), statistics, args.output / "plots"
    )
    write_report(statistics, plot_paths, args.output)
    print(pd.Series(statistics).to_string())


if __name__ == "__main__":
    main()