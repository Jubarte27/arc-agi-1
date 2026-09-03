"""Statistical analysis and plots for baseline versus CEGIS results."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import PathLike
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

StrPath = str | PathLike

if __package__:
    from .load_results_experiment import load_results
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.load_results_experiment import load_results


def _paired_permutation_p_value(
    baseline: np.ndarray, cegis: np.ndarray, permutations: int = 10_000, seed: int = 42
) -> float:
    """
    Return the one-sided p-value from a paired sign-flip permutation test.

    Null hypothesis: there is no difference in accuracy between baseline and CEGIS
    (i.e., the assignment of outcomes to methods is exchangeable within each task).

    Only discordant pairs (where the two methods disagree) contribute variance.
    For each permutation their signs are independently flipped with probability 0.5,
    simulating random exchangeability under H0.  The one-sided p-value is the
    fraction of null mean-differences >= the observed mean-difference.
    """
    differences = cegis.astype(int) - baseline.astype(int)
    observed = differences.mean()

    discordant = differences[differences != 0].astype(float)
    n_discordant = len(discordant)
    if n_discordant == 0:
        return 1.0

    n_total = len(differences)
    rng = np.random.default_rng(seed)
    # Random sign matrix: (permutations × n_discordant), values ±1
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, n_discordant))
    null_means = (signs * discordant).sum(axis=1) / n_total

    return float((np.sum(null_means >= observed) + 1) / (permutations + 1))



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
    permutation_samples: int = 10_000,
    seed: int = 42,
    strategy: str = "cegis",
) -> dict[str, Any]:
    """Calculate a paired sign-flip permutation test and a directional bootstrap test on paired binary outcomes."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    if permutation_samples < 1:
        raise ValueError("permutation_samples must be positive.")

    paired = _validate_results(dataframe, strategy=strategy)
    if paired.empty:
        raise ValueError("The results file must contain at least one task.")
    baseline = paired["baseline.success"].to_numpy(dtype=bool)
    cegis = paired["cegis.success"].to_numpy(dtype=bool)
    differences = cegis.astype(int) - baseline.astype(int)
    baseline_only = int(((baseline == 1) & (cegis == 0)).sum())
    cegis_only = int(((baseline == 0) & (cegis == 1)).sum())
    sample_size = len(paired)

    permutation_p_value = _paired_permutation_p_value(
        baseline, cegis, permutations=permutation_samples, seed=seed
    )

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
        "permutation_p_value": permutation_p_value,
        "bootstrap_better_p_value": bootstrap_p_value,
        "alpha": alpha,
        "reject_difference": permutation_p_value < alpha,
        "reject_cegis_better": observed_difference > 0 and bootstrap_p_value < alpha,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
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

    ci_low = statistics.get("bootstrap_ci_low", 0.0)
    ci_high = statistics.get("bootstrap_ci_high", 0.0)
    baseline_acc = statistics["baseline_accuracy"]

    fig, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(labels, accuracies, color=["#4267ac", "#d97736"], width=0.55)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy by method")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar, accuracy in zip(bars, accuracies):
        axis.text(bar.get_x() + bar.get_width() / 2, accuracy + 0.03, f"{accuracy:.1%}", ha="center")

    # 95% bootstrap CI for the accuracy difference, shown as full-width lines
    ci_high_color = "red"
    ci_low_color = "red"
    axis.axhline(baseline_acc + ci_low, linestyle="--", color=ci_low_color, linewidth=3, zorder=1,
                 label=f"CI low ({ci_low:+.1%})")
    axis.axhline(baseline_acc + ci_high, linestyle="--", color=ci_high_color, linewidth=3, zorder=1,
                 label=f"CI high ({ci_high:+.1%})")
    axis.legend(fontsize=8, loc="upper left")

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
    }
    strategy_label = labels_map.get(statistics.get("strategy", ""), statistics.get("strategy", "CEGIS"))
    difference_result = "rejeitada" if statistics["reject_difference"] else "não rejeitada"
    better_result = "rejeitada" if statistics["reject_cegis_better"] else "não rejeitada"
    report = f"""# Baseline vs. {strategy_label}

Análise pareada de **{statistics['n']} tarefas**, com α = {statistics['alpha']:.2f}.

- Baseline: {statistics['baseline_accuracy']:.1%} ({statistics['baseline_correct']}/{statistics['n']})
- {strategy_label}: {statistics['cegis_accuracy']:.1%} ({statistics['cegis_correct']}/{statistics['n']})
- Diferença {strategy_label} − Baseline: {statistics['accuracy_difference']:.1%}
- Permutação pareada unilateral (H0: sem diferença de acurácia): p = {statistics['permutation_p_value']:.4f}; H0 **{difference_result}**
- Bootstrap pareado unilateral ({strategy_label} melhor): p = {statistics['bootstrap_better_p_value']:.4f}; H0 **{better_result}**
- IC bootstrap de 95% para a diferença: [{statistics['bootstrap_ci_low']:.1%}, {statistics['bootstrap_ci_high']:.1%}]

## Plots

![Accuracy comparison]({plot_paths[0]})

![Paired outcomes]({plot_paths[1]})
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def generate_report_for_path(
    result_path: StrPath,
    output_dir: StrPath,
    *,
    alpha: float = 0.05,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 10_000,
    seed: int = 42,
    strategy: str = "cegis",
) -> Path:
    """Generate a report for a single results.json file and save it under output_dir."""
    result_path = Path(result_path)
    output_dir = Path(output_dir)

    dataframe = load_results(result_path)
    statistics = calculate_significance(
        dataframe,
        alpha,
        bootstrap_samples,
        permutation_samples,
        seed,
        strategy=strategy,
    )

    report_dir = output_dir/dataframe.attrs["metadata"]["config"]["model"]/strategy/result_path.stem
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = generate_plots(
        _validate_results(dataframe, strategy=strategy),
        statistics,
        report_dir / "plots",
    )
    write_report(statistics, plot_paths, report_dir)
    return report_dir


def generate_reports_for_paths(
    result_paths: list[StrPath],
    output_dir: StrPath,
    *,
    alpha: float = 0.05,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 10_000,
    seed: int = 42,
    strategy: str = "cegis",
) -> list[Path]:
    """Generate one report per result file under a shared output directory.

    Each file is processed in a separate thread for faster throughput.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=seed,
        strategy=strategy,
    )

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(generate_report_for_path, path, output_dir, **kwargs): i
            for i, path in enumerate(result_paths)
        }

        report_dirs: list[Path | None] = [None] * len(result_paths)
        for future in as_completed(futures):
            idx = futures[future]
            report_dirs[idx] = future.result()

    return report_dirs  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze baseline versus CEGIS results.")
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--strategy",
        type=str,
        default="cegis",
        choices=["cegis", "cegis_anticheat"],
        help="Comparison strategy (default: cegis)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_reports_for_paths(
        args.input,
        args.output,
        alpha=args.alpha,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()