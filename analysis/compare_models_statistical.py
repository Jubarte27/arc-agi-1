#!/usr/bin/env python3
"""Generate a statistical comparison report across multiple model result files."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

if __package__:
    from .load_results_experiment import load_results
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.load_results_experiment import load_results


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError("Success values must be booleans.")


def _load_label(path: Path, default: str) -> str:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return default
    config = payload.get("config")
    if isinstance(config, dict):
        model = config.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return default


def _load_model_frame(path: Path, strategy: str, label: str) -> pd.DataFrame:
    dataframe = load_results(path)
    strategy_column = f"{strategy}.success"
    required_columns = {"task_id", strategy_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} missing required columns: {missing}")

    model_frame = dataframe[["task_id", strategy_column]].copy()
    if model_frame["task_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated task_id values.")
    model_frame[strategy_column] = model_frame[strategy_column].map(_as_bool)
    model_frame = model_frame.rename(columns={strategy_column: label}).set_index("task_id")
    return model_frame


def _to_markdown_table(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No data available._"
    columns = [str(column) for column in dataframe.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in dataframe.iterrows():
        values = [str(row[column]) for column in dataframe.columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def compare_models(
    input_paths: list[Path],
    strategy: str,
    alpha: float,
    correction: str,
    labels: list[str] | None = None,
) -> str:
    if len(input_paths) < 2:
        raise ValueError("Provide at least two results files to compare models.")

    frames: list[pd.DataFrame] = []
    resolved_labels: list[str] = []
    for index, path in enumerate(input_paths):
        cli_label = labels[index].strip() if labels and index < len(labels) and labels[index] else ""
        default_label = path.stem
        auto_label = _load_label(path, default_label)
        label = cli_label or auto_label
        if label in resolved_labels:
            label = f"{label} ({index + 1})"
        resolved_labels.append(label)
        frames.append(_load_model_frame(path, strategy=strategy, label=label))

    common_tasks = set(frames[0].index)
    for frame in frames[1:]:
        common_tasks &= set(frame.index)
    if not common_tasks:
        raise ValueError("No overlapping task_id values found across all files.")

    ordered_tasks = sorted(common_tasks)
    aligned = [frame.loc[ordered_tasks] for frame in frames]
    outcome_matrix = pd.concat(aligned, axis=1)

    n_tasks = len(outcome_matrix)
    performance_rows: list[dict[str, Any]] = []
    for label in resolved_labels:
        solved = int(outcome_matrix[label].sum())
        accuracy = solved / n_tasks
        ci_low, ci_high = proportion_confint(count=solved, nobs=n_tasks, alpha=alpha, method="wilson")
        performance_rows.append(
            {
                "Model": label,
                "Solved": solved,
                "Tasks": n_tasks,
                "Accuracy": accuracy,
                f"{int((1 - alpha) * 100)}% CI (Wilson)": f"[{ci_low:.3f}, {ci_high:.3f}]",
            }
        )

    performance = pd.DataFrame(performance_rows).sort_values(by="Accuracy", ascending=False)
    performance.insert(0, "Rank", range(1, len(performance) + 1))
    performance["Accuracy"] = performance["Accuracy"].map(lambda value: f"{value:.3f}")

    pairwise_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(resolved_labels, 2):
        left_success = outcome_matrix[left]
        right_success = outcome_matrix[right]
        left_only = int((left_success & ~right_success).sum())
        right_only = int((~left_success & right_success).sum())
        both_correct = int((left_success & right_success).sum())
        both_wrong = int((~left_success & ~right_success).sum())
        table = [[both_correct, left_only], [right_only, both_wrong]]
        p_value = float(mcnemar(table, exact=True, correction=False).pvalue)
        accuracy_delta = left_success.mean() - right_success.mean()
        pairwise_rows.append(
            {
                "Model A": left,
                "Model B": right,
                "Accuracy Δ (A-B)": accuracy_delta,
                "A only correct": left_only,
                "B only correct": right_only,
                "McNemar p-value": p_value,
            }
        )

    pairwise = pd.DataFrame(pairwise_rows)
    if not pairwise.empty:
        rejected, adjusted_p_values, _, _ = multipletests(
            pairwise["McNemar p-value"].to_numpy(),
            alpha=alpha,
            method=correction,
        )
        pairwise["Adjusted p-value"] = adjusted_p_values
        pairwise["Significant"] = rejected
        pairwise["Accuracy Δ (A-B)"] = pairwise["Accuracy Δ (A-B)"].map(lambda value: f"{value:+.3f}")
        pairwise["McNemar p-value"] = pairwise["McNemar p-value"].map(lambda value: f"{value:.4g}")
        pairwise["Adjusted p-value"] = pairwise["Adjusted p-value"].map(lambda value: f"{value:.4g}")

    coverage_rows = []
    for label, frame in zip(resolved_labels, frames):
        coverage_rows.append(
            {
                "Model": label,
                "Tasks in file": int(frame.shape[0]),
                "Tasks used in comparison": n_tasks,
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    report = [
        "# Statistical Comparison Across Models",
        "",
        f"- Strategy compared: `{strategy}`",
        f"- Files compared: {len(input_paths)}",
        f"- Overlapping tasks used: {n_tasks}",
        f"- Significance level (alpha): {alpha}",
        f"- Multiple-testing correction: `{correction}`",
        "",
        "## Coverage",
        "",
        _to_markdown_table(coverage),
        "",
        "## Performance Summary",
        "",
        _to_markdown_table(performance),
        "",
        "## Pairwise McNemar Tests",
        "",
        _to_markdown_table(pairwise),
        "",
    ]
    return "\n".join(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate statistical comparison reports across model results files."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Paths to results_experiment JSON files from different models.",
    )
    parser.add_argument(
        "--strategy",
        default="cegis",
        choices=["baseline", "cegis", "cegis_anticheat"],
        help="Result strategy success field to compare.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold used in confidence intervals and tests.",
    )
    parser.add_argument(
        "--correction",
        default="holm",
        choices=["bonferroni", "holm", "fdr_bh"],
        help="Multiple-testing correction method for pairwise p-values.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="NAME",
        help="Optional label for each input file (repeat in the same order as inputs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path; prints to stdout when omitted.",
    )
    args = parser.parse_args()

    report = compare_models(
        input_paths=args.input,
        strategy=args.strategy,
        alpha=args.alpha,
        correction=args.correction,
        labels=args.label,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
