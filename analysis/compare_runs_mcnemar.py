"""Framework for paired comparison of two run outputs with McNemar's test."""

from __future__ import annotations

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


ALTERNATIVES = {"two-sided", "greater", "less"}


def _exact_mcnemar_p_value(a_only: int, b_only: int, alternative: str = "two-sided") -> float:
    """Exact p-value for McNemar's test based on discordant pairs."""
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of: {', '.join(sorted(ALTERNATIVES))}")

    discordant = a_only + b_only
    if discordant == 0:
        return 1.0

    if alternative == "greater":
        k = b_only
        return float(sum(math.comb(discordant, x) for x in range(k, discordant + 1)) / (2**discordant))

    if alternative == "less":
        k = b_only
        return float(sum(math.comb(discordant, x) for x in range(0, k + 1)) / (2**discordant))

    lower = sum(math.comb(discordant, x) for x in range(0, min(a_only, b_only) + 1)) / (2**discordant)
    return float(min(1.0, 2 * lower))


def _coerce_success_column(dataframe: pd.DataFrame, strategy: str, label: str) -> pd.DataFrame:
    column = f"{strategy}.success"
    required = {"task_id", column}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{label}: missing columns: {', '.join(sorted(missing))}")

    run = dataframe[["task_id", column]].copy()
    run = run.rename(columns={column: "success"})

    if run["task_id"].duplicated().any():
        raise ValueError(f"{label}: duplicate task_id values are not allowed")

    if run["success"].isna().any() or not run["success"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{label}: '{column}' must contain only boolean values")

    run["success"] = run["success"].astype(bool)
    return run


def _load_run(path: Path, strategy: str, label: str) -> pd.DataFrame:
    return _coerce_success_column(load_results(path), strategy, label)


def compare_runs(
    run_a_path: Path,
    run_b_path: Path,
    strategy_a: str,
    strategy_b: str,
    label_a: str,
    label_b: str,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    allow_partial_match: bool = False,
) -> dict[str, Any]:
    """Compare two paired runs and return McNemar statistics."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    run_a = _load_run(run_a_path, strategy_a, label_a)
    run_b = _load_run(run_b_path, strategy_b, label_b)

    paired = run_a.merge(run_b, on="task_id", how="inner", suffixes=("_a", "_b"))
    if paired.empty:
        raise ValueError("No overlapping task_id between runs")

    if not allow_partial_match:
        if len(paired) != len(run_a) or len(paired) != len(run_b):
            raise ValueError(
                "Runs do not have identical task_id sets; use --allow-partial-match to compare on the overlap"
            )

    success_a = paired["success_a"]
    success_b = paired["success_b"]

    a_only = int((success_a & ~success_b).sum())
    b_only = int((~success_a & success_b).sum())
    both_correct = int((success_a & success_b).sum())
    both_wrong = int((~success_a & ~success_b).sum())

    p_value = _exact_mcnemar_p_value(a_only, b_only, alternative=alternative)
    n = len(paired)
    accuracy_a = float(success_a.mean())
    accuracy_b = float(success_b.mean())

    result = {
        "n": n,
        "alpha": alpha,
        "alternative": alternative,
        "label_a": label_a,
        "label_b": label_b,
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "run_a_path": str(run_a_path),
        "run_b_path": str(run_b_path),
        "tasks_in_run_a": int(len(run_a)),
        "tasks_in_run_b": int(len(run_b)),
        "tasks_compared": int(n),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_only": a_only,
        "b_only": b_only,
        "discordant": a_only + b_only,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "accuracy_diff_b_minus_a": float(accuracy_b - accuracy_a),
        "mcnemar_p_value": p_value,
        "reject_h0": bool(p_value < alpha),
    }
    return result


def format_report(statistics: dict[str, Any]) -> str:
    """Format a compact text report."""
    return "\n".join(
        [
            f"Compared tasks: {statistics['tasks_compared']} ({statistics['label_a']} vs {statistics['label_b']})",
            f"Accuracy {statistics['label_a']}: {statistics['accuracy_a']:.2%}",
            f"Accuracy {statistics['label_b']}: {statistics['accuracy_b']:.2%}",
            f"Difference ({statistics['label_b']} - {statistics['label_a']}): {statistics['accuracy_diff_b_minus_a']:.2%}",
            "McNemar contingency:",
            f"  both_correct={statistics['both_correct']} both_wrong={statistics['both_wrong']}",
            f"  {statistics['label_a']}_only={statistics['a_only']} {statistics['label_b']}_only={statistics['b_only']}",
            f"  discordant={statistics['discordant']}",
            f"Exact McNemar p-value ({statistics['alternative']}): {statistics['mcnemar_p_value']:.6f}",
            f"Reject H0 at alpha={statistics['alpha']:.3f}: {statistics['reject_h0']}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two model/strategy runs on the same dataset using exact McNemar test."
    )
    parser.add_argument("run_a", type=Path, help="Path to first results_experiment JSON")
    parser.add_argument("run_b", type=Path, help="Path to second results_experiment JSON")
    parser.add_argument("--strategy-a", default="cegis", help="Strategy name in run_a (default: cegis)")
    parser.add_argument("--strategy-b", default=None, help="Strategy name in run_b (default: same as --strategy-a)")
    parser.add_argument("--label-a", default="run_a", help="Display label for run_a")
    parser.add_argument("--label-b", default="run_b", help="Display label for run_b")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--alternative",
        choices=sorted(ALTERNATIVES),
        default="two-sided",
        help="Hypothesis alternative for McNemar test",
    )
    parser.add_argument(
        "--allow-partial-match",
        action="store_true",
        help="Compare only overlapping task_ids if runs differ",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write full statistics JSON",
    )
    args = parser.parse_args()

    strategy_b = args.strategy_b or args.strategy_a
    stats = compare_runs(
        run_a_path=args.run_a,
        run_b_path=args.run_b,
        strategy_a=args.strategy_a,
        strategy_b=strategy_b,
        label_a=args.label_a,
        label_b=args.label_b,
        alpha=args.alpha,
        alternative=args.alternative,
        allow_partial_match=args.allow_partial_match,
    )

    print(format_report(stats))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(pd.Series(stats).to_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
