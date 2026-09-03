#!/usr/bin/env python3
"""Generate a markdown experiment summary matching the project's report style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _format_percent(value: float) -> str:
    return f"{value:.1f}%"


def _format_signed_percent(value: float) -> str:
    prefix = "+" if value >= 0 else ""
    return f"{prefix}{value:.1f}%"


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _load_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _task_count(payload: dict[str, Any]) -> int:
    config = payload.get("config", {})
    if isinstance(config, dict):
        for key in ("completed_tasks", "total_tasks_in_dataset"):
            value = config.get(key)
            if isinstance(value, int):
                return value
    results = payload.get("results")
    if isinstance(results, list):
        return len(results)
    return 0


def _strategy_requests(results: list[dict[str, Any]], strategy: str) -> int:
    total = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        strategy_result = item.get(strategy)
        if not isinstance(strategy_result, dict):
            continue
        history = strategy_result.get("iteration_history")
        if isinstance(history, list):
            total += len(history)
        else:
            total += 1
    return total


def _summary_metrics(payload: dict[str, Any]) -> tuple[float, float, float, int, int, int]:
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        baseline_acc = _as_float(summary.get("baseline_accuracy"))
        cegis_acc = _as_float(summary.get("cegis_accuracy"))
        anticheat_acc = _as_float(summary.get("cegis_anticheat_accuracy"))
        baseline_correct = int(summary.get("baseline_correct", 0) or 0)
        cegis_correct = int(summary.get("cegis_correct", 0) or 0)
        anticheat_correct = int(summary.get("cegis_anticheat_correct", 0) or 0)
        return baseline_acc, cegis_acc, anticheat_acc, baseline_correct, cegis_correct, anticheat_correct

    results = payload.get("results", [])
    if not isinstance(results, list):
        return 0.0, 0.0, 0.0, 0, 0, 0

    total = len(results)
    baseline_correct = sum(1 for item in results if isinstance(item, dict) and bool(item.get("baseline", {}).get("success")))
    cegis_correct = sum(1 for item in results if isinstance(item, dict) and bool(item.get("cegis", {}).get("success")))
    anticheat_correct = sum(1 for item in results if isinstance(item, dict) and bool(item.get("cegis_anticheat", {}).get("success")))
    return (
        _safe_divide(baseline_correct, total) * 100.0,
        _safe_divide(cegis_correct, total) * 100.0,
        _safe_divide(anticheat_correct, total) * 100.0,
        baseline_correct,
        cegis_correct,
        anticheat_correct,
    )


def _derived_findings(payload: dict[str, Any]) -> tuple[int, int, int, int]:
    results = payload.get("results")
    if not isinstance(results, list):
        return 0, 0, 0, 0

    semantic_recovery = 0
    regression = 0
    spurious_overfitting = 0
    representation_ceiling = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        baseline = item.get("baseline")
        cegis = item.get("cegis")
        if not isinstance(baseline, dict) or not isinstance(cegis, dict):
            continue

        baseline_success = bool(baseline.get("success"))
        cegis_success = bool(cegis.get("success"))
        converged_train = bool(cegis.get("converged_train"))

        if not baseline_success and cegis_success:
            semantic_recovery += 1
        if baseline_success and not cegis_success:
            regression += 1
        if converged_train and not cegis_success:
            spurious_overfitting += 1
        if not converged_train and not cegis_success:
            representation_ceiling += 1

    return semantic_recovery, regression, spurious_overfitting, representation_ceiling


def _default_output_path(input_path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    summary_dir = project_root / "experiments" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / f"{input_path.stem}.md"


def summarize_file(input_path: Path, output_path: Path | None = None) -> Path:
    payload = _load_results(input_path)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{input_path} does not contain a list of results.")

    task_count = _task_count(payload)
    config = payload.get("config", {})
    model_name = config.get("model") if isinstance(config, dict) else input_path.stem

    baseline_accuracy, cegis_accuracy, anticheat_accuracy, baseline_correct, cegis_correct, anticheat_correct = _summary_metrics(payload)
    baseline_requests = _strategy_requests(results, "baseline")
    cegis_requests = _strategy_requests(results, "cegis")
    anticheat_requests = _strategy_requests(results, "cegis_anticheat")

    if anticheat_requests > 0:
        primary_accuracy = anticheat_accuracy
        primary_correct = anticheat_correct
        primary_requests = anticheat_requests
        primary_label = "CEGIS AntiCheat"
    else:
        primary_accuracy = cegis_accuracy
        primary_correct = cegis_correct
        primary_requests = cegis_requests
        primary_label = "CEGIS (Semantic Feedback)"

    delta_accuracy = primary_accuracy - baseline_accuracy
    relative_delta = (_safe_divide(delta_accuracy, baseline_accuracy) * 100.0) if baseline_accuracy else 0.0
    delta_tasks = primary_correct - baseline_correct
    request_delta = primary_requests - baseline_requests
    avg_requests_baseline = _safe_divide(baseline_requests, task_count)
    avg_requests_primary = _safe_divide(primary_requests, task_count)
    avg_requests_delta = avg_requests_primary - avg_requests_baseline

    rows: list[str] = [
        f"| **Baseline (1-Shot)** | **{_format_percent(baseline_accuracy)}** | {baseline_correct} / {task_count} | {baseline_requests} | {avg_requests_baseline:.2f} |",
        f"| **CEGIS (Semantic Feedback)** | **{_format_percent(cegis_accuracy)}** | {cegis_correct} / {task_count} | {cegis_requests} | {_safe_divide(cegis_requests, task_count):.2f} |",
    ]
    if anticheat_requests > 0:
        rows.append(
            f"| **CEGIS AntiCheat** | **{_format_percent(anticheat_accuracy)}** | {anticheat_correct} / {task_count} | {anticheat_requests} | {_safe_divide(anticheat_requests, task_count):.2f} |"
        )

    delta_task_text = f"**{delta_tasks:+d} tasks**" if delta_tasks != 0 else "**0 tasks**"
    rows.append(
        f"| **Delta / Impact** | **{_format_signed_percent(delta_accuracy)}** *({_format_signed_percent(relative_delta)} rel.)* | {delta_task_text} | {request_delta:+d} | {avg_requests_delta:+.2f} |"
    )

    semantic_recovery, regression, spurious_overfitting, representation_ceiling = _derived_findings(payload)
    content = f"""## Experimental Results

A comparative evaluation was performed across `{task_count} tasks` using `{model_name}`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
{chr(10).join(rows)}

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** {semantic_recovery} task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** {regression} task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** {spurious_overfitting} task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** {representation_ceiling} task(s) failed without converging on training.
"""

    output_path = output_path or _default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown summaries from ARC experiment JSON files.")
    parser.add_argument("input", type=Path, help="Path to a results_experiment.json file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output markdown path.")
    args = parser.parse_args()

    output = summarize_file(args.input, args.output)
    print(f"Wrote summary to {output}")


if __name__ == "__main__":
    main()
