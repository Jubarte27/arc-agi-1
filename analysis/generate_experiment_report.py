"""Generate a concise Markdown report from an experiment results JSON file."""

import argparse
import json
from pathlib import Path
from typing import Any



def _as_bool(value: Any) -> bool:
    return value is True



def _split_success(result: dict[str, Any], split: str) -> bool:
    field = f"{split}_success"
    if field in result:
        return _as_bool(result[field])

    details = result.get(f"{split}_results", [])
    return bool(details) and all(_as_bool(item.get("is_correct")) for item in details)



def _strategy_stats(results: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    entries = [item.get(strategy) for item in results]
    entries = [entry for entry in entries if isinstance(entry, dict) and entry]
    successes = sum(_as_bool(entry.get("success")) for entry in entries)
    return {
        "count": len(entries),
        "correct": successes,
        "accuracy": successes / len(entries) if entries else 0.0,
    }



def _request_count(results: list[dict[str, Any]], config: dict[str, Any]) -> int | None:
    recorded = config.get("total_requests_used")
    if isinstance(recorded, int):
        return recorded

    baseline_calls = 0
    cegis_calls = 0
    found_cegis_history = False
    for item in results:
        baseline = item.get("baseline")
        cegis = item.get("cegis")
        if isinstance(baseline, dict) and baseline:
            baseline_calls += 1
        if isinstance(cegis, dict) and cegis:
            history = cegis.get("iteration_history")
            if isinstance(history, list):
                found_cegis_history = True
                cegis_calls += len(history)

    if found_cegis_history:
        return baseline_calls + cegis_calls
    return baseline_calls or None



def _percent(value: float) -> str:
    fraction = value / 100 if value > 1 else value
    return f"{fraction * 100:.1f}%"


def _fraction(value: Any) -> float:
    numeric_value = float(value)
    return numeric_value / 100 if numeric_value > 1 else numeric_value



def build_report(payload: dict[str, Any]) -> str:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("The JSON file must contain a 'results' list.")

    config = payload.get("config")
    config = config if isinstance(config, dict) else {}
    summary = payload.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    baseline = _strategy_stats(results, "baseline")
    cegis = _strategy_stats(results, "cegis")
    task_count = max(baseline["count"], cegis["count"])

    baseline_accuracy = _fraction(summary.get("baseline_accuracy", baseline["accuracy"]))
    cegis_accuracy = _fraction(summary.get("cegis_accuracy", cegis["accuracy"]))
    baseline_correct = summary.get("baseline_correct", baseline["correct"])
    cegis_correct = summary.get("cegis_correct", cegis["correct"])
    accuracy_delta = float(cegis_accuracy) - float(baseline_accuracy)
    relative_delta = (accuracy_delta / baseline_accuracy) if baseline_accuracy else 0.0
    requests = _request_count(results, config)
    baseline_requests = baseline["count"]
    cegis_requests = None
    if requests is not None:
        cegis_requests = requests - baseline_requests

    recovered = 0
    regressions = 0
    false_convergence = 0
    exhausted = 0
    for item in results:
        baseline_result = item.get("baseline", {})
        cegis_result = item.get("cegis", {})
        if not isinstance(baseline_result, dict) or not isinstance(cegis_result, dict):
            continue
        baseline_success = _as_bool(baseline_result.get("success"))
        cegis_success = _as_bool(cegis_result.get("success"))
        recovered += not baseline_success and cegis_success
        regressions += baseline_success and not cegis_success
        test_success = _split_success(cegis_result, "test")
        false_convergence += _as_bool(cegis_result.get("converged_train")) and not test_success
        exhausted += not _as_bool(cegis_result.get("converged_train")) and not cegis_success

    model = config.get("model", "unspecified model")
    title = f"{task_count} tasks" if task_count else "Experiment"
    request_cell = str(requests) if requests is not None else "n/a"
    cegis_request_cell = str(cegis_requests) if cegis_requests is not None else "n/a"
    incremental_requests = (
        cegis_requests - baseline_requests if cegis_requests is not None else None
    )
    incremental_avg = (
        incremental_requests / task_count
        if incremental_requests is not None and task_count
        else None
    )
    baseline_avg_str = f"{baseline_requests / task_count:.2f}" if task_count else "n/a"
    cegis_avg_str = (
        f"{cegis_requests / task_count:.2f}"
        if cegis_requests is not None and task_count
        else "n/a"
    )
    return f"""## Experimental Results

A comparative evaluation was performed across `{title}` using `{model}`.

### Summary Statistics

| Approach | Exact Accuracy | Solved Tasks | API Requests | Avg. Requests/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1-Shot)** | **{_percent(baseline_accuracy)}** | {baseline_correct} / {task_count} | {baseline_requests} | {baseline_avg_str} |
| **CEGIS (Semantic Feedback)** | **{_percent(cegis_accuracy)}** | {cegis_correct} / {task_count} | {cegis_request_cell} | {cegis_avg_str} |
| **Delta / Impact** | **{accuracy_delta * 100:+.1f}%** *({relative_delta * 100:+.1f}% rel.)* | **{int(cegis_correct) - int(baseline_correct):+d} tasks** | {incremental_requests if incremental_requests is not None else 'n/a'} | {incremental_avg if incremental_avg is not None else 'n/a'} |

### Derived Findings & Error Breakdown

1. **Semantic Recovery:** {recovered} task(s) where CEGIS succeeded after the baseline failed.
2. **Regression:** {regressions} task(s) where the baseline succeeded but CEGIS did not.
3. **Spurious Overfitting (False Convergence):** {false_convergence} task(s) converged on training but failed on the test split.
4. **Representation Ceiling:** {exhausted} task(s) failed without converging on training.

"""



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown experiment report from results_experiment.json."
    )
    parser.add_argument("input", type=Path, help="Path to results_experiment.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path (default: input path with .md suffix)",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as results_file:
        payload = json.load(results_file)
    report = build_report(payload)
    output = args.output or args.input.with_suffix(".md")
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
