#!/usr/bin/env python3
"""Generate a LaTeX table from one or more results_experiment.json files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse helpers from the markdown summary generator.
if __package__:
    from .generate_experiment_summary import (
        _load_results,
        _task_count,
        _summary_metrics,
        _strategy_requests,
        _safe_divide,
        _derived_findings,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_experiment_summary import (
        _load_results,
        _task_count,
        _summary_metrics,
        _strategy_requests,
        _safe_divide,
        _derived_findings,
    )


def _escape_latex(text: str) -> str:
    """Escape characters that are special in LaTeX."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}\\%"


def _fmt_signed_pct(value: float) -> str:
    prefix = "+" if value >= 0 else ""
    return f"{prefix}{value:.1f}\\%"


def generate_latex_table(input_path: Path) -> str:
    """Return a LaTeX table string for a single results_experiment.json."""
    payload = _load_results(input_path)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{input_path} does not contain a list of results.")

    task_count = _task_count(payload)
    config = payload.get("config", {})
    model_name = config.get("model", input_path.stem) if isinstance(config, dict) else input_path.stem

    (
        baseline_accuracy,
        cegis_accuracy,
        anticheat_accuracy,
        baseline_correct,
        cegis_correct,
        anticheat_correct,
    ) = _summary_metrics(payload)

    baseline_requests = _strategy_requests(results, "baseline")
    cegis_requests = _strategy_requests(results, "cegis")
    anticheat_requests = _strategy_requests(results, "cegis_anticheat")

    has_anticheat = anticheat_requests > 0

    if has_anticheat:
        primary_accuracy = anticheat_accuracy
        primary_correct = anticheat_correct
        primary_requests = anticheat_requests
    else:
        primary_accuracy = cegis_accuracy
        primary_correct = cegis_correct
        primary_requests = cegis_requests

    delta_accuracy = primary_accuracy - baseline_accuracy
    relative_delta = (_safe_divide(delta_accuracy, baseline_accuracy) * 100.0) if baseline_accuracy else 0.0
    delta_tasks = primary_correct - baseline_correct
    request_delta = primary_requests - baseline_requests
    avg_baseline = _safe_divide(baseline_requests, task_count)
    avg_cegis = _safe_divide(cegis_requests, task_count)
    avg_anticheat = _safe_divide(anticheat_requests, task_count) if has_anticheat else 0.0
    avg_primary = avg_anticheat if has_anticheat else avg_cegis
    avg_delta = avg_primary - avg_baseline

    semantic_recovery, regression, spurious_overfitting, representation_ceiling = _derived_findings(payload)

    # -- Build LaTeX --
    model_escaped = _escape_latex(model_name)

    data_rows: list[str] = []

    # Baseline row
    data_rows.append(
        f"    Baseline (1-Shot) & {_fmt_pct(baseline_accuracy)} & "
        f"{baseline_correct}/{task_count} & {baseline_requests} & {avg_baseline:.2f} \\\\"
    )

    # CEGIS row
    data_rows.append(
        f"    CEGIS & {_fmt_pct(cegis_accuracy)} & "
        f"{cegis_correct}/{task_count} & {cegis_requests} & {avg_cegis:.2f} \\\\"
    )

    # CEGIS AntiCheat row (optional)
    if has_anticheat:
        data_rows.append(
            f"    CEGIS AntiCheat & {_fmt_pct(anticheat_accuracy)} & "
            f"{anticheat_correct}/{task_count} & {anticheat_requests} & {avg_anticheat:.2f} \\\\"
        )

    # Delta row
    delta_task_text = f"{delta_tasks:+d}" if delta_tasks != 0 else "0"
    data_rows.append(
        f"    \\midrule\n"
        f"    $\\Delta$ / Impact & "
        f"{_fmt_signed_pct(delta_accuracy)} ({_fmt_signed_pct(relative_delta)} rel.) & "
        f"{delta_task_text} tasks & {request_delta:+d} & {avg_delta:+.2f} \\\\"
    )

    rows_block = "\n".join(data_rows)

    table = rf"""% Auto-generated from {input_path.name}
% Model: {model_escaped}, {task_count} tasks
\begin{{table}}[htbp]
  \centering
  \caption{{Experimental results for \texttt{{{model_escaped}}} ({task_count} tasks).}}
  \label{{tab:{model_name.replace(' ', '_').replace('.', '_')}}}
  \begin{{tabular}}{{lcccc}}
    \toprule
    Approach & Exact Accuracy & Solved Tasks & API Requests & Avg.\ Req./Task \\
    \midrule
{rows_block}
    \bottomrule
  \end{{tabular}}

  \vspace{{0.5em}}
  \footnotesize
  \begin{{tabular}}{{rl}}
    Semantic Recovery:        & {semantic_recovery} task(s) \\
    Regression:               & {regression} task(s) \\
    Spurious Overfitting:     & {spurious_overfitting} task(s) \\
    Representation Ceiling:   & {representation_ceiling} task(s) \\
  \end{{tabular}}
\end{{table}}
"""
    return table


def _extract_row_data(input_path: Path) -> dict:
    """Extract summary data from a single results file for the combined table."""
    payload = _load_results(input_path)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{input_path} does not contain a list of results.")

    task_count = _task_count(payload)
    config = payload.get("config", {})
    model_name = config.get("model", input_path.stem) if isinstance(config, dict) else input_path.stem

    (
        baseline_accuracy,
        cegis_accuracy,
        anticheat_accuracy,
        baseline_correct,
        cegis_correct,
        anticheat_correct,
    ) = _summary_metrics(payload)

    baseline_requests = _strategy_requests(results, "baseline")
    cegis_requests = _strategy_requests(results, "cegis")
    anticheat_requests = _strategy_requests(results, "cegis_anticheat")

    has_anticheat = anticheat_requests > 0

    if has_anticheat:
        primary_accuracy = anticheat_accuracy
        primary_correct = anticheat_correct
        primary_requests = anticheat_requests
    else:
        primary_accuracy = cegis_accuracy
        primary_correct = cegis_correct
        primary_requests = cegis_requests

    delta_accuracy = primary_accuracy - baseline_accuracy

    semantic_recovery, regression, spurious_overfitting, representation_ceiling = _derived_findings(payload)

    return {
        "model": model_name,
        "tasks": task_count,
        "baseline_acc": baseline_accuracy,
        "cegis_acc": cegis_accuracy,
        "anticheat_acc": anticheat_accuracy,
        "baseline_correct": baseline_correct,
        "cegis_correct": cegis_correct,
        "anticheat_correct": anticheat_correct,
        "has_anticheat": has_anticheat,
        "primary_acc": primary_accuracy,
        "primary_correct": primary_correct,
        "delta_acc": delta_accuracy,
        "baseline_requests": baseline_requests,
        "cegis_requests": cegis_requests,
        "anticheat_requests": anticheat_requests,
        "primary_requests": primary_requests,
        "avg_req": _safe_divide(primary_requests, task_count),
        "semantic_recovery": semantic_recovery,
        "regression": regression,
        "spurious_overfitting": spurious_overfitting,
        "representation_ceiling": representation_ceiling,
    }


def generate_combined_latex_table(
    input_paths: list[Path],
    labels: list[str] | None = None,
) -> str:
    """Return a single LaTeX table with one row per results file.

    Parameters
    ----------
    labels : list[str] | None
        Optional display names for each input file (positional).
        Entries that are empty strings fall back to the model name from the JSON.
    """
    rows_data = [_extract_row_data(p) for p in input_paths]

    # Apply label overrides
    if labels:
        for i, label in enumerate(labels):
            if i < len(rows_data) and label:
                rows_data[i]["label"] = label
    for r in rows_data:
        r.setdefault("label", r["model"])

    # Sort by CEGIS accuracy descending
    rows_data.sort(key=lambda r: r["primary_acc"], reverse=True)

    any_anticheat = any(r["has_anticheat"] for r in rows_data)

    # Build rows
    data_rows: list[str] = []
    for r in rows_data:
        display = _escape_latex(r["label"])
        n = r["tasks"]

        if any_anticheat:
            ac_cell = f" & {_fmt_pct(r['anticheat_acc'])}" if r["has_anticheat"] else " & ---"
        else:
            ac_cell = ""

        data_rows.append(
            f"    \\texttt{{{display}}} & {n} "
            f"& {_fmt_pct(r['baseline_acc'])} "
            f"& {_fmt_pct(r['cegis_acc'])}"
            f"{ac_cell} "
            f"& {_fmt_signed_pct(r['delta_acc'])} "
            f"& {r['primary_correct']}/{n} "
            f"& {r['avg_req']:.2f} "
            f"& {r['semantic_recovery']} "
            f"& {r['regression']} \\\\"
        )

    rows_block = "\n".join(data_rows)

    ac_header = " & AntiCheat" if any_anticheat else ""
    ac_col = "c" if any_anticheat else ""

    table = rf"""% Auto-generated combined summary table
\begin{{table}}[htbp]
  \centering
  \caption{{Combined experimental results across models.}}
  \label{{tab:combined_results}}
  \begin{{tabular}}{{lc ccc{ac_col} cccc}}
    \toprule
    Model & Tasks & Baseline & CEGIS{ac_header} & $\Delta$ & Solved & Avg.\ Req./Task & Recovery & Regression \\
    \midrule
{rows_block}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX summary tables from ARC experiment JSON files."
    )
    parser.add_argument("input", type=Path, nargs="+", help="Path(s) to results_experiment.json files.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .tex file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Generate a single table with one row per input file instead of separate tables.",
    )
    parser.add_argument(
        "--label", action="append", default=[], metavar="NAME",
        help="Display label for the corresponding input file (positional). "
             "Can be repeated, e.g. --label 'Flash Lite' --label 'Medium'. "
             "Use an empty string '' to keep the default model name.",
    )
    args = parser.parse_args()

    if args.combined:
        output_text = generate_combined_latex_table(args.input, args.label or None)
    else:
        sections: list[str] = []
        for path in args.input:
            sections.append(generate_latex_table(path))
        output_text = "\n".join(sections)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Wrote LaTeX table(s) to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
