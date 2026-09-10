#!/usr/bin/env python3
"""Compile ARC-AGI experiment results.json files into a single CSV spreadsheet.

Supports two primary compilation modes:
  - 'summary' (default): One row per experiment / model run, containing benchmark
    accuracies, solved task counts, delta improvement, request counts, latencies,
    and CEGIS error diagnostics (semantic recovery, spurious overfitting, etc.).
  - 'tasks': One row per individual task evaluation across all selected experiment
    files, containing per-task outcomes, latencies, iterations, and failure modes.
  - 'both': Generates both summary and task-level CSV files in a single run.

Usage examples:
  # 1. Compile all experiments in ./experiments into a summary CSV
  python analysis/compile_results_to_csv.py -o summary.csv

  # 2. Compile all task-level results across all models into a single CSV
  python analysis/compile_results_to_csv.py --tasks -o all_tasks.csv

  # 3. Generate both summary and tasks CSV files
  python analysis/compile_results_to_csv.py --both -o benchmark_results.csv
  # -> produces benchmark_results_summary.csv and benchmark_results_tasks.csv

  # 4. Compile specific results.json files or wildcards
  python analysis/compile_results_to_csv.py experiments/google/*/results_experiment.json -o google_summary.csv

  # 5. Output CSV to stdout (pipe to other tools)
  python analysis/compile_results_to_csv.py experiments/google/gemini-3.1-flash-lite/results_experiment.json

  # 6. Include full generated Python code in task mode
  python analysis/compile_results_to_csv.py --tasks --include-code -o tasks_with_code.csv

  # 7. Write only the code to a separate JSON file (minified)
  python analysis/compile_results_to_csv.py --code-json code.json

  # 8. Output compact CSV without code, while saving all code in a separate JSON file
  python analysis/compile_results_to_csv.py --tasks -o tasks.csv --code-json code.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import io
import json
from pathlib import Path
import re
import sys
import tokenize
from typing import Any, Sequence
import warnings

# Regex patterns for identifying backup or intermediate checkpoint files
CHECKPOINT_REGEX = re.compile(r"_\d+_\d+\.json$")
BACKUP_REGEX = re.compile(r"\.(emergency|old)\.json$")


class _CodeMinifier(ast.NodeTransformer):
    """AST transformer that removes standalone string expressions (docstrings/comments) and type annotations."""

    def _strip_string_exprs(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        new_stmts = []
        for stmt in stmts:
            # Drop standalone string expressions (module/function/class docstrings, explanation blocks)
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                continue
            new_stmts.append(stmt)
        return new_stmts if new_stmts else [ast.Pass()]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        for arg in getattr(node.args, "posonlyargs", []):
            arg.annotation = None
        for arg in getattr(node.args, "kwonlyargs", []):
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        for arg in getattr(node.args, "posonlyargs", []):
            arg.annotation = None
        for arg in getattr(node.args, "kwonlyargs", []):
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if node.value is not None:
            return ast.Assign(targets=[node.target], value=node.value)
        return None

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        for attr in ("body", "orelse", "finalbody"):
            if hasattr(node, attr):
                val = getattr(node, attr)
                if isinstance(val, list) and val:
                    setattr(node, attr, self._strip_string_exprs(val))
        return node


def minify_python_code(code: str) -> str:
    """Minify a Python code snippet by stripping comments, docstrings, and blank lines.

    Uses AST unparsing as the primary method, with tokenize and regex fallbacks
    for code snippets containing syntax irregularities produced by LLMs.
    """
    if not code or not code.strip():
        return ""

    text = code.strip()

    # Strip markdown code fences if present (e.g. ```python ... ```)
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1. Primary: AST-based minification (removes all comments and docstrings cleanly)
        try:
            tree = ast.parse(text)
            _CodeMinifier().visit(tree)
            ast.fix_missing_locations(tree)
            return ast.unparse(tree).strip()
        except Exception:
            pass

        # 2. Secondary fallback: Tokenize-based minification
        try:
            clean_tokens = []
            prev_tok = tokenize.INDENT
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.COMMENT:
                    continue
                if tok.type == tokenize.STRING and prev_tok in (
                    tokenize.INDENT,
                    tokenize.ENCODING,
                    tokenize.NEWLINE,
                ):
                    continue
                clean_tokens.append(tok)
                if tok.type not in (tokenize.NL, tokenize.COMMENT):
                    prev_tok = tok.type
            untokenized = tokenize.untokenize(clean_tokens)
            non_empty = [line.rstrip() for line in untokenized.splitlines() if line.strip()]
            return "\n".join(non_empty).strip()
        except Exception:
            pass

    # 3. Final fallback: Line-by-line comment and empty line stripping
    cleaned = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip()


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers without ZeroDivisionError."""
    if denominator == 0:
        return default
    return numerator / denominator


def _as_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _as_int(val: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _extract_first_error(test_results: Any) -> str:
    """Extract first non-empty error message from a list of test results."""
    if isinstance(test_results, list):
        for item in test_results:
            if isinstance(item, dict):
                msg = item.get("error_msg", "")
                if msg:
                    return str(msg).strip()
    return ""


def discover_result_files(
    inputs: Sequence[str | Path] | None = None,
    pattern: str = "*result*.json",
    include_checkpoints: bool = False,
    include_backups: bool = False,
) -> list[Path]:
    """Find and return all matching result JSON files.

    Parameters
    ----------
    inputs : Sequence[str | Path], optional
        List of files, directories, or glob strings. If empty or None,
        defaults to searching './experiments' (or current directory if './experiments' doesn't exist).
    pattern : str, optional
        Glob pattern to match when searching directories (default: '*result*.json').
    include_checkpoints : bool, optional
        If False, excludes intermediate checkpoint files matching `_X_Y.json` (e.g. `_50_50.json`).
    include_backups : bool, optional
        If False, excludes backup files matching `.emergency.json` or `.old.json`.

    Returns
    -------
    list[Path]
        Sorted list of unique matching Path objects.
    """
    if not inputs:
        exp_dir = Path("experiments")
        inputs = [exp_dir] if exp_dir.is_dir() else [Path(".")]

    discovered: set[Path] = set()

    for item in inputs:
        item_str = str(item)

        # Check if the string contains glob wildcards
        if any(char in item_str for char in ("*", "?", "[", "]")):
            matches = glob.glob(item_str, recursive=True)
            for m in matches:
                p = Path(m)
                if p.is_file():
                    discovered.add(p.resolve())
            continue

        p = Path(item).resolve()
        if p.is_file():
            # Explicit files provided by the user are always included
            discovered.add(p)
        elif p.is_dir():
            # Recursively scan directory
            for found in p.rglob(pattern):
                if not found.is_file():
                    continue
                if not include_backups and BACKUP_REGEX.search(found.name):
                    continue
                if not include_checkpoints and CHECKPOINT_REGEX.search(found.name):
                    continue
                discovered.add(found.resolve())

    return sorted(discovered)


def load_result_payload(path: Path) -> dict[str, Any]:
    """Load JSON payload from a result file.

    Handles both standard dictionary results and raw list formats.
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {"results": data, "config": {}, "summary": {}}
    if isinstance(data, dict):
        return data
    raise ValueError(f"Invalid JSON in {path}: expected object or list, got {type(data).__name__}")


def extract_provider_and_model(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    """Extract model name and provider from file path and payload config."""
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    model = config.get("model")

    parts = path.resolve().parts
    provider = ""

    # Detect provider from directory hierarchy (e.g. experiments/<provider>/<model>/...)
    if "experiments" in parts:
        exp_idx = parts.index("experiments")
        if len(parts) > exp_idx + 1 and parts[exp_idx + 1] != path.name:
            provider = parts[exp_idx + 1]

    # Fallback provider detection from config or model prefix
    if not provider:
        provider = config.get("provider", "")

    if not model:
        # Infer model from parent folder name
        model = path.parent.name if path.parent.name != provider else path.stem

    if not provider and "/" in model:
        provider = model.split("/")[0]

    return provider or "unknown", model


def extract_summary_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a summary-level record (1 row) from an experiment results file."""
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    results = payload.get("results", []) if isinstance(payload.get("results"), list) else []

    provider, model = extract_provider_and_model(path, payload)
    n_tasks = len(results)

    # Calculate or retrieve correct counts and accuracies
    baseline_correct = sum(
        1 for r in results if isinstance(r, dict) and bool(r.get("baseline", {}).get("success"))
    )
    cegis_correct = sum(
        1 for r in results if isinstance(r, dict) and bool(r.get("cegis", {}).get("success"))
    )

    anticheat_items = [
        r.get("cegis_anticheat")
        for r in results
        if isinstance(r, dict) and r.get("cegis_anticheat") is not None
    ]
    has_anticheat = len(anticheat_items) > 0
    anticheat_correct = (
        sum(1 for r in anticheat_items if isinstance(r, dict) and bool(r.get("success")))
        if has_anticheat
        else None
    )

    baseline_acc = (baseline_correct / n_tasks * 100.0) if n_tasks else 0.0
    cegis_acc = (cegis_correct / n_tasks * 100.0) if n_tasks else 0.0
    anticheat_acc = (anticheat_correct / n_tasks * 100.0) if (has_anticheat and n_tasks) else None

    # Primary strategy resolution (CEGIS AntiCheat takes precedence if present)
    if has_anticheat:
        primary_strategy = "cegis_anticheat"
        primary_correct = anticheat_correct if anticheat_correct is not None else 0
        primary_acc = anticheat_acc if anticheat_acc is not None else 0.0
    else:
        primary_strategy = "cegis"
        primary_correct = cegis_correct
        primary_acc = cegis_acc

    delta_acc = primary_acc - baseline_acc
    rel_improvement = (
        _safe_divide(delta_acc, baseline_acc) * 100.0 if baseline_acc > 0 else 0.0
    )

    # CEGIS diagnostic error breakdown
    semantic_recovery = 0
    regression = 0
    spurious_overfitting = 0
    representation_ceiling = 0

    baseline_requests = 0
    cegis_requests = 0
    anticheat_requests = 0

    baseline_latencies: list[float] = []
    cegis_latencies: list[float] = []
    anticheat_latencies: list[float] = []

    for r in results:
        if not isinstance(r, dict):
            continue

        b = r.get("baseline", {}) if isinstance(r.get("baseline"), dict) else {}
        c = r.get("cegis", {}) if isinstance(r.get("cegis"), dict) else {}
        ac = r.get("cegis_anticheat", {}) if isinstance(r.get("cegis_anticheat"), dict) else {}

        b_succ = bool(b.get("success"))
        c_succ = bool(c.get("success"))
        c_conv = bool(c.get("converged_train"))

        if not b_succ and c_succ:
            semantic_recovery += 1
        if b_succ and not c_succ:
            regression += 1
        if c_conv and not c_succ:
            spurious_overfitting += 1
        if not c_conv and not c_succ:
            representation_ceiling += 1

        # Requests count
        b_hist = b.get("iteration_history")
        baseline_requests += len(b_hist) if isinstance(b_hist, list) else 1

        c_hist = c.get("iteration_history")
        cegis_requests += len(c_hist) if isinstance(c_hist, list) else 1

        if has_anticheat and ac:
            ac_hist = ac.get("iteration_history")
            anticheat_requests += len(ac_hist) if isinstance(ac_hist, list) else 1

        # Latencies
        if b.get("latency") is not None:
            baseline_latencies.append(_as_float(b.get("latency")))
        if c.get("latency") is not None:
            cegis_latencies.append(_as_float(c.get("latency")))
        if ac.get("latency") is not None:
            anticheat_latencies.append(_as_float(ac.get("latency")))

    primary_requests = anticheat_requests if has_anticheat else cegis_requests
    avg_req_per_task = _safe_divide(primary_requests, n_tasks)

    avg_lat_b = (sum(baseline_latencies) / len(baseline_latencies)) if baseline_latencies else 0.0
    avg_lat_c = (sum(cegis_latencies) / len(cegis_latencies)) if cegis_latencies else 0.0
    avg_lat_ac = (sum(anticheat_latencies) / len(anticheat_latencies)) if anticheat_latencies else 0.0

    # Relative path if within current working directory, else absolute
    try:
        rel_path = str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        rel_path = str(path.resolve())

    return {
        "model": model,
        "provider": provider,
        "tasks_evaluated": n_tasks,
        "completed_tasks": config.get("completed_tasks", n_tasks),
        "faulty_tasks": config.get("faulty_tasks", 0),
        "baseline_correct": baseline_correct,
        "baseline_accuracy": round(baseline_acc, 2),
        "cegis_correct": cegis_correct,
        "cegis_accuracy": round(cegis_acc, 2),
        "has_anticheat": has_anticheat,
        "anticheat_correct": anticheat_correct if anticheat_correct is not None else "",
        "anticheat_accuracy": round(anticheat_acc, 2) if anticheat_acc is not None else "",
        "primary_strategy": primary_strategy,
        "primary_correct": primary_correct,
        "primary_accuracy": round(primary_acc, 2),
        "delta_accuracy": round(delta_acc, 2),
        "relative_improvement_pct": round(rel_improvement, 2),
        "semantic_recovery": semantic_recovery,
        "regression": regression,
        "spurious_overfitting": spurious_overfitting,
        "representation_ceiling": representation_ceiling,
        "baseline_requests": baseline_requests,
        "cegis_requests": cegis_requests,
        "anticheat_requests": anticheat_requests if has_anticheat else "",
        "primary_requests": primary_requests,
        "avg_requests_per_task": round(avg_req_per_task, 2),
        "avg_latency_baseline_sec": round(avg_lat_b, 2),
        "avg_latency_cegis_sec": round(avg_lat_c, 2),
        "avg_latency_anticheat_sec": round(avg_lat_ac, 2) if has_anticheat else "",
        "max_cegis_iters": config.get("max_cegis_iters", ""),
        "timeout_seconds": config.get("timeout_seconds", ""),
        "file_path": rel_path,
    }


def extract_task_records(
    path: Path,
    payload: dict[str, Any],
    include_code: bool = False,
    minify_code: bool = True,
) -> list[dict[str, Any]]:
    """Extract granular per-task records (N rows) from an experiment results file."""
    provider, model = extract_provider_and_model(path, payload)
    results = payload.get("results", []) if isinstance(payload.get("results"), list) else []

    try:
        rel_path = str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        rel_path = str(path.resolve())

    records: list[dict[str, Any]] = []

    for item in results:
        if not isinstance(item, dict):
            continue

        task_id = item.get("task_id", "")
        b = item.get("baseline", {}) if isinstance(item.get("baseline"), dict) else {}
        c = item.get("cegis", {}) if isinstance(item.get("cegis"), dict) else {}
        ac = item.get("cegis_anticheat")

        b_success = bool(b.get("success"))
        c_success = bool(c.get("success"))
        c_converged = bool(c.get("converged_train"))

        # Outcome classification
        if not b_success and c_success:
            outcome = "semantic_recovery"
        elif b_success and not c_success:
            outcome = "regression"
        elif b_success and c_success:
            outcome = "both_correct"
        else:
            outcome = "both_failed"

        # CEGIS failure mode
        if c_success:
            cegis_failure_type = "none"
        elif c_converged:
            cegis_failure_type = "spurious_overfitting"
        else:
            cegis_failure_type = "representation_ceiling"

        b_code = str(b.get("generated_code") or "")
        c_code = str(c.get("generated_code") or "")
        ac_code = str(ac.get("generated_code") or "") if isinstance(ac, dict) else ""

        if include_code and minify_code:
            b_code_clean = minify_python_code(b_code)
            c_code_clean = minify_python_code(c_code)
            ac_code_clean = minify_python_code(ac_code) if ac_code else ""
        else:
            b_code_clean = b_code
            c_code_clean = c_code
            ac_code_clean = ac_code

        b_lines = len(b_code_clean.splitlines()) if b_code_clean else 0
        c_lines = len(c_code_clean.splitlines()) if c_code_clean else 0

        row: dict[str, Any] = {
            "source_file": rel_path,
            "provider": provider,
            "model": model,
            "task_id": task_id,
            # Baseline outcomes
            "baseline_success": b_success,
            "baseline_train_success": bool(b.get("train_success")),
            "baseline_test_success": bool(b.get("test_success")),
            "baseline_api_error": bool(b.get("api_error")),
            "baseline_latency_sec": round(_as_float(b.get("latency")), 3),
            "baseline_code_lines": b_lines,
            "baseline_error": _extract_first_error(b.get("test_results") or b.get("train_results")),
            # CEGIS outcomes
            "cegis_success": c_success,
            "cegis_converged_train": c_converged,
            "cegis_train_success": bool(c.get("train_success")),
            "cegis_test_success": bool(c.get("test_success")),
            "cegis_iterations_used": _as_int(c.get("iterations_used"), 1),
            "cegis_api_error": bool(c.get("api_error")),
            "cegis_latency_sec": round(_as_float(c.get("latency")), 3),
            "cegis_code_lines": c_lines,
            "cegis_error": _extract_first_error(c.get("test_results") or c.get("train_results")),
            # Categorical analysis
            "outcome_category": outcome,
            "cegis_failure_type": cegis_failure_type,
            # AntiCheat (if evaluated)
            "has_anticheat": ac is not None,
        }

        if isinstance(ac, dict):
            ac_lines = len(ac_code_clean.splitlines()) if ac_code_clean else 0
            row.update(
                {
                    "anticheat_success": bool(ac.get("success")),
                    "anticheat_converged_train": bool(ac.get("converged_train")),
                    "anticheat_train_success": bool(ac.get("train_success")),
                    "anticheat_test_success": bool(ac.get("test_success")),
                    "anticheat_iterations_used": _as_int(ac.get("iterations_used"), 1),
                    "anticheat_api_error": bool(ac.get("api_error")),
                    "anticheat_latency_sec": round(_as_float(ac.get("latency")), 3),
                    "anticheat_code_lines": ac_lines,
                    "anticheat_error": _extract_first_error(
                        ac.get("test_results") or ac.get("train_results")
                    ),
                }
            )
        else:
            row.update(
                {
                    "anticheat_success": "",
                    "anticheat_converged_train": "",
                    "anticheat_train_success": "",
                    "anticheat_test_success": "",
                    "anticheat_iterations_used": "",
                    "anticheat_api_error": "",
                    "anticheat_latency_sec": "",
                    "anticheat_code_lines": "",
                    "anticheat_error": "",
                }
            )

        if include_code:
            row["baseline_code"] = b_code_clean
            row["cegis_code"] = c_code_clean
            row["anticheat_code"] = ac_code_clean

        records.append(row)

    return records


def extract_file_code(
    path: Path,
    payload: dict[str, Any],
    minify_code: bool = True,
) -> dict[str, dict[str, str]]:
    """Extract code snippets for all tasks in a results file.

    Returns a dict mapping task_id -> {strategy_name: code_str}.
    """
    results = payload.get("results", []) if isinstance(payload.get("results"), list) else []
    task_code: dict[str, dict[str, str]] = {}

    for item in results:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id", "")
        if not task_id:
            continue

        strategies_code: dict[str, str] = {}
        for strat in ("baseline", "cegis", "cegis_anticheat"):
            s_obj = item.get(strat)
            if isinstance(s_obj, dict) and s_obj.get("generated_code"):
                raw_code = str(s_obj["generated_code"])
                code_str = minify_python_code(raw_code) if minify_code else raw_code
                if code_str:
                    strategies_code[strat] = code_str

        if strategies_code:
            task_code[task_id] = strategies_code

    return task_code


def compile_code_data(
    files: Sequence[Path] | None = None,
    minify_code: bool = True,
    code_format: str = "nested",
    verbose: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Compile code across all specified results files.

    If code_format == 'nested':
        Returns {model_name: {task_id: {strategy: code_str}}}
    If code_format == 'flat':
        Returns [{provider: ..., model: ..., task_id: ..., baseline: ..., cegis: ...}]
    """
    if files is None:
        files = discover_result_files()
    if code_format == "flat":
        flat_records: list[dict[str, Any]] = []
        for path in files:
            try:
                payload = load_result_payload(path)
                provider, model = extract_provider_and_model(path, payload)
                file_tasks = extract_file_code(path, payload, minify_code=minify_code)
                for task_id, strats in file_tasks.items():
                    entry: dict[str, Any] = {
                        "provider": provider,
                        "model": model,
                        "task_id": task_id,
                    }
                    entry.update(strats)
                    flat_records.append(entry)
            except Exception as err:
                if verbose:
                    sys.stderr.write(f"Warning: Skipping {path} during code extraction: {err}\n")
        return flat_records

    # Nested format: {model: {task_id: {strategy: code}}}
    nested_data: dict[str, dict[str, dict[str, str]]] = {}
    for path in files:
        try:
            payload = load_result_payload(path)
            provider, model = extract_provider_and_model(path, payload)
            model_key = model
            if model_key in nested_data:
                # Disambiguate if model name already exists from another folder/file
                if path.parent.name and path.parent.name != model_key:
                    model_key = f"{model} ({path.parent.name})"
                else:
                    model_key = f"{model} ({path.stem})"

            file_tasks = extract_file_code(path, payload, minify_code=minify_code)
            nested_data[model_key] = file_tasks
        except Exception as err:
            if verbose:
                sys.stderr.write(f"Warning: Skipping {path} during code extraction: {err}\n")

    return nested_data


def write_code_json(
    code_data: dict[str, Any] | list[dict[str, Any]],
    output_path: Path | str,
) -> None:
    """Write compiled code data to a JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(code_data, f, indent=2, ensure_ascii=False)


def extract_model_metadata(files: Sequence[Path]) -> dict[str, dict[str, str]]:
    """Extract metadata for each model (provider, relative file path)."""
    meta: dict[str, dict[str, str]] = {}
    for path in files:
        try:
            payload = load_result_payload(path)
            provider, model = extract_provider_and_model(path, payload)
            try:
                rel_path = str(path.resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                rel_path = str(path.resolve())
            model_key = model
            if model_key in meta:
                if path.parent.name and path.parent.name != model_key:
                    model_key = f"{model} ({path.parent.name})"
                else:
                    model_key = f"{model} ({path.stem})"
            meta[model_key] = {"provider": provider, "file_path": rel_path}
        except Exception:
            pass
    return meta


def format_code_markdown(
    nested_code: dict[str, dict[str, dict[str, str]]],
    model_metadata: dict[str, dict[str, str]] | None = None,
) -> str:
    """Format nested code dictionary as a clean, syntax-highlighted Markdown document."""
    strategy_labels = {
        "baseline": "Baseline (1-Shot)",
        "cegis": "CEGIS",
        "cegis_anticheat": "CEGIS AntiCheat",
    }

    lines: list[str] = [
        "# ARC-AGI Generated Code\n",
        "## Index of Models\n",
    ]

    for model_name, tasks in nested_code.items():
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", model_name).strip("-").lower()
        lines.append(f"- [{model_name}](#{slug}) ({len(tasks)} tasks)")
    lines.append("")

    for model_name, tasks in nested_code.items():
        lines.append(f"## {model_name}\n")
        meta = (model_metadata or {}).get(model_name, {})
        details: list[str] = []
        if meta.get("provider"):
            details.append(f"**Provider:** `{meta['provider']}`")
        if meta.get("file_path"):
            details.append(f"**Source:** `{meta['file_path']}`")
        if details:
            lines.append(" | ".join(details) + "\n")

        if not tasks:
            lines.append("_No generated code recorded for this model._\n")
            continue

        for task_id in sorted(tasks.keys()):
            strategies = tasks[task_id]
            lines.append(f"### Task `{task_id}`\n")
            for strat_key in ("baseline", "cegis", "cegis_anticheat"):
                code = strategies.get(strat_key)
                if not code:
                    continue
                label = strategy_labels.get(strat_key, strat_key.title())
                lines.append(f"#### {label}\n")
                lines.append(f"```python\n{code}\n```\n")

    return "\n".join(lines)


def write_code_markdown(
    code_data: dict[str, dict[str, dict[str, str]]],
    output_path: Path | str,
    model_metadata: dict[str, dict[str, str]] | None = None,
) -> None:
    """Write compiled code data to a Markdown file."""
    content = format_code_markdown(code_data, model_metadata=model_metadata)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def compile_summary(
    files: Sequence[Path],
    sort_by: str = "primary_accuracy",
    descending: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Compile summary records across all specified results files."""
    rows: list[dict[str, Any]] = []

    for path in files:
        try:
            payload = load_result_payload(path)
            record = extract_summary_record(path, payload)
            rows.append(record)
        except Exception as err:
            if verbose:
                sys.stderr.write(f"Warning: Skipping {path}: {err}\n")

    if sort_by:
        def sort_key(item: dict[str, Any]) -> Any:
            val = item.get(sort_by)
            if val is None or val == "":
                return -1e9 if descending else 1e9
            return val

        rows.sort(key=sort_key, reverse=descending)

    return rows


def compile_tasks(
    files: Sequence[Path],
    include_code: bool = False,
    minify_code: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Compile all per-task records across all specified results files."""
    all_rows: list[dict[str, Any]] = []

    for path in files:
        try:
            payload = load_result_payload(path)
            task_rows = extract_task_records(
                path, payload, include_code=include_code, minify_code=minify_code
            )
            all_rows.extend(task_rows)
        except Exception as err:
            if verbose:
                sys.stderr.write(f"Warning: Skipping {path}: {err}\n")

    return all_rows


def write_csv(
    records: Sequence[dict[str, Any]],
    output_path: Path | str | None = None,
) -> None:
    """Write records to a CSV file or stdout."""
    if not records:
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text("", encoding="utf-8")
        return

    # Derive fieldnames preserving order of first appearance
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in records:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# --- Optional Pandas helpers for python script/notebook usage ---
def compile_summary_df(
    files: Sequence[Path] | None = None,
    sort_by: str = "primary_accuracy",
    descending: bool = True,
) -> Any:
    """Compile summary records as a pandas DataFrame."""
    import pandas as pd  # type: ignore

    if files is None:
        files = discover_result_files()
    rows = compile_summary(files, sort_by=sort_by, descending=descending)
    return pd.DataFrame(rows)


def compile_tasks_df(
    files: Sequence[Path] | None = None,
    include_code: bool = False,
    minify_code: bool = True,
) -> Any:
    """Compile task-level records as a pandas DataFrame."""
    import pandas as pd  # type: ignore

    if files is None:
        files = discover_result_files()
    rows = compile_tasks(files, include_code=include_code, minify_code=minify_code)
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Compile a set of ARC experiment results.json files into a single CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s -o summary.csv
  %(prog)s --tasks -o all_tasks.csv
  %(prog)s --tasks --include-code -o all_tasks_with_code.csv
  %(prog)s --both -o benchmark.csv
  %(prog)s experiments/google/gemini-3.1-flash-lite/results_experiment.json
  %(prog)s "experiments/**/results_experiment_100_100.json" -o runs_100.csv
""",
    )

    parser.add_argument(
        "paths",
        type=str,
        nargs="*",
        help="Paths to results JSON files, directories, or glob patterns. "
             "If omitted, defaults to './experiments'.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to output CSV file. If omitted, prints CSV to stdout. "
             "In '--mode both', writes <output>_summary.csv and <output>_tasks.csv.",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["summary", "tasks", "both"],
        default="summary",
        help="Aggregation mode: 'summary' (1 row per experiment), "
             "'tasks' (1 row per task evaluation across models), "
             "or 'both' (generate both CSV files). Default: summary.",
    )
    parser.add_argument(
        "--summary",
        action="store_const",
        dest="mode",
        const="summary",
        help="Shortcut for --mode summary.",
    )
    parser.add_argument(
        "--tasks",
        action="store_const",
        dest="mode",
        const="tasks",
        help="Shortcut for --mode tasks.",
    )
    parser.add_argument(
        "--both",
        action="store_const",
        dest="mode",
        const="both",
        help="Shortcut for --mode both.",
    )
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="*result*.json",
        help="Glob pattern to match when searching directories (default: '*result*.json').",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include checkpoint files matching '_X_Y.json' (e.g. '_50_50.json') when scanning directories.",
    )
    parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Include backup files like '*.emergency.json' and '*.old.json'.",
    )
    parser.add_argument(
        "--include-code",
        action="store_true",
        help="In 'tasks' mode, include generated Python code columns (minified by default).",
    )
    parser.add_argument(
        "--no-minify",
        action="store_false",
        dest="minify_code",
        default=True,
        help="Do not minify code when using --include-code or --code-json (keeps original formatting, docstrings, and comments).",
    )
    parser.add_argument(
        "--code-json",
        type=Path,
        default=None,
        metavar="JSON_FILE",
        help="Write only the generated code to a separate JSON file (minified by default).",
    )
    parser.add_argument(
        "--code-md",
        type=Path,
        default=None,
        metavar="MD_FILE",
        help="Write only the generated code to a separate Markdown (.md) file (minified by default).",
    )
    parser.add_argument(
        "--code-format",
        choices=["nested", "flat"],
        default="nested",
        help="JSON structure for --code-json: 'nested' (model -> task_id -> strategy) "
             "or 'flat' (list of records). Default: nested.",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Extract and write only the code to a JSON/Markdown file (skips CSV generation).",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="primary_accuracy",
        help="Column to sort by in summary mode (default: 'primary_accuracy').",
    )
    parser.add_argument(
        "--sort-asc",
        action="store_true",
        help="Sort ascending instead of descending.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose status messages to stderr.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Discover files
    files = discover_result_files(
        inputs=args.paths,
        pattern=args.pattern,
        include_checkpoints=args.include_checkpoints,
        include_backups=args.include_backups,
    )

    if not files:
        sys.stderr.write("No matching results JSON files found.\n")
        return 1

    if args.verbose:
        sys.stderr.write(f"Found {len(files)} result file(s) to process.\n")

    # Handle code-only export (JSON and/or Markdown) if requested
    is_code_md = bool(args.code_md)
    is_code_json = bool(args.code_json)

    if args.code_only and not is_code_md and not is_code_json:
        if args.output and args.output.suffix.lower() == ".md":
            is_code_md = True
        else:
            is_code_json = True

    if is_code_json:
        code_out = args.code_json or args.output
        if not code_out:
            sys.stderr.write(
                "Error: Specify a destination JSON path via --code-json <path> or -o <path>.\n"
            )
            return 2

        code_data = compile_code_data(
            files,
            minify_code=args.minify_code,
            code_format=args.code_format,
            verbose=args.verbose,
        )
        write_code_json(code_data, code_out)
        count_desc = (
            f"{len(code_data)} models"
            if args.code_format == "nested"
            else f"{len(code_data)} task records"
        )
        sys.stderr.write(f"Wrote code JSON ({count_desc}) -> {code_out}\n")

    if is_code_md:
        md_out = args.code_md or args.output
        if not md_out:
            sys.stderr.write(
                "Error: Specify a destination Markdown path via --code-md <path> or -o <path>.\n"
            )
            return 2

        nested_code = compile_code_data(
            files,
            minify_code=args.minify_code,
            code_format="nested",
            verbose=args.verbose,
        )
        model_meta = extract_model_metadata(files)
        write_code_markdown(nested_code, md_out, model_metadata=model_meta)
        sys.stderr.write(f"Wrote code Markdown ({len(nested_code)} models) -> {md_out}\n")

    # If code export was performed and no CSV output was requested, finish here
    if (is_code_json or is_code_md) and (args.code_only or args.output is None):
        return 0

    descending = not args.sort_asc

    if args.mode == "both":
        if args.output is None:
            sys.stderr.write("Error: --output is required when using --mode both.\n")
            return 2

        out_stem = args.output.stem
        out_suffix = args.output.suffix or ".csv"
        out_parent = args.output.parent

        summary_file = out_parent / f"{out_stem}_summary{out_suffix}"
        tasks_file = out_parent / f"{out_stem}_tasks{out_suffix}"

        summary_records = compile_summary(
            files, sort_by=args.sort_by, descending=descending, verbose=args.verbose
        )
        write_csv(summary_records, summary_file)
        sys.stderr.write(f"Wrote summary CSV ({len(summary_records)} rows) -> {summary_file}\n")

        task_records = compile_tasks(
            files,
            include_code=args.include_code,
            minify_code=args.minify_code,
            verbose=args.verbose,
        )
        write_csv(task_records, tasks_file)
        sys.stderr.write(f"Wrote tasks CSV ({len(task_records)} rows) -> {tasks_file}\n")
        return 0

    if args.mode == "summary":
        summary_records = compile_summary(
            files, sort_by=args.sort_by, descending=descending, verbose=args.verbose
        )
        write_csv(summary_records, args.output)
        if args.output:
            sys.stderr.write(f"Wrote summary CSV ({len(summary_records)} rows) -> {args.output}\n")
        return 0

    if args.mode == "tasks":
        task_records = compile_tasks(
            files,
            include_code=args.include_code,
            minify_code=args.minify_code,
            verbose=args.verbose,
        )
        write_csv(task_records, args.output)
        if args.output:
            sys.stderr.write(f"Wrote tasks CSV ({len(task_records)} rows) -> {args.output}\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

