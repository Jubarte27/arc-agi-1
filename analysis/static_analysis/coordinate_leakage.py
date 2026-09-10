#!/usr/bin/env python3
"""Coordinate Leakage & Hardcoding Static Scanner.

Statically scans synthesized code in benchmark_code.json for coordinate-based
memoization, hardcoded grid equality patches, and dimensional guards.
Independent of specific model names or hardcoded benchmark results.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_LEAKAGE_PATTERNS = [
    # Explicit coordinate lookups: if r == 3 and c == 5
    (r"if\s+\(?[rcxyij]\s*==\s*\d+\s+and\s+[rcxyij]\s*==\s*\d+\)?", "coordinate_point_match"),
    # Coordinate tuple check: if (r, c) == (3, 5) or (r, c) in {(3, 5), ...}
    (r"if\s+\(?[rcxyij],\s*[rcxyij]\)?\s*(?:==|\s*in\s*\{\s*)\(?\d+,\s*\d+\)?", "coordinate_tuple_match"),
    # Grid dimension + cell fingerprint: if len(grid) == 10 and grid[0][0] == 4
    (r"if\s+len\(grid\)\s*==\s*\d+\s+and\s+(?:len\(grid\[0\]\)\s*==\s*\d+\s+and\s+)?grid\[\d+\]\[\d+\]\s*==", "grid_fingerprint_check"),
    # Literal grid comparison: if grid == [[...
    (r"if\s+grid\s*==\s*\[\s*\[", "literal_grid_equality"),
]


def scan_code_for_leakage(code: str, patterns=DEFAULT_LEAKAGE_PATTERNS) -> list[dict[str, Any]]:
    """Scan a Python code string for hardcoded coordinate checks.

    Parameters
    ----------
    code : str
        Code string.
    patterns : list[tuple[str, str]]
        Regex patterns and label.

    Returns
    -------
    list[dict[str, Any]]
        List of match dicts with line number, pattern, and snippet.
    """
    if not code:
        return []

    lines = code.splitlines()
    matches = []

    for line_no, line in enumerate(lines, 1):
        for pattern_str, pattern_type in patterns:
            if re.search(pattern_str, line, re.IGNORECASE):
                matches.append({
                    "line_number": line_no,
                    "pattern_type": pattern_type,
                    "matched_text": line.strip(),
                })

    return matches


def analyze_coordinate_leakage(
    code_json_path: str | Path,
    patterns=DEFAULT_LEAKAGE_PATTERNS,
) -> dict[str, Any]:
    """Scan all models, tasks, and strategies in a code JSON file.

    Parameters
    ----------
    code_json_path : str | Path
        Path to benchmark_code.json.
    patterns : list[tuple[str, str]]
        List of regex patterns.

    Returns
    -------
    dict[str, Any]
        Dictionary with detected leakages, counts, and per-strategy breakdown.
    """
    with open(code_json_path, "r") as f:
        code_data = json.load(f)

    total_snippets = 0
    flagged_snippets = []
    strategy_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}

    for model, tasks in code_data.items():
        for task_id, strategies in tasks.items():
            for strategy, code in strategies.items():
                total_snippets += 1
                matches = scan_code_for_leakage(code, patterns=patterns)
                if matches:
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                    model_counts[model] = model_counts.get(model, 0) + 1
                    flagged_snippets.append({
                        "model": model,
                        "task_id": task_id,
                        "strategy": strategy,
                        "match_count": len(matches),
                        "matches": matches,
                        "code_excerpt": "\n".join(code.splitlines()[:20]),
                    })

    return {
        "total_snippets_scanned": total_snippets,
        "flagged_snippets_count": len(flagged_snippets),
        "leakage_rate_pct": float(len(flagged_snippets) / total_snippets * 100) if total_snippets > 0 else 0.0,
        "strategy_counts": strategy_counts,
        "model_counts": model_counts,
        "flagged_snippets": flagged_snippets,
    }


def main():
    parser = argparse.ArgumentParser(description="Scan benchmark code for coordinate hardcoding / cheating.")
    parser.add_argument("--code-json", type=str, default="benchmark_code.json", help="Path to code JSON.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    results = analyze_coordinate_leakage(args.code_json)

    print("=" * 70)
    print("COORDINATE LEAKAGE & HARDCODING SCANNER")
    print("=" * 70)
    print(f"Total snippets scanned: {results['total_snippets_scanned']}")
    print(f"Flagged snippets with hardcoded coordinates: {results['flagged_snippets_count']} ({results['leakage_rate_pct']:.2f}%)\n")

    print("Breakdown by Strategy:")
    for strat, count in results["strategy_counts"].items():
        print(f"  {strat:20s}: {count:3d} snippets")

    print("\nBreakdown by Model:")
    for model, count in results["model_counts"].items():
        print(f"  {model:35s}: {count:3d} snippets")

    if results["flagged_snippets"]:
        print("\nSample Detections:")
        for sample in results["flagged_snippets"][:5]:
            print(f"  Model: {sample['model']} | Task: {sample['task_id']} | Strategy: {sample['strategy']}")
            for m in sample["matches"][:2]:
                print(f"    Line {m['line_number']} [{m['pattern_type']}]: {m['matched_text']}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON results to {out_p}")


if __name__ == "__main__":
    main()

