"""
Prompt engineering and semantic counterexample feedback builders for ARC tasks.
"""

import json
from typing import Dict, List, Optional

ANTI_IMPORT_RULES = (
    "CODE RULES:\n"
    "- DO NOT use any `import` statements (e.g., collections, numpy, math, etc.).\n"
    "- USE ONLY pure native Python types and functions (list, dict, set, tuple, range, len, min, max, sum, enumerate, zip, abs).\n"
)

ANTI_CHEAT_RULES = (
    "- DO NOT add conditional branches (if/else) targeting specific example indices, hardcoded coordinates, or specific row values of a single training case. The transformation must be a single, uniform mathematical/geometric rule that applies to ALL grids identically.\n"
    "- Before writing the code, explicitly state the single abstract invariant that explains why the previous code failed on Example X and how the new logic naturally handles Examples 0, 1, and 2 without special cases.\n"
)

def build_initial_prompt(task_train_pairs: List[Dict[str, List[List[int]]]]) -> str:
    """
    Constructs the initial prompt presenting train demonstration pairs and requesting transform().
    """
    prompt = (
        "You are an expert Python programmer solving an ARC (Abstraction and Reasoning Corpus) puzzle.\n"
        "Analyze the following input-output grid demonstration pairs to discover the underlying transformation rule.\n\n"
    )
    for i, pair in enumerate(task_train_pairs):
        prompt += f"--- Example {i} ---\n"
        prompt += f"Input:\n{json.dumps(pair['input'])}\n"
        prompt += f"Output:\n{json.dumps(pair['output'])}\n\n"

    prompt += (
        "Write a Python function `transform(grid: list[list[int]]) -> list[list[int]]` that implements the solution.\n\n"
        f"{ANTI_IMPORT_RULES}\n"
        "Requirements:\n"
        "- The function must accept a 2D list of integers representing the input grid.\n"
        "- The function must return a 2D list of integers representing the transformed output grid.\n"
        "- Return only valid Python code inside a ```python ... ``` block without additional chatter.\n"
    )
    return prompt


def build_counterexample_feedback(
    example_idx: int,
    input_grid: List[List[int]],
    expected_grid: List[List[int]],
    actual_grid: Optional[List[List[int]]],
    error_msg: str,
) -> str:
    """
    Builds semantic counterexample feedback when code fails on a training pair.
    """
    output_repr = json.dumps(actual_grid) if not error_msg else f"Execution Error: {error_msg}"
    feedback = (
        f"Your code failed on Training Example {example_idx}.\n"
        f"- Input:\n{json.dumps(input_grid)}\n"
        f"- Expected Output:\n{json.dumps(expected_grid)}\n"
        f"- Produced Output:\n{output_repr}\n\n"
        f"{ANTI_IMPORT_RULES}\n"
        f"{ANTI_CHEAT_RULES}\n"
        "Analyze the discrepancy, fix your logic, and provide the updated `transform(grid)` function "
        "inside a ```python ... ``` block."
    )
    return feedback
