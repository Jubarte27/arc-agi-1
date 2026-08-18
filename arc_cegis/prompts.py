"""
Prompt engineering and semantic counterexample feedback builders for ARC tasks.
"""

import json
from typing import Dict, List, Optional


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
        "Write a Python function `transform(grid: list[list[int]]) -> list[list[int]]` that implements this rule.\n"
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
        "Please analyze the discrepancy, fix your logic, and provide the updated `transform(grid)` function "
        "inside a ```python ... ``` block."
    )
    return feedback
