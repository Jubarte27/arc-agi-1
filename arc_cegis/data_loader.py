"""
Data loading utilities for ARC-AGI JSON task benchmarks.
"""

import json
import os
from typing import Any, Dict


def load_tasks(data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads ARC tasks from a single JSON file or a directory containing task JSON files.
    Falls back to a built-in demonstration task if the path does not exist.
    """
    tasks: Dict[str, Dict[str, Any]] = {}
    
    if os.path.isfile(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, dict) and "train" in content and "test" in content:
                task_id = os.path.splitext(os.path.basename(data_path))[0]
                tasks[task_id] = content
            elif isinstance(content, dict):
                tasks = content
    elif os.path.isdir(data_path):
        for root, _, files in os.walk(data_path):
            for fname in sorted(files):
                if fname.endswith(".json"):
                    task_id = os.path.splitext(fname)[0]
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = json.load(f)
                            if isinstance(content, dict) and "train" in content and "test" in content:
                                tasks[task_id] = content
                            elif isinstance(content, dict) and not any(k in content for k in ("train", "test")):
                                tasks.update(content)
                    except Exception as e:
                        print(f"[Warning] Failed to load {fpath}: {e}")
    else:
        print(f"[Info] Path '{data_path}' not found. Using built-in sample ARC task.")
        tasks["sample_task_001"] = {
            "train": [
                {"input": [[0, 1], [0, 0]], "output": [[1, 0], [0, 0]]},
                {"input": [[0, 0], [2, 0]], "output": [[0, 0], [0, 2]]},
            ],
            "test": [
                {"input": [[0, 3], [0, 0]], "output": [[3, 0], [0, 0]]}
            ],
        }

    return tasks
