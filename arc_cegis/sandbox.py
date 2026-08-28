"""
Safe execution environment and Python code extraction for ARC grid transformations.
"""

import multiprocessing as mp
import re
from typing import Any, Dict, List, Optional, Tuple

from . import config


def extract_python_code(response_text: str) -> str:
    """
    Extracts executable Python code from markdown response text.
    Matches ```python ... ```, ```py ... ```, or ``` ... ``` code fences, prioritizing blocks
    containing the `def transform` definition, or falls back to raw text.
    """
    code_blocks = re.findall(r"```(?:python\d*|py)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    if code_blocks:
        for block in code_blocks:
            if "def transform" in block:
                return block.strip()
        return code_blocks[-1].strip()

    if "def transform" in response_text:
        start_idx = response_text.find("def transform")
        return response_text[start_idx:].strip()

    return response_text.strip()


def _worker_process_target(code_str: str, input_grid: List[List[int]], queue: mp.Queue) -> None:
    """
    Subprocess worker that executes transform(grid) in a restricted namespace.

    NOTE on Restricted Builtins (Design Choice):
    The safe_globals namespace is intentionally locked down to a minimal whitelist
    of pure-Python functional and sequence primitives. This strictly enforces the
    ANTI_IMPORT and anti-side-effect experimental constraints (prohibiting file I/O,
    module imports, OS access, and non-whitelisted reflections) to ensure safe,
    reproducible, and deterministic evaluation across all benchmark tasks.
    """
    try:
        safe_globals: Dict[str, Any] = {
            "__builtins__": {
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "int": int,
                "float": float,
                "bool": bool,
                "all": all,
                "any": any,
                "sorted": sorted,
                "reversed": reversed,
                "isinstance": isinstance,
                "print": print,
            }
        }
        exec_scope: Dict[str, Any] = dict(safe_globals)
        
        exec(code_str, exec_scope)
        
        if "transform" not in exec_scope or not callable(exec_scope["transform"]):
            queue.put((False, None, "Function 'transform(grid)' not found or not callable."))
            return
        
        # Deep copy to ensure input grid is not mutated
        input_copy = [row[:] for row in input_grid]
        result = exec_scope["transform"](input_copy)
        
        if not isinstance(result, list):
            queue.put((False, None, f"Expected return type list[list[int]], got {type(result).__name__}"))
            return
        
        queue.put((True, result, ""))
    except Exception as e:
        queue.put((False, None, f"{type(e).__name__}: {str(e)}"))


def run_transform(
    code_str: str,
    input_grid: List[List[int]],
    timeout_seconds: float = config.TIMEOUT_SECONDS
) -> Tuple[bool, Optional[List[List[int]]], str]:
    """
    Executes transform() against input_grid inside a dedicated subprocess with strict timeout.
    Returns:
        (success: bool, output_grid: Optional[List[List[int]]], error_msg: str)
    """
    if not code_str:
        return False, None, "Empty code string provided."

    queue: mp.Queue = mp.Queue()
    proc: Optional[mp.Process] = None
    try:
        proc = mp.Process(target=_worker_process_target, args=(code_str, input_grid, queue))
        proc.start()

        proc.join(timeout=timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.5)
            if proc.is_alive():
                proc.kill()
            queue.cancel_join_thread()
            return False, None, f"Execution timed out (> {timeout_seconds}s)."

        try:
            return queue.get(timeout=0.2)
        except Exception:
            pass
    finally:
        queue.close()
        if proc is not None:
            try:
                proc.close()
            except Exception:
                pass

    return False, None, "Process terminated unexpectedly without returning output."
