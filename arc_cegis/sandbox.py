"""
Safe execution environment and Python code extraction for ARC grid transformations.
"""

import multiprocessing as mp
import queue as _queue_mod
import re
import sys
from types import MappingProxyType
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


# Immutable builtins whitelist — shared across all invocations, safe because
# MappingProxyType prevents exec'd code from mutating the dict (e.g. injecting
# __import__), while still allowing recursion and helper function lookups since
# we use a single exec namespace (not split globals/locals).
_SAFE_BUILTINS = MappingProxyType({
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
    "str": str,
    "all": all,
    "any": any,
    "sorted": sorted,
    "reversed": reversed,
    "isinstance": isinstance,
    "print": print,
    "True": True,
    "False": False,
    "None": None,
})

# Audit events that are blocked inside the sandbox subprocess.
# Once sys.addaudithook is installed, it cannot be removed — the child
# process is short-lived so this is safe and effective.
_BLOCKED_AUDIT_EVENTS = frozenset({
    "import",
    "builtins.input",
    "open",
    "os.system",
    "os.popen",
    "os.exec",
    "os.spawn",
    "subprocess.Popen",
    "subprocess.call",
    "socket.connect",
    "socket.bind",
    "webbrowser.open",
    "ctypes.dlopen",
})

# Events that should only be blocked AFTER the top-level exec() completes.
# CPython fires 'compile' and 'exec' internally during our own exec() call,
# so we gate these with a flag that is set after the initial exec finishes.
_DEFERRED_BLOCK_EVENTS = frozenset({"compile", "exec"})

# Flag set after the top-level exec() completes; once True, nested
# exec()/compile() from user code are blocked.
_sandbox_armed = False


def _sandbox_audit_hook(event: str, args: tuple) -> None:
    """CPython audit hook that blocks dangerous operations inside the sandbox."""
    if event in _BLOCKED_AUDIT_EVENTS:
        raise PermissionError(f"Sandbox: blocked '{event}'")
    if _sandbox_armed and event in _DEFERRED_BLOCK_EVENTS:
        raise PermissionError(f"Sandbox: blocked '{event}'")


# Dangerous dunder attributes used in MRO escape chains and __import__ recovery.
# These patterns catch: .__subclasses__(), .__bases__, .__globals__, etc.
# Deliberately does NOT block __init__, __name__, __len__, __eq__ etc. which are
# commonly used in legitimate transform code.
_DANGEROUS_DUNDER_PATTERN = re.compile(
    r"__(?:"
    r"subclasses|bases|mro|globals|import|builtins|code|loader|spec"
    r"|qualname|dict|reduce|reduce_ex|getattr|setattr|delattr"
    r")__",
    re.IGNORECASE,
)


def _check_dangerous_patterns(code_str: str) -> Optional[str]:
    """
    Statically scans code for dangerous dunder attribute patterns before exec.
    Returns the matched pattern string if found, or None if the code is safe.
    """
    match = _DANGEROUS_DUNDER_PATTERN.search(code_str)
    if match:
        return match.group(0)
    return None


def _worker_process_target(code_str: str, input_grid: List[List[int]], queue: mp.Queue) -> None:
    """
    Subprocess worker that executes transform(grid) in a restricted namespace.

    Security layers (defense in depth):
      1. Static code validation — rejects dangerous dunder patterns before exec.
      2. Restricted __builtins__ whitelist — no __import__, no open(), no eval().
      3. MappingProxyType on __builtins__ — prevents runtime mutation of the whitelist.
      4. sys.addaudithook — irrevocable CPython-level block on import, file I/O,
         subprocess, network, and ctypes.
    """
    # Install audit hook BEFORE any exec(). Once installed, it cannot be removed.
    sys.addaudithook(_sandbox_audit_hook)

    try:
        # Static validation: reject code with dangerous dunder attribute access
        # that could be used for MRO escape or __import__ recovery.
        violation = _check_dangerous_patterns(code_str)
        if violation:
            queue.put((False, None, f"Sandbox: blocked dangerous pattern: {violation}"))
            return

        safe_globals: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}

        exec(code_str, safe_globals)

        # Now that the top-level exec has compiled and defined all functions,
        # arm the deferred audit block so user code cannot call exec()/compile().
        global _sandbox_armed
        _sandbox_armed = True

        if "transform" not in safe_globals or not callable(safe_globals["transform"]):
            queue.put((False, None, "Function 'transform(grid)' not found or not callable."))
            return

        # Deep copy to ensure input grid is not mutated
        input_copy = [row[:] for row in input_grid]
        result = safe_globals["transform"](input_copy)

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

    Queue is drained BEFORE joining the process to prevent deadlock when the
    result payload exceeds the OS pipe buffer (per Python multiprocessing docs).

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

        # Drain the queue FIRST to unblock the child's put(), then join.
        try:
            result = queue.get(timeout=timeout_seconds)
        except _queue_mod.Empty:
            result = None

        # Reap the child process.
        proc.join(timeout=1.0)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.5)
            queue.cancel_join_thread()

        if result is None:
            # Distinguish between timeout and unexpected exit
            if proc.exitcode is None or proc.exitcode != 0:
                return False, None, f"Execution timed out (> {timeout_seconds}s)."
            return False, None, "Process terminated unexpectedly without returning output."

        return result
    finally:
        queue.close()
        if proc is not None:
            try:
                proc.close()
            except Exception:
                pass
