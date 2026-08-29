#!/usr/bin/env bash
# ==============================================================================
# Runner script for ARC-CEGIS using Local Ollama
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "$PYTHON_EXEC" ]]; then
    PYTHON_EXEC="python3"
fi

DOTENV=envs/.env.ollama "$PYTHON_EXEC" "${SCRIPT_DIR}/main.py" "$@"

