#!/bin/bash
# ==============================================================================
# Runner script for ARC-CEGIS using Local Ollama
# ==============================================================================

set -euo pipefail

SCRIPT_DIR=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$SCRIPT_DIR/.venv/bin/activate"

# MODEL_NAME=qwen2.5-coder:7b
# MODEL_NAME=qwen2.5-coder:1.5b
# MODEL_NAME=llama3.2:3b
# MODEL_NAME=deepseek-coder:6.7b
MODEL_NAME=gemma4:e4b

export MODEL_NAME

cd "$SCRIPT_DIR"
TARGET="experiments/ollama/$MODEL_NAME"
ENV_DIR="$SCRIPT_DIR/envs"

mkdir -p "$TARGET"
cd "$TARGET"

DOTENV="$ENV_DIR/.env:$ENV_DIR/.env.ollama" python3 "$SCRIPT_DIR/main.py" --tasks "$SCRIPT_DIR/data" --output results_experiment.json "$@"
