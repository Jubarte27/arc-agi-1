#!/bin/bash
set -e

SCRIPT_DIR=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))

# MODEL_NAME=gemini-3.5-flash-lite
# MODEL_NAME=gemini-3.1-flash-lite
# MODEL_NAME=gemini-2.5-pro
MODEL_NAME=gemma-4-31b-it

export MODEL_NAME

cd "$SCRIPT_DIR"
TARGET="experiments/google/$MODEL_NAME"
ENV_DIR="$SCRIPT_DIR/envs"

source "$SCRIPT_DIR/.venv/bin/activate"

mkdir -p "$TARGET"
cd "$TARGET"

DOTENV="$ENV_DIR/.env:$ENV_DIR/.env.secret:$ENV_DIR/.env.google" python3 "$SCRIPT_DIR/main.py" --tasks "$SCRIPT_DIR/data" --output results_experiment.json