#!/bin/bash
# ==============================================================================
# Runner script for ARC-CEGIS using vLLM
# Starts a vLLM server, runs the experiment, and cleans up on exit.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
UV_RUN="uv run --python ${VENV_DIR}/bin/python3"

# ── Configuration ───────────────────────────────────────────────────────────
# MODEL_NAME: Hugging Face model ID to serve and evaluate.
# Uncomment the model you want to use:
# MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct
# MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
MODEL_NAME=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
# MODEL_NAME=google/gemma-4-26b-a4b-it

export MODEL_NAME

# VLLM_HOST: Address and port for the vLLM OpenAI-compatible server.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"

# VLLM_EXTRA_ARGS: Additional arguments passed to `vllm serve`.
# Examples: --tensor-parallel-size 2, --max-model-len 4096, --gpu-memory-utilization 0.9
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# ── Server Lifecycle ───────────────────────────────────────────────────────
VLLM_PID=""

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Stopping vLLM server (PID: ${VLLM_PID})..."
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting vLLM server for model: ${MODEL_NAME}"
# shellcheck disable=SC2086
$UV_RUN vllm serve "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    ${VLLM_EXTRA_ARGS} &
VLLM_PID=$!

echo "Waiting for vLLM server to become ready at ${VLLM_HOST}:${VLLM_PORT}..."
MAX_WAIT=120
WAIT_COUNT=0
until curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Error: vLLM server process terminated unexpectedly." >&2
        exit 1
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    if [[ $WAIT_COUNT -ge $MAX_WAIT ]]; then
        echo "Error: vLLM server did not start within ${MAX_WAIT} seconds." >&2
        exit 1
    fi
done
echo "vLLM server is ready."

# ── Run Experiment ──────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
TARGET="experiments/vllm/$MODEL_NAME"
ENV_DIR="$SCRIPT_DIR/envs"

mkdir -p "$TARGET"
cd "$TARGET"

DOTENV="$ENV_DIR/.env:$ENV_DIR/.env.vllm" $UV_RUN "$SCRIPT_DIR/main.py" \
    --tasks "$SCRIPT_DIR/data" \
    --output results_experiment.json \
    "$@"


