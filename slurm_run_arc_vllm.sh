#!/bin/bash
# ==============================================================================
# GPU Resources (adjust these for your cluster):
# ==============================================================================
#SBATCH --job-name=arc-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=lunaris
#SBATCH --nodelist=lunaris
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_arc_vllm_%j.out
#SBATCH --error=logs/slurm_arc_vllm_%j.err
# ==============================================================================
# To change GPU allocation, edit --gres above (e.g. gpu:4 for 4 GPUs).
# To match, set TENSOR_PARALLEL_SIZE below to the same GPU count.
# ==============================================================================

# ==============================================================================
# Slurm: Run ARC-CEGIS Experiment on vLLM
# Mirrors slurm_run_arc_ollama.sh for the vLLM backend.
# ==============================================================================

set -euo pipefail

cd "$SCRATCH/arc-agi-1"
SCRIPT_DIR="$(pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

mkdir -p "$SCRIPT_DIR/logs"

# ── Model Configuration ────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct}"
export MODEL_NAME
export LLM_PROVIDER="vllm"

# ── vLLM Server Configuration ──────────────────────────────────────────────
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export API_BASE_URL="$VLLM_BASE_URL"

# GPU parallelism: set this to match --gres=gpu:N above.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"

# Additional vLLM serve args (e.g. --max-model-len 4096, --gpu-memory-utilization 0.9)
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

export MAX_DAILY_REQUESTS=10000000
export MAX_CONCURRENT_TASKS=1
export REQUEST_DELAY=0.1

# ── Hugging Face Token ──────────────────────────────────────────────────────
# Some models (Llama, Gemma, etc.) are gated and require a HF token.
# Set HF_TOKEN in your environment before submitting this job:
#   export HF_TOKEN='hf_...'
# Obtain a token at: https://huggingface.co/settings/tokens

TARGET_DIR="$SCRIPT_DIR/experiments/vllm/$MODEL_NAME"
mkdir -p "$TARGET_DIR"

echo "============================================================"
echo "          Slurm: ARC-CEGIS on vLLM                          "
echo "============================================================"
echo "Workspace:  $SCRIPT_DIR"
echo "Model:      $MODEL_NAME"
echo "Provider:   $LLM_PROVIDER"
echo "vLLM Host:  ${VLLM_HOST}:${VLLM_PORT}"
echo "Tensor Par: $TENSOR_PARALLEL_SIZE"
echo "Output dir: $TARGET_DIR"

# ── Start vLLM Server ──────────────────────────────────────────────────────
VLLM_PID=""

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Stopping vLLM server (PID: ${VLLM_PID})..."
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting vLLM server..."
# shellcheck disable=SC2086
vllm serve "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    ${VLLM_EXTRA_ARGS} \
    > "$SCRIPT_DIR/logs/vllm_server.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server to become ready..."
MAX_WAIT=180
WAIT_COUNT=0
until curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Error: vLLM server process terminated unexpectedly." >&2
        echo "Check logs at: $SCRIPT_DIR/logs/vllm_server.log" >&2
        exit 1
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    if [[ $WAIT_COUNT -ge $MAX_WAIT ]]; then
        echo "Error: vLLM server did not start within ${MAX_WAIT} seconds." >&2
        echo "Check logs at: $SCRIPT_DIR/logs/vllm_server.log" >&2
        exit 1
    fi
done
echo "vLLM server is ready (PID: ${VLLM_PID})."

# ── Run Experiment ──────────────────────────────────────────────────────────
cd "$TARGET_DIR"
DOTENV="$SCRIPT_DIR/envs/.env:$SCRIPT_DIR/envs/.env.vllm" \
python3 "$SCRIPT_DIR/main.py" \
    --provider vllm \
    --model "$MODEL_NAME" \
    --tasks "$SCRIPT_DIR/data" \
    --output "$TARGET_DIR/results_experiment.json" \
    "$@"

echo "============================================================"
echo "ARC-CEGIS run finished."
echo "Results written to: $TARGET_DIR/results_experiment.json"
echo "============================================================"

