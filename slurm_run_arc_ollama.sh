#!/bin/bash
#SBATCH --job-name=arc-ollama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=lunaris
#SBATCH --nodelist=lunaris
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_arc_ollama_%j.out
#SBATCH --error=logs/slurm_arc_ollama_%j.err

set -euo pipefail

report_end() {
    local exit_code=$? 
    echo "Something made me finnish. Last exit code was: $exit_code"
}
trap report_end EXIT

cd "$SCRATCH/arc-agi-1"
SCRIPT_DIR="$(pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

mkdir -p "$SCRIPT_DIR/logs"

MODEL_NAME="${MODEL_NAME:-"gemma4:31b-it-q4_K_M"}"
export MODEL_NAME
export LLM_PROVIDER="ollama"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_BASE_URL="http://${OLLAMA_HOST}/v1"
export API_BASE_URL="$OLLAMA_BASE_URL"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$SCRIPT_DIR/.ollama/models}"
export OLLAMA_BIN="${OLLAMA_BIN:-$SCRIPT_DIR/.ollama/bin/ollama}"
export PATH="$SCRIPT_DIR/.ollama/bin:$PATH"

# ── ROCm Environment Setup ──────────────────────────────────────────────────
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
if [[ -d "${ROCM_PATH}/bin" ]]; then
    export PATH="${ROCM_PATH}/bin:$PATH"
fi
if [[ -d "${SCRIPT_DIR}/.ollama/lib/ollama" ]]; then
    export LD_LIBRARY_PATH="${SCRIPT_DIR}/.ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
elif [[ -d "${ROCM_PATH}/lib" ]]; then
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${SCRIPT_DIR}/.ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
fi

# Ensure HSA/ROCm target override if set
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    export HSA_OVERRIDE_GFX_VERSION
fi

# Ensure ROCm GPU device visibility
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export ROCR_VISIBLE_DEVICES
fi
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES
fi

export MAX_DAILY_REQUESTS=10000000
export MAX_CONCURRENT_TASKS=1
export REQUEST_DELAY=0.1

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "Error: Ollama binary not found at $OLLAMA_BIN" >&2
    echo "Run sbatch slurm_install_ollama.sh first or set OLLAMA_BIN." >&2
    exit 1
fi

if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

TARGET_DIR="$SCRIPT_DIR/experiments/ollama/$MODEL_NAME"
mkdir -p "$TARGET_DIR"

echo "============================================================"
echo "          Slurm: ARC-CEGIS on local Ollama                  "
echo "============================================================"
echo "Workspace: $SCRIPT_DIR"
echo "Model: $MODEL_NAME"
echo "Provider: $LLM_PROVIDER"
echo "Ollama host: $OLLAMA_HOST"
if [[ -d "${ROCM_PATH}" ]]; then
    echo "ROCm Path: $ROCM_PATH"
fi
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    echo "HSA GFX Override: $HSA_OVERRIDE_GFX_VERSION"
fi
echo "Output dir: $TARGET_DIR"

echo "Ensuring Ollama server is available..."
if ! curl -fsS "http://${OLLAMA_HOST}/api/version" >/dev/null 2>&1; then
    echo "Ollama server is not running; starting it in the background..."
    "$OLLAMA_BIN" serve >/tmp/arc_ollama_slurm.log 2>&1 &
    OLLAMA_PID=$!
    trap 'kill "$OLLAMA_PID" >/dev/null 2>&1 || true' EXIT

    for _ in $(seq 1 30); do
        if curl -fsS "http://${OLLAMA_HOST}/api/version" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -fsS "http://${OLLAMA_HOST}/api/version" >/dev/null 2>&1; then
        echo "Error: Ollama server failed to start." >&2
        exit 1
    fi
fi

# if ! "$OLLAMA_BIN" list | grep -q "${MODEL_NAME}"; then
#     echo "Model ${MODEL_NAME} is not installed locally. Pulling it now..."
#     "$OLLAMA_BIN" pull "$MODEL_NAME"
# fi

echo "Running test inference on model: $MODEL_NAME"
"$OLLAMA_BIN" run "$MODEL_NAME" "Respond with OK." >/dev/null

# 3. Check processor allocation (GPU vs CPU)
echo "=== Ollama Device Allocation ==="
"$OLLAMA_BIN" ps

cd "$TARGET_DIR"
python3 "$SCRIPT_DIR/main.py" \
    --provider ollama \
    --model "$MODEL_NAME" \
    --tasks "$SCRIPT_DIR/data" \
    --output "$TARGET_DIR/results_experiment.json" \
    "$@"

echo "============================================================"
echo "ARC-CEGIS run finished."
echo "Results written to: $TARGET_DIR/results_experiment.json"
echo "============================================================"
