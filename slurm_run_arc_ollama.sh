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

if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

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

# ── Ollama Bundled Libraries ────────────────────────────────────────────────
if [[ -d "${SCRIPT_DIR}/.ollama/lib/ollama" ]]; then
    export LD_LIBRARY_PATH="${SCRIPT_DIR}/.ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
fi

# ── NVIDIA / CUDA Environment Setup ─────────────────────────────────────────
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
if [[ -d "${CUDA_PATH}/bin" ]]; then
    export PATH="${CUDA_PATH}/bin:$PATH"
fi
if [[ -d "${CUDA_PATH}/lib64" ]]; then
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
elif [[ -d "${CUDA_PATH}/lib" ]]; then
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi

# Ensure CUDA GPU device visibility (Slurm sets CUDA_VISIBLE_DEVICES)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
fi

# ── AMD / ROCm Environment Setup ────────────────────────────────────────────
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
if [[ -d "${ROCM_PATH}/bin" ]]; then
    export PATH="${ROCM_PATH}/bin:$PATH"
fi
if [[ -d "${ROCM_PATH}/lib" ]]; then
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi

# Ensure HSA/ROCm target override if set
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    export HSA_OVERRIDE_GFX_VERSION
fi

# Ensure ROCm GPU device visibility (Slurm sets ROCR/HIP_VISIBLE_DEVICES on ROCm nodes)
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

TARGET_DIR="$SCRIPT_DIR/experiments/ollama/$MODEL_NAME"
mkdir -p "$TARGET_DIR"

echo "============================================================"
echo "          Slurm: ARC-CEGIS on local Ollama                  "
echo "============================================================"
echo "Workspace:      $SCRIPT_DIR"
echo "Model:          $MODEL_NAME"
echo "Provider:       $LLM_PROVIDER"
echo "Ollama host:    $OLLAMA_HOST"
echo "Model storage:  $OLLAMA_MODELS"
echo "Ollama binary:  $OLLAMA_BIN"

# Display detected GPU hardware info
echo "------------------------------------------------------------"
echo "GPU Environment Diagnostics:"
DETECTED_GPU=0

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPU=1
    echo "[NVIDIA CUDA Detected]"
    nvidia-smi -L 2>/dev/null || nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  NVIDIA GPU present"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    fi
    if [[ -d "${CUDA_PATH}" ]]; then
        echo "  CUDA_PATH:            $CUDA_PATH"
    fi
fi

if command -v rocminfo >/dev/null 2>&1 || [[ -d "${ROCM_PATH}" ]] || command -v rocm-smi >/dev/null 2>&1; then
    DETECTED_GPU=1
    echo "[AMD ROCm Detected]"
    if [[ -d "${ROCM_PATH}" ]]; then
        echo "  ROCm Path:            $ROCM_PATH"
    fi
    if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
        echo "  HSA GFX Override:     $HSA_OVERRIDE_GFX_VERSION"
    fi
    if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
        echo "  ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES"
    fi
    if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
        echo "  HIP_VISIBLE_DEVICES:  $HIP_VISIBLE_DEVICES"
    fi
fi

if [[ $DETECTED_GPU -eq 0 ]]; then
    echo "[No Dedicated GPU Detected - Falling back to CPU]"
fi
echo "------------------------------------------------------------"
echo "Output dir:     $TARGET_DIR"

echo "Ensuring Ollama server is available..."
if ! curl -fsS "http://${OLLAMA_HOST}/api/version" >/dev/null 2>&1; then
    echo "Ollama server is not running; starting it in the background..."
    "$OLLAMA_BIN" serve > "$SCRIPT_DIR/logs/ollama_server.log" 2>&1 &
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
        echo "Check logs at: $SCRIPT_DIR/logs/ollama_server.log" >&2
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
DOTENV="$SCRIPT_DIR/envs/.env:$SCRIPT_DIR/envs/.env.ollama" \
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
