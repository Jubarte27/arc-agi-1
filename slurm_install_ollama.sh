#!/bin/bash
#SBATCH --job-name=arc-ollama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=lunaris
#SBATCH --nodelist=lunaris
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_arc_ollama_%j.out
#SBATCH --error=logs/slurm_arc_ollama_%j.err

set -euo pipefail

report_end() {
    local exit_code=$? 
    echo "Something made me finnish. Last exit code was: $exit_code"
}
trap report_end EXIT

cd "$SCRATCH"
if [[ ! -d "arc-agi-1" ]]; then
    echo "Cloning arc-agi-1 repository..."
    git clone --branch ollama https://github.com/Jubarte27/arc-agi-1
    cd arc-agi-1
else
    echo "arc-agi-1 repository already exists. Pulling latest changes..."
    cd arc-agi-1
    git pull origin ollama
fi
SCRIPT_DIR="$(pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
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
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    export HSA_OVERRIDE_GFX_VERSION
fi
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export ROCR_VISIBLE_DEVICES
fi
if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES
fi

echo "============================================================"
echo "          Slurm: install portable Ollama + models           "
echo "============================================================"
echo "Workspace:      $SCRIPT_DIR"
echo "Ollama host:    $OLLAMA_HOST"
echo "Model storage:  $OLLAMA_MODELS"
echo "Binary:         $OLLAMA_BIN"

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
    echo "[No Dedicated GPU Detected - Defaulting to CPU / portable build]"
fi
echo "------------------------------------------------------------"

echo "Installing portable Ollama..."
EXTRA_INSTALL_ARGS=()
if [[ -n "${GPU_BACKEND:-}" ]]; then
    EXTRA_INSTALL_ARGS+=(--gpu "$GPU_BACKEND")
elif [[ "${OLLAMA_ROCM:-auto}" == "1" || "${OLLAMA_ROCM:-auto}" == "true" || "${ROCM:-0}" == "1" ]]; then
    EXTRA_INSTALL_ARGS+=(--rocm)
elif [[ "${OLLAMA_CUDA:-auto}" == "1" || "${OLLAMA_CUDA:-auto}" == "true" || "${CUDA:-0}" == "1" ]]; then
    EXTRA_INSTALL_ARGS+=(--gpu "cuda")
fi

if [[ -n "${ROCM_ARCH:-}" ]]; then
    EXTRA_INSTALL_ARGS+=(--arch "$ROCM_ARCH")
fi

./install_ollama_portable.sh --dir "$SCRIPT_DIR/.ollama" --version "${OLLAMA_VERSION:-latest}" "${EXTRA_INSTALL_ARGS[@]}"

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "Error: Ollama binary not found at $OLLAMA_BIN" >&2
    exit 1
fi

echo "Downloading configured models..."
./download_models.sh \
    --file "$SCRIPT_DIR/models.txt" \
    --binary "$OLLAMA_BIN" \
    --host "$OLLAMA_HOST" \
    --models-dir "$OLLAMA_MODELS"

cd "$SCRIPT_DIR"

if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "============================================================"
echo "Portable Ollama install and model pull complete."
echo "Run the ARC-CEGIS job with:"
echo "  sbatch slurm_run_arc_ollama.sh"
echo "============================================================"
