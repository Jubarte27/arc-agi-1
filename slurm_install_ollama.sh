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

# ── ROCm Environment Setup ──────────────────────────────────────────────────
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
if [[ -d "${ROCM_PATH}/bin" ]]; then
    export PATH="${ROCM_PATH}/bin:$PATH"
fi
if [[ -d "${ROCM_PATH}/lib" ]]; then
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${SCRIPT_DIR}/.ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    export HSA_OVERRIDE_GFX_VERSION
fi

echo "============================================================"
echo "          Slurm: install portable Ollama + models           "
echo "============================================================"
echo "Workspace: $SCRIPT_DIR"
echo "Ollama host: $OLLAMA_HOST"
echo "Model storage: $OLLAMA_MODELS"
echo "Binary: $OLLAMA_BIN"
if [[ -d "${ROCM_PATH}" ]]; then
    echo "ROCm Path: $ROCM_PATH"
fi

echo "Installing portable Ollama..."
EXTRA_INSTALL_ARGS=()
if [[ "${OLLAMA_ROCM:-auto}" == "1" || "${OLLAMA_ROCM:-auto}" == "true" || "${ROCM:-0}" == "1" || "${GPU_BACKEND:-}" == "rocm" ]]; then
    EXTRA_INSTALL_ARGS+=(--rocm)
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

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


echo "============================================================"
echo "Portable Ollama install and model pull complete."
echo "Run the ARC-CEGIS job with:"
echo "  sbatch slurm_run_arc_ollama.sh"
echo "============================================================"
