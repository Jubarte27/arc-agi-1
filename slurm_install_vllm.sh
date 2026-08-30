#!/bin/bash
#SBATCH --job-name=arc-vllm-install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=lunaris
#SBATCH --nodelist=lunaris
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_arc_vllm_install_%j.out
#SBATCH --error=logs/slurm_arc_vllm_install_%j.err

# ==============================================================================
# Slurm: Install vLLM + Download Models
# Mirrors slurm_install_ollama.sh for the vLLM backend.
# ==============================================================================

set -euo pipefail

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

# ── Hugging Face Token ──────────────────────────────────────────────────────
# Some models (Llama, Gemma, etc.) are gated and require a HF token.
# Set HF_TOKEN in your environment or add it to envs/.env.vllm:
#   export HF_TOKEN='hf_...'
# Obtain a token at: https://huggingface.co/settings/tokens
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is set."
else
    echo "Warning: HF_TOKEN is not set. Gated model downloads will fail."
    echo "Set it with: export HF_TOKEN='hf_...'"
fi

echo "============================================================"
echo "          Slurm: Install vLLM + Download Models             "
echo "============================================================"
echo "Workspace: $SCRIPT_DIR"

# ── Create venv and install vLLM ────────────────────────────────────────────
echo "Installing vLLM..."
EXTRA_INSTALL_ARGS=()
if [[ "${USE_PREBUILT:-0}" == "1" || "${VLLM_PREBUILT:-0}" == "1" ]]; then
    EXTRA_INSTALL_ARGS+=(--prebuilt)
fi
if [[ -n "${ROCM_ARCH:-}" ]]; then
    EXTRA_INSTALL_ARGS+=(--arch "$ROCM_ARCH")
elif [[ -n "${PYTORCH_ROCM_ARCH:-}" ]]; then
    EXTRA_INSTALL_ARGS+=(--arch "$PYTORCH_ROCM_ARCH")
fi
./install_vllm.sh --venv "$SCRIPT_DIR/.venv" "${EXTRA_INSTALL_ARGS[@]}"

# ── Download models ─────────────────────────────────────────────────────────
echo "Downloading configured models..."
./download_models_vllm.sh \
    --file "$SCRIPT_DIR/models_vllm.txt" \
    --venv "$SCRIPT_DIR/.venv"

echo "============================================================"
echo "vLLM install and model download complete."
echo "Run the ARC-CEGIS job with:"
echo "  sbatch slurm_run_arc_vllm.sh"
echo "============================================================"

