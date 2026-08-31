#!/bin/bash
# ==============================================================================
# Runner script for ARC-CEGIS using Local Ollama
# ==============================================================================

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Ollama Bundled Libraries ────────────────────────────────────────────────
if [[ -d "${SCRIPT_DIR}/.ollama/lib/ollama" ]]; then
    export LD_LIBRARY_PATH="${SCRIPT_DIR}/.ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "$SCRIPT_DIR/.ollama/bin" ]]; then
    export PATH="$SCRIPT_DIR/.ollama/bin:$PATH"
fi

# ── NVIDIA / CUDA Environment ───────────────────────────────────────────────
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

# ── ROCm / AMD Environment ──────────────────────────────────────────────────
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
