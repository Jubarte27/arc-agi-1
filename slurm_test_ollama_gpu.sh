#!/bin/bash
#SBATCH --job-name=ollama-gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1


#SBATCH --cpus-per-task=16
#SBATCH --partition=lunaris
#SBATCH --nodelist=lunaris

#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --output=logs/ollama_gpu_%j.out
#SBATCH --error=logs/ollama_gpu_%j.err

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-"hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M"}"

cd "$SCRATCH/arc-agi-1"

# Set paths for local/portable Ollama installation if present
PATH="$(pwd)/.ollama/bin:${PATH}"
LD_LIBRARY_PATH="$(pwd)/.ollama/lib/ollama:${LD_LIBRARY_PATH:-}"
OLLAMA_MODELS="$(pwd)/.ollama/models"

export PATH LD_LIBRARY_PATH OLLAMA_MODELS

# 1. Start Ollama server in background
ollama serve &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" >/dev/null 2>&1 || true' EXIT

# Wait for server readiness
until curl -fsS "http://127.0.0.1:11434/api/version" >/dev/null 2>&1; do
    sleep 1
done

# 2. Warm up model with a short prompt
echo "Running inference on model: $MODEL_NAME"
ollama run "$MODEL_NAME" "Respond with OK." >/dev/null

# 3. Check processor allocation (GPU vs CPU)
echo "=== Ollama Device Allocation ==="
ollama ps

