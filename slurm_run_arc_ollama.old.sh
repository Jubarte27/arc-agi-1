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

cd "$SCRATCH/arc-agi-1"
SCRIPT_DIR="$(pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

mkdir -p "$SCRIPT_DIR/logs"

MODEL_NAME="${MODEL_NAME:-lfm2.5}"
export MODEL_NAME
export LLM_PROVIDER="ollama"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_BASE_URL="http://${OLLAMA_HOST}/v1"
export API_BASE_URL="$OLLAMA_BASE_URL"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$SCRIPT_DIR/.ollama/models}"
export OLLAMA_BIN="${OLLAMA_BIN:-$SCRIPT_DIR/.ollama/bin/ollama}"
export PATH="$SCRIPT_DIR/.ollama/bin:$PATH"

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

if ! "$OLLAMA_BIN" list | grep -q "${MODEL_NAME}"; then
    echo "Model ${MODEL_NAME} is not installed locally. Pulling it now..."
    "$OLLAMA_BIN" pull "$MODEL_NAME"
fi

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
