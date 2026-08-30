#!/bin/bash
#SBATCH --job-name=arc-ollama-uninstall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=lunaris
#SBATCH --nodelist=lunaris
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_arc_ollama_uninstall_%j.out
#SBATCH --error=logs/slurm_arc_ollama_uninstall_%j.err

set -euo pipefail

cd "$SCRATCH/arc-agi-1"
SCRIPT_DIR="$(pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$SCRIPT_DIR/.ollama/models}"
export OLLAMA_BIN="${OLLAMA_BIN:-$SCRIPT_DIR/.ollama/bin/ollama}"
export PATH="$SCRIPT_DIR/.ollama/bin:$PATH"

echo "============================================================"
echo "          Slurm: Uninstall Ollama Models                    "
echo "============================================================"
echo "Workspace:     $SCRIPT_DIR"
echo "Ollama host:   $OLLAMA_HOST"
echo "Model storage: $OLLAMA_MODELS"
echo "Binary:        $OLLAMA_BIN"

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "Error: Ollama binary not found at $OLLAMA_BIN" >&2
    echo "Run sbatch slurm_install_ollama.sh first or set OLLAMA_BIN." >&2
    exit 1
fi

./uninstall_models.sh \
    --file "$SCRIPT_DIR/models.txt" \
    --binary "$OLLAMA_BIN" \
    --host "$OLLAMA_HOST"

echo "============================================================"
echo "Ollama model uninstall complete."
echo "============================================================"

