#!/usr/bin/bash
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

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")
ENV_DIR="$SCRIPT_DIR/envs"

PROVIDER="$1"
shift

if [ -z "$PROVIDER" ]; then
    echo "Error: No provider specified. Please provide a provider (e.g., 'google', 'pool')."
    exit 1
fi

if ! [ -f "$ENV_DIR/.env.$PROVIDER" ]; then
    echo "Error: Environment file for provider '$PROVIDER' does not exist at '$ENV_DIR/.env.$PROVIDER'."
    exit 2
fi

source "$ENV_DIR/.env.$PROVIDER"

#fragil demais
if [ "$PROVIDER" == "pool" ]; then
    MODEL_NAME="$(echo "$LLM_POOL" | cut -d':' -f2 | cut -d',' -f1)"
    LLM_PROVIDER="$(echo "$LLM_POOL" | cut -d':' -f1)"
fi

if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME is not set. Please ensure it is defined in the environment file."
    exit 3
fi

source "$SCRIPT_DIR/.venv/bin/activate"

TARGET_DIR="${TARGET_DIR:-"$SCRIPT_DIR/experiments/$LLM_PROVIDER/$MODEL_NAME"}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

DOTENV="$ENV_DIR/.env:$ENV_DIR/.env.secret:$ENV_DIR/.env.$PROVIDER"  \
    python3 "$SCRIPT_DIR/main.py" \
        --tasks "$SCRIPT_DIR/data100/" \
        --output "$TARGET_DIR/results_experiment.json" \
        "$@"