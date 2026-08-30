#!/usr/bin/env bash
# ==============================================================================
# Model Download Script for vLLM (Hugging Face Hub)
# Reads HF model IDs from a file and downloads them for offline use.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
MODELS_FILE=""

usage() {
    cat <<EOF
Usage: $(basename "$0") <MODELS_FILE> [OPTIONS]
       $(basename "$0") -f <MODELS_FILE> [OPTIONS]

Downloads Hugging Face models listed in MODELS_FILE for offline vLLM serving.

Arguments:
  MODELS_FILE            Path to text file with HF model IDs (one per line)

Options:
  -f, --file <PATH>      Path to models file
  -d, --venv <PATH>      Path to virtual environment (default: ${VENV_DIR})
  -c, --cache <DIR>      Custom HF cache directory (overrides HF_HOME)
  -h, --help             Show this help message and exit

Environment Variables:
  HF_TOKEN               Hugging Face token for gated models (e.g. Llama, Gemma).
                         Obtain a token at https://huggingface.co/settings/tokens
                         and set it before running this script:
                           export HF_TOKEN='hf_...'
  HF_HOME                Root directory for Hugging Face cache (default: ~/.cache/huggingface)
  HUGGINGFACE_HUB_CACHE  Specific cache directory for downloaded model files

Format of MODELS_FILE:
  # Comments and empty lines are ignored
  deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
  Qwen/Qwen3-Coder-30B-A3B-Instruct

Examples:
  ./$(basename "$0") models_vllm.txt
  HF_TOKEN=hf_... ./$(basename "$0") models_vllm.txt
  ./$(basename "$0") models_vllm.txt --cache /scratch/hf_cache
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file)
            MODELS_FILE="$2"
            shift 2
            ;;
        -d|--venv)
            VENV_DIR="$2"
            shift 2
            ;;
        -c|--cache)
            export HF_HOME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option '$1'" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -z "$MODELS_FILE" ]]; then
                MODELS_FILE="$1"
                shift
            else
                echo "Error: Unexpected positional argument '$1'" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$MODELS_FILE" ]]; then
    echo "Error: Missing required MODELS_FILE argument." >&2
    usage
    exit 1
fi

if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Error: Models file '$MODELS_FILE' not found." >&2
    exit 1
fi

# Activate virtual environment
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
else
    echo "Error: Virtual environment not found at '${VENV_DIR}'." >&2
    echo "Run ./install_vllm.sh first." >&2
    exit 1
fi

# Ensure huggingface_hub CLI is available
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "Installing huggingface_hub CLI..."
    pip install --upgrade huggingface_hub
fi

echo "============================================================"
echo "        vLLM Model Download (Hugging Face Hub)             "
echo "============================================================"
echo "Models File:      ${MODELS_FILE}"
echo "Virtual Env:      ${VENV_DIR}"
if [[ -n "${HF_HOME:-}" ]]; then
    echo "HF Cache Dir:     ${HF_HOME}"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF Token:         (set)"
else
    echo "HF Token:         (not set — gated models will fail)"
    echo ""
    echo "  To set a Hugging Face token for gated models:"
    echo "    export HF_TOKEN='hf_...'"
    echo "  Get a token at: https://huggingface.co/settings/tokens"
fi

# Read models from file
models=()
while IFS= read -r line || [[ -n "$line" ]]; do
    trimmed="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    if [[ -z "$trimmed" || "$trimmed" =~ ^# ]]; then
        continue
    fi
    models+=("$trimmed")
done < "$MODELS_FILE"

TOTAL_MODELS=${#models[@]}
if [[ $TOTAL_MODELS -eq 0 ]]; then
    echo "Warning: No valid model entries found in '${MODELS_FILE}'."
    exit 0
fi

echo ""
echo "Found ${TOTAL_MODELS} model(s) to download:"
for m in "${models[@]}"; do
    echo "  - $m"
done
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

INDEX=1
for model in "${models[@]}"; do
    echo "------------------------------------------------------------"
    echo "[$INDEX/$TOTAL_MODELS] Downloading: ${model}"
    echo "------------------------------------------------------------"

    if huggingface-cli download "${model}"; then
        echo "Successfully downloaded: ${model}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "Error: Failed to download model: ${model}" >&2
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("${model}")
    fi
    INDEX=$((INDEX + 1))
    echo ""
done

echo "============================================================"
echo "                   Download Summary                         "
echo "============================================================"
echo "Total models processed: ${TOTAL_MODELS}"
echo "Successfully downloaded: ${SUCCESS_COUNT}"
echo "Failed:                  ${FAILED_COUNT}"

if [[ $FAILED_COUNT -gt 0 ]]; then
    echo "Failed models:"
    for fm in "${FAILED_MODELS[@]}"; do
        echo "  - ${fm}"
    done
    echo ""
    echo "Tip: If a model is gated, set HF_TOKEN before running:"
    echo "  export HF_TOKEN='hf_...'"
fi

echo "============================================================"

if [[ $FAILED_COUNT -gt 0 ]]; then
    exit 1
fi

