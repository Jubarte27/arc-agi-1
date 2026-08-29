#!/usr/bin/env bash
# ==============================================================================
# Model Uninstall Script for Ollama
# Reads a list of models from a file and removes them from Ollama.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_FILE=""
OLLAMA_BIN_OVERRIDE=""
SERVER_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
AUTO_STARTED_SERVER=0
SERVER_PID=""

usage() {
    cat <<EOF
Usage: $(basename "$0") <MODELS_FILE> [OPTIONS]
       $(basename "$0") -f <MODELS_FILE> [OPTIONS]

Removes/deletes all models listed in MODELS_FILE from Ollama.

Arguments:
  MODELS_FILE            Path to text file with model names (one per line)

Options:
  -f, --file <PATH>      Path to models file
  -b, --binary <PATH>    Path to ollama binary (overrides auto-detection)
  -H, --host <HOST:PORT> Ollama server host/address (default: ${SERVER_HOST})
  -h, --help             Show this help message and exit

Environment Variables:
  OLLAMA_BIN             Path to ollama executable
  OLLAMA_HOST            Ollama server address (e.g. 127.0.0.1:11434)

Format of MODELS_FILE:
  # Comments and empty lines are ignored
  qwen2.5-coder:7b
  llama3.2:3b
  deepseek-coder:6.7b

Examples:
  ./$(basename "$0") models.txt
  ./$(basename "$0") models.txt --binary ./.ollama/bin/ollama
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file)
            MODELS_FILE="$2"
            shift 2
            ;;
        -b|--binary)
            OLLAMA_BIN_OVERRIDE="$2"
            shift 2
            ;;
        -H|--host)
            SERVER_HOST="$2"
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

# Locate Ollama binary (same logic as download_models.sh)
find_ollama_binary() {
    if [[ -n "$OLLAMA_BIN_OVERRIDE" && -x "$OLLAMA_BIN_OVERRIDE" ]]; then
        echo "$OLLAMA_BIN_OVERRIDE"
        return 0
    fi
    if [[ -n "${OLLAMA_BIN:-}" && -x "${OLLAMA_BIN:-}" ]]; then
        echo "$OLLAMA_BIN"
        return 0
    fi
    if [[ -x "${SCRIPT_DIR}/.ollama/bin/ollama" ]]; then
        echo "${SCRIPT_DIR}/.ollama/bin/ollama"
        return 0
    fi
    if [[ -x "${SCRIPT_DIR}/bin/ollama" ]]; then
        echo "${SCRIPT_DIR}/bin/ollama"
        return 0
    fi
    if command -v ollama >/dev/null 2>&1; then
        command -v ollama
        return 0
    fi
    return 1
}

if ! OLLAMA_EXEC="$(find_ollama_binary)"; then
    echo "Error: Could not find 'ollama' binary." >&2
    echo "Please run ./install_ollama_portable.sh or set OLLAMA_BIN=/path/to/ollama" >&2
    exit 1
fi

export OLLAMA_HOST="${SERVER_HOST}"

# Helper to normalize host for HTTP requests
get_http_url() {
    local host="$1"
    if [[ "$host" =~ ^https?:// ]]; then
        echo "$host"
    else
        echo "http://${host}"
    fi
}

SERVER_HTTP_URL="$(get_http_url "${SERVER_HOST}")"

is_server_alive() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsS "${SERVER_HTTP_URL}/api/version" >/dev/null 2>&1
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "${SERVER_HTTP_URL}/api/version" >/dev/null 2>&1
    else
        return 1
    fi
}

cleanup_server() {
    if [[ $AUTO_STARTED_SERVER -eq 1 && -n "${SERVER_PID:-}" ]]; then
        echo "Stopping temporary background Ollama server (PID: ${SERVER_PID})..."
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT INT TERM

echo "============================================================"
echo "            Ollama Model Uninstall / Removal                "
echo "============================================================"
echo "Ollama Binary:    ${OLLAMA_EXEC}"
echo "Models File:      ${MODELS_FILE}"
echo "Ollama Host:      ${SERVER_HOST}"

# Ensure server is running
if is_server_alive; then
    echo "Ollama server is already active at ${SERVER_HTTP_URL}."
else
    echo "Ollama server is not active. Starting local instance in background..."
    "${OLLAMA_EXEC}" serve >/dev/null 2>&1 &
    SERVER_PID=$!
    AUTO_STARTED_SERVER=1

    echo "Waiting for Ollama server to become ready..."
    MAX_WAIT=30
    WAIT_COUNT=0
    until is_server_alive; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Error: Ollama server failed to start." >&2
            exit 1
        fi
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
        if [[ $WAIT_COUNT -ge $MAX_WAIT ]]; then
            echo "Error: Timed out waiting for Ollama server after ${MAX_WAIT} seconds." >&2
            exit 1
        fi
    done
    echo "Ollama server started successfully (PID: ${SERVER_PID})."
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

echo "Found ${TOTAL_MODELS} model(s) to remove from '${MODELS_FILE}':"
for m in "${models[@]}"; do
    echo "  - $m"
done
echo ""

SUCCESS_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

INDEX=1
for model in "${models[@]}"; do
    echo "------------------------------------------------------------"
    echo "[$INDEX/$TOTAL_MODELS] Removing model: ${model}"
    echo "------------------------------------------------------------"

    # Check if model is installed before trying to remove
    if ! "${OLLAMA_EXEC}" list 2>/dev/null | grep -q "${model}"; then
        echo "Skipped: model '${model}' is not installed."
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    elif "${OLLAMA_EXEC}" rm "${model}"; then
        echo "Successfully removed: ${model}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "Error: Failed to remove model: ${model}" >&2
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("${model}")
    fi
    INDEX=$((INDEX + 1))
    echo ""
done

echo "============================================================"
echo "                   Uninstall Summary                        "
echo "============================================================"
echo "Total models processed: ${TOTAL_MODELS}"
echo "Successfully removed:   ${SUCCESS_COUNT}"
echo "Skipped (not found):    ${SKIPPED_COUNT}"
echo "Failed:                 ${FAILED_COUNT}"

if [[ $FAILED_COUNT -gt 0 ]]; then
    echo "Failed models:"
    for fm in "${FAILED_MODELS[@]}"; do
        echo "  - ${fm}"
    done
fi

echo ""
echo "Remaining Ollama Models:"
"${OLLAMA_EXEC}" list || true
echo "============================================================"

if [[ $FAILED_COUNT -gt 0 ]]; then
    exit 1
fi

