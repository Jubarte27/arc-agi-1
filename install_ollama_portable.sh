#!/usr/bin/env bash
# ==============================================================================
# Portable Ollama Installer Script
# Downloads and extracts the standalone Ollama Linux distribution without sudo.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INSTALL_DIR="${SCRIPT_DIR}/.ollama"
INSTALL_DIR="${DEFAULT_INSTALL_DIR}"
VERSION="latest"
FORCE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Downloads and extracts a portable/standalone version of Ollama for Linux without root privileges.

Options:
  -d, --dir <PATH>      Target installation directory (default: ${DEFAULT_INSTALL_DIR})
  -v, --version <TAG>   Ollama version or release tag (default: latest, e.g. v0.5.12)
  -f, --force           Overwrite existing installation
  -h, --help            Show this help message and exit

Environment Variables:
  OLLAMA_INSTALL_DIR    Overrides default installation directory
  OLLAMA_VERSION        Overrides default version

Examples:
  ./$(basename "$0")
  ./$(basename "$0") --dir ./.ollama --version latest
  ./$(basename "$0") -d /home/user/ollama_portable
EOF
}

# Override from environment variables if set
if [[ -n "${OLLAMA_INSTALL_DIR:-}" ]]; then
    INSTALL_DIR="${OLLAMA_INSTALL_DIR}"
fi
if [[ -n "${OLLAMA_VERSION:-}" ]]; then
    VERSION="${OLLAMA_VERSION}"
fi

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            usage
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "          Portable Ollama Standalone Installer             "
echo "============================================================"

# Check OS
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
if [[ "$OS" != "linux" ]]; then
    echo "Error: Portable installer currently supports Linux only (detected: $OS)." >&2
    echo "For macOS or Windows, please visit https://ollama.com/download" >&2
    exit 1
fi

# Detect Architecture
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)
        OLLAMA_ARCH="amd64"
        ;;
    aarch64|arm64)
        OLLAMA_ARCH="arm64"
        ;;
    *)
        echo "Error: Unsupported CPU architecture '$ARCH'." >&2
        echo "Supported architectures are x86_64 (amd64) and aarch64 (arm64)." >&2
        exit 1
        ;;
esac

echo "Detected Platform: Linux (${OLLAMA_ARCH})"
echo "Target Directory:  ${INSTALL_DIR}"
echo "Requested Version: ${VERSION}"

OLLAMA_BIN="${INSTALL_DIR}/bin/ollama"

# Check if already installed
if [[ -x "$OLLAMA_BIN" && $FORCE -eq 0 ]]; then
    echo "Ollama is already installed at: ${OLLAMA_BIN}"
    echo -n "Current version: "
    "${OLLAMA_BIN}" --version || true
    echo ""
    echo "Use --force to overwrite the existing installation."
    exit 0
fi

# Determine download URL
if [[ "$VERSION" == "latest" ]]; then
    DOWNLOAD_URL="https://ollama.com/download/ollama-linux-${OLLAMA_ARCH}.tar.gz"
else
    # Normalize version tag to include 'v' prefix if omitted
    if [[ ! "$VERSION" =~ ^v ]]; then
        VERSION="v${VERSION}"
    fi
    DOWNLOAD_URL="https://github.com/ollama/ollama/releases/download/${VERSION}/ollama-linux-${OLLAMA_ARCH}.tar.gz"
fi

echo "Download URL:      ${DOWNLOAD_URL}"
echo "Creating directory: ${INSTALL_DIR} ..."
mkdir -p "${INSTALL_DIR}"

TMP_TAR="$(mktemp "${TMPDIR:-/tmp}/ollama-portable-XXXXXX.tar.gz")"
cleanup() {
    rm -f "${TMP_TAR}"
}
trap cleanup EXIT

echo "Downloading Ollama standalone package..."
if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -o "${TMP_TAR}" "${DOWNLOAD_URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "${TMP_TAR}" "${DOWNLOAD_URL}"
else
    echo "Error: Neither 'curl' nor 'wget' was found on PATH." >&2
    exit 1
fi

echo "Extracting package into '${INSTALL_DIR}'..."
tar -xzf "${TMP_TAR}" -C "${INSTALL_DIR}"

# Ensure executable permissions on binaries and libraries
if [[ -f "${OLLAMA_BIN}" ]]; then
    chmod +x "${OLLAMA_BIN}"
fi
if [[ -d "${INSTALL_DIR}/lib/ollama" ]]; then
    chmod -R u+rX,go+rX "${INSTALL_DIR}/lib/ollama" || true
fi

if [[ ! -x "${OLLAMA_BIN}" ]]; then
    echo "Error: Ollama binary not found or not executable at '${OLLAMA_BIN}'" >&2
    exit 1
fi

echo ""
echo "============================================================"
echo "             Installation Complete!                         "
echo "============================================================"
echo -n "Installed Version: "
"${OLLAMA_BIN}" --version
echo ""
echo "To use this portable Ollama installation:"
echo ""
echo "1. Add to your PATH (optional):"
echo "   export PATH=\"${INSTALL_DIR}/bin:\$PATH\""
echo ""
echo "2. Or set OLLAMA_BIN in your environment:"
echo "   export OLLAMA_BIN=\"${OLLAMA_BIN}\""
echo ""
echo "3. Custom model storage directory (optional):"
echo "   export OLLAMA_MODELS=\"${INSTALL_DIR}/models\""
echo ""
echo "4. Start local server:"
echo "   ${OLLAMA_BIN} serve"
echo "============================================================"

