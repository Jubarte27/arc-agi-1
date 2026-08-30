#!/usr/bin/env bash
# ==============================================================================
# Portable Ollama Installer Script
# Downloads and extracts the standalone Ollama Linux distribution without sudo.
# Supports CPU, NVIDIA (CUDA), and AMD (ROCm) GPUs.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INSTALL_DIR="${SCRIPT_DIR}/.ollama"
INSTALL_DIR="${DEFAULT_INSTALL_DIR}"
VERSION="latest"
FORCE=0
GPU_BACKEND="${GPU_BACKEND:-auto}"
ROCM_ARCH=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Downloads and extracts a portable/standalone version of Ollama for Linux without root privileges.
Supports CPU, NVIDIA (CUDA), and AMD (ROCm) GPUs.

Options:
  -d, --dir <PATH>      Target installation directory (default: ${DEFAULT_INSTALL_DIR})
  -v, --version <TAG>   Ollama version or release tag (default: latest, e.g. v0.5.12)
  -g, --gpu <BACKEND>   GPU backend: 'rocm', 'cuda', 'cpu', or 'auto' (default: auto)
  -r, --rocm            Enable AMD ROCm GPU acceleration (downloads ROCm runner package)
      --no-rocm         Disable ROCm GPU package download
  -a, --arch <ARCH>     ROCm GPU target architecture override (e.g. gfx90a, gfx942, gfx1100, gfx1030)
  -f, --force           Overwrite existing installation
  -h, --help            Show this help message and exit

Environment Variables:
  OLLAMA_INSTALL_DIR    Overrides default installation directory
  OLLAMA_VERSION        Overrides default version
  OLLAMA_ROCM           Set to 1/true to force ROCm package installation, 0/false to disable
  GPU_BACKEND           Set to 'rocm', 'cuda', 'cpu', or 'auto'
  ROCM_PATH             Path to ROCm installation (default: /opt/rocm)
  ROCM_ARCH             ROCm target architecture (e.g. gfx90a, gfx942, gfx1100, gfx1030)
  HSA_OVERRIDE_GFX_VERSION Target HSA GFX version override (e.g. 10.3.0, 11.0.0, 9.0.0)

Examples:
  ./$(basename "$0")
  ./$(basename "$0") --rocm
  ./$(basename "$0") --gpu rocm --arch gfx90a
  ./$(basename "$0") --dir ./.ollama --version latest --force
EOF
}

# Override from environment variables if set
if [[ -n "${OLLAMA_INSTALL_DIR:-}" ]]; then
    INSTALL_DIR="${OLLAMA_INSTALL_DIR}"
fi
if [[ -n "${OLLAMA_VERSION:-}" ]]; then
    VERSION="${OLLAMA_VERSION}"
fi
if [[ -n "${OLLAMA_ROCM:-}" ]]; then
    case "${OLLAMA_ROCM}" in
        1|true|TRUE|yes|YES) GPU_BACKEND="rocm" ;;
        0|false|FALSE|no|NO) GPU_BACKEND="cpu" ;;
    esac
fi
if [[ -n "${ROCM_ARCH:-}" ]]; then
    ROCM_ARCH="${ROCM_ARCH}"
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
        -g|--gpu)
            GPU_BACKEND="$2"
            shift 2
            ;;
        -r|--rocm)
            GPU_BACKEND="rocm"
            shift
            ;;
        --no-rocm)
            GPU_BACKEND="cpu"
            shift
            ;;
        -a|--arch|--rocm-arch)
            ROCM_ARCH="$2"
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

# ── Detect GPU Hardware / ROCm ──────────────────────────────────────────────
detect_gpu_backend() {
    if [[ "$GPU_BACKEND" != "auto" ]]; then
        echo "$GPU_BACKEND"
        return
    fi

    # 1. Check for AMD ROCm GPU
    local has_rocm=0
    if command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1; then
        has_rocm=1
    elif [[ -x "${ROCM_PATH:-/opt/rocm}/bin/rocminfo" ]] && "${ROCM_PATH:-/opt/rocm}/bin/rocminfo" >/dev/null 2>&1; then
        has_rocm=1
    elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then
        has_rocm=1
    elif [[ -d "/opt/rocm" ]]; then
        has_rocm=1
    elif [[ -d "/sys/class/kfd/kfd/topology/nodes" ]]; then
        has_rocm=1
    elif command -v lspci >/dev/null 2>&1 && (lspci -d '1002:' 2>/dev/null | grep -iqE "VGA|3D|Display"); then
        has_rocm=1
    elif lsmod 2>/dev/null | grep -q amdgpu; then
        has_rocm=1
    fi

    if [[ $has_rocm -eq 1 ]]; then
        echo "rocm"
        return
    fi

    # 2. Check for NVIDIA CUDA GPU
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "cuda"
        return
    fi

    echo "cpu"
}

# ── Detect ROCm Architecture ────────────────────────────────────────────────
detect_rocm_arch() {
    if [[ -n "$ROCM_ARCH" ]]; then
        echo "$ROCM_ARCH"
        return
    fi

    local detected=""
    local rocminfo_bin=""
    if command -v rocminfo >/dev/null 2>&1; then
        rocminfo_bin="$(command -v rocminfo)"
    elif [[ -x "${ROCM_PATH:-/opt/rocm}/bin/rocminfo" ]]; then
        rocminfo_bin="${ROCM_PATH:-/opt/rocm}/bin/rocminfo"
    fi

    if [[ -n "$rocminfo_bin" ]]; then
        detected="$("$rocminfo_bin" 2>/dev/null | grep -E "(Name:\s+gfx|amdgcn-amd-amdhsa--gfx)" | grep -o 'gfx[0-9a-z]*' | head -n 1 || true)"
    fi

    if [[ -z "$detected" && -d "/sys/class/kfd/kfd/topology/nodes" ]]; then
        for prop in /sys/class/kfd/kfd/topology/nodes/*/properties; do
            if [[ -f "$prop" ]]; then
                local ver
                ver="$(grep -i "gfx_target_version" "$prop" 2>/dev/null | awk '{print $2}' || true)"
                case "$ver" in
                    90010) detected="gfx90a" ;;
                    90400|90401|90402) detected="gfx942" ;;
                    90006) detected="gfx906" ;;
                    90008) detected="gfx908" ;;
                    100300) detected="gfx1030" ;;
                    110000) detected="gfx1100" ;;
                    110001) detected="gfx1101" ;;
                    110002) detected="gfx1102" ;;
                    *) ;;
                esac
                if [[ -n "$detected" ]]; then
                    break
                fi
            fi
        done
    fi

    if [[ -z "$detected" ]] && command -v lspci >/dev/null 2>&1; then
        local pci_info
        pci_info="$(lspci 2>/dev/null | grep -iE "VGA|3D|Display" | grep -i "AMD" || true)"
        if echo "$pci_info" | grep -iqE "MI200|MI210|MI250"; then
            detected="gfx90a"
        elif echo "$pci_info" | grep -iqE "MI300"; then
            detected="gfx942"
        elif echo "$pci_info" | grep -iqE "MI100"; then
            detected="gfx908"
        elif echo "$pci_info" | grep -iqE "7900"; then
            detected="gfx1100"
        elif echo "$pci_info" | grep -iqE "6800|6900"; then
            detected="gfx1030"
        fi
    fi

    echo "${detected:-unknown}"
}

RESOLVED_GPU="$(detect_gpu_backend)"
INSTALL_ROCM=0
if [[ "$RESOLVED_GPU" == "rocm" ]]; then
    INSTALL_ROCM=1
fi

echo "Detected Platform: Linux (${OLLAMA_ARCH})"
echo "GPU Backend:       ${RESOLVED_GPU}"
if [[ $INSTALL_ROCM -eq 1 ]]; then
    DETECTED_ARCH="$(detect_rocm_arch)"
    echo "ROCm Arch:         ${DETECTED_ARCH}"
fi
echo "Target Directory:  ${INSTALL_DIR}"
echo "Requested Version: ${VERSION}"

OLLAMA_BIN="${INSTALL_DIR}/bin/ollama"
ROCM_RUNNERS_INSTALLED=0
if [[ -d "${INSTALL_DIR}/lib/ollama" ]] && find "${INSTALL_DIR}/lib/ollama" -maxdepth 3 -name "*rocm*" -print -quit 2>/dev/null | grep -q .; then
    ROCM_RUNNERS_INSTALLED=1
fi

# Check if already installed
if [[ -x "$OLLAMA_BIN" && $FORCE -eq 0 ]]; then
    if [[ $INSTALL_ROCM -eq 1 && $ROCM_RUNNERS_INSTALLED -eq 0 ]]; then
        echo ""
        echo "Base Ollama binary found at ${OLLAMA_BIN}, but ROCm runners are missing."
        echo "Proceeding to install ROCm GPU runners package..."
    else
        echo "Ollama is already installed at: ${OLLAMA_BIN}"
        echo -n "Current version: "
        "${OLLAMA_BIN}" --version || true
        if [[ $ROCM_RUNNERS_INSTALLED -eq 1 ]]; then
            echo "ROCm GPU runners: Installed"
        fi
        echo ""
        echo "Use --force to overwrite the existing installation."
        exit 0
    fi
fi

# Determine download URLs
if [[ "$VERSION" == "latest" ]]; then
    BASE_DOWNLOAD_URL="https://ollama.com/download/ollama-linux-${OLLAMA_ARCH}.tar.zst"
    ROCM_DOWNLOAD_URL="https://ollama.com/download/ollama-linux-${OLLAMA_ARCH}-rocm.tar.zst"
else
    # Normalize version tag to include 'v' prefix if omitted
    if [[ ! "$VERSION" =~ ^v ]]; then
        VERSION="v${VERSION}"
    fi
    BASE_DOWNLOAD_URL="https://github.com/ollama/ollama/releases/download/${VERSION}/ollama-linux-${OLLAMA_ARCH}.tar.zst"
    ROCM_DOWNLOAD_URL="https://github.com/ollama/ollama/releases/download/${VERSION}/ollama-linux-${OLLAMA_ARCH}-rocm.tar.zst"
fi

download_file() {
    local url="$1"
    local dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar -o "${dest}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "${dest}" "${url}"
    else
        echo "Error: Neither 'curl' nor 'wget' was found on PATH." >&2
        exit 1
    fi
}

echo "Creating directory: ${INSTALL_DIR} ..."
mkdir -p "${INSTALL_DIR}"

TMP_TAR="$(mktemp "${TMPDIR:-/tmp}/ollama-portable-XXXXXX.tar.zst")"
TMP_ROCM_TAR=""
cleanup() {
    rm -f "${TMP_TAR}"
    if [[ -n "${TMP_ROCM_TAR:-}" ]]; then
        rm -f "${TMP_ROCM_TAR}"
    fi
}
trap cleanup EXIT

# ── Download Base Ollama ───────────────────────────────────────────────────
if [[ ! -x "$OLLAMA_BIN" || $FORCE -eq 1 ]]; then
    echo "Downloading Ollama base package..."
    echo "URL: ${BASE_DOWNLOAD_URL}"
    download_file "${BASE_DOWNLOAD_URL}" "${TMP_TAR}"

    echo "Extracting base package into '${INSTALL_DIR}'..."
    tar -xaf "${TMP_TAR}" -C "${INSTALL_DIR}"
fi

# ── Download ROCm Extension Package ────────────────────────────────────────
if [[ $INSTALL_ROCM -eq 1 ]]; then
    if [[ "$OLLAMA_ARCH" != "amd64" ]]; then
        echo "Warning: ROCm GPU package is currently only provided by Ollama for Linux x86_64 (amd64). Skipping ROCm package." >&2
    else
        TMP_ROCM_TAR="$(mktemp "${TMPDIR:-/tmp}/ollama-rocm-XXXXXX.tar.zst")"
        echo ""
        echo "Downloading Ollama ROCm GPU package..."
        echo "URL: ${ROCM_DOWNLOAD_URL}"
        if download_file "${ROCM_DOWNLOAD_URL}" "${TMP_ROCM_TAR}"; then
            echo "Extracting ROCm GPU runners into '${INSTALL_DIR}'..."
            tar -xaf "${TMP_ROCM_TAR}" -C "${INSTALL_DIR}"
            echo "ROCm GPU package installed successfully."
        else
            echo "Warning: Failed to download ROCm package from ${ROCM_DOWNLOAD_URL}." >&2
            echo "Ollama will fall back to CPU or CUDA." >&2
        fi
    fi
fi

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
echo "GPU Backend:       ${RESOLVED_GPU}"
if [[ $INSTALL_ROCM -eq 1 ]]; then
    echo "ROCm GPU Support:  Enabled"
    if [[ -d "${INSTALL_DIR}/lib/ollama" ]]; then
        echo "ROCm Library Dir:  ${INSTALL_DIR}/lib/ollama"
    fi
fi
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
if [[ $INSTALL_ROCM -eq 1 ]]; then
    echo "4. ROCm environment setup (if needed for unsupported GPU gfx revisions):"
    echo "   export ROCM_PATH=\"\${ROCM_PATH:-/opt/rocm}\""
    echo "   export LD_LIBRARY_PATH=\"${INSTALL_DIR}/lib/ollama:\${ROCM_PATH}/lib:\${LD_LIBRARY_PATH:-}\""
    echo "   # export HSA_OVERRIDE_GFX_VERSION=10.3.0  # (for RX 6000 series, e.g. gfx1032)"
    echo "   # export HSA_OVERRIDE_GFX_VERSION=11.0.0  # (for RX 7000 series, e.g. gfx1102)"
    echo ""
    echo "5. Start local server:"
    echo "   ${OLLAMA_BIN} serve"
else
    echo "4. Start local server:"
    echo "   ${OLLAMA_BIN} serve"
fi
echo "============================================================"


