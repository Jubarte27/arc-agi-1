#!/usr/bin/env bash
# ==============================================================================
# vLLM Installer Script
# Installs vLLM into the project's .venv virtual environment.
# Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
BUILD_DIR="${VLLM_BUILD_DIR:-${SCRIPT_DIR}/.vllm-src}"
WHEEL_DIR="${VLLM_WHEEL_DIR:-${SCRIPT_DIR}/wheels}"
VLLM_VERSION="${VLLM_VERSION:-v0.7.3}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 4)}"
FORCE=0
GPU_BACKEND=""  # auto-detect by default

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Compiles and installs the vLLM Python wheel into the project virtual environment (.venv).
Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs.

Options:
  -d, --venv <PATH>       Path to virtual environment (default: ${VENV_DIR})
  -g, --gpu <BACKEND>     Force GPU backend: 'cuda' or 'rocm' (default: auto-detect)
  -a, --arch <ARCH>       ROCm target architecture (e.g. gfx90a, gfx942, gfx1100, gfx1030)
  -v, --version <TAG>     vLLM git branch/tag to build (default: ${VLLM_VERSION})
  -w, --wheel-dir <DIR>   Output directory for compiled wheel (default: ${WHEEL_DIR})
  -b, --build-dir <DIR>   Source checkout / build directory (default: ${BUILD_DIR})
  -j, --jobs <N>          Parallel build jobs (default: ${MAX_JOBS})
  -f, --force             Recompile and reinstall even if vLLM is already present
  -h, --help              Show this help message and exit

Environment Variables:
  VENV_DIR                Overrides default virtual environment path
  VLLM_BUILD_DIR          Source checkout directory
  VLLM_WHEEL_DIR          Output directory for generated wheels
  VLLM_VERSION            vLLM git tag or branch to build (default: v0.7.3)
  MAX_JOBS                Parallel compilation worker count
  ROCM_PATH               Path to ROCm installation (default: /opt/rocm)
  PYTORCH_ROCM_ARCH       ROCm target architecture (e.g. gfx90a, gfx942, gfx1100)
  HF_TOKEN                Hugging Face token for gated model access (optional)

Examples:
  ./$(basename "$0")
  ./$(basename "$0") --gpu rocm --arch gfx90a -j 8
  ./$(basename "$0") --version v0.7.3 --force
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--venv)
            VENV_DIR="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_BACKEND="$2"
            shift 2
            ;;
        -v|--version)
            VLLM_VERSION="$2"
            shift 2
            ;;
        -w|--wheel-dir)
            WHEEL_DIR="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -j|--jobs)
            MAX_JOBS="$2"
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
echo "          vLLM Python Wheel Builder & Installer             "
echo "============================================================"

# ── Detect GPU Backend ──────────────────────────────────────────────────────
detect_gpu_backend() {
    if [[ -n "$GPU_BACKEND" ]]; then
        echo "$GPU_BACKEND"
        return
    fi

    # Check for NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "cuda"
        return
    fi

    # Check for AMD GPU (ROCm)
    if command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1; then
        echo "rocm"
        return
    fi
    if [[ -d "/opt/rocm" ]]; then
        echo "rocm"
        return
    fi

    echo "unknown"
}

GPU_BACKEND="$(detect_gpu_backend)"
echo "Detected GPU Backend: ${GPU_BACKEND}"

# ── Detect ROCm Architecture ────────────────────────────────────────────────
detect_rocm_arch() {
    # 1. Direct user override via CLI (--arch) or environment variable
    if [[ -n "${PYTORCH_ROCM_ARCH:-}" && "${PYTORCH_ROCM_ARCH}" != "native" ]]; then
        echo "${PYTORCH_ROCM_ARCH}"
        return
    fi
    if [[ -n "${ROCM_ARCH:-}" && "${ROCM_ARCH}" != "native" ]]; then
        echo "${ROCM_ARCH}"
        return
    fi

    local detected=""
    local supported="gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx950 gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx1250"

    # 2. Query rocminfo
    local rocminfo_bin=""
    if command -v rocminfo >/dev/null 2>&1; then
        rocminfo_bin="$(command -v rocminfo)"
    elif [[ -x "${ROCM_PATH:-/opt/rocm}/bin/rocminfo" ]]; then
        rocminfo_bin="${ROCM_PATH:-/opt/rocm}/bin/rocminfo"
    fi

    if [[ -n "$rocminfo_bin" ]]; then
        detected="$("$rocminfo_bin" 2>/dev/null | grep -E "(Name:\s+gfx|amdgcn-amd-amdhsa--gfx)" | grep -o 'gfx[0-9a-z]*' | head -n 1 || true)"
    fi

    # 3. Query KFD sysfs topology (useful on Linux without rocminfo in PATH)
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

    # 4. Query rocm-smi / lspci device names
    if [[ -z "$detected" ]] && command -v lspci >/dev/null 2>&1; then
        local pci_info
        pci_info="$(lspci 2>/dev/null | grep -i "VGA\|3D\|Display" | grep -i "AMD" || true)"
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

    # 5. Validate detected architecture against supported list
    if [[ -n "$detected" ]]; then
        for arch in $supported; do
            if [[ "$detected" == "$arch" ]]; then
                echo "$detected"
                return
            fi
        done
    fi

    # 6. Default fallback for cluster nodes (MI200 / MI300 series)
    echo "gfx90a;gfx942"
}

if [[ "$GPU_BACKEND" == "rocm" ]]; then
    echo "Selected ROCm Arch: $(detect_rocm_arch)"
fi

if [[ "$GPU_BACKEND" == "unknown" ]]; then
    echo "Warning: No NVIDIA (CUDA) or AMD (ROCm) GPU detected." >&2
    echo "vLLM requires a supported GPU. Attempting CUDA build as fallback." >&2
    GPU_BACKEND="cuda"
fi

# ── Ensure uv is available ──────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "Using uv: $(command -v uv) ($(uv --version))"

# ── Create / Activate Virtual Environment ───────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at: ${VENV_DIR}"
    uv venv "$VENV_DIR"
fi

echo "Activating virtual environment: ${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ── Check if Already Installed ──────────────────────────────────────────────
if [ "$FORCE" == "0" ]; then
    if python3 -c "import vllm; print(f'vLLM {vllm.__version__} is already installed.')" 2>/dev/null; then
        echo "vLLM is already installed. Use --force to recompile and reinstall."
        exit 0
    fi
fi

# ── Ensure Rust toolchain is available (required for setuptools-rust in vLLM) ─
if ! command -v cargo >/dev/null 2>&1; then
    if [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "Installing Rust toolchain (cargo/rustc)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    if [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "Using Cargo: $(command -v cargo 2>/dev/null || echo 'not found') ($(cargo --version 2>/dev/null || true))"

# ── Handle Prebuilt Installation ───────────────────────────────────────────
if [ "$USE_PREBUILT" == "1" ]; then
    echo "============================================================"
    echo "Installing precompiled vLLM (${GPU_BACKEND})..."
    echo "============================================================"
    if [[ "$GPU_BACKEND" == "rocm" ]]; then
        DETECTED_ROCM_VER="$(detect_rocm_version)"
        ROCM_TORCH_INDEX="${ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm${DETECTED_ROCM_VER}}"
        echo "Using PyTorch ROCm index: ${ROCM_TORCH_INDEX}"
        uv pip install torch torchvision --index-url "${ROCM_TORCH_INDEX}"
        uv pip install vllm --extra-index-url "${ROCM_TORCH_INDEX}"
    else
        uv pip install torch torchvision
        uv pip install vllm
    fi

    cd "$SCRIPT_DIR"
    if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
        echo "Installing project requirements..."
        uv pip install -r "${SCRIPT_DIR}/requirements.txt"
    fi

    echo ""
    echo "Verifying vLLM installation..."
    if python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"; then
        echo "============================================================"
        echo "Precompiled vLLM installed successfully!"
        echo "============================================================"
        exit 0
    else
        echo "Error: Precompiled vLLM installation failed." >&2
        exit 1
    fi
fi

# ── Install Build Prerequisites ─────────────────────────────────────────────
echo "Installing build prerequisites (ninja, cmake, wheel, build, setuptools, setuptools-rust, setuptools-scm, packaging, jinja2, psutil)..."
uv pip install ninja cmake wheel build setuptools setuptools-rust setuptools-scm packaging jinja2 psutil

if [[ "$GPU_BACKEND" == "rocm" ]]; then
    DETECTED_ROCM_VER="$(detect_rocm_version)"
    ROCM_TORCH_INDEX="${ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm${DETECTED_ROCM_VER}}"
    echo "Installing prebuilt PyTorch for ROCm (${DETECTED_ROCM_VER}) from: ${ROCM_TORCH_INDEX}..."
    uv pip install torch torchvision --index-url "${ROCM_TORCH_INDEX}" || uv pip install torch torchvision
else
    echo "Installing prebuilt PyTorch for CUDA..."
    uv pip install torch torchvision
fi

# ── Fetch vLLM Source Code ──────────────────────────────────────────────────
mkdir -p "$(dirname "$BUILD_DIR")"
mkdir -p "$WHEEL_DIR"

if [[ -d "$BUILD_DIR/.git" ]]; then
    echo "Updating existing vLLM source at: ${BUILD_DIR}"
    cd "$BUILD_DIR"
    git fetch --tags origin
    if [ "$FORCE" == "1" ]; then
        git reset --hard "origin/${VLLM_VERSION}" 2>/dev/null || git checkout "${VLLM_VERSION}"
        rm -rf build .deps *.egg-info
    fi
    git submodule update --init --recursive || true
else
    echo "Cloning vLLM source (${VLLM_VERSION}) into: ${BUILD_DIR}..."
    git clone --branch "${VLLM_VERSION}" --depth 1 --recurse-submodules https://github.com/vllm-project/vllm.git "$BUILD_DIR" 2>/dev/null || \
    git clone --recurse-submodules https://github.com/vllm-project/vllm.git "$BUILD_DIR"
    cd "$BUILD_DIR"
    git checkout "${VLLM_VERSION}" 2>/dev/null || true
    git submodule update --init --recursive || true
fi

cd "$BUILD_DIR"

# Clean stale build cache if present
rm -rf build .deps

# Install build/runtime requirements from source if available
if [[ -f "requirements-build.txt" ]]; then
    uv pip install -r requirements-build.txt || true
fi
if [[ "$GPU_BACKEND" == "rocm" && -f "requirements-rocm.txt" ]]; then
    uv pip install -r requirements-rocm.txt || true
elif [[ "$GPU_BACKEND" == "cuda" && -f "requirements-cuda.txt" ]]; then
    uv pip install -r requirements-cuda.txt || true
fi

# ── Configure Compilation Environment ───────────────────────────────────────
export MAX_JOBS="${MAX_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export VERBOSE=1

# Disable unstable extensions on ROCm/CUDA
export VLLM_BUILD_LIBTORCH_EXT=0
export CMAKE_ARGS="-DVLLM_BUILD_LIBTORCH_EXT=OFF -DVLLM_INSTALL_PUNICA_WRAPPER=OFF ${CMAKE_ARGS:-}"

if [[ "$GPU_BACKEND" == "rocm" ]]; then
    export VLLM_TARGET_DEVICE="rocm"
    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export HIP_PATH="${ROCM_PATH}"
    export HIP_PLATFORM="amd"
    if [[ -d "${ROCM_PATH}/llvm/bin" ]]; then
        export HIP_CLANG_PATH="${ROCM_PATH}/llvm/bin"
        export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
    else
        export PATH="${ROCM_PATH}/bin:${PATH}"
    fi
    export CMAKE_PREFIX_PATH="${ROCM_PATH}:${ROCM_PATH}/lib/cmake:${CMAKE_PREFIX_PATH:-}"
    
    TARGET_ROCM_ARCH="$(detect_rocm_arch)"
    export PYTORCH_ROCM_ARCH="${TARGET_ROCM_ARCH}"
    echo "Configured ROCm build environment: ROCM_PATH=${ROCM_PATH}, PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
else
    export VLLM_TARGET_DEVICE="cuda"
    echo "Configured CUDA build environment (MAX_JOBS=${MAX_JOBS})"
fi

# ── Compile Python Wheel ────────────────────────────────────────────────────
echo "============================================================"
echo "Compiling vLLM Python wheel (parallel jobs: ${MAX_JOBS})..."
echo "============================================================"

# Clean previous wheel builds in output dir if forcing
if [ "$FORCE" == " 1" ]; then
    rm -f "${WHEEL_DIR}"/vllm*.whl
fi

if ! python3 setup.py bdist_wheel --dist-dir "$WHEEL_DIR"; then
    echo ""
    echo "============================================================"
    echo "Warning: Source wheel compilation failed."
    echo "Falling back to official precompiled ROCm/CUDA vLLM wheel..."
    echo "============================================================"
    if [[ "$GPU_BACKEND" == "rocm" ]]; then
        DETECTED_ROCM_VER="$(detect_rocm_version)"
        ROCM_TORCH_INDEX="${ROCM_TORCH_INDEX:-https://download.pytorch.org/whl/rocm${DETECTED_ROCM_VER}}"
        uv pip install vllm --extra-index-url "${ROCM_TORCH_INDEX}"
    else
        uv pip install vllm
    fi
fi

COMPILED_WHEEL="$(find "$WHEEL_DIR" -maxdepth 1 -name "vllm*.whl" -type f 2>/dev/null | sort -V | tail -n 1 || true)"

if [[ -n "$COMPILED_WHEEL" && -f "$COMPILED_WHEEL" ]]; then
    echo "Successfully built wheel: ${COMPILED_WHEEL}"
    echo "Installing compiled wheel into virtual environment..."
    uv pip install --no-deps --force-reinstall "$COMPILED_WHEEL"
    uv pip install -r requirements-common.txt 2>/dev/null || true
fi

# ── Install Project Requirements ────────────────────────────────────────────
cd "$SCRIPT_DIR"
if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
    echo "Installing project requirements..."
    uv pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

# ── Verify Installation ────────────────────────────────────────────────────
echo ""
echo "Verifying vLLM installation..."
if python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"; then
    echo ""
    echo "============================================================"
    echo "             Installation Complete!                         "
    echo "============================================================"
    echo ""
    echo "GPU Backend:     ${GPU_BACKEND}"
    echo "Virtual Env:     ${VENV_DIR}"
    echo "Compiled Wheel:  ${COMPILED_WHEEL}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Download models:"
    echo "   ./download_models_vllm.sh models_vllm.txt"
    echo ""
    echo "2. (Optional) Set Hugging Face token for gated models:"
    echo "   export HF_TOKEN='hf_...'"
    echo "   Or add HF_TOKEN=hf_... to envs/.env.vllm"
    echo ""
    echo "3. Run an experiment:"
    echo "   ./run_vllm.sh"
    echo "============================================================"
else
    echo "Error: vLLM installation verification failed." >&2
    exit 1
fi

