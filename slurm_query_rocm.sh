#!/bin/bash
#SBATCH --job-name=arc-ollama-rocm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=lunaris
#SBATCH --nodelist=lunaris
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_arc_ollama_uninstall_%j.out
#SBATCH --error=logs/slurm_arc_ollama_uninstall_%j.err

set -euo pipefail

report_end() {
    local exit_code=$? 
    echo "Something made me finnish. Last exit code was: $exit_code"
}
trap report_end EXIT

cd "$SCRATCH/arc-agi-1"
SCRIPT_DIR="$(pwd)"
cd "$SCRIPT_DIR"


echo "============================================================"
echo "              ROCm / AMD GPU Information                    "
echo "============================================================"

# ── ROCm Version ────────────────────────────────────────────────────────────
echo ""
echo "--- ROCm Version ---"

if [[ -f /opt/rocm/.info/version ]]; then
    echo "ROCm version (from /opt/rocm/.info/version): $(cat /opt/rocm/.info/version)"
elif [[ -f /opt/rocm/include/rocm-core/rocm_version.h ]]; then
    echo "ROCm version (from rocm_version.h):"
    grep -E "ROCM_VERSION_(MAJOR|MINOR|PATCH)" /opt/rocm/include/rocm-core/rocm_version.h | head -3
elif [[ -d /opt/rocm ]]; then
    # Try to infer from directory name
    rocm_dir="$(readlink -f /opt/rocm 2>/dev/null || echo /opt/rocm)"
    echo "ROCm installation found at: ${rocm_dir}"
    echo "Version file not found; check 'apt list --installed 2>/dev/null | grep rocm' or 'dnf list installed | grep rocm'."
else
    echo "ROCm is NOT installed (no /opt/rocm found)."
fi

# ── rocminfo ────────────────────────────────────────────────────────────────
echo ""
echo "--- rocminfo ---"

if command -v rocminfo >/dev/null 2>&1; then
    echo "rocminfo binary: $(command -v rocminfo)"
    echo ""
    # Show agent (GPU) summary lines
    rocminfo 2>/dev/null | grep -E "^\s*(Name|Marketing Name|Vendor Name|Device Type|Compute Unit|Max Clock)" || echo "(no agent info found)"
else
    echo "rocminfo: not found on PATH."
fi

# ── rocm-smi ────────────────────────────────────────────────────────────────
echo ""
echo "--- rocm-smi ---"

if command -v rocm-smi >/dev/null 2>&1; then
    echo "rocm-smi binary: $(command -v rocm-smi)"
    echo ""
    rocm-smi 2>/dev/null || echo "(rocm-smi returned an error)"
else
    echo "rocm-smi: not found on PATH."
fi

# ── hipcc / HIP Version ────────────────────────────────────────────────────
echo ""
echo "--- HIP Version ---"

if command -v hipcc >/dev/null 2>&1; then
    hipcc --version 2>/dev/null || echo "(hipcc --version failed)"
elif [[ -f /opt/rocm/bin/hipcc ]]; then
    /opt/rocm/bin/hipcc --version 2>/dev/null || echo "(hipcc --version failed)"
else
    echo "hipcc: not found."
fi

# ── Kernel Driver ───────────────────────────────────────────────────────────
echo ""
echo "--- Kernel Driver (amdgpu) ---"

if lsmod 2>/dev/null | grep -q amdgpu; then
    echo "amdgpu kernel module: loaded"
    modinfo amdgpu 2>/dev/null | grep -E "^(version|filename):" || true
else
    echo "amdgpu kernel module: not loaded"
fi

# ── GPU PCI Devices ─────────────────────────────────────────────────────────
echo ""
echo "--- AMD GPU PCI Devices ---"

if command -v lspci >/dev/null 2>&1; then
    lspci 2>/dev/null | grep -iE "vga|display|3d" | grep -i amd || echo "(no AMD GPU found via lspci)"
else
    echo "lspci: not found."
fi

echo ""
echo "============================================================"

echo "============================================================"
echo "complete."
echo "============================================================"

