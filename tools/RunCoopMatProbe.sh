#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/coopmat_probe_report.txt"
if [ ! -x "$DIR/xefg_coopmat_probe" ]; then
    echo "[ERROR] xefg_coopmat_probe not found or not executable next to this script." >&2
    exit 1
fi
echo "Running XeFG Vulkan Cooperative-Matrix Probe (Linux)..."
"$DIR/xefg_coopmat_probe" "$DIR/xefg_mfg_programs/vulkan/coopmat" 2>&1 | tee "$OUT"
echo ""
echo "Report saved to: $OUT"
