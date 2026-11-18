#!/usr/bin/env python3
"""
Builds the 16 unique SPV_KHR_cooperative_matrix compute shaders for XeFG's 17 ported INT8 layers
from src/vulkan/shaders/xefg_coopmat_layers.comp into <output>/vulkan/coopmat/<kernel>.spv.

Usage:
    python tools/GenerateXeFgCoopMat.py --output third_party/xefg_mfg_programs/vulkan/coopmat
"""
import argparse
import hashlib
import struct
import pathlib
import shutil
import subprocess
import sys

# The 16 unique kernels in xefg_cuda::kKernels (src/MfgPort/Cuda/cuda_layers.h)
KERNELS = [
    # (name, family, in_blocks, out_blocks, bias, pool, relu, tile_h, zsplit, acc_bits, nt)
    ("L_aa016078",     0,  1,  1, 1, 1, 1, 16, 1, 16, 4),
    ("L_198d6bf8",     0,  1,  2, 0, 1, 1,  8, 1, 16, 4),
    ("L_dcd8becf",     0,  2,  4, 0, 1, 1, 16, 2, 16, 4),
    ("L_efbe3788",     0,  4,  8, 0, 1, 1,  8, 4, 16, 4),
    ("L_50f0c205",     0,  8,  4, 0, 0, 1,  4, 4, 16, 4),
    ("L_a3c3caa1",     0,  4,  2, 0, 0, 0,  8, 1, 16, 4),
    ("L_9481d452",     0,  1,  1, 0, 0, 0, 16, 1, 16, 4),
    ("L_06e97933",     0,  2,  1, 0, 0, 0, 16, 1, 16, 4),
    ("L_3ed03daf",     1,  1,  1, 0, 0, 0, 16, 1, 16, 4),
    ("L_748b0baf",     1,  4,  4, 0, 0, 0,  8, 1, 16, 4),
    ("L_cb391582",     1,  2,  2, 0, 0, 0, 16, 1, 16, 4),
    ("L_fc1b8a04",     2, 14,  8, 0, 0, 0,  8, 7, 16, 4),
    ("L_3459485",      2, 14,  8, 0, 0, 0,  8, 7, 18, 4),
    ("L_3452733",      2, 14,  8, 0, 0, 0,  8, 7, 19, 4),
    ("L_flow16_acc17", 2,  4, 16, 0, 0, 0,  8, 2, 17, 4),
    ("L_3453044",      2,  4, 16, 0, 0, 0,  8, 2, 18, 4),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "src" / "vulkan" / "shaders" / "xefg_coopmat_layers.comp",
        help="Path to xefg_coopmat_layers.comp",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output directory for <kernel>.spv files",
    )
    parser.add_argument(
        "--glslang",
        type=str,
        default=shutil.which("glslangValidator") or "glslangValidator",
        help="Path to glslangValidator executable",
    )
    args = parser.parse_args()

    if not args.source.is_file():
        print(f"error: shader source not found: {args.source}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    manifest = args.output / "coopmat_layers.tsv"
    manifest.unlink(missing_ok=True)
    rows = []

    for (name, family, in_b, out_b, bias, pool, relu, tile_h, zsplit, acc_bits, nt) in KERNELS:
        out_spv = args.output / f"{name}.spv"
        out_spv.unlink(missing_ok=True)
        cmd = [
            args.glslang,
            "-V",
            "--target-env", "vulkan1.3",
            "-S", "comp",
            f"-DLAYER_FAMILY={family}",
            f"-DLAYER_IN={in_b}",
            f"-DLAYER_OUT={out_b}",
            f"-DLAYER_BIAS={bias}",
            f"-DLAYER_POOL={pool}",
            f"-DLAYER_RELU={relu}",
            f"-DLAYER_TILE_H={tile_h}",
            f"-DLAYER_ZSPLIT={zsplit}",
            f"-DLAYER_ACC_BITS={acc_bits}",
            f"-DLAYER_NT={nt}",
            "-o", str(out_spv),
            str(args.source),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAIL {name}: {res.stderr or res.stdout}", file=sys.stderr)
            return res.returncode
        data = out_spv.read_bytes()
        if len(data) < 20 or len(data) % 4 or struct.unpack_from("<I", data)[0] != 0x07230203:
            print(f"error: invalid SPIR-V output: {out_spv}", file=sys.stderr)
            return 1
        rows.append(f"{name}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\n")
        print(f"ok   {out_spv.name} ({len(data)} bytes)")

    manifest.write_text("".join(rows), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
