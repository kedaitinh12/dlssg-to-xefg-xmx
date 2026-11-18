#!/usr/bin/env python3
"""
Builds the 16 unique SPV_KHR_cooperative_matrix compute shaders (and 17 program aliases + two SHA-256 manifests)
for XeFG's 17 ported INT8 layers from src/vulkan/shaders/xefg_coopmat_layers.comp into <output>/<kernel>.spv.

Usage:
    python tools/GenerateXeFgCoopMat.py --output third_party/xefg_mfg_programs/vulkan/coopmat
"""
import argparse
import hashlib
import pathlib
import shutil
import subprocess
import struct
import sys

# The 16 unique kernels in xefg_cuda::kKernels (src/MfgPort/Cuda/cuda_layers.h & xefg_cuda_layers.cu)
KERNELS = [
    # (name, family, in_blocks, out_blocks, bias, pool, relu, tile_h, zsplit, acc_bits, nt, programs)
    ("L_aa016078",     0,  1,  1, 1, 1, 1, 16, 1, 16, 4, ["aa016078"]),
    ("L_198d6bf8",     0,  1,  2, 0, 1, 1,  8, 1, 16, 4, ["198d6bf8"]),
    ("L_dcd8becf",     0,  2,  4, 0, 1, 1, 16, 2, 16, 4, ["dcd8becf"]),
    ("L_efbe3788",     0,  4,  8, 0, 1, 1,  8, 4, 16, 4, ["efbe3788"]),
    ("L_50f0c205",     0,  8,  4, 0, 0, 1,  4, 4, 16, 4, ["50f0c205"]),
    ("L_a3c3caa1",     0,  4,  2, 0, 0, 0,  8, 1, 16, 4, ["a3c3caa1"]),
    ("L_9481d452",     0,  1,  1, 0, 0, 0, 16, 1, 16, 4, ["9481d452"]),
    ("L_06e97933",     0,  2,  1, 0, 0, 0, 16, 1, 16, 4, ["06e97933"]),
    ("L_3ed03daf",     1,  1,  1, 0, 0, 1, 16, 1, 16, 4, ["3ed03daf"]),
    ("L_748b0baf",     1,  4,  4, 0, 0, 1,  8, 1, 16, 4, ["748b0baf"]),
    ("L_cb391582",     1,  2,  2, 0, 0, 1, 16, 1, 16, 4, ["cb391582"]),
    ("L_fc1b8a04",     2, 14,  8, 0, 0, 0,  8, 7, 16, 4, ["fc1b8a04"]),
    ("L_3459485",      2, 14,  8, 0, 0, 0,  8, 7, 18, 4, ["3459485"]),
    ("L_3452733",      2, 14,  8, 0, 0, 0,  8, 7, 19, 4, ["3452733"]),
    ("L_flow16_acc17", 2,  4, 16, 0, 0, 0,  8, 2, 17, 4, ["55e5d6b9", "34517926"]),
    ("L_3453044",      2,  4, 16, 0, 0, 0,  8, 2, 18, 4, ["3453044"]),
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
    alias_manifest = args.output / "coopmat_programs.tsv"
    manifest.unlink(missing_ok=True)
    alias_manifest.unlink(missing_ok=True)
    rows = []
    tsv_lines = ["program\tkernel\tfamily\tin\tout\ttile_h\tgrid_z\tbytes\tsha256"]

    for (name, family, in_b, out_b, bias, pool, relu, tile_h, zsplit, acc_bits, nt, programs) in KERNELS:
        out_spv = args.output / f"{name}.spv"
        out_spv.unlink(missing_ok=True)
        for prog in programs:
            (args.output / f"{prog}.spv").unlink(missing_ok=True)
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
        sha256 = hashlib.sha256(data).hexdigest()
        rows.append(f"{name}\t{sha256}\t{len(data)}\n")
        for prog in programs:
            prog_spv = args.output / f"{prog}.spv"
            shutil.copyfile(out_spv, prog_spv)
            tsv_lines.append(f"{prog}\t{name}\t{family}\t{in_b}\t{out_b}\t{tile_h}\t{zsplit}\t{len(data)}\t{sha256}")
        print(f"ok   {out_spv.name} ({len(data)} bytes, sha256={sha256[:12]}...)")

    alias_manifest.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    manifest.write_text("".join(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
