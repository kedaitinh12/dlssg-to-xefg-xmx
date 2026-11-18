#!/usr/bin/env python3
"""
Static cycle-level performance benchmark of the 17 INT8 layers of XeFG MfgModel (Model 16):
Compares XMX DP4a INT8 (v_dot4_i32_iu8, 512 ops/clk/CU) vs. WMMA INT8 (__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12,
2048 ops/clk/CU) on AMD RDNA 4 GPUs (Radeon RX 9070 XT 64 CU @ 2.97 GHz and Radeon RX 9070 56 CU @ 2.52 GHz).
"""

# Layer table from src/MfgPort/Cuda/cuda_layers.h + xefg_cuda_layers.cu
# (program, kernel, family, in_ch, out_ch, pool, scale_div, runs_per_batch_or_frame)
LAYERS = [
    ("aa016078", "L_aa016078",     "Pool",    16,  16, True,   4, "per_frame"),
    ("198d6bf8", "L_198d6bf8",     "Pool",    16,  32, True,   8, "per_frame"),
    ("dcd8becf", "L_dcd8becf",     "Pool",    32,  64, True,  16, "per_frame"),
    ("efbe3788", "L_efbe3788",     "Pool",    64, 128, True,  32, "per_frame"),
    ("50f0c205", "L_50f0c205",     "Pool",   128,  64, False, 32, "per_frame"),
    ("a3c3caa1", "L_a3c3caa1",     "Pool",    64,  32, False, 16, "per_frame"),
    ("9481d452", "L_9481d452",     "Pool",    16,  16, False,  4, "per_frame"),
    ("06e97933", "L_06e97933",     "Pool",    32,  16, False,  4, "per_frame"),
    ("3ed03daf", "L_3ed03daf",     "Decoder", 16,  16, False,  4, "per_frame"),
    ("748b0baf", "L_748b0baf",     "Decoder", 64,  64, False, 16, "per_frame"),
    ("cb391582", "L_cb391582",     "Decoder", 32,  32, False,  8, "per_frame"),
    ("fc1b8a04", "L_fc1b8a04",     "Flow",   224,  56, False, 16, "per_pair"),
    ("3459485",  "L_3459485",      "Flow",   224,  56, False,  8, "per_pair"),
    ("3452733",  "L_3452733",      "Flow",   224,  56, False,  4, "per_pair"),
    ("55e5d6b9", "L_flow16_acc17", "Flow",    64,  32, False,  8, "per_pair"),
    ("34517926", "L_flow16_acc17", "Flow",    64,  32, False,  4, "per_pair"),
    ("3453044",  "L_3453044",      "Flow",    64,  32, False,  4, "per_pair"),
]

GPUS = [
    ("AMD Radeon RX 9070 (RDNA 4, 56 CU @ 2.52 GHz)",    56, 2.52e9, 640e9),
    ("AMD Radeon RX 9070 XT (RDNA 4, 64 CU @ 2.97 GHz)", 64, 2.97e9, 640e9),
]

def analyze_resolution(width, height, fg_mult):
    gen_frames = fg_mult - 1
    print(f"=== Resolution {width}x{height}, {fg_mult}x FG ({gen_frames} generated frames per real frame) ===")
    for gpu_name, cus, clock, bw in GPUS:
        dp4a_peak_ops_sec = cus * 512 * clock   # 512 INT8 ops/clock/CU for v_dot4_i32_iu8
        wmma_peak_ops_sec = cus * 2048 * clock  # 2048 INT8 ops/clock/CU for RDNA 4 WMMA 16x16x16/32
        total_dp4a_us = 0.0
        total_wmma_us = 0.0
        for prog, kern, fam, cin, cout, pool, sdiv, freq in LAYERS:
            out_w = width // sdiv
            out_h = height // sdiv
            conv_w = out_w * (2 if pool else 1)
            conv_h = out_h * (2 if pool else 1)
            ops_per_call = conv_w * conv_h * 9 * cin * cout * 2
            bytes_per_call = (conv_w * conv_h * cin) + (out_w * out_h * cout) + (9 * cin * cout)
            calls = gen_frames if freq == "per_frame" else 1

            # Roofline + dispatch/LDS latency (~6.5 us per dispatch for DP4a, ~4.2 us for WMMA with SwizzleWeightIdx)
            t_dp4a = max(ops_per_call / (dp4a_peak_ops_sec * 0.62), bytes_per_call / (bw * 0.65)) + 6.5e-6
            t_wmma = max(ops_per_call / (wmma_peak_ops_sec * 0.72), bytes_per_call / (bw * 0.72)) + 4.2e-6

            total_dp4a_us += (t_dp4a * 1e6) * calls / gen_frames
            total_wmma_us += (t_wmma * 1e6) * calls / gen_frames

        speedup = total_dp4a_us / total_wmma_us
        print(f"  {gpu_name}:")
        print(f"    17 INT8 layers per generated frame: XMX DP4a INT8 = {total_dp4a_us/1000:.3f} ms | WMMA INT8 = {total_wmma_us/1000:.3f} ms | Speedup = {speedup:.2f}x")

if __name__ == "__main__":
    analyze_resolution(2560, 1440, 2)
    analyze_resolution(3840, 2160, 4)
