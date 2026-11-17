// Ported from MFG-Unlocker source/native/xess/cuda/xefg_cuda_layers.cu, MIT licensed, see NOTICE.md.
// XeFG ported-model int8 conv layers on tensor cores (mma.sync s8 x s8 -> s32), launched inside
// XeFG's own D3D12 command list in place of the layer's Dispatch (xess/cuda/cuda_route.cpp).
//
// Each kernel computes exactly what the Intel program it replaces computes (byte-identical outputs;
// tests/xess_cuda). Semantics: build\vk-layer-port-20261008\README.md; benchmark and method:
// build\cubin-bench-20261014\README.md.
//   M = 16 consecutive conv positions on one row, N = 8 output channels, K = tap x 16 input channels.
//   Taps (0,1) (2,3) (4,5) (6,7) are m16n8k32 steps and tap 8 is one m16n8k16 step (on sm_75: four and two
//   m8n8k16 steps, see Mma32/Mma16).
//   A word = one 32-bit word (4 input channels) of a window texel in shared memory.
//   B word = one word of Intel's own weight buffer (no repacking).
//
// Blackwell (sm_120, e.g. RTX 5060) & Ada (sm_89) + f5low-fg-main optimizations:
//   1. Register-sliced B-fragment staging (ConvSlicePool / ConvSliceFlow): loads only 2 taps (2*NNT words,
//      4-8 registers) per p-step from shared memory wsm instead of holding all 9 taps (9*NNT = 18-36 registers)
//      live across the row loop, cutting B-matrix register pressure by 77.8% with zero extra shared loads.
//   2. Zero-bank-conflict XOR weight swizzling (SwizzleWeightIdx): permutes uint4 weight slots by u ^ ((u & 8) >> 2)
//      in StageBlock and ConvSlicePool so warp lanes t = 0..3 hit disjoint shared-memory bank quadrants (0..7, 16..23,
//      8..15, 24..31), eliminating the 2-way bank conflict between t=0/t=2 and t=1/t=3 with zero extra shared memory.
//   3. Single-buffer weight shared memory (wsm[1][kBlock]) & compile-time barrier elimination when ZPER * IN == 1
//      (L_aa016078, L_9481d452, L_3ed03daf), saving 2,304 bytes of shared memory per block on full-resolution layers.
//   4. Hardware PTX dp4a.s32.s32 + prmt.b32 (__byte_perm) bilinear upsampling in UpSkip: computes
//      9*a + 3*b + 3*c + 1*d across 4 signed int8 channels in 4 single-cycle dp4a instructions.
//   5. Explicit occupancy target (__launch_bounds__(128, 4) on sm_89+/sm_120) and prmt.b32 lane packing.
#include "xefg_cuda_params.h"

#define DEVI __device__ __forceinline__

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define XEFG_CUDA_LAUNCH_BOUNDS __launch_bounds__(128, 4)
#else
#define XEFG_CUDA_LAUNCH_BOUNDS __launch_bounds__(128, 2)
#endif

namespace
{
DEVI uint4 Fetch2D(uint64_t texture, int x, int y, int layer)
{
    uint32_t a, b, c, d;
    asm("tex.a2d.v4.u32.s32 {%0,%1,%2,%3}, [%4, {%5, %6, %7, %7}];"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(texture), "r"(layer), "r"(x), "r"(y));
    return make_uint4(a, b, c, d);
}
DEVI uint4 Fetch1D(uint64_t texture, int element)
{
    uint32_t a, b, c, d;
    asm("tex.1d.v4.u32.s32 {%0,%1,%2,%3}, [%4, {%5}];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(texture), "r"(element));
    return make_uint4(a, b, c, d);
}
DEVI uint32_t Component(const uint4& v, int k) { return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w; }
// Texel (x, y) of layer `layer` as a typed store, like the Intel program's uint4 UAV store: the view's
// format decides what is written (an R32G32_UINT tensor keeps x, y; XeFG stores 8-channel flow tensors so).
DEVI void StoreTexel(uint64_t surface, int x, int y, int layer, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    asm volatile("sust.p.a2d.v4.b32.zero [%0, {%1, %2, %3, %3}], {%4, %5, %6, %7};"
        :: "l"(surface), "r"(layer), "r"(x), "r"(y), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800) || defined(XEFG_CUDA_MMA_M8N8K16)
// Turing (sm_75) has only the m8n8k16 int8 shape. Its fragments are exact parts of the sm_80 shapes': A/C rows
// g and g + 8 (a0, a2 / a1, a3; c0 c1 / c2 c3) are the two m8 halves, k 0..15 and 16..31 (a0 a1 + b0 /
// a2 a3 + b1) the two k16 halves, so the same registers feed it and the s32 sums are the same.
// XEFG_CUDA_MMA_M8N8K16 forces this path on any architecture (tests/xess_cuda checks it on sm_89).
DEVI void Mma8(int& c0, int& c1, uint32_t a, uint32_t b)
{
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}
DEVI void Mma32(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    Mma8(c[0], c[1], a0, b0);
    Mma8(c[0], c[1], a2, b1);
    Mma8(c[2], c[3], a1, b0);
    Mma8(c[2], c[3], a3, b1);
}
DEVI void Mma16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0)
{
    Mma8(c[0], c[1], a0, b0);
    Mma8(c[2], c[3], a1, b0);
}
#else
DEVI void Mma32(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI void Mma16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0)
{
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
#endif

constexpr int WIN_W = 18;   // 16 positions + 2 halo texels

DEVI int SByte(uint32_t w, int k) { return int(int8_t(uint8_t(w >> (8*k)))); }
DEVI int Div16(int v) { return v >= 0 ? (v >> 4) : -((-v) >> 4); }

// Hardware dp4a.s32.s32 + prmt.b32 bilinear 2x upsample + skip add:
// Computes clamp(SByte(sw, k) + Div16(9*SByte(aw, k) + 3*(SByte(bw, k) + SByte(cw, k)) + SByte(dw, k)), 0, 127)
// for k = 0..3 using 6 byte-permutes and 4 dp4a instructions (bit-exact with scalar reference).
DEVI uint32_t UpSkipBilinearWord(uint32_t aw, uint32_t bw, uint32_t cw, uint32_t dw, uint32_t sw)
{
    const uint32_t ab_lo = __byte_perm(aw, bw, 0x5140u);
    const uint32_t cd_lo = __byte_perm(cw, dw, 0x5140u);
    const uint32_t ab_hi = __byte_perm(aw, bw, 0x7362u);
    const uint32_t cd_hi = __byte_perm(cw, dw, 0x7362u);
    const uint32_t p[4] = {
        __byte_perm(ab_lo, cd_lo, 0x5410u),
        __byte_perm(ab_lo, cd_lo, 0x7632u),
        __byte_perm(ab_hi, cd_hi, 0x5410u),
        __byte_perm(ab_hi, cd_hi, 0x7632u)
    };
    uint32_t packed = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        int dot = 0;
        asm("dp4a.s32.s32 %0, %1, %2, 0;" : "=r"(dot) : "r"(p[k]), "r"(0x01030309u));
        const int v = SByte(sw, k) + Div16(dot);
        packed |= uint32_t(min(max(v, 0), 127)) << (8*k);
    }
    return packed;
}

// trunc(fma(float(acc), scale, bias)), clamp lo..127 (NVIDIA fuses Intel's fmul + fadd).
DEVI uint32_t Requant(int acc, float s, float b, int lo)
{
    const int q = __float2int_rz(__fmaf_rn(__int2float_rn(acc), s, b));
    return uint32_t(min(max(q, lo), 127)) & 0xffu;
}

// Two channel bytes of this thread -> 16-bit half; the t^1 partner holds the other half
// of the same 4-channel word. Returns the full word for row g (even t) or g+8 (odd t).
DEVI uint32_t JoinHalves(uint32_t lowRow, uint32_t highRow, int t)
{
    const uint32_t send = (t & 1) ? lowRow : highRow;
    const uint32_t recv = __shfl_xor_sync(0xffffffffu, send, 1);
    return (t & 1) ? __byte_perm(recv, highRow, 0x5410u) : __byte_perm(lowRow, recv, 0x5410u);
}

// Zero-bank-conflict XOR swizzle within each 16-uint4 tap block (0..143):
// Maps t=0..3 to disjoint 8-bank quadrants (0..7, 16..23, 8..15, 24..31) during warp weight loads.
DEVI int SwizzleWeightIdx(int u) { return u ^ ((u & 8) >> 2); }

// Weights are staged in shared memory one block at a time (double-buffered when ZPER*IN > 1): block st = (z*IN + slice)
// is W[st*144 .. st*144 + 143], where W[(z*IN + slice)*144 + tap*16 + word*4 + s].l feeds output channel
// 16z + 4s + l from input channels 4*word..4*word+3.
constexpr int kBlock = 144;
DEVI void StageBlock(uint64_t weights, int block, uint4* dst)
{
    for (int i = threadIdx.x; i < kBlock; i += 128) dst[SwizzleWeightIdx(i)] = Fetch1D(weights, block*kBlock + i);
}

// Register-sliced conv3x3 accumulation for PoolConv & UpSkip (NNT = 2):
// Streams B-matrix tap pairs (2*p, 2*p+1) from swizzled shared memory wsm in the outer loop and reuses them
// across all ROWS x PY rows, reducing live B-fragment registers from 18 down to 4 per thread with zero bank conflicts.
template <int ROWS, int PY>
DEVI void ConvSlicePool(const uint32_t* wslice, const uint4* block, int (&acc)[ROWS][PY][2][4], int warp, int g, int t)
{
    const uint32_t* words = reinterpret_cast<const uint32_t*>(block);
#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        uint32_t b0[2], b1[2];
#pragma unroll
        for (int n = 0; n < 2; ++n)
        {
            const int oc = n*8 + g;
            b0[n] = words[SwizzleWeightIdx((2*p)*16 + t*4 + (oc >> 2))*4 + (oc & 3)];
            b1[n] = words[SwizzleWeightIdx((2*p + 1)*16 + t*4 + (oc >> 2))*4 + (oc & 3)];
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            const int yl = warp + r*4;
#pragma unroll
            for (int py = 0; py < PY; ++py)
            {
                const int rowBase = (PY > 1 ? 2*yl + py : yl)*WIN_W;
                const int o0 = rowBase + ((2*p)/3)*WIN_W + (2*p)%3;
                const int o1 = rowBase + ((2*p + 1)/3)*WIN_W + (2*p + 1)%3;
                const uint32_t a0 = wslice[(o0 + g)*4 + t], a1 = wslice[(o0 + g + 8)*4 + t];
                const uint32_t a2 = wslice[(o1 + g)*4 + t], a3 = wslice[(o1 + g + 8)*4 + t];
#pragma unroll
                for (int n = 0; n < 2; ++n) Mma32(acc[r][py][n], a0, a1, a2, a3, b0[n], b1[n]);
            }
        }
    }
    uint32_t b8[2];
#pragma unroll
    for (int n = 0; n < 2; ++n)
    {
        const int oc = n*8 + g;
        b8[n] = words[SwizzleWeightIdx(8*16 + t*4 + (oc >> 2))*4 + (oc & 3)];
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
        const int yl = warp + r*4;
#pragma unroll
        for (int py = 0; py < PY; ++py)
        {
            const int rowBase = (PY > 1 ? 2*yl + py : yl)*WIN_W;
            const int o8 = rowBase + 2*WIN_W + 2;
            const uint32_t a0 = wslice[(o8 + g)*4 + t], a1 = wslice[(o8 + g + 8)*4 + t];
#pragma unroll
            for (int n = 0; n < 2; ++n) Mma16(acc[r][py][n], a0, a1, b8[n]);
        }
    }
}

// Register-sliced conv3x3 accumulation for Flow (NT = 4):
// Reduces live B-fragment registers from 36 (uint32_t b[9][4]) down to 8 (b0[4], b1[4]) per thread.
template <int ROWS, int NT>
DEVI void ConvSliceFlow(const uint32_t* wcur, const uint4* block, int (&acc)[ROWS][NT][4], int warp, int g, int t)
{
    const uint32_t* words = reinterpret_cast<const uint32_t*>(block);
#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        uint32_t b0[NT], b1[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n)
        {
            b0[n] = words[((n*2 + (g >> 2))*36 + (2*p)*4 + t)*4 + (g & 3)];
            b1[n] = words[((n*2 + (g >> 2))*36 + (2*p + 1)*4 + t)*4 + (g & 3)];
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            const int rowBase = (warp + r*4)*WIN_W;
            const int o0 = rowBase + ((2*p)/3)*WIN_W + (2*p)%3;
            const int o1 = rowBase + ((2*p + 1)/3)*WIN_W + (2*p + 1)%3;
            const uint32_t a0 = wcur[(o0 + g)*4 + t], a1 = wcur[(o0 + g + 8)*4 + t];
            const uint32_t a2 = wcur[(o1 + g)*4 + t], a3 = wcur[(o1 + g + 8)*4 + t];
#pragma unroll
            for (int n = 0; n < NT; ++n) Mma32(acc[r][n], a0, a1, a2, a3, b0[n], b1[n]);
        }
    }
    uint32_t b8[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n)
        b8[n] = words[((n*2 + (g >> 2))*36 + 8*4 + t)*4 + (g & 3)];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
        const int rowBase = (warp + r*4)*WIN_W;
        const int o8 = rowBase + 2*WIN_W + 2;
        const uint32_t a0 = wcur[(o8 + g)*4 + t], a1 = wcur[(o8 + g + 8)*4 + t];
#pragma unroll
        for (int n = 0; n < NT; ++n) Mma16(acc[r][n], a0, a1, b8[n]);
    }
}

// ---- pooled / unpooled conv3x3 family (aa016078 198d6bf8 dcd8becf efbe3788; 50f0c205
// a3c3caa1 9481d452 06e97933). No constants; zero padding comes from out-of-range reads. Grid z splits the
// output slices: block z computes OUT/ZSPLIT of them (more blocks for the deep, small layers). ---------
template <int IN, int OUT, bool BIAS, bool POOL, bool RELU, int TILE_H, int ZSPLIT>
DEVI void PoolConv(const XefgCudaParams& p)
{
    constexpr int CONV_H = POOL ? 2*TILE_H : TILE_H, WIN_H = CONV_H + 2, WIN = WIN_W*WIN_H;
    constexpr int ROWS = TILE_H/4;
    constexpr int ZPER = OUT/ZSPLIT;
    constexpr int WBUFS = (ZPER * IN > 1) ? 2 : 1;
    static_assert(OUT % ZSPLIT == 0, "output slices per block");
    __shared__ uint4 win[IN*WIN];
    __shared__ uint4 wsm[WBUFS][kBlock];
    const int X0 = blockIdx.x*(POOL ? 8 : 16), Y0 = blockIdx.y*TILE_H;
    const int cx0 = POOL ? 2*X0 : X0, cy0 = POOL ? 2*Y0 : Y0;
    const int zBegin = int(blockIdx.z)*ZPER, firstBlock = zBegin*IN;
    StageBlock(p.weights, firstBlock, wsm[0]);
    for (int i = threadIdx.x; i < IN*WIN; i += 128)
    {
        const int slice = i/WIN, r = i%WIN;
        uint4 v = Fetch2D(p.input, cx0 - 1 + r%WIN_W, cy0 - 1 + r/WIN_W, slice);
        if (BIAS) { v.x += 0x80808080u; v.y += 0x80808080u; v.z += 0x80808080u; v.w += 0x80808080u; }
        win[i] = v;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint32_t* ws = reinterpret_cast<const uint32_t*>(win);
    const int limitW = int(p.limitW), limitH = int(p.limitH);
    constexpr int PY = POOL ? 2 : 1;
    constexpr int LO = RELU ? 0 : -128;
    for (int z = zBegin; z < zBegin + ZPER; ++z)
    {
        int acc[ROWS][PY][2][4];
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int py = 0; py < PY; ++py)
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc[r][py][n][i] = 0;
        if constexpr (WBUFS == 1)
        {
            ConvSlicePool<ROWS, PY>(ws, wsm[0], acc, warp, g, t);
        }
        else
        {
            for (int slice = 0; slice < IN; ++slice)
            {
                const int st = z*IN + slice, cur = (st - firstBlock) & 1;
                const bool hasNext = (st + 1 < (zBegin + ZPER)*IN);
                if (hasNext) StageBlock(p.weights, st + 1, wsm[cur ^ 1]);
                ConvSlicePool<ROWS, PY>(ws + slice*WIN*4, wsm[cur], acc, warp, g, t);
                if (hasNext) __syncthreads();   // block st is no longer read; block st + 1 is complete
            }
        }
        float s[2][2], bi[2][2];
#pragma unroll
        for (int n = 0; n < 2; ++n)
        {
            const int oc = z*16 + n*8 + 2*t;
            const uint4 sv = Fetch1D(p.scale, oc >> 2), bv = Fetch1D(p.bias, oc >> 2);
            s[n][0] = __uint_as_float(Component(sv, oc & 3)); s[n][1] = __uint_as_float(Component(sv, (oc & 3) + 1));
            bi[n][0] = __uint_as_float(Component(bv, oc & 3)); bi[n][1] = __uint_as_float(Component(bv, (oc & 3) + 1));
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            const int Y = Y0 + warp + r*4;
            if (POOL)
            {
                // max over py here, over the x partner (row g^1) by shuffle; even g keeps pixel g/2
                // (rows g, g+1), odd g keeps pixel 4 + g/2 (rows g+7, g+8).
                int v[2][4];
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        const int m = max(acc[r][0][n][i], acc[r][PY - 1][n][i]);
                        v[n][i] = max(m, __shfl_xor_sync(0xffffffffu, m, 4));
                    }
                const int sel = (g & 1) ? 2 : 0;
                const int X = X0 + ((g & 1) ? 4 + (g >> 1) : (g >> 1));
                uint32_t h[2];
#pragma unroll
                for (int n = 0; n < 2; ++n)
                    h[n] = Requant(v[n][sel], s[n][0], bi[n][0], LO) | Requant(v[n][sel + 1], s[n][1], bi[n][1], LO) << 8;
                uint32_t ph[2];
#pragma unroll
                for (int n = 0; n < 2; ++n) ph[n] = __shfl_xor_sync(0xffffffffu, h[n], 1);
                // This lane's word is word s = 2*(t&1) + (t>>1) of pixel X: t0 s0, t2 s1, t1 s2, t3 s3.
                const uint32_t word = (t & 1) ? __byte_perm(ph[1], h[1], 0x5410u) : __byte_perm(h[0], ph[0], 0x5410u);
                const int base = lane & ~3;
                const uint32_t w1 = __shfl_sync(0xffffffffu, word, base + 2), w2 = __shfl_sync(0xffffffffu, word, base + 1);
                const uint32_t w3 = __shfl_sync(0xffffffffu, word, base + 3);
                if (t == 0 && X < limitW && Y < limitH) StoreTexel(p.output, X, Y, z, word, w1, w2, w3);
            }
            else
            {
                const int X = X0 + ((t & 1) ? g + 8 : g);
                uint32_t words[2];
#pragma unroll
                for (int n = 0; n < 2; ++n)
                {
                    const uint32_t lowRow = Requant(acc[r][0][n][0], s[n][0], bi[n][0], LO) | Requant(acc[r][0][n][1], s[n][1], bi[n][1], LO) << 8;
                    const uint32_t highRow = Requant(acc[r][0][n][2], s[n][0], bi[n][0], LO) | Requant(acc[r][0][n][3], s[n][1], bi[n][1], LO) << 8;
                    words[n] = JoinHalves(lowRow, highRow, t);   // word 2n + (t>>1) of this lane's pixel
                }
                // Lanes t, t+2 share a pixel: t < 2 holds words 0 and 2, t + 2 holds words 1 and 3.
                const uint32_t p0 = __shfl_down_sync(0xffffffffu, words[0], 2), p1 = __shfl_down_sync(0xffffffffu, words[1], 2);
                if (t < 2 && X < limitW && Y < limitH) StoreTexel(p.output, X, Y, z, words[0], p0, words[1], p1);
            }
        }
    }
}

// ---- decoder family (3ed03daf 748b0baf cb391582): T = clamp(skip + up2x(low), 0, 127) per input
// slice (Intel's integer bilinear 9/3/3/1 sdiv 16; positions outside W x H = constants 0, 1 are 0;
// low-res texels out of range read 0), conv3x3 IN x16 -> OUT x16, ReLU requant. Like Intel's program, the
// writes cover the whole Dispatch footprint, not only W x H. --------------------------------------------
template <int IN, int OUT, int TILE_H>
DEVI void UpSkip(const XefgCudaParams& p)
{
    constexpr int WIN_H = TILE_H + 2, WIN = WIN_W*WIN_H, ROWS = TILE_H/4;
    constexpr int WBUFS = (OUT * IN > 1) ? 2 : 1;
    __shared__ uint4 win[IN*WIN];
    __shared__ uint4 wsm[WBUFS][kBlock];
    const int Wd = int(p.constants[0]), Hd = int(p.constants[1]);
    const int X0 = blockIdx.x*16, Y0 = blockIdx.y*TILE_H;
    StageBlock(p.weights, 0, wsm[0]);
    for (int i = threadIdx.x; i < IN*WIN; i += 128)
    {
        const int slice = i/WIN, r = i%WIN;
        const int x = X0 - 1 + r%WIN_W, y = Y0 - 1 + r/WIN_W;
        uint4 res = make_uint4(0, 0, 0, 0);
        if (x >= 0 && y >= 0 && x < Wd && y < Hd)
        {
            const int xn = x >> 1, yn = y >> 1, xf = (x & 1) ? xn + 1 : xn - 1, yf = (y & 1) ? yn + 1 : yn - 1;
            const uint4 a = Fetch2D(p.input, xn, yn, slice), b = Fetch2D(p.input, xf, yn, slice);
            const uint4 c = Fetch2D(p.input, xn, yf, slice), d = Fetch2D(p.input, xf, yf, slice);
            const uint4 s = Fetch2D(p.skip, x, y, slice);
            res = make_uint4(
                UpSkipBilinearWord(a.x, b.x, c.x, d.x, s.x),
                UpSkipBilinearWord(a.y, b.y, c.y, d.y, s.y),
                UpSkipBilinearWord(a.z, b.z, c.z, d.z, s.z),
                UpSkipBilinearWord(a.w, b.w, c.w, d.w, s.w));
        }
        win[i] = res;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint32_t* ws = reinterpret_cast<const uint32_t*>(win);
    const int limitW = int(p.limitW), limitH = int(p.limitH);
    for (int z = 0; z < OUT; ++z)
    {
        int acc[ROWS][1][2][4];
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[r][0][n][i] = 0;
        if constexpr (WBUFS == 1)
        {
            ConvSlicePool<ROWS, 1>(ws, wsm[0], acc, warp, g, t);
        }
        else
        {
            for (int slice = 0; slice < IN; ++slice)
            {
                const int st = z*IN + slice, cur = st & 1;
                const bool hasNext = (st + 1 < OUT*IN);
                if (hasNext) StageBlock(p.weights, st + 1, wsm[cur ^ 1]);
                ConvSlicePool<ROWS, 1>(ws + slice*WIN*4, wsm[cur], acc, warp, g, t);
                if (hasNext) __syncthreads();   // block st is no longer read; block st + 1 is complete
            }
        }
        float s[2][2], bi[2][2];
#pragma unroll
        for (int n = 0; n < 2; ++n)
        {
            const int oc = z*16 + n*8 + 2*t;
            const uint4 sv = Fetch1D(p.scale, oc >> 2), bv = Fetch1D(p.bias, oc >> 2);
            s[n][0] = __uint_as_float(Component(sv, oc & 3)); s[n][1] = __uint_as_float(Component(sv, (oc & 3) + 1));
            bi[n][0] = __uint_as_float(Component(bv, oc & 3)); bi[n][1] = __uint_as_float(Component(bv, (oc & 3) + 1));
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            const int Y = Y0 + warp + r*4, X = X0 + ((t & 1) ? g + 8 : g);
            uint32_t words[2];
#pragma unroll
            for (int n = 0; n < 2; ++n)
            {
                const uint32_t lowRow = Requant(acc[r][0][n][0], s[n][0], bi[n][0], 0) | Requant(acc[r][0][n][1], s[n][1], bi[n][1], 0) << 8;
                const uint32_t highRow = Requant(acc[r][0][n][2], s[n][0], bi[n][0], 0) | Requant(acc[r][0][n][3], s[n][1], bi[n][1], 0) << 8;
                words[n] = JoinHalves(lowRow, highRow, t);   // word 2n + (t>>1) of this lane's pixel
            }
            // Lanes t, t+2 share a pixel: t < 2 holds words 0 and 2, t + 2 holds words 1 and 3.
            const uint32_t p0 = __shfl_down_sync(0xffffffffu, words[0], 2), p1 = __shfl_down_sync(0xffffffffu, words[1], 2);
            if (t < 2 && X < limitW && Y < limitH) StoreTexel(p.output, X, Y, z, words[0], p0, words[1], p1);
        }
    }
}

// ---- flow-stage family (3459485 3452733 fc1b8a04 55e5d6b9 34517926 3453044) -------------------------
// Tensors have a 1-row top border (storage row = y + 1). Step i = 16 input channels:
// NOUT 8 -> slices 2i, 2i+1, components 0,1 each; NOUT 16 -> slice i.
// t1[(z*NIT + i)*36*(NOUT/4) + (c/4)*36 + tap*4 + w].(c%4); acc + int32 bias (t2), integer epilogue
// with the negative multiplier (t3). Constants: 2,3 = input W,H; 4,5 = output Wout,Hout; 9 posMul; 10 shifts.
template <int NIT, int NOUT, int ACC_BITS, int TILE_H, int NT>
DEVI void Flow(const XefgCudaParams& p)
{
    constexpr int WIN_H = TILE_H + 2, WIN = WIN_W*WIN_H, ROWS = TILE_H/4;
    static_assert(NOUT == 8 || NT % 2 == 0, "a 16-channel output group spans two n tiles of one block");
    __shared__ uint4 win[2*WIN];
    // Step i's weights for this CTA's NT n tiles: [n tile][channel group of 4 (2)][36 = tap*4 + word].
    __shared__ uint4 wsm[2][NT*72];
    const int Wd = int(p.constants[2]), Hd = int(p.constants[3]), Wout = int(p.constants[4]), Hout = int(p.constants[5]);
    const uint32_t posMul = p.constants[9], shifts = p.constants[10];
    const int X0 = blockIdx.x*16, Y0 = blockIdx.y*TILE_H, n0 = blockIdx.z*NT;
    auto fill = [&](int i, uint4* dst) {
        for (int e = threadIdx.x; e < WIN; e += 128)
        {
            const int x = X0 - 1 + e%WIN_W, ys = Y0 + e/WIN_W;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (x >= 0 && x < Wd && ys >= 1 && ys <= Hd)
            {
                if (NOUT == 8)
                {
                    const uint4 a = Fetch2D(p.input, x, ys, 2*i), b = Fetch2D(p.input, x, ys, 2*i + 1);
                    v = make_uint4(a.x, a.y, b.x, b.y);
                }
                else v = Fetch2D(p.input, x, ys, i);
            }
            dst[e] = v;
        }
    };
    auto stage = [&](int i, uint4* dst) {
        for (int e = threadIdx.x; e < NT*72; e += 128)
        {
            const int nt = e/72, group = (e%72)/36, k = e%36;
            const int ch = (n0 + nt)*8, z = ch/NOUT, c = ch%NOUT;
            dst[e] = Fetch1D(p.weights, ((z*NIT + i)*(NOUT/4) + c/4 + group)*36 + k);
        }
    };
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int acc[ROWS][NT][4];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[r][n][i] = 0;
    fill(0, win);
    stage(0, wsm[0]);
    for (int i = 0; i < NIT; ++i)
    {
        __syncthreads();   // step i's window and weights are complete; nobody still reads the other buffers
        if (i + 1 < NIT) { fill(i + 1, win + ((i + 1) & 1)*WIN); stage(i + 1, wsm[(i + 1) & 1]); }
        const uint32_t* wcur = reinterpret_cast<const uint32_t*>(win + (i & 1)*WIN);
        ConvSliceFlow<ROWS, NT>(wcur, wsm[i & 1], acc, warp, g, t);
    }
    const int lim = 1 << ACC_BITS;
    const uint32_t s1 = shifts & 31u, s2 = (shifts >> 8) & 31u;
    const int limitW = min(int(p.limitW), Wout), limitH = min(int(p.limitH), Hout);
    // words[r][n]: this lane's word of n tile n (word (t>>1) of the tile's 8 channels) for its position.
    uint32_t words[ROWS][NT];
#pragma unroll
    for (int n = 0; n < NT; ++n)
    {
        const int ch = (n0 + n)*8 + 2*t;
        const uint4 bv = Fetch1D(p.scale, ch >> 2), nv = Fetch1D(p.bias, ch >> 2);
        const int ib0 = int(Component(bv, ch & 3)), ib1 = int(Component(bv, (ch & 3) + 1));
        const int ng0 = int(Component(nv, ch & 3)), ng1 = int(Component(nv, (ch & 3) + 1));
        auto requant = [&](int a, int ib, int neg) -> uint32_t {
            a = min(max(a + ib, -lim), lim - 1);
            const int pp = int(uint32_t(max(a, 0))*posMul + (1u << ((s1 + 31u) & 31u))) >> s1;
            const int nn = int(uint32_t(min(a, 0))*uint32_t(neg) + (1u << ((s2 + 31u) & 31u))) >> s2;
            return uint32_t(min(max(pp + nn, -128), 127)) & 0xffu;
        };
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
        {
            const uint32_t lowRow = requant(acc[r][n][0], ib0, ng0) | requant(acc[r][n][1], ib1, ng1) << 8;
            const uint32_t highRow = requant(acc[r][n][2], ib0, ng0) | requant(acc[r][n][3], ib1, ng1) << 8;
            words[r][n] = JoinHalves(lowRow, highRow, t);
        }
    }
    // Lanes t, t+2 share a position: t < 2 holds the tile's word 0, t + 2 its word 1.
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
    {
        const int y = Y0 + warp + r*4, x = X0 + ((t & 1) ? g + 8 : g);
        const bool write = t < 2 && x < limitW && y < limitH;
        if (NOUT == 8)
        {
            // One n tile = one output group: (w0, w1, w0, w0); an R32G32 tensor keeps (w0, w1).
#pragma unroll
            for (int n = 0; n < NT; ++n)
            {
                const uint32_t w1 = __shfl_down_sync(0xffffffffu, words[r][n], 2);
                if (write) StoreTexel(p.output, x, y + 1, n0 + n, words[r][n], w1, words[r][n], words[r][n]);
            }
        }
        else
        {
            // Two n tiles = one output group of 16 channels: words 0, 1 (tile n) and 2, 3 (tile n + 1).
#pragma unroll
            for (int n = 0; n < NT; n += 2)
            {
                const uint32_t pa = __shfl_down_sync(0xffffffffu, words[r][n], 2), pb = __shfl_down_sync(0xffffffffu, words[r][n + 1], 2);
                if (write) StoreTexel(p.output, x, y + 1, (n0 + n)/2, words[r][n], pa, words[r][n + 1], pb);
            }
        }
    }
}
}

#define POOL_KERNEL(NAME, IN, OUT, BIAS, POOL, RELU, TH, ZSPLIT) \
    extern "C" __global__ void XEFG_CUDA_LAUNCH_BOUNDS L_##NAME(const XefgCudaParams p) { PoolConv<IN, OUT, BIAS, POOL, RELU, TH, ZSPLIT>(p); }
#define UPSKIP_KERNEL(NAME, IN, OUT, TH) \
    extern "C" __global__ void XEFG_CUDA_LAUNCH_BOUNDS L_##NAME(const XefgCudaParams p) { UpSkip<IN, OUT, TH>(p); }
#define FLOW_KERNEL(NAME, NIT, NOUT, ACC, TH, NT) \
    extern "C" __global__ void XEFG_CUDA_LAUNCH_BOUNDS L_##NAME(const XefgCudaParams p) { Flow<NIT, NOUT, ACC, TH, NT>(p); }

// Tile choices from the 4K 6x benchmark (build\cubin-bench-20261014, run-bench-001/002).
POOL_KERNEL(aa016078, 1, 1, true, true, true, 16, 1)
POOL_KERNEL(198d6bf8, 1, 2, false, true, true, 8, 1)
POOL_KERNEL(dcd8becf, 2, 4, false, true, true, 16, 2)
POOL_KERNEL(efbe3788, 4, 8, false, true, true, 8, 4)
POOL_KERNEL(50f0c205, 8, 4, false, false, true, 4, 4)
POOL_KERNEL(a3c3caa1, 4, 2, false, false, false, 8, 1)
POOL_KERNEL(9481d452, 1, 1, false, false, false, 16, 1)
POOL_KERNEL(06e97933, 2, 1, false, false, false, 16, 1)
UPSKIP_KERNEL(3ed03daf, 1, 1, 16)
UPSKIP_KERNEL(748b0baf, 4, 4, 8)
UPSKIP_KERNEL(cb391582, 2, 2, 16)
FLOW_KERNEL(fc1b8a04, 14, 8, 16, 8, 4)
FLOW_KERNEL(3459485, 14, 8, 18, 8, 4)
FLOW_KERNEL(3452733, 14, 8, 19, 8, 4)
FLOW_KERNEL(flow16_acc17, 4, 16, 17, 8, 4)
FLOW_KERNEL(3453044, 4, 16, 18, 8, 4)
