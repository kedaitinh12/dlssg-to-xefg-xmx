// SPDX-License-Identifier: MIT
//
// Full 9-tap 3x3 INT8 convolution D3D12 HLSL (Shader Model 6.6) compute shader template for XeFG's
// 17 ported INT8 layers on AMD RDNA 4 / RDNA 3 (Wave32 dot4add_i8packed + WaveReadLaneAt).
//
// Direct bit-exact port of PoolConv, UpSkip (Decoder), and Flow from src/MfgPort/Cuda/xefg_cuda_layers.cu:
//   - Root Parameter 0 (Descriptor Table):
//       t0 : Texture2DArray<uint4> tInput
//       t4 : Texture2DArray<uint4> tSkip (Decoder LAYER_FAMILY == 1 only)
//       u0 : RWTexture2DArray<uint2> uOutput (8-ch Flow) or RWTexture2DArray<uint4> uOutput (all others)
//   - Root Parameter 1 (Descriptor Table):
//       t1 : StructuredBuffer<uint4> tWeights  (matches XeFG's SRVStructured / StructureByteStride=16)
//       t2 : StructuredBuffer<uint4> tScale    (IEEE-754 float scale for Pool/Decoder, int32 bias for Flow)
//       t3 : StructuredBuffer<uint4> tBias     (IEEE-754 float bias for Pool/Decoder, int32 negMul for Flow)
//   - Root Parameter 2 (32-bit Root Constants b0):
//       4 constants (1 uint4) for Pool/Decoder; 12 constants (3 uint4) for Flow.
//
// Compiled per layer via tools/GenerateXeFgWmma.ps1 with dxc.exe -T cs_6_6 -O3 into
// xefg_mfg_programs\wmma\<program>.dxbc.

#ifndef LAYER_FAMILY
#define LAYER_FAMILY 0 // 0 = PoolConv, 1 = UpSkip (Decoder), 2 = Flow
#endif

#ifndef LAYER_IN
#define LAYER_IN 1
#endif

#ifndef LAYER_OUT
#define LAYER_OUT 1
#endif

#ifndef LAYER_BIAS
#define LAYER_BIAS 0
#endif

#ifndef LAYER_POOL
#define LAYER_POOL 0
#endif

#ifndef LAYER_RELU
#define LAYER_RELU 0
#endif

#ifndef LAYER_TILE_H
#define LAYER_TILE_H 16
#endif

#ifndef LAYER_ZSPLIT
#define LAYER_ZSPLIT 1
#endif

#ifndef LAYER_ACC_BITS
#define LAYER_ACC_BITS 16
#endif

#ifndef LAYER_NT
#define LAYER_NT 4
#endif

Texture2DArray<uint4>   tInput   : register(t0);
StructuredBuffer<uint4> tWeights : register(t1);
StructuredBuffer<uint4> tScale   : register(t2);
StructuredBuffer<uint4> tBias    : register(t3);
#if LAYER_FAMILY == 1
Texture2DArray<uint4>   tSkip    : register(t4);
#endif
#if LAYER_FAMILY == 2 && LAYER_OUT == 8
RWTexture2DArray<uint2> uOutput  : register(u0);
#else
RWTexture2DArray<uint4> uOutput  : register(u0);
#endif

cbuffer RootConstants : register(b0)
{
#if LAYER_FAMILY == 2
    uint4 cWords[3];
#else
    uint4 cWords[1];
#endif
};

static const int WIN_W = 18;
#if LAYER_FAMILY == 0 && LAYER_POOL != 0
static const int PY = 2;
#else
static const int PY = 1;
#endif
static const int CONV_H = LAYER_TILE_H * PY;
static const int WIN_H  = CONV_H + 2;
static const int WIN    = WIN_W * WIN_H;
static const int ROWS   = LAYER_TILE_H / 4;

#if LAYER_FAMILY == 2
// Double-buffered input window and weight slab cut Flow loop barriers in half (1 barrier/step instead of 2)
groupshared uint4 win[2][WIN];
groupshared uint4 wsm[2][LAYER_NT * 72];
#else
static const int ZPER  = LAYER_OUT / LAYER_ZSPLIT;
static const int WBUFS = (ZPER * LAYER_IN > 1) ? 2 : 1;
groupshared uint4 win[LAYER_IN * WIN];
groupshared uint4 wsm[WBUFS][144];
#endif

// Zero-bank-conflict XOR weight swizzle within each 16-uint4 tap block (0..143)
uint SwizzleWeightIdx(uint u)
{
    return u ^ ((u & 8u) >> 2u);
}

int SByte(uint w, uint k)
{
    uint shift = k * 8u;
    int b = int((w >> shift) & 0xFFu);
    return (b >= 128) ? (b - 256) : b;
}

int Div16Signed(int v)
{
    return (v >= 0) ? (v >> 4) : -((-v) >> 4);
}

uint4 FetchInputDim(int x, int y, int slice, uint inW, uint inH, uint inD)
{
    if (x < 0 || y < 0 || slice < 0 || uint(x) >= inW || uint(y) >= inH || uint(slice) >= inD)
        return uint4(0u, 0u, 0u, 0u);
    return tInput.Load(int4(x, y, slice, 0));
}

#if LAYER_FAMILY == 1
uint4 FetchSkipDim(int x, int y, int slice, uint skipW, uint skipH, uint skipD)
{
    if (x < 0 || y < 0 || slice < 0 || uint(x) >= skipW || uint(y) >= skipH || uint(slice) >= skipD)
        return uint4(0u, 0u, 0u, 0u);
    return tSkip.Load(int4(x, y, slice, 0));
}

// Exact port of UpSkipBilinearWord (9*a + 3*b + 3*c + d, Div16 + skip clamped to 0..127)
uint UpSkipBilinearWord(uint aw, uint bw, uint cw, uint dw, uint sw)
{
    uint packed = 0u;
    [unroll]
    for (uint k = 0u; k < 4u; ++k)
    {
        int dot = 9 * SByte(aw, k) + 3 * (SByte(bw, k) + SByte(cw, k)) + SByte(dw, k);
        int v = clamp(SByte(sw, k) + Div16Signed(dot), 0, 127);
        packed |= (uint(v) & 0xFFu) << (k * 8u);
    }
    return packed;
}
#endif

#if LAYER_FAMILY != 2
// Exact port of PoolConv / UpSkip Requant: trunc(fma(float(acc), scale, bias)), clamp(lo..127)
uint RequantFloat(int acc, float s, float b, int lo)
{
    int q = int(mad(float(acc), s, b));
    return uint(clamp(q, lo, 127)) & 0xFFu;
}
#else
// Exact port of Flow requantization from src/MfgPort/Cuda/xefg_cuda_layers.cu lines 533-538
uint RequantFlow(int a, int ib, int neg, int lim, uint posMul, uint s1, uint s2)
{
    int clamped = clamp(a + ib, -lim, lim - 1);
    uint roundPos = 1u << ((s1 + 31u) & 31u);
    uint roundNeg = 1u << ((s2 + 31u) & 31u);
    int pp = int(uint(max(clamped, 0)) * posMul + roundPos) >> int(s1);
    int nn = int(uint(min(clamped, 0)) * uint(neg) + roundNeg) >> int(s2);
    return uint(clamp(pp + nn, -128, 127)) & 0xFFu;
}
#endif

[WaveSize(32)]
[numthreads(128, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint tid : SV_GroupIndex)
{
    const uint warp   = tid >> 5u;
    const uint lane   = tid & 31u;
    const uint px     = lane & 15u;
    const uint halfCh = lane >> 4u; // 0 = first 8 channels (lanes 0..15), 1 = next 8 channels (lanes 16..31)

    uint inW, inH, inD;
    tInput.GetDimensions(inW, inH, inD);

    uint outW, outH, outD;
    uOutput.GetDimensions(outW, outH, outD);

#if LAYER_FAMILY != 2
    // ---- Family 0 (PoolConv) & Family 1 (UpSkip Decoder) ----
    const int X0         = int(groupID.x) * ((LAYER_POOL != 0) ? 8 : 16);
    const int Y0         = int(groupID.y) * LAYER_TILE_H;
    const int limitW     = int(outW);
    const int limitH     = int(outH);
    const int lo         = (LAYER_FAMILY == 1 || LAYER_RELU != 0) ? 0 : -128;
    const int zBegin     = int(groupID.z) * ZPER;
    const int firstBlock = zBegin * LAYER_IN;

    // Stage the first weight block while loading the input window
    for (uint wIdx = tid; wIdx < 144u; wIdx += 128u)
        wsm[0][SwizzleWeightIdx(wIdx)] = tWeights[firstBlock * 144 + int(wIdx)];

#if LAYER_FAMILY == 0
    const int cx0 = (LAYER_POOL != 0) ? (2 * X0) : X0;
    const int cy0 = (LAYER_POOL != 0) ? (2 * Y0) : Y0;
    for (uint i = tid; i < uint(LAYER_IN * WIN); i += 128u)
    {
        int slice = int(i) / WIN;
        int r     = int(i) % WIN;
        uint4 v   = FetchInputDim(cx0 - 1 + (r % WIN_W), cy0 - 1 + (r / WIN_W), slice, inW, inH, inD);
#if LAYER_BIAS != 0
        v += uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u);
#endif
        win[i] = v;
    }
#else
    uint skipW, skipH, skipD;
    tSkip.GetDimensions(skipW, skipH, skipD);
    const int Wd = int(cWords[0].x);
    const int Hd = int(cWords[0].y);
    for (uint i = tid; i < uint(LAYER_IN * WIN); i += 128u)
    {
        int slice = int(i) / WIN;
        int r     = int(i) % WIN;
        int x     = X0 - 1 + (r % WIN_W);
        int y     = Y0 - 1 + (r / WIN_W);
        uint4 res = uint4(0u, 0u, 0u, 0u);
        if (x >= 0 && y >= 0 && x < Wd && y < Hd)
        {
            int xn = x >> 1;
            int yn = y >> 1;
            int xf = ((x & 1) != 0) ? (xn + 1) : (xn - 1);
            int yf = ((y & 1) != 0) ? (yn + 1) : (yn - 1);
            uint4 a = FetchInputDim(xn, yn, slice, inW, inH, inD);
            uint4 b = FetchInputDim(xf, yn, slice, inW, inH, inD);
            uint4 c = FetchInputDim(xn, yf, slice, inW, inH, inD);
            uint4 d = FetchInputDim(xf, yf, slice, inW, inH, inD);
            uint4 s = FetchSkipDim(x, y, slice, skipW, skipH, skipD);
            res = uint4(
                UpSkipBilinearWord(a.x, b.x, c.x, d.x, s.x),
                UpSkipBilinearWord(a.y, b.y, c.y, d.y, s.y),
                UpSkipBilinearWord(a.z, b.z, c.z, d.z, s.z),
                UpSkipBilinearWord(a.w, b.w, c.w, d.w, s.w));
        }
        win[i] = res;
    }
#endif
    GroupMemoryBarrierWithGroupSync();

    const uint s0 = halfCh * 2u + 0u;
    const uint s1 = halfCh * 2u + 1u;

    for (int z = zBegin; z < zBegin + ZPER; ++z)
    {
        int acc[ROWS][PY][8];
        [unroll]
        for (int r = 0; r < ROWS; ++r)
        {
            [unroll]
            for (int py = 0; py < PY; ++py)
            {
                [unroll]
                for (int c = 0; c < 8; ++c)
                    acc[r][py][c] = 0;
            }
        }

        for (int slice = 0; slice < LAYER_IN; ++slice)
        {
            const int st = z * LAYER_IN + slice;
            const int cur = (WBUFS > 1) ? ((st - firstBlock) & 1) : 0;
            const bool hasNext = (WBUFS > 1) && (st + 1 < (zBegin + ZPER) * LAYER_IN);
            if (hasNext)
            {
                const int nxt = cur ^ 1;
                for (uint wIdx = tid; wIdx < 144u; wIdx += 128u)
                    wsm[nxt][SwizzleWeightIdx(wIdx)] = tWeights[(st + 1) * 144 + int(wIdx)];
            }

            const int sliceBase = slice * WIN;
            [unroll]
            for (int tap = 0; tap < 9; ++tap)
            {
                const int dy = tap / 3;
                const int dx = tap % 3;
                [unroll]
                for (int w = 0; w < 4; ++w)
                {
                    uint4 wv0 = wsm[cur][SwizzleWeightIdx(uint(tap * 16 + w * 4) + s0)];
                    uint4 wv1 = wsm[cur][SwizzleWeightIdx(uint(tap * 16 + w * 4) + s1)];
                    uint b[8] = { wv0.x, wv0.y, wv0.z, wv0.w, wv1.x, wv1.y, wv1.z, wv1.w };

                    [unroll]
                    for (int r = 0; r < ROWS; ++r)
                    {
                        const int yl = int(warp) + r * 4;
                        [unroll]
                        for (int py = 0; py < PY; ++py)
                        {
                            const int rowY = (PY > 1) ? (2 * yl + py) : yl;
                            uint4 av = win[sliceBase + (rowY + dy) * WIN_W + int(px) + dx];
                            uint a = (w == 0) ? av.x : ((w == 1) ? av.y : ((w == 2) ? av.z : av.w));
                            [unroll]
                            for (int c = 0; c < 8; ++c)
                                acc[r][py][c] = dot4add_i8packed(a, b[c], acc[r][py][c]);
                        }
                    }
                }
            }
            if (hasNext)
                GroupMemoryBarrierWithGroupSync();
        }

        // Load only this thread's 8 channels of scale and bias (cuts 16 float VGPRs vs holding all 16)
        uint4 sv0 = tScale[z * 4 + int(s0)];
        uint4 sv1 = tScale[z * 4 + int(s1)];
        uint4 bv0 = tBias[z * 4 + int(s0)];
        uint4 bv1 = tBias[z * 4 + int(s1)];
        float s[8] = {
            asfloat(sv0.x), asfloat(sv0.y), asfloat(sv0.z), asfloat(sv0.w),
            asfloat(sv1.x), asfloat(sv1.y), asfloat(sv1.z), asfloat(sv1.w)
        };
        float bi[8] = {
            asfloat(bv0.x), asfloat(bv0.y), asfloat(bv0.z), asfloat(bv0.w),
            asfloat(bv1.x), asfloat(bv1.y), asfloat(bv1.z), asfloat(bv1.w)
        };

        // Barrier-free Wave32 epilogue using WaveReadLaneAt (zero LDS, zero GroupMemoryBarrierWithGroupSync)
        [unroll]
        for (int r = 0; r < ROWS; ++r)
        {
            const int Y = Y0 + int(warp) + r * 4;
#if LAYER_POOL != 0
            uint2 halfWords = uint2(0u, 0u);
            [unroll]
            for (int w = 0; w < 2; ++w)
            {
                uint packed = 0u;
                [unroll]
                for (int c = 0; c < 4; ++c)
                {
                    int localC = w * 4 + c;
                    int vMax = max(acc[r][0][localC], acc[r][1][localC]);
                    int hMax = max(vMax, WaveReadLaneAt(vMax, lane ^ 1u));
                    packed |= RequantFloat(hMax, s[localC], bi[localC], lo) << uint(c * 8);
                }
                if (w == 0) halfWords.x = packed;
                else        halfWords.y = packed;
            }
            uint2 hi2 = WaveReadLaneAt(halfWords, lane ^ 16u);
            if (halfCh == 0u && ((px & 1u) == 0u) && Y < limitH)
            {
                const int X = X0 + int(px >> 1u);
                if (X < limitW)
                    uOutput[uint3(uint(X), uint(Y), uint(z))] = uint4(halfWords.x, halfWords.y, hi2.x, hi2.y);
            }
#else
            uint2 halfWords = uint2(0u, 0u);
            [unroll]
            for (int w = 0; w < 2; ++w)
            {
                uint packed = 0u;
                [unroll]
                for (int c = 0; c < 4; ++c)
                {
                    int localC = w * 4 + c;
                    packed |= RequantFloat(acc[r][0][localC], s[localC], bi[localC], lo) << uint(c * 8);
                }
                if (w == 0) halfWords.x = packed;
                else        halfWords.y = packed;
            }
            uint2 hi2 = WaveReadLaneAt(halfWords, lane ^ 16u);
            if (halfCh == 0u && Y < limitH)
            {
                const int X = X0 + int(px);
                if (X < limitW)
                    uOutput[uint3(uint(X), uint(Y), uint(z))] = uint4(halfWords.x, halfWords.y, hi2.x, hi2.y);
            }
#endif
        }
    }

#else
    // ---- Family 2 (Flow: NIT = LAYER_IN, NOUT = LAYER_OUT, TILE_H = 8, NT = 4) ----
    const int  Wd     = int(cWords[0].z);
    const int  Hd     = int(cWords[0].w);
    const int  Wout   = int(cWords[1].x);
    const int  Hout   = int(cWords[1].y);
    const uint posMul = cWords[2].y;
    const uint shifts = cWords[2].z;
    const uint sh1    = shifts & 31u;
    const uint sh2    = (shifts >> 8u) & 31u;
    const int  lim    = 1 << LAYER_ACC_BITS;
    const int  X0     = int(groupID.x) * 16;
    const int  Y0     = int(groupID.y) * LAYER_TILE_H;
    const int  n0     = int(groupID.z) * LAYER_NT;
    const int  limitW = min(int(outW), Wout);
    const int  limitH = min(int(outH), Hout);

    // Each thread accumulates 16 output channels: halfCh == 0 covers tiles nt = 0,1; halfCh == 1 covers tiles nt = 2,3
    int acc[ROWS][16];
    [unroll]
    for (int r = 0; r < ROWS; ++r)
    {
        [unroll]
        for (int c = 0; c < 16; ++c)
            acc[r][c] = 0;
    }

    // Stage step 0 into buffer 0
    for (uint e = tid; e < uint(WIN); e += 128u)
    {
        int x  = X0 - 1 + (int(e) % WIN_W);
        int ys = Y0 + (int(e) / WIN_W);
        uint4 v = uint4(0u, 0u, 0u, 0u);
        if (x >= 0 && x < Wd && ys >= 1 && ys <= Hd)
        {
#if LAYER_OUT == 8
            uint4 a = FetchInputDim(x, ys, 0, inW, inH, inD);
            uint4 b = FetchInputDim(x, ys, 1, inW, inH, inD);
            v = uint4(a.x, a.y, b.x, b.y);
#else
            v = FetchInputDim(x, ys, 0, inW, inH, inD);
#endif
        }
        win[0][e] = v;
    }
    for (uint wIdx = tid; wIdx < uint(LAYER_NT * 72); wIdx += 128u)
    {
        int nt    = int(wIdx) / 72;
        int group = (int(wIdx) % 72) / 36;
        int k     = int(wIdx) % 36;
        int ch    = (n0 + nt) * 8;
        int z     = ch / LAYER_OUT;
        int c     = ch % LAYER_OUT;
        int idx   = ((z * LAYER_IN + 0) * (LAYER_OUT / 4) + (c / 4) + group) * 36 + k;
        wsm[0][wIdx] = tWeights[idx];
    }

    const uint basent = halfCh * 2u;

    for (int step = 0; step < LAYER_IN; ++step)
    {
        GroupMemoryBarrierWithGroupSync();
        const int cur = step & 1;

        // Overlap staging step + 1 into the opposite double-buffer
        if (step + 1 < LAYER_IN)
        {
            const int nxt = cur ^ 1;
            const int nextStep = step + 1;
            for (uint e = tid; e < uint(WIN); e += 128u)
            {
                int x  = X0 - 1 + (int(e) % WIN_W);
                int ys = Y0 + (int(e) / WIN_W);
                uint4 v = uint4(0u, 0u, 0u, 0u);
                if (x >= 0 && x < Wd && ys >= 1 && ys <= Hd)
                {
#if LAYER_OUT == 8
                    uint4 a = FetchInputDim(x, ys, 2 * nextStep, inW, inH, inD);
                    uint4 b = FetchInputDim(x, ys, 2 * nextStep + 1, inW, inH, inD);
                    v = uint4(a.x, a.y, b.x, b.y);
#else
                    v = FetchInputDim(x, ys, nextStep, inW, inH, inD);
#endif
                }
                win[nxt][e] = v;
            }
            for (uint wIdx = tid; wIdx < uint(LAYER_NT * 72); wIdx += 128u)
            {
                int nt    = int(wIdx) / 72;
                int group = (int(wIdx) % 72) / 36;
                int k     = int(wIdx) % 36;
                int ch    = (n0 + nt) * 8;
                int z     = ch / LAYER_OUT;
                int c     = ch % LAYER_OUT;
                int idx   = ((z * LAYER_IN + nextStep) * (LAYER_OUT / 4) + (c / 4) + group) * 36 + k;
                wsm[nxt][wIdx] = tWeights[idx];
            }
        }

        [unroll]
        for (int tap = 0; tap < 9; ++tap)
        {
            const int dy = tap / 3;
            const int dx = tap % 3;
            [unroll]
            for (int w = 0; w < 4; ++w)
            {
                const uint k = uint(tap * 4 + w);
                uint4 wv0 = wsm[cur][(basent + 0u) * 72u + 0u * 36u + k];
                uint4 wv1 = wsm[cur][(basent + 0u) * 72u + 1u * 36u + k];
                uint4 wv2 = wsm[cur][(basent + 1u) * 72u + 0u * 36u + k];
                uint4 wv3 = wsm[cur][(basent + 1u) * 72u + 1u * 36u + k];
                uint b[16] = {
                    wv0.x, wv0.y, wv0.z, wv0.w,
                    wv1.x, wv1.y, wv1.z, wv1.w,
                    wv2.x, wv2.y, wv2.z, wv2.w,
                    wv3.x, wv3.y, wv3.z, wv3.w
                };

                [unroll]
                for (int r = 0; r < ROWS; ++r)
                {
                    const int yl = int(warp) + r * 4;
                    uint4 av = win[cur][(yl + dy) * WIN_W + int(px) + dx];
                    uint a = (w == 0) ? av.x : ((w == 1) ? av.y : ((w == 2) ? av.z : av.w));
                    [unroll]
                    for (int c = 0; c < 16; ++c)
                        acc[r][c] = dot4add_i8packed(a, b[c], acc[r][c]);
                }
            }
        }
    }

    // Barrier-free Wave32 epilogue using WaveReadLaneAt
    [unroll]
    for (int r = 0; r < ROWS; ++r)
    {
        uint4 halfWords = uint4(0u, 0u, 0u, 0u);
        [unroll]
        for (int w = 0; w < 4; ++w)
        {
            int ch4 = (n0 * 8 + int(halfCh) * 16) + w * 4;
            uint4 bv = tScale[ch4 >> 2];
            uint4 nv = tBias[ch4 >> 2];
            int ib[4] = { int(bv.x), int(bv.y), int(bv.z), int(bv.w) };
            int ng[4] = { int(nv.x), int(nv.y), int(nv.z), int(nv.w) };
            uint packed = 0u;
            [unroll]
            for (int c = 0; c < 4; ++c)
            {
                int a = acc[r][w * 4 + c];
                packed |= RequantFlow(a, ib[c], ng[c], lim, posMul, sh1, sh2) << uint(c * 8);
            }
            if (w == 0)      halfWords.x = packed;
            else if (w == 1) halfWords.y = packed;
            else if (w == 2) halfWords.z = packed;
            else             halfWords.w = packed;
        }
        uint4 hi4 = WaveReadLaneAt(halfWords, lane ^ 16u);

        const int y = Y0 + int(warp) + r * 4;
        const int x = X0 + int(px);
        if (halfCh == 0u && x < limitW && y < limitH)
        {
#if LAYER_OUT == 8
            uOutput[uint3(uint(x), uint(y + 1), uint(n0 + 0))] = halfWords.xy;
            uOutput[uint3(uint(x), uint(y + 1), uint(n0 + 1))] = halfWords.zw;
            uOutput[uint3(uint(x), uint(y + 1), uint(n0 + 2))] = hi4.xy;
            uOutput[uint3(uint(x), uint(y + 1), uint(n0 + 3))] = hi4.zw;
#else
            uOutput[uint3(uint(x), uint(y + 1), uint((n0 / 2) + 0))] = halfWords;
            uOutput[uint3(uint(x), uint(y + 1), uint((n0 / 2) + 1))] = hi4;
#endif
        }
    }
#endif
}
