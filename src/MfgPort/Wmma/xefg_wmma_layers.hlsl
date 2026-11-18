// SPDX-License-Identifier: MIT
//
// Full 9-tap 3x3 INT8 convolution D3D12 HLSL (Shader Model 6.6) compute shader template for XeFG's
// 17 ported INT8 layers on AMD RDNA 4 / RDNA 3 (WMMA / dot4add_i8packed).
//
// Direct bit-exact port of PoolConv, UpSkip (Decoder), and Flow from src/MfgPort/Cuda/xefg_cuda_layers.cu:
//   - Root Parameter 0 (Descriptor Table):
//       t0 : Texture2DArray<uint4> tInput
//       t4 : Texture2DArray<uint4> tSkip (Decoder LAYER_FAMILY == 1 only)
//       u0 : RWTexture2DArray<uint2> uOutput (8-ch Flow) or RWTexture2DArray<uint4> uOutput (all others)
//   - Root Parameter 1 (Descriptor Table):
//       t1 : Buffer<uint4> tWeights
//       t2 : Buffer<uint4> tScale (IEEE-754 float scale for Pool/Decoder, int32 bias for Flow)
//       t3 : Buffer<uint4> tBias  (IEEE-754 float bias for Pool/Decoder, int32 negMul for Flow)
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

Texture2DArray<uint4> tInput   : register(t0);
Buffer<uint4>         tWeights : register(t1);
Buffer<uint4>         tScale   : register(t2);
Buffer<uint4>         tBias    : register(t3);
#if LAYER_FAMILY == 1
Texture2DArray<uint4> tSkip    : register(t4);
#endif
#if LAYER_FAMILY == 2 && LAYER_OUT == 8
RWTexture2DArray<uint2> uOutput : register(u0);
#else
RWTexture2DArray<uint4> uOutput : register(u0);
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
groupshared uint4 win[WIN];
groupshared uint4 wsm[LAYER_NT * 72];
groupshared uint4 sFlowJoin[4][2][16];
#else
static const int ZPER = LAYER_OUT / LAYER_ZSPLIT;
groupshared uint4 win[LAYER_IN * WIN];
groupshared uint4 wsm[144];
#if LAYER_POOL != 0
groupshared int   sPool[4][16][16];
#else
groupshared uint2 sJoin[4][2][16];
#endif
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

uint4 FetchInput(int x, int y, int slice)
{
    uint w, h, d;
    tInput.GetDimensions(w, h, d);
    if (x < 0 || y < 0 || slice < 0 || uint(x) >= w || uint(y) >= h || uint(slice) >= d)
        return uint4(0u, 0u, 0u, 0u);
    return tInput.Load(int4(x, y, slice, 0));
}

#if LAYER_FAMILY == 1
uint4 FetchSkip(int x, int y, int slice)
{
    uint w, h, d;
    tSkip.GetDimensions(w, h, d);
    if (x < 0 || y < 0 || slice < 0 || uint(x) >= w || uint(y) >= h || uint(slice) >= d)
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

[numthreads(128, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint tid : SV_GroupIndex)
{
    const uint warp   = tid >> 5u;
    const uint lane   = tid & 31u;
    const uint px     = lane & 15u;
    const uint halfCh = lane >> 4u; // 0 = first 8 channels, 1 = next 8 channels

    uint outW, outH, outD;
    uOutput.GetDimensions(outW, outH, outD);

#if LAYER_FAMILY != 2
    // ---- Family 0 (PoolConv) & Family 1 (UpSkip Decoder) ----
    const int X0     = int(groupID.x) * ((LAYER_POOL != 0) ? 8 : 16);
    const int Y0     = int(groupID.y) * LAYER_TILE_H;
    const int limitW = int(outW);
    const int limitH = int(outH);
    const int lo     = (LAYER_FAMILY == 1 || LAYER_RELU != 0) ? 0 : -128;

#if LAYER_FAMILY == 0
    const int cx0 = (LAYER_POOL != 0) ? (2 * X0) : X0;
    const int cy0 = (LAYER_POOL != 0) ? (2 * Y0) : Y0;
    for (uint i = tid; i < uint(LAYER_IN * WIN); i += 128u)
    {
        int slice = int(i) / WIN;
        int r     = int(i) % WIN;
        uint4 v   = FetchInput(cx0 - 1 + (r % WIN_W), cy0 - 1 + (r / WIN_W), slice);
#if LAYER_BIAS != 0
        v += uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u);
#endif
        win[i] = v;
    }
#else
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
            uint4 a = FetchInput(xn, yn, slice);
            uint4 b = FetchInput(xf, yn, slice);
            uint4 c = FetchInput(xn, yf, slice);
            uint4 d = FetchInput(xf, yf, slice);
            uint4 s = FetchSkip(x, y, slice);
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

    const int zBegin = int(groupID.z) * ZPER;
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
            int block = z * LAYER_IN + slice;
            for (uint wIdx = tid; wIdx < 144u; wIdx += 128u)
                wsm[SwizzleWeightIdx(wIdx)] = tWeights.Load(block * 144 + int(wIdx));
            GroupMemoryBarrierWithGroupSync();

            int sliceBase = slice * WIN;
            [unroll]
            for (int tap = 0; tap < 9; ++tap)
            {
                int dy = tap / 3;
                int dx = tap % 3;
                [unroll]
                for (int w = 0; w < 4; ++w)
                {
                    uint s0 = halfCh * 2u + 0u;
                    uint s1 = halfCh * 2u + 1u;
                    uint4 wv0 = wsm[SwizzleWeightIdx(uint(tap * 16 + w * 4) + s0)];
                    uint4 wv1 = wsm[SwizzleWeightIdx(uint(tap * 16 + w * 4) + s1)];
                    uint b[8] = { wv0.x, wv0.y, wv0.z, wv0.w, wv1.x, wv1.y, wv1.z, wv1.w };

                    [unroll]
                    for (int r = 0; r < ROWS; ++r)
                    {
                        int yl = int(warp) + r * 4;
                        [unroll]
                        for (int py = 0; py < PY; ++py)
                        {
                            int rowY = (PY > 1) ? (2 * yl + py) : yl;
                            uint4 av = win[sliceBase + (rowY + dy) * WIN_W + int(px) + dx];
                            uint a = (w == 0) ? av.x : ((w == 1) ? av.y : ((w == 2) ? av.z : av.w));
                            [unroll]
                            for (int c = 0; c < 8; ++c)
                                acc[r][py][c] = dot4add_i8packed(a, b[c], acc[r][py][c]);
                        }
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        float s[16];
        float bi[16];
        [unroll]
        for (int w = 0; w < 4; ++w)
        {
            uint4 sv = tScale.Load(z * 4 + w);
            uint4 bv = tBias.Load(z * 4 + w);
            s[w * 4 + 0]  = asfloat(sv.x);
            s[w * 4 + 1]  = asfloat(sv.y);
            s[w * 4 + 2]  = asfloat(sv.z);
            s[w * 4 + 3]  = asfloat(sv.w);
            bi[w * 4 + 0] = asfloat(bv.x);
            bi[w * 4 + 1] = asfloat(bv.y);
            bi[w * 4 + 2] = asfloat(bv.z);
            bi[w * 4 + 3] = asfloat(bv.w);
        }

        [unroll]
        for (int r = 0; r < ROWS; ++r)
        {
            int Y = Y0 + int(warp) + r * 4;
#if LAYER_POOL != 0
            [unroll]
            for (int c = 0; c < 8; ++c)
                sPool[warp][px][halfCh * 8u + uint(c)] = max(acc[r][0][c], acc[r][1][c]);
            GroupMemoryBarrierWithGroupSync();

            if (halfCh == 0u && px < 8u && Y < limitH)
            {
                int X = X0 + int(px);
                if (X < limitW)
                {
                    uint words[4];
                    [unroll]
                    for (int w = 0; w < 4; ++w)
                    {
                        uint packed = 0u;
                        [unroll]
                        for (int c = 0; c < 4; ++c)
                        {
                            int oc = w * 4 + c;
                            int m = max(sPool[warp][2u * px][oc], sPool[warp][2u * px + 1u][oc]);
                            packed |= RequantFloat(m, s[oc], bi[oc], lo) << uint(c * 8);
                        }
                        words[w] = packed;
                    }
                    uOutput[uint3(uint(X), uint(Y), uint(z))] = uint4(words[0], words[1], words[2], words[3]);
                }
            }
            GroupMemoryBarrierWithGroupSync();
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
                    int oc = int(halfCh) * 8 + localC;
                    packed |= RequantFloat(acc[r][0][localC], s[oc], bi[oc], lo) << uint(c * 8);
                }
                if (w == 0) halfWords.x = packed;
                else        halfWords.y = packed;
            }
            sJoin[warp][halfCh][px] = halfWords;
            GroupMemoryBarrierWithGroupSync();

            if (halfCh == 0u && Y < limitH)
            {
                int X = X0 + int(px);
                if (X < limitW)
                {
                    uint2 lo2 = sJoin[warp][0][px];
                    uint2 hi2 = sJoin[warp][1][px];
                    uOutput[uint3(uint(X), uint(Y), uint(z))] = uint4(lo2.x, lo2.y, hi2.x, hi2.y);
                }
            }
            GroupMemoryBarrierWithGroupSync();
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
    const uint s1     = shifts & 31u;
    const uint s2     = (shifts >> 8u) & 31u;
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

    for (int step = 0; step < LAYER_IN; ++step)
    {
        for (uint e = tid; e < uint(WIN); e += 128u)
        {
            int x  = X0 - 1 + (int(e) % WIN_W);
            int ys = Y0 + (int(e) / WIN_W);
            uint4 v = uint4(0u, 0u, 0u, 0u);
            if (x >= 0 && x < Wd && ys >= 1 && ys <= Hd)
            {
#if LAYER_OUT == 8
                uint4 a = FetchInput(x, ys, 2 * step);
                uint4 b = FetchInput(x, ys, 2 * step + 1);
                v = uint4(a.x, a.y, b.x, b.y);
#else
                v = FetchInput(x, ys, step);
#endif
            }
            win[e] = v;
        }

        for (uint wIdx = tid; wIdx < uint(LAYER_NT * 72); wIdx += 128u)
        {
            int nt    = int(wIdx) / 72;
            int group = (int(wIdx) % 72) / 36;
            int k     = int(wIdx) % 36;
            int ch    = (n0 + nt) * 8;
            int z     = ch / LAYER_OUT;
            int c     = ch % LAYER_OUT;
            int idx   = ((z * LAYER_IN + step) * (LAYER_OUT / 4) + (c / 4) + group) * 36 + k;
            wsm[wIdx] = tWeights.Load(idx);
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (int tap = 0; tap < 9; ++tap)
        {
            int dy = tap / 3;
            int dx = tap % 3;
            [unroll]
            for (int w = 0; w < 4; ++w)
            {
                int k = tap * 4 + w;
                uint basent = halfCh * 2u;
                uint4 wv0 = wsm[(basent + 0u) * 72u + 0u * 36u + uint(k)];
                uint4 wv1 = wsm[(basent + 0u) * 72u + 1u * 36u + uint(k)];
                uint4 wv2 = wsm[(basent + 1u) * 72u + 0u * 36u + uint(k)];
                uint4 wv3 = wsm[(basent + 1u) * 72u + 1u * 36u + uint(k)];
                uint b[16] = {
                    wv0.x, wv0.y, wv0.z, wv0.w,
                    wv1.x, wv1.y, wv1.z, wv1.w,
                    wv2.x, wv2.y, wv2.z, wv2.w,
                    wv3.x, wv3.y, wv3.z, wv3.w
                };

                [unroll]
                for (int r = 0; r < ROWS; ++r)
                {
                    int yl = int(warp) + r * 4;
                    uint4 av = win[(yl + dy) * WIN_W + int(px) + dx];
                    uint a = (w == 0) ? av.x : ((w == 1) ? av.y : ((w == 2) ? av.z : av.w));
                    [unroll]
                    for (int c = 0; c < 16; ++c)
                        acc[r][c] = dot4add_i8packed(a, b[c], acc[r][c]);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (int r = 0; r < ROWS; ++r)
    {
        uint4 halfWords = uint4(0u, 0u, 0u, 0u);
        [unroll]
        for (int w = 0; w < 4; ++w)
        {
            int ch4 = (n0 * 8 + int(halfCh) * 16) + w * 4;
            uint4 bv = tScale.Load(ch4 >> 2);
            uint4 nv = tBias.Load(ch4 >> 2);
            int ib[4] = { int(bv.x), int(bv.y), int(bv.z), int(bv.w) };
            int ng[4] = { int(nv.x), int(nv.y), int(nv.z), int(nv.w) };
            uint packed = 0u;
            [unroll]
            for (int c = 0; c < 4; ++c)
            {
                int a = acc[r][w * 4 + c];
                packed |= RequantFlow(a, ib[c], ng[c], lim, posMul, s1, s2) << uint(c * 8);
            }
            if (w == 0)      halfWords.x = packed;
            else if (w == 1) halfWords.y = packed;
            else if (w == 2) halfWords.z = packed;
            else             halfWords.w = packed;
        }
        sFlowJoin[warp][halfCh][px] = halfWords;
        GroupMemoryBarrierWithGroupSync();

        int y = Y0 + int(warp) + r * 4;
        int x = X0 + int(px);
        if (halfCh == 0u && x < limitW && y < limitH)
        {
            uint4 lo4 = sFlowJoin[warp][0][px];
            uint4 hi4 = sFlowJoin[warp][1][px];
#if LAYER_OUT == 8
            uOutput[uint3(uint(x), uint(y + 1), uint(n0 + 0))] = lo4.xy;
            uOutput[uint3(uint(x), uint(y + 1), uint(n0 + 1))] = lo4.zw;
            uOutput[uint3(uint(x), uint(y + 1), uint(n0 + 2))] = hi4.xy;
            uOutput[uint3(uint(x), uint(y + 1), uint(n0 + 3))] = hi4.zw;
#else
            uOutput[uint3(uint(x), uint(y + 1), uint((n0 / 2) + 0))] = lo4;
            uOutput[uint3(uint(x), uint(y + 1), uint((n0 / 2) + 1))] = hi4;
#endif
        }
        GroupMemoryBarrierWithGroupSync();
    }
#endif
}
