#include "HudTierPass.h"
#include "Log.h"
#include "Util.h"
#include "D3D12VulkanWmma.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{
	using D3DCompileFn = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*,
		LPCSTR, LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**);

	D3DCompileFn LoadD3DCompile() noexcept
	{
		static D3DCompileFn fn = []() noexcept -> D3DCompileFn {
			HMODULE mod = LoadLibraryW(L"d3dcompiler_47.dll");
			if (!mod)
				return nullptr;
			return reinterpret_cast<D3DCompileFn>(GetProcAddress(mod, "D3DCompile"));
		}();
		return fn;
	}

	DXGI_FORMAT SrvFormat(DXGI_FORMAT fmt) noexcept
	{
		switch (fmt)
		{
		case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
		case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
		case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
		case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
		default: return fmt;
		}
	}

	DXGI_FORMAT UavFormat(DXGI_FORMAT fmt) noexcept
	{
		switch (fmt)
		{
		case DXGI_FORMAT_R8G8B8A8_TYPELESS:
		case DXGI_FORMAT_R8G8B8A8_UNORM:
			return DXGI_FORMAT_R8G8B8A8_UNORM;
		case DXGI_FORMAT_B8G8R8A8_TYPELESS:
		case DXGI_FORMAT_B8G8R8A8_UNORM:
			return DXGI_FORMAT_B8G8R8A8_UNORM;
		case DXGI_FORMAT_R10G10B10A2_TYPELESS:
		case DXGI_FORMAT_R10G10B10A2_UNORM:
			return DXGI_FORMAT_R10G10B10A2_UNORM;
		case DXGI_FORMAT_R16G16B16A16_TYPELESS:
		case DXGI_FORMAT_R16G16B16A16_FLOAT:
			return DXGI_FORMAT_R16G16B16A16_FLOAT;
		default:
			return DXGI_FORMAT_UNKNOWN;
		}
	}

	// Shared HLSL source for all 3 HUD tier compute shaders (cs_5_0).
	// Root signature layout:
	//   b0: 4 root 32-bit constants (width, height, tolerance, flags)
	//   t0, t1: 2 SRV textures
	//   u0, u1: 2 UAVs (texture + RWByteAddressBuffer counter)
	constexpr char kHudTierHlsl[] = R"(
bool IsFiniteF(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
bool3 IsFiniteF3(float3 v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
cbuffer Params : register(b0)
{
    uint Width;
    uint Height;
    float Tolerance;
    uint Flags;
};

Texture2D<float4> TexA : register(t0);
Texture2D<float4> TexB : register(t1);
RWTexture2D<float4> OutColor : register(u0);
RWStructuredBuffer<uint> Counter : register(u1);

float3 SanitizeRgb(float3 c)
{
    return float3(
        IsFiniteF(c.r) ? max(c.r, 0.0f) : 0.0f,
        IsFiniteF(c.g) ? max(c.g, 0.0f) : 0.0f,
        IsFiniteF(c.b) ? max(c.b, 0.0f) : 0.0f);
}

float Rec709Luma(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

bool IsHudDiffPair(float3 a, float3 b)
{
    float3 sa = SanitizeRgb(a);
    float3 sb = SanitizeRgb(b);
    float3 d = abs(sa - sb);
    float lumaD = abs(Rec709Luma(sa) - Rec709Luma(sb));
    return (lumaD > Tolerance * 0.65f) || (max(d.r, max(d.g, d.b)) > Tolerance * 1.15f);
}

// Tier 1 Guard: analyze-only 1/16 strided pixel sampling (id.xy * 4 -> 16x fewer thread groups, identical sampled pixels)
// with groupshared per-group reduction (up to 64x less global L2 atomic contention).
groupshared uint g_guardGroupCount;

[numthreads(8, 8, 1)]
void HudDiffGuardCS(uint3 id : SV_DispatchThreadID, uint gidx : SV_GroupIndex)
{
    if (gidx == 0u) g_guardGroupCount = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint2 p = id.xy * 4u;
    if (p.x < Width && p.y < Height && IsHudDiffPair(TexA[int2(p)].rgb, TexB[int2(p)].rgb))
    {
        uint dummy;
        InterlockedAdd(g_guardGroupCount, 1u, dummy);
    }

    GroupMemoryBarrierWithGroupSync();
    if (gidx == 0u && g_guardGroupCount != 0u)
    {
        uint dummy;
        InterlockedAdd(Counter[0], g_guardGroupCount, dummy);
    }
}

// Tier 2 Composite (MFGUnlock xmx_hud_composite style): 3x3 dilated Rec.709 Luma + RGB diff > Tolerance.
// Flags bit 0: update diff counter (1/16 sample). Flags bit 1: debug magenta overlay.
bool IsHudDiff(int2 p)
{
    return IsHudDiffPair(TexA[p].rgb, TexB[p].rgb);
}

// 10x10 shared tile for 8x8 group + 1-pixel halo (200 vs 1152 texture loads per group = 5.76x fewer VRAM reads)
// plus uniform tile early-exit (~90%+ non-HUD background tiles skip dilation, counter & composite passes completely)
groupshared uint g_sDiff[100];
groupshared uint g_diffGroupCount;
groupshared uint g_tileHasAnyDiff;

[numthreads(8, 8, 1)]
void HudDiffCompositeCS(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex)
{
    if (gidx == 0u)
    {
        g_diffGroupCount = 0u;
        g_tileHasAnyDiff = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    int2 maxCoord = int2(Width - 1u, Height - 1u);
    int2 tileOrigin = int2(gid.xy * 8u) - int2(1, 1);

    for (uint idx = gidx; idx < 100u; idx += 64u)
    {
        int tx = int(idx % 10u);
        int ty = int(idx / 10u);
        int2 q = clamp(tileOrigin + int2(tx, ty), int2(0, 0), maxCoord);
        uint d = IsHudDiff(q) ? 1u : 0u;
        g_sDiff[idx] = d;
        if (d != 0u) InterlockedOr(g_tileHasAnyDiff, 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    // Uniform across all 64 threads: if no pixel in the 10x10 halo differs, no pixel in the 8x8 tile is HUD.
    if (g_tileHasAnyDiff == 0u) return;

    if (id.x < Width && id.y < Height)
    {
        uint base = gtid.y * 10u + gtid.x;
        uint mask = g_sDiff[base + 0u]  | g_sDiff[base + 1u]  | g_sDiff[base + 2u]
                  | g_sDiff[base + 10u] | g_sDiff[base + 11u] | g_sDiff[base + 12u]
                  | g_sDiff[base + 20u] | g_sDiff[base + 21u] | g_sDiff[base + 22u];
        bool hud = (mask != 0u);

        if ((Flags & 1u) != 0u && hud && (id.x & 3u) == 0u && (id.y & 3u) == 0u)
        {
            uint dummy;
            InterlockedAdd(g_diffGroupCount, 1u, dummy);
        }

        if (hud)
        {
            if ((Flags & 2u) != 0u)
            {
                OutColor[id.xy] = float4(1.0f, 0.0f, 1.0f, 1.0f);
            }
            else
            {
                float4 src = TexA[int2(id.xy)];
                OutColor[id.xy] = float4(SanitizeRgb(src.rgb), IsFiniteF(src.a) ? src.a : 1.0f);
            }
        }
    }

    if ((Flags & 1u) != 0u)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gidx == 0u && g_diffGroupCount != 0u)
        {
            uint dummy;
            InterlockedAdd(Counter[0], g_diffGroupCount, dummy);
        }
    }
}

// Opt-in split-screen debug comparison (0% overhead when disabled):
// Left of SplitX = original real frame (TexA), divider = cyan, right = generated/HUD-protected OutColor.
[numthreads(8, 8, 1)]
void SplitCompareCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    uint splitX = Flags;
    if (id.x < splitX)
    {
        float4 src = TexA[int2(id.xy)];
        OutColor[id.xy] = float4(SanitizeRgb(src.rgb), IsFiniteF(src.a) ? src.a : 1.0f);
    }
    else if (id.x == splitX || id.x == splitX + 1u)
    {
        OutColor[id.xy] = float4(0.35f, 0.72f, 1.0f, 1.0f);
    }
}
)";

	// Separate HLSL block for Tier 3 so u0 can be RWTexture2D<uint> in UpdateCS and RWTexture2D<float4> in CompositeCS.
	constexpr char kStaticHudUpdateHlsl[] = R"(
bool IsFiniteF(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
bool3 IsFiniteF3(float3 v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
cbuffer Params : register(b0)
{
    uint Width;
    uint Height;
    float Tolerance;
    uint Flags; // bit 0: reset history, bit 1: update Counter[1] readback
};

Texture2D<float4> CurrReal : register(t0);
Texture2D<float4> PrevReal : register(t1);
RWTexture2D<uint> StaticCount : register(u0);
RWStructuredBuffer<uint> Counter : register(u1);

groupshared uint g_staticGroupCount;

[numthreads(8, 8, 1)]
void StaticHudUpdateCS(uint3 id : SV_DispatchThreadID, uint gidx : SV_GroupIndex)
{
    if ((Flags & 2u) != 0u)
    {
        if (gidx == 0u) g_staticGroupCount = 0u;
        GroupMemoryBarrierWithGroupSync();
    }

    if (id.x < Width && id.y < Height)
    {
        uint count = 0u;
        if ((Flags & 1u) == 0u)
        {
            float3 cA = CurrReal[int2(id.xy)].rgb;
            float3 cB = PrevReal[int2(id.xy)].rgb;
            cA = float3(IsFiniteF(cA.r) ? max(cA.r, 0.0f) : 0.0f, IsFiniteF(cA.g) ? max(cA.g, 0.0f) : 0.0f, IsFiniteF(cA.b) ? max(cA.b, 0.0f) : 0.0f);
            cB = float3(IsFiniteF(cB.r) ? max(cB.r, 0.0f) : 0.0f, IsFiniteF(cB.g) ? max(cB.g, 0.0f) : 0.0f, IsFiniteF(cB.b) ? max(cB.b, 0.0f) : 0.0f);
            float3 d = abs(cA - cB);
            bool unchanged = max(d.r, max(d.g, d.b)) <= Tolerance;
            uint prev = StaticCount[id.xy];
            count = unchanged ? min(prev + 1u, 8u) : 0u;
        }

        StaticCount[id.xy] = count;

        // Count 1/16 sampled pixels that reached the static-HUD threshold (>= 4 consecutive real frames).
        if ((Flags & 2u) != 0u && count >= 4u && (id.x & 3u) == 0u && (id.y & 3u) == 0u)
        {
            uint dummy;
            InterlockedAdd(g_staticGroupCount, 1u, dummy);
        }
    }

    if ((Flags & 2u) != 0u)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gidx == 0u && g_staticGroupCount != 0u)
        {
            uint dummy;
            InterlockedAdd(Counter[1], g_staticGroupCount, dummy);
        }
    }
}
)";

	constexpr char kStaticHudCompositeHlsl[] = R"(
bool IsFiniteF(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
bool3 IsFiniteF3(float3 v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
cbuffer Params : register(b0)
{
    uint Width;
    uint Height;
    float Tolerance;
    uint Flags; // bit 1: debug magenta
};

Texture2D<float4> CurrReal : register(t0);
Texture2D<uint> StaticCount : register(t1);
RWTexture2D<float4> OutColor : register(u0);
RWStructuredBuffer<uint> Counter : register(u1);

[numthreads(8, 8, 1)]
void StaticHudCompositeCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;

    if (StaticCount[int2(id.xy)] >= 4u)
    {
        if ((Flags & 2u) != 0u)
        {
            OutColor[id.xy] = float4(1.0f, 0.0f, 1.0f, 1.0f);
        }
        else
        {
            float4 src = CurrReal[int2(id.xy)];
            float3 safeRgb = float3(
                IsFiniteF(src.r) ? max(src.r, 0.0f) : 0.0f,
                IsFiniteF(src.g) ? max(src.g, 0.0f) : 0.0f,
                IsFiniteF(src.b) ? max(src.b, 0.0f) : 0.0f);
            OutColor[id.xy] = float4(safeRgb, IsFiniteF(src.a) ? src.a : 1.0f);
        }
    }
}
)";

	constexpr char kHybridHudCompositeHlsl[] = R"(
bool IsFiniteF(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
bool3 IsFiniteF3(float3 v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
cbuffer Params : register(b0)
{
    uint Width;
    uint Height;
    float Tolerance;
    uint Flags; // bit 0: count 1/16 diff pixels, bit 1: debug magenta
};

Texture2D<float4> TexA : register(t0);
Texture2D<float4> TexB : register(t1);
Texture2D<uint> StaticCount : register(t2);
RWTexture2D<float4> OutColor : register(u0);
RWStructuredBuffer<uint> Counter : register(u1);

float3 SanitizeHybridRgb(float3 c)
{
    return float3(
        IsFiniteF(c.r) ? max(c.r, 0.0f) : 0.0f,
        IsFiniteF(c.g) ? max(c.g, 0.0f) : 0.0f,
        IsFiniteF(c.b) ? max(c.b, 0.0f) : 0.0f);
}

bool IsHudDiffHybrid(int2 p)
{
    float3 a = SanitizeHybridRgb(TexA[p].rgb);
    float3 b = SanitizeHybridRgb(TexB[p].rgb);
    float3 d = abs(a - b);
    float lumaD = abs(dot(a, float3(0.2126f, 0.7152f, 0.0722f)) - dot(b, float3(0.2126f, 0.7152f, 0.0722f)));
    return (lumaD > Tolerance * 0.65f) || (max(d.r, max(d.g, d.b)) > Tolerance * 1.15f);
}

// 10x10 shared tile for 8x8 group + 1-pixel halo (5.76x fewer TexA/TexB VRAM reads)
groupshared uint g_sDiffHybrid[100];
groupshared uint g_hybridGroupCount;

[numthreads(8, 8, 1)]
void HybridHudCompositeCS(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex)
{
    if (gidx == 0u) g_hybridGroupCount = 0u;
    int2 maxCoord = int2(Width - 1u, Height - 1u);
    int2 tileOrigin = int2(gid.xy * 8u) - int2(1, 1);

    for (uint idx = gidx; idx < 100u; idx += 64u)
    {
        int tx = int(idx % 10u);
        int ty = int(idx / 10u);
        int2 q = clamp(tileOrigin + int2(tx, ty), int2(0, 0), maxCoord);
        g_sDiffHybrid[idx] = IsHudDiffHybrid(q) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    if (id.x < Width && id.y < Height)
    {
        uint base = gtid.y * 10u + gtid.x;
        uint mask = g_sDiffHybrid[base + 0u]  | g_sDiffHybrid[base + 1u]  | g_sDiffHybrid[base + 2u]
                  | g_sDiffHybrid[base + 10u] | g_sDiffHybrid[base + 11u] | g_sDiffHybrid[base + 12u]
                  | g_sDiffHybrid[base + 20u] | g_sDiffHybrid[base + 21u] | g_sDiffHybrid[base + 22u];
        bool hudDiff = (mask != 0u);

        if ((Flags & 1u) != 0u && hudDiff && (id.x & 3u) == 0u && (id.y & 3u) == 0u)
        {
            uint dummy;
            InterlockedAdd(g_hybridGroupCount, 1u, dummy);
        }

        int2 p = int2(id.xy);
        bool isStatic = StaticCount[p] >= 4u;
        if (hudDiff || isStatic)
        {
            if ((Flags & 2u) != 0u)
            {
                OutColor[id.xy] = float4(1.0f, 0.0f, 1.0f, 1.0f);
            }
            else
            {
                float4 src = TexA[p];
                OutColor[id.xy] = float4(SanitizeHybridRgb(src.rgb), IsFiniteF(src.a) ? src.a : 1.0f);
            }
        }
    }

    if ((Flags & 1u) != 0u)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gidx == 0u && g_hybridGroupCount != 0u)
        {
            uint dummy;
            InterlockedAdd(Counter[0], g_hybridGroupCount, dummy);
        }
    }
}
)";


	// Tier 4 Tone-Map Match (ported from MFG-Unlocker V1.7.2 hudless_match.cpp):
	// Lazy-compiled only when Tier 4 activates on a color/format-mismatched HUDLess (0 cost on normal games).
	// 5 passes: clear -> accumulate (1/16 stride) -> median (groupshared 256-thread scan) -> fill (5-bin outlier filter + interp) -> apply.
	constexpr uint32_t kToneMatchBins = 256u;
	constexpr uint32_t kToneMatchCodes = 1024u;
	constexpr uint32_t kToneMatchMinCount = 8u;
	constexpr float kToneMatchLogMin = -16.0f;
	constexpr float kToneMatchBinsPerOctave = 12.0f;
	constexpr uint32_t kToneMatchStride = 4u;

	constexpr char kToneMatchHlsl[] = R"(
bool IsFiniteF(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
bool3 IsFiniteF3(float3 v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
#define STRIDE 4
#define OUTLIER (32.0 / 1023.0)
cbuffer Match : register(b0)
{
    uint width, height;
    float logMin, binsPerOctave;
    uint bins, codes, minimumCount, unused;
};
Texture2D<float4> hudless : register(t0);
Texture2D<float4> backBuffer : register(t1);
RWTexture2D<float4> output : register(u0);
RWByteAddressBuffer histogram : register(u1);
RWByteAddressBuffer lut : register(u2);

uint BinOf(float v)
{
    uint bin = 0;
    if (v > 0.0)
        bin = (uint)clamp((log2(v) - logMin) * binsPerOctave, 0.0, (float)(bins - 1));
    return bin;
}

[numthreads(64, 1, 1)]
void clear(uint3 id : SV_DispatchThreadID)
{
    if (id.x < 3 * bins * codes) histogram.Store(id.x * 4u, 0u);
}

[numthreads(8, 8, 1)]
void accumulate(uint3 id : SV_DispatchThreadID)
{
    const uint2 p = id.xy * STRIDE;
    if (p.x >= width || p.y >= height) return;
    const float3 h = hudless.Load(int3(p, 0)).rgb;
    const float3 b = backBuffer.Load(int3(p, 0)).rgb;
    [unroll] for (uint c = 0; c < 3; ++c)
    {
        const uint code = (uint)(saturate(b[c]) * (codes - 1) + 0.5);
        uint previous;
        histogram.InterlockedAdd(((c * bins + BinOf(h[c])) * codes + code) * 4u, 1u, previous);
    }
}

groupshared uint partial[256];
[numthreads(256, 1, 1)]
void median(uint3 group : SV_GroupID, uint gi : SV_GroupIndex)
{
    const uint bin = group.x, c = group.y;
    const uint base = (c * bins + bin) * codes + gi * 4;
    uint counts[4];
    uint sum = 0;
    [unroll] for (uint k = 0; k < 4; ++k) { counts[k] = histogram.Load((base + k) * 4u); sum += counts[k]; }
    partial[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint offset = 1; offset < 256; offset <<= 1)
    {
        uint v = partial[gi];
        if (gi >= offset) v += partial[gi - offset];
        GroupMemoryBarrierWithGroupSync();
        partial[gi] = v;
        GroupMemoryBarrierWithGroupSync();
    }
    const uint total = partial[255];
    const uint before = gi ? partial[gi - 1] : 0;
    const uint slot = (c * bins + bin) * 4;
    if (total < minimumCount) { if (gi == 0) lut.Store(slot, asuint(-1.0)); return; }
    const uint middle = (total + 1) / 2;
    if (before < middle && partial[gi] >= middle)
    {
        uint seen = before, code = gi * 4 + 3;
        for (uint j = 0; j < 4; ++j) { seen += counts[j]; if (seen >= middle) { code = gi * 4 + j; break; } }
        lut.Store(slot, asuint((float)code / (float)(codes - 1)));
    }
}

groupshared float fillRaw[256];
groupshared float fillLut[256];
[numthreads(256, 1, 1)]
void fill(uint3 group : SV_GroupID, uint gi : SV_GroupIndex)
{
    const uint c = group.x, base = c * bins;
    const float rawVal = asfloat(lut.Load((base + gi) * 4));
    fillRaw[gi] = rawVal;
    fillLut[gi] = rawVal;
    GroupMemoryBarrierWithGroupSync();

    if (rawVal >= 0.0)
    {
        float w[5];
        uint n = 0;
        for (int d = -2; d <= 2; ++d)
        {
            const int j = (int)gi + d;
            if (j >= 0 && j < 256 && fillRaw[j] >= 0.0) { w[n] = fillRaw[j]; ++n; }
        }
        for (uint a = 1; a < n; ++a)
        {
            const float x = w[a];
            int p = (int)a - 1;
            while (p >= 0 && w[p] > x) { w[p + 1] = w[p]; --p; }
            w[p + 1] = x;
        }
        if (abs(rawVal - w[n / 2]) > OUTLIER) fillLut[gi] = w[n / 2];
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi == 0u)
    {
        int first = -1, last = -1;
        for (uint i = 0; i < 256; ++i) if (fillLut[i] >= 0.0) { if (first < 0) first = (int)i; last = (int)i; }
        if (first < 0)
        {
            for (uint z = 0; z < 256; ++z) fillLut[z] = 0.0;
        }
        else
        {
            int previous = first;
            for (int b = first + 1; b <= last; ++b)
            {
                if (fillLut[b] < 0.0) continue;
                for (int g = previous + 1; g < b; ++g) fillLut[g] = lerp(fillLut[previous], fillLut[b], (float)(g - previous) / (float)(b - previous));
                previous = b;
            }
            const int span = min(4, last - first);
            const float headSlope = span > 0 ? max((fillLut[first + span] - fillLut[first]) / (float)span, 0.0) : 0.0;
            const float tailSlope = span > 0 ? max((fillLut[last] - fillLut[last - span]) / (float)span, 0.0) : 0.0;
            for (int h = 0; h < first; ++h) fillLut[h] = saturate(fillLut[first] - headSlope * (float)(first - h));
            for (int t = last + 1; t < 256; ++t) fillLut[t] = saturate(fillLut[last] + tailSlope * (float)(t - last));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    lut.Store((base + gi) * 4, asuint(fillLut[gi]));
}

groupshared float g_applyLut[768];

[numthreads(8, 8, 1)]
void apply(uint3 id : SV_DispatchThreadID, uint gidx : SV_GroupIndex)
{
    const uint totalLut = min(3u * bins, 768u);
    for (uint li = gidx; li < totalLut; li += 64u)
        g_applyLut[li] = asfloat(lut.Load(li * 4u));
    GroupMemoryBarrierWithGroupSync();

    if (id.x >= width || id.y >= height) return;
    const float3 h = hudless.Load(int3(id.xy, 0)).rgb;
    float3 o = 0;
    [unroll] for (uint c = 0; c < 3; ++c)
    {
        const float v = h[c];
        float f = v > 0.0 ? (log2(v) - logMin) * binsPerOctave - 0.5 : 0.0;
        f = clamp(f, 0.0, (float)(bins - 1));
        const uint i = min((uint)f, bins - 2);
        const float t = saturate(f - (float)i);
        const uint lutIdx = c * bins + i;
        o[c] = lerp(g_applyLut[lutIdx], g_applyLut[lutIdx + 1u], t);
    }
    o = float3(
        IsFiniteF(o.r) ? saturate(o.r) : 0.0f,
        IsFiniteF(o.g) ? saturate(o.g) : 0.0f,
        IsFiniteF(o.b) ? saturate(o.b) : 0.0f);
    output[id.xy] = float4(o, 1.0);
}
)";

	// Tier 4 3D Bilateral-Grid Local Affine (3x4) Tone-Map Matcher:
	// Handles spatial tone-mapping (RenoDX/ReShade vignette, local exposure, regional color grading).
	// Compact L2-resident grid (cell=128, bins=8 -> 1,920 cells @ 1440p), 1/16 strided fixed-point atomic
	// accumulation (overflow-free, fast fxc cs_5_0 async compile), amortized fit every 4 real frames.
	constexpr uint32_t kBilateralCell = 128u;
	constexpr uint32_t kBilateralBins = 8u;
	constexpr uint32_t kBilateralStride = 4u;
	constexpr uint32_t kBilateralFitInterval = 4u;
	constexpr float kBilateralLambda = 1e-3f;

	constexpr char kUnjitterMVecsHlsl[] = R"(
bool IsFiniteF(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }

cbuffer UnjitterCB : register(b0)
{
    uint  BaseX;
    uint  BaseY;
    uint  SizeX;
    uint  SizeY;
    float DeltaX;
    float DeltaY;
    uint  Pad0;
    uint  Pad1;
};

RWTexture2D<float2> MVec2 : register(u0);
RWTexture2D<float4> MVec4 : register(u0);

[numthreads(8, 8, 1)]
void UnjitterMVecs2CS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SizeX || id.y >= SizeY)
        return;
    uint2 p = uint2(BaseX + id.x, BaseY + id.y);
    float2 mv = MVec2[p];
    if (IsFiniteF(mv.x) && IsFiniteF(mv.y))
        MVec2[p] = mv + float2(DeltaX, DeltaY);
}

[numthreads(8, 8, 1)]
void UnjitterMVecs4CS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= SizeX || id.y >= SizeY)
        return;
    uint2 p = uint2(BaseX + id.x, BaseY + id.y);
    float4 mv = MVec4[p];
    if (IsFiniteF(mv.x) && IsFiniteF(mv.y))
    {
        mv.xy += float2(DeltaX, DeltaY);
        MVec4[p] = mv;
    }
}
)";

	constexpr char kBilateralToneMatchHlsl[] = R"(
bool IsFiniteF(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
bool3 IsFiniteF3(float3 v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
#define B_STRIDE 4
#define B_SCALE 1024.0f
#define B_INV_SCALE (1.0f / 1024.0f)

cbuffer BilateralParams : register(b0)
{
    uint width, height;
    uint cellsX, cellsY;
    uint cell, bins;
    float lambda;
    uint totalStatsUints;
};

Texture2D<float4> hudless : register(t0);
Texture2D<float4> backBuffer : register(t1);
RWTexture2D<float4> output : register(u0);
RWByteAddressBuffer statsBuf : register(u1);
RWByteAddressBuffer modelBuf : register(u2);

float3 SafeSaturateRgb(float3 c)
{
    return float3(
        IsFiniteF(c.r) ? saturate(c.r) : 0.0f,
        IsFiniteF(c.g) ? saturate(c.g) : 0.0f,
        IsFiniteF(c.b) ? saturate(c.b) : 0.0f);
}

float Luma709(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

void AccumBin(uint cellIdx, uint b, float w, float3 x, float3 y)
{
    if (w <= 1e-4f) return;
    uint base = (cellIdx * bins + b) * 22u;
    uint dummy;
    statsBuf.InterlockedAdd((base + 0u) * 4u, (uint)(w * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 1u) * 4u, (uint)(w * x.r * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 2u) * 4u, (uint)(w * x.g * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 3u) * 4u, (uint)(w * x.b * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 4u) * 4u, (uint)(w * x.r * x.r * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 5u) * 4u, (uint)(w * x.r * x.g * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 6u) * 4u, (uint)(w * x.r * x.b * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 7u) * 4u, (uint)(w * x.g * x.g * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 8u) * 4u, (uint)(w * x.g * x.b * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 9u) * 4u, (uint)(w * x.b * x.b * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 10u) * 4u, (uint)(w * x.r * y.r * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 11u) * 4u, (uint)(w * x.r * y.g * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 12u) * 4u, (uint)(w * x.r * y.b * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 13u) * 4u, (uint)(w * x.g * y.r * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 14u) * 4u, (uint)(w * x.g * y.g * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 15u) * 4u, (uint)(w * x.g * y.b * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 16u) * 4u, (uint)(w * x.b * y.r * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 17u) * 4u, (uint)(w * x.b * y.g * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 18u) * 4u, (uint)(w * x.b * y.b * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 19u) * 4u, (uint)(w * y.r * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 20u) * 4u, (uint)(w * y.g * B_SCALE + 0.5f), dummy);
    statsBuf.InterlockedAdd((base + 21u) * 4u, (uint)(w * y.b * B_SCALE + 0.5f), dummy);
}

[numthreads(64, 1, 1)]
void BilateralClearCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x < totalStatsUints)
        statsBuf.Store(id.x * 4u, 0u);
}

[numthreads(8, 8, 1)]
void BilateralAccumCS(uint3 id : SV_DispatchThreadID)
{
    uint2 p = id.xy * B_STRIDE;
    if (p.x >= width || p.y >= height) return;

    float3 x = SafeSaturateRgb(hudless.Load(int3(p, 0)).rgb);
    float3 y = SafeSaturateRgb(backBuffer.Load(int3(p, 0)).rgb);
    uint cx = min(p.x / cell, cellsX - 1u);
    uint cy = min(p.y / cell, cellsY - 1u);
    uint cellIdx = cy * cellsX + cx;

    float z = saturate(Luma709(x)) * (float)(bins - 1u);
    uint b0 = min((uint)floor(z), bins - 1u);
    uint b1 = min(b0 + 1u, bins - 1u);
    float fz = saturate(z - (float)b0);

    if (b0 == b1)
    {
        AccumBin(cellIdx, b0, 1.0f, x, y);
    }
    else
    {
        AccumBin(cellIdx, b0, 1.0f - fz, x, y);
        AccumBin(cellIdx, b1, fz, x, y);
    }
}

[numthreads(64, 1, 1)]
void BilateralSolveCS(uint3 id : SV_DispatchThreadID)
{
    uint totalCells = cellsX * cellsY * bins;
    if (id.x >= totalCells) return;

    uint b = id.x % bins;
    uint xy = id.x / bins;
    int cx = (int)(xy % cellsX);
    int cy = (int)(xy / cellsX);

    float s[22];
    for (uint k = 0u; k < 22u; ++k) s[k] = 0.0f;

    // 3x3 spatial tent filter for smooth regional regularization
    for (int dy = -1; dy <= 1; ++dy)
    {
        int ny = cy + dy;
        if (ny < 0 || ny >= (int)cellsY) continue;
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = cx + dx;
            if (nx < 0 || nx >= (int)cellsX) continue;
            float w = (float)((2 - abs(dx)) * (2 - abs(dy)));
            uint nBase = (((uint)ny * cellsX + (uint)nx) * bins + b) * 22u;
            for (uint m = 0u; m < 22u; ++m)
                s[m] += w * ((float)statsBuf.Load((nBase + m) * 4u) * B_INV_SCALE);
        }
    }

    float4 row0 = float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 row1 = float4(0.0f, 1.0f, 0.0f, 0.0f);
    float4 row2 = float4(0.0f, 0.0f, 1.0f, 0.0f);

    float n = s[0];
    if (n >= 8.0f)
    {
        float invN = 1.0f / n;
        float3 mx = float3(s[1], s[2], s[3]) * invN;
        float3 my = float3(s[19], s[20], s[21]) * invN;

        float3x3 cxx = float3x3(
            s[4], s[5], s[6],
            s[5], s[7], s[8],
            s[6], s[8], s[9]) * invN;
        float3x3 cxy = float3x3(
            s[10], s[11], s[12],
            s[13], s[14], s[15],
            s[16], s[17], s[18]) * invN;

        [unroll] for (uint i = 0u; i < 3u; ++i)
        {
            [unroll] for (uint j = 0u; j < 3u; ++j)
            {
                cxx[i][j] -= mx[i] * mx[j];
                cxy[i][j] -= mx[i] * my[j];
                if (i == j)
                {
                    cxx[i][j] += lambda;
                    cxy[i][j] += lambda;
                }
            }
        }

        float3 r0 = cxx[0];
        float3 r1 = cxx[1];
        float3 r2 = cxx[2];
        float3 k0 = cross(r1, r2);
        float3 k1 = cross(r2, r0);
        float3 k2 = cross(r0, r1);
        float det = dot(r0, k0);
        if (IsFiniteF(det) && abs(det) > 1e-9f)
        {
            float3x3 invCxx = transpose(float3x3(k0, k1, k2)) / det;
            float3x3 a = mul(invCxx, cxy);
            float3 w0 = float3(a[0][0], a[1][0], a[2][0]);
            float3 w1 = float3(a[0][1], a[1][1], a[2][1]);
            float3 w2 = float3(a[0][2], a[1][2], a[2][2]);
            float4 cand0 = float4(w0, my.x - dot(mx, w0));
            float4 cand1 = float4(w1, my.y - dot(mx, w1));
            float4 cand2 = float4(w2, my.z - dot(mx, w2));
            if (all(IsFiniteF3(cand0.xyz)) && IsFiniteF(cand0.w) && all(IsFiniteF3(cand1.xyz)) && IsFiniteF(cand1.w) && all(IsFiniteF3(cand2.xyz)) && IsFiniteF(cand2.w))
            {
                row0 = cand0;
                row1 = cand1;
                row2 = cand2;
            }
        }
    }

    uint mBase = id.x * 12u;
    modelBuf.Store4((mBase + 0u) * 4u, asuint(row0));
    modelBuf.Store4((mBase + 4u) * 4u, asuint(row1));
    modelBuf.Store4((mBase + 8u) * 4u, asuint(row2));
}

// 4 spatial corners x 8 luma bins x 3 float4 rows = 96 float4 (1.5 KB LDS):
// Because cell=128 is a multiple of 16, all 64 threads in every 8x8 group share identical (x0,y0,x1,y1),
// cutting global modelBuf Load4 traffic by 16x (96 loads/group vs 1536 loads/group) with bit-exact output.
groupshared float4 g_cellModels[96];

[numthreads(8, 8, 1)]
void BilateralApplyCS(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex)
{
    const float grpGx = ((float)(gid.x * 8u) + 0.5f) / (float)cell - 0.5f;
    const float grpGy = ((float)(gid.y * 8u) + 0.5f) / (float)cell - 0.5f;
    const int gx0 = clamp((int)floor(grpGx), 0, (int)cellsX - 1);
    const int gy0 = clamp((int)floor(grpGy), 0, (int)cellsY - 1);
    const int gx1 = min(gx0 + 1, (int)cellsX - 1);
    const int gy1 = min(gy0 + 1, (int)cellsY - 1);

    const uint cachedBins = min(bins, 8u);
    const uint totalVecs = 4u * cachedBins * 3u;
    for (uint v = gidx; v < totalVecs; v += 64u)
    {
        const uint row = v % 3u;
        const uint cb = v / 3u;
        const uint cz = cb % cachedBins;
        const uint sc = cb / cachedBins;
        const uint cx = (uint)(((sc & 1u) != 0u) ? gx1 : gx0);
        const uint cy = (uint)(((sc & 2u) != 0u) ? gy1 : gy0);
        const uint mBase = ((cy * cellsX + cx) * bins + cz) * 12u;
        g_cellModels[v] = asfloat(modelBuf.Load4((mBase + row * 4u) * 4u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (id.x >= width || id.y >= height) return;

    float3 h = SafeSaturateRgb(hudless.Load(int3(id.xy, 0)).rgb);

    float gx = ((float)id.x + 0.5f) / (float)cell - 0.5f;
    float gy = ((float)id.y + 0.5f) / (float)cell - 0.5f;
    float gz = saturate(Luma709(h)) * (float)(bins - 1u);

    uint z0 = min((uint)floor(gz), cachedBins - 1u);
    uint z1 = min(z0 + 1u, cachedBins - 1u);

    float fx = saturate(gx - floor(gx));
    float fy = saturate(gy - floor(gy));
    float fz = saturate(gz - (float)z0);

    float4 m0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 m1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 m2 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    [unroll] for (uint corner = 0u; corner < 8u; ++corner)
    {
        bool bx = (corner & 1u) != 0u;
        bool by = (corner & 2u) != 0u;
        bool bz = (corner & 4u) != 0u;
        float w = (bx ? fx : (1.0f - fx)) * (by ? fy : (1.0f - fy)) * (bz ? fz : (1.0f - fz));
        uint sc = corner & 3u;
        uint cz = bz ? z1 : z0;
        uint baseV = (sc * cachedBins + cz) * 3u;
        m0 += w * g_cellModels[baseV + 0u];
        m1 += w * g_cellModels[baseV + 1u];
        m2 += w * g_cellModels[baseV + 2u];
    }

    float3 o = float3(
        dot(m0.xyz, h) + m0.w,
        dot(m1.xyz, h) + m1.w,
        dot(m2.xyz, h) + m2.w);
    output[id.xy] = float4(SafeSaturateRgb(o), 1.0f);
}
)";

	struct BilateralConstants
	{
		uint32_t width, height;
		uint32_t cellsX, cellsY;
		uint32_t cell, bins;
		float lambda;
		uint32_t totalStatsUints;
	};

	struct ToneMatchConstants
	{
		uint32_t width, height;
		float logMin, binsPerOctave;
		uint32_t bins, codes, minimumCount, unused;
	};

	struct RootConstants
	{
		uint32_t width;
		uint32_t height;
		float tolerance;
		uint32_t flags;
	};

	void TransitionBarrier(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* res,
		D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) noexcept
	{
		if (!res || before == after)
			return;
		D3D12_RESOURCE_BARRIER b{};
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		b.Transition.pResource = res;
		b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		b.Transition.StateBefore = before;
		b.Transition.StateAfter = after;
		cmdList->ResourceBarrier(1, &b);
	}

	void UavBarrier(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* res) noexcept
	{
		D3D12_RESOURCE_BARRIER b{};
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		b.UAV.pResource = res;
		cmdList->ResourceBarrier(1, &b);
	}

	void AddTransition(D3D12_RESOURCE_BARRIER* arr, UINT& count, ID3D12Resource* res,
		D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) noexcept
	{
		if (!res || before == after)
			return;
		for (UINT i = 0; i < count; ++i)
		{
			if (arr[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION && arr[i].Transition.pResource == res)
				return;
		}
		D3D12_RESOURCE_BARRIER& b = arr[count++];
		b = {};
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		b.Transition.pResource = res;
		b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		b.Transition.StateBefore = before;
		b.Transition.StateAfter = after;
	}

	void FlushBarriers(ID3D12GraphicsCommandList* cmdList, const D3D12_RESOURCE_BARRIER* arr, UINT count) noexcept
	{
		if (count > 0)
			cmdList->ResourceBarrier(count, arr);
	}

	bool CompileCs(D3DCompileFn compile, const char* src, size_t len, const char* entry,
		Microsoft::WRL::ComPtr<ID3DBlob>& outBlob) noexcept
	{
		Microsoft::WRL::ComPtr<ID3DBlob> err;
		const HRESULT hr = compile(src, len, "HudTierPass.hlsl", nullptr, nullptr, entry, "cs_5_0",
			D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &outBlob, &err);
		if (FAILED(hr) || !outBlob)
		{
			const char* msg = err ? static_cast<const char*>(err->GetBufferPointer()) : "unknown error";
			Log::Warn("HudTierPass: failed to compile {} ({:#010x}): {}", entry, static_cast<uint32_t>(hr), msg);
			return false;
		}
		return true;
	}
	template <typename T>
	void DeferRelease(Microsoft::WRL::ComPtr<T>& obj) noexcept
	{
		if (obj)
			DeferredDestroy::Add(std::make_shared<Microsoft::WRL::ComPtr<T>>(std::move(obj)));
	}
} // namespace

HudTierPass::~HudTierPass()
{
	StopToneMatchCompile();
	StopBilateralCompile();
	ReleaseResources();
}

bool HudTierPass::IsUavCompatibleFormat(DXGI_FORMAT format) noexcept
{
	return UavFormat(format) != DXGI_FORMAT_UNKNOWN;
}

void HudTierPass::ReleaseResources() noexcept
{
	ReleaseToneMapMatchResources();
	for (uint32_t i = 0; i < kReadbackSlots; ++i)
	{
		if (counterReadback_[i] && mappedReadback_[i])
			counterReadback_[i]->Unmap(0, nullptr);
		mappedReadback_[i] = nullptr;
		DeferRelease(counterReadback_[i]);
		readbackHasDiff_[i] = false;
		readbackDiffFromToneMatch_[i] = false;
		readbackHasStatic_[i] = false;
	}
	DeferRelease(staticCountTex_);
	DeferRelease(counterGpu_);
	DeferRelease(counterZeroUpload_);
	DeferRelease(srvUavHeap_);
	resourcesReady_ = false;
	width_ = 0;
	height_ = 0;
}

bool HudTierPass::CreatePipelines(ID3D12Device* device) noexcept
{
	const auto compile = LoadD3DCompile();
	if (!compile)
	{
		Log::Warn("HudTierPass: d3dcompiler_47.dll unavailable; GPU HUD guard/composite disabled");
		return false;
	}

	// Root signature: [0] 4 root constants (b0), [1] descriptor table (t0..t1, u0..u1)
	D3D12_DESCRIPTOR_RANGE ranges[2]{};
	ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	ranges[0].NumDescriptors = 2;
	ranges[0].BaseShaderRegister = 0;
	ranges[0].OffsetInDescriptorsFromTableStart = 0;

	ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	ranges[1].NumDescriptors = 2;
	ranges[1].BaseShaderRegister = 0;
	ranges[1].OffsetInDescriptorsFromTableStart = 2;

	D3D12_ROOT_PARAMETER params[2]{};
	params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	params[0].Constants.ShaderRegister = 0;
	params[0].Constants.Num32BitValues = 4;
	params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	params[1].DescriptorTable.NumDescriptorRanges = 2;
	params[1].DescriptorTable.pDescriptorRanges = ranges;
	params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	D3D12_ROOT_SIGNATURE_DESC rsDesc{};
	rsDesc.NumParameters = 2;
	rsDesc.pParameters = params;

	Microsoft::WRL::ComPtr<ID3DBlob> sigBlob, errBlob;
	HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
	if (FAILED(hr) || !sigBlob)
	{
		Log::Warn("HudTierPass: D3D12SerializeRootSignature failed ({:#010x})", static_cast<uint32_t>(hr));
		return false;
	}

	hr = device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig_));
	if (FAILED(hr))
	{
		Log::Warn("HudTierPass: CreateRootSignature failed ({:#010x})", static_cast<uint32_t>(hr));
		return false;
	}

	auto makePso = [&](const char* src, size_t len, const char* entry,
		Microsoft::WRL::ComPtr<ID3D12PipelineState>& outPso) -> bool {
		Microsoft::WRL::ComPtr<ID3DBlob> cs;
		if (!CompileCs(compile, src, len, entry, cs))
			return false;
		D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
		desc.pRootSignature = rootSig_.Get();
		desc.CS.pShaderBytecode = cs->GetBufferPointer();
		desc.CS.BytecodeLength = cs->GetBufferSize();
		const HRESULT psoHr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&outPso));
		if (FAILED(psoHr))
		{
			Log::Warn("HudTierPass: CreateComputePipelineState({}) failed ({:#010x})", entry, static_cast<uint32_t>(psoHr));
			return false;
		}
		return true;
	};

	if (!makePso(kHudTierHlsl, sizeof(kHudTierHlsl) - 1, "HudDiffGuardCS", psoDiffGuard_) ||
		!makePso(kHudTierHlsl, sizeof(kHudTierHlsl) - 1, "HudDiffCompositeCS", psoDiffComposite_) ||
		!makePso(kStaticHudUpdateHlsl, sizeof(kStaticHudUpdateHlsl) - 1, "StaticHudUpdateCS", psoStaticUpdate_) ||
		!makePso(kStaticHudCompositeHlsl, sizeof(kStaticHudCompositeHlsl) - 1, "StaticHudCompositeCS", psoStaticComposite_))
	{
		return false;
	}
	(void)makePso(kHudTierHlsl, sizeof(kHudTierHlsl) - 1, "SplitCompareCS", psoSplitCompare_);

	// Check per-format UAV_TYPED_LOAD + UAV_TYPED_STORE for in-place float2/float4 motion-vector unjittering
	D3D12_FEATURE_DATA_D3D12_OPTIONS d3dOpts{};
	typedUavLoadSupported_ = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &d3dOpts, sizeof(d3dOpts))) &&
		d3dOpts.TypedUAVLoadAdditionalFormats != FALSE;
	static constexpr DXGI_FORMAT kUnjitterFmts[4] = {
		DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32G32_FLOAT,
		DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT
	};
	for (uint32_t i = 0; i < 4u; ++i)
	{
		unjitterFormatSupported_[i] = false;
		if (!typedUavLoadSupported_)
			continue;
		D3D12_FEATURE_DATA_FORMAT_SUPPORT fmtSup{};
		fmtSup.Format = kUnjitterFmts[i];
		if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fmtSup, sizeof(fmtSup))))
		{
			constexpr D3D12_FORMAT_SUPPORT2 kNeed =
				D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
			unjitterFormatSupported_[i] = (fmtSup.Support2 & kNeed) == kNeed;
		}
	}
	if (typedUavLoadSupported_)
	{
		D3D12_DESCRIPTOR_RANGE uRange{};
		uRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		uRange.NumDescriptors = 1;
		uRange.BaseShaderRegister = 0;
		uRange.OffsetInDescriptorsFromTableStart = 0;

		D3D12_ROOT_PARAMETER uParams[2]{};
		uParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		uParams[0].Constants.ShaderRegister = 0;
		uParams[0].Constants.Num32BitValues = 8;
		uParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		uParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		uParams[1].DescriptorTable.NumDescriptorRanges = 1;
		uParams[1].DescriptorTable.pDescriptorRanges = &uRange;
		uParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

		D3D12_ROOT_SIGNATURE_DESC uRsDesc{};
		uRsDesc.NumParameters = 2;
		uRsDesc.pParameters = uParams;
		Microsoft::WRL::ComPtr<ID3DBlob> uSigBlob, uSigErr;
		if (SUCCEEDED(D3D12SerializeRootSignature(&uRsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &uSigBlob, &uSigErr)) &&
			SUCCEEDED(device->CreateRootSignature(0, uSigBlob->GetBufferPointer(), uSigBlob->GetBufferSize(),
				IID_PPV_ARGS(&unjitterRootSig_))))
		{
			auto makeUnjitterPso = [&](const char* entry, Microsoft::WRL::ComPtr<ID3D12PipelineState>& outPso) {
				Microsoft::WRL::ComPtr<ID3DBlob> cs;
				if (!CompileCs(compile, kUnjitterMVecsHlsl, sizeof(kUnjitterMVecsHlsl) - 1, entry, cs))
					return;
				D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
				desc.pRootSignature = unjitterRootSig_.Get();
				desc.CS.pShaderBytecode = cs->GetBufferPointer();
				desc.CS.BytecodeLength = cs->GetBufferSize();
				(void)device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&outPso));
			};
			makeUnjitterPso("UnjitterMVecs2CS", psoUnjitterMv2_);
			makeUnjitterPso("UnjitterMVecs4CS", psoUnjitterMv4_);
		}
	}

	// Optional fused Tier 3 Hybrid root signature (t0..t2, u0..u1) + PSO
	D3D12_DESCRIPTOR_RANGE hybridRanges[2]{};
	hybridRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	hybridRanges[0].NumDescriptors = 3;
	hybridRanges[0].BaseShaderRegister = 0;
	hybridRanges[0].OffsetInDescriptorsFromTableStart = 0;
	hybridRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
	hybridRanges[1].NumDescriptors = 2;
	hybridRanges[1].BaseShaderRegister = 0;
	hybridRanges[1].OffsetInDescriptorsFromTableStart = 3;

	D3D12_ROOT_PARAMETER hybridParams[2]{};
	hybridParams[0] = params[0];
	hybridParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	hybridParams[1].DescriptorTable.NumDescriptorRanges = 2;
	hybridParams[1].DescriptorTable.pDescriptorRanges = hybridRanges;
	hybridParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	D3D12_ROOT_SIGNATURE_DESC hybridRsDesc{};
	hybridRsDesc.NumParameters = 2;
	hybridRsDesc.pParameters = hybridParams;

	Microsoft::WRL::ComPtr<ID3DBlob> hybridSigBlob, hybridErrBlob;
	if (SUCCEEDED(D3D12SerializeRootSignature(&hybridRsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &hybridSigBlob, &hybridErrBlob)) &&
		hybridSigBlob &&
		SUCCEEDED(device->CreateRootSignature(0, hybridSigBlob->GetBufferPointer(), hybridSigBlob->GetBufferSize(), IID_PPV_ARGS(&hybridRootSig_))))
	{
		Microsoft::WRL::ComPtr<ID3DBlob> cs;
		if (CompileCs(compile, kHybridHudCompositeHlsl, sizeof(kHybridHudCompositeHlsl) - 1, "HybridHudCompositeCS", cs))
		{
			D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
			desc.pRootSignature = hybridRootSig_.Get();
			desc.CS.pShaderBytecode = cs->GetBufferPointer();
			desc.CS.BytecodeLength = cs->GetBufferSize();
			device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&psoHybridComposite_));
		}
	}

	descSize_ = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	return true;
}

bool HudTierPass::CreateResources(ID3D12Device* device, uint32_t width, uint32_t height) noexcept
{
	ReleaseResources();
	if (width == 0 || height == 0)
		return false;

	D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
	heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heapDesc.NumDescriptors = kDescriptorBlocks * kDescriptorsPerBlock;
	heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&srvUavHeap_))))
		return false;

	D3D12_HEAP_PROPERTIES defaultHeap{ D3D12_HEAP_TYPE_DEFAULT };
	D3D12_HEAP_PROPERTIES uploadHeap{ D3D12_HEAP_TYPE_UPLOAD };
	D3D12_HEAP_PROPERTIES readbackHeap{ D3D12_HEAP_TYPE_READBACK };

	// R8_UINT stability map for Tier 3 Static-HUD Mask (~2 MiB at 1080p, ~3.5 MiB at 1440p)
	D3D12_RESOURCE_DESC texDesc{};
	texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Width = width;
	texDesc.Height = height;
	texDesc.DepthOrArraySize = 1;
	texDesc.MipLevels = 1;
	texDesc.Format = DXGI_FORMAT_R8_UINT;
	texDesc.SampleDesc.Count = 1;
	texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	if (FAILED(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&staticCountTex_))))
	{
		ReleaseResources();
		return false;
	}

	D3D12_RESOURCE_DESC bufDesc{};
	bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	bufDesc.Width = 16;
	bufDesc.Height = 1;
	bufDesc.DepthOrArraySize = 1;
	bufDesc.MipLevels = 1;
	bufDesc.SampleDesc.Count = 1;
	bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

	D3D12_RESOURCE_DESC gpuBufDesc = bufDesc;
	gpuBufDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	if (FAILED(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &gpuBufDesc,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&counterGpu_))))
	{
		ReleaseResources();
		return false;
	}

	if (FAILED(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&counterZeroUpload_))))
	{
		ReleaseResources();
		return false;
	}

	void* zeroPtr = nullptr;
	if (SUCCEEDED(counterZeroUpload_->Map(0, nullptr, &zeroPtr)) && zeroPtr)
	{
		std::memset(zeroPtr, 0, 16);
		counterZeroUpload_->Unmap(0, nullptr);
	}

	for (uint32_t i = 0; i < kReadbackSlots; ++i)
	{
		if (FAILED(device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
			D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&counterReadback_[i]))))
		{
			ReleaseResources();
			return false;
		}
		void* ptr = nullptr;
		const D3D12_RANGE fullRange{ 0, 16 };
		if (FAILED(counterReadback_[i]->Map(0, &fullRange, &ptr)) || !ptr)
		{
			ReleaseResources();
			return false;
		}
		mappedReadback_[i] = static_cast<const uint32_t*>(ptr);
		readbackHasDiff_[i] = false;
		readbackHasStatic_[i] = false;
	}

	width_ = width;
	height_ = height;
	realFrameCount_ = 0;
	staticReadbackFrameCounter_ = 0;
	readbackSlot_ = 0;
	descBlockCursor_ = 0;
	resourcesReady_ = true;
	return true;
}

bool HudTierPass::EnsureInitialized(ID3D12Device* device, uint32_t width, uint32_t height) noexcept
{
	if (!device || width == 0 || height == 0)
		return false;

	if (device_.Get() != device)
	{
		StopToneMatchCompile();
		StopBilateralCompile();
		ReleaseResources();
		for (auto& p : toneMatchPso_) p.Reset();
		toneMatchRootSig_.Reset();
		toneMatchPipeAttempted_ = false;
		toneMatchPipeReady_ = false;
		for (auto& p : bilateralPso_) p.Reset();
		bilateralRootSig_.Reset();
		bilateralPipeAttempted_ = false;
		bilateralPipeReady_ = false;
		psoDiffGuard_.Reset();
		psoDiffComposite_.Reset();
		psoStaticUpdate_.Reset();
		psoStaticComposite_.Reset();
		psoSplitCompare_.Reset();
		psoHybridComposite_.Reset();
		hybridRootSig_.Reset();
		rootSig_.Reset();
		device_ = device;
		initAttempted_ = false;
		pipelinesReady_ = false;
	}

	if (!initAttempted_)
	{
		initAttempted_ = true;
		pipelinesReady_ = CreatePipelines(device);
	}

	if (!pipelinesReady_)
		return false;

	if (!resourcesReady_ || width_ != width || height_ != height)
	{
		if (!CreateResources(device, width, height))
			return false;
	}

	return true;
}

uint32_t HudTierPass::AllocDescriptorBlock() noexcept
{
	const uint32_t idx = descBlockCursor_;
	descBlockCursor_ = (descBlockCursor_ + 1u) % kDescriptorBlocks;
	return idx;
}

void HudTierPass::BeginRealFrame() noexcept
{
	++realFrameCount_;
	readbackSlot_ = static_cast<uint32_t>(realFrameCount_ % kReadbackSlots);
	// Keep descBlockCursor_ advancing monotonically around the 96-block ring so CPU descriptor writes never
	// overwrite blocks still referenced by in-flight GPU command lists (Streamline DLSS-G queues 3+ frames).
}

bool HudTierPass::PollDiffFraction(float& outFraction, bool& outFromToneMatched) noexcept
{
	if (!resourcesReady_ || realFrameCount_ < kReadbackSlots)
		return false;

	const uint32_t slot = readbackSlot_;
	if (!readbackHasDiff_[slot] || !mappedReadback_[slot])
		return false;

	const uint64_t sampledTotal = static_cast<uint64_t>((width_ + 3u) / 4u) * static_cast<uint64_t>((height_ + 3u) / 4u);
	if (sampledTotal == 0)
		return false;

	const uint32_t diffPixels = mappedReadback_[slot][0];
	outFraction = std::clamp(static_cast<float>(diffPixels) / static_cast<float>(sampledTotal), 0.0f, 1.0f);
	outFromToneMatched = readbackDiffFromToneMatch_[slot];
	if (outFromToneMatched && activeToneMapMode_ == 1 && !autoPromotedBilateral_)
	{
		if (outFraction > 0.18f)
		{
			if (++toneMatchHighResidualCount_ >= 3u)
			{
				autoPromotedBilateral_ = true;
				Log::Info("HudTierPass: promoting Tier 4 Tone-Map Matcher from 1D LUT to 3D Bilateral Grid (residual diff={:.1f}%)",
					outFraction * 100.0f);
			}
		}
		else
		{
			toneMatchHighResidualCount_ = 0u;
		}
	}
	readbackHasDiff_[slot] = false;
	readbackDiffFromToneMatch_[slot] = false;
	return true;
}

bool HudTierPass::PollStaticFraction(float& outFraction) noexcept
{
	if (!resourcesReady_ || realFrameCount_ < kReadbackSlots)
		return false;

	const uint32_t slot = readbackSlot_;
	if (!readbackHasStatic_[slot] || !mappedReadback_[slot])
		return false;

	const uint64_t sampledTotal = static_cast<uint64_t>((width_ + 3u) / 4u) * static_cast<uint64_t>((height_ + 3u) / 4u);
	if (sampledTotal == 0)
		return false;

	const uint32_t staticPixels = mappedReadback_[slot][1];
	outFraction = std::clamp(static_cast<float>(staticPixels) / static_cast<float>(sampledTotal), 0.0f, 1.0f);
	readbackHasStatic_[slot] = false;
	return true;
}

void HudTierPass::RecordHudDiffGuard(ID3D12GraphicsCommandList* cmdList,
	ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
	ID3D12Resource* hudlessTex, D3D12_RESOURCE_STATES hudlessState) noexcept
{
	if (!Available() || !cmdList || !realTex || !hudlessTex)
		return;

	const uint32_t block = AllocDescriptorBlock();
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = srvUavHeap_->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu = srvUavHeap_->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;

	D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
	srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Texture2D.MipLevels = 1;

	srv.Format = SrvFormat(realTex->GetDesc().Format);
	device_->CreateShaderResourceView(realTex, &srv, cpu);
	cpu.ptr += descSize_;

	srv.Format = SrvFormat(hudlessTex->GetDesc().Format);
	device_->CreateShaderResourceView(hudlessTex, &srv, cpu);
	cpu.ptr += descSize_;

	// u0 unused by HudDiffGuardCS, bind a safe null/dummy UAV (staticCountTex_ as R8_UINT) to satisfy descriptor table.
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavTex{};
	uavTex.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	uavTex.Format = DXGI_FORMAT_R8_UINT;
	device_->CreateUnorderedAccessView(staticCountTex_.Get(), nullptr, &uavTex, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavBuf{};
	uavBuf.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavBuf.Format = DXGI_FORMAT_UNKNOWN;
	uavBuf.Buffer.NumElements = 4;
	uavBuf.Buffer.StructureByteStride = 4;
	uavBuf.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavBuf, cpu);

	// Zero counter[0] via CopyBufferRegion, batching texture transitions with counter transition
	TransitionBarrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
	cmdList->CopyBufferRegion(counterGpu_.Get(), 0, counterZeroUpload_.Get(), 0, 4);
	D3D12_RESOURCE_BARRIER pre[3]{};
	UINT preCount = 0;
	AddTransition(pre, preCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	AddTransition(pre, preCount, realTex, realState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	AddTransition(pre, preCount, hudlessTex, hudlessState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	FlushBarriers(cmdList, pre, preCount);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(rootSig_.Get());
	cmdList->SetPipelineState(psoDiffGuard_.Get());

	const RootConstants rc{ width_, height_, 0.02f, 1u };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	const uint32_t sampleW = (width_ + 3u) / 4u;
	const uint32_t sampleH = (height_ + 3u) / 4u;
	cmdList->Dispatch((sampleW + 7u) / 8u, (sampleH + 7u) / 8u, 1u);

	// UAV -> COPY_SOURCE already flushes UAV writes (no redundant UavBarrier needed).
	TransitionBarrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	cmdList->CopyBufferRegion(counterReadback_[readbackSlot_].Get(), 0, counterGpu_.Get(), 0, 4);
	readbackHasDiff_[readbackSlot_] = true;
	readbackDiffFromToneMatch_[readbackSlot_] = false;

	D3D12_RESOURCE_BARRIER post[3]{};
	UINT postCount = 0;
	AddTransition(post, postCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	AddTransition(post, postCount, realTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, realState);
	AddTransition(post, postCount, hudlessTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, hudlessState);
	FlushBarriers(cmdList, post, postCount);
}

void HudTierPass::RecordHudDiffComposite(ID3D12GraphicsCommandList* cmdList,
	ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
	ID3D12Resource* hudlessTex, D3D12_RESOURCE_STATES hudlessState,
	ID3D12Resource* outputTex, D3D12_RESOURCE_STATES outputState,
	bool updateCounter, bool debugMagenta, bool fromToneMatched) noexcept
{
	if (!Available() || !cmdList || !realTex || !hudlessTex || !outputTex)
		return;

	const DXGI_FORMAT outUavFmt = UavFormat(outputTex->GetDesc().Format);
	if (outUavFmt == DXGI_FORMAT_UNKNOWN)
		return;

	const uint32_t block = AllocDescriptorBlock();
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = srvUavHeap_->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu = srvUavHeap_->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;

	D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
	srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Texture2D.MipLevels = 1;

	srv.Format = SrvFormat(realTex->GetDesc().Format);
	device_->CreateShaderResourceView(realTex, &srv, cpu);
	cpu.ptr += descSize_;

	srv.Format = SrvFormat(hudlessTex->GetDesc().Format);
	device_->CreateShaderResourceView(hudlessTex, &srv, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavOut{};
	uavOut.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	uavOut.Format = outUavFmt;
	device_->CreateUnorderedAccessView(outputTex, nullptr, &uavOut, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavBuf{};
	uavBuf.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavBuf.Format = DXGI_FORMAT_UNKNOWN;
	uavBuf.Buffer.NumElements = 4;
	uavBuf.Buffer.StructureByteStride = 4;
	uavBuf.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavBuf, cpu);

	D3D12_RESOURCE_BARRIER pre[4]{};
	UINT preCount = 0;
	if (updateCounter)
	{
		TransitionBarrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
		cmdList->CopyBufferRegion(counterGpu_.Get(), 0, counterZeroUpload_.Get(), 0, 4);
		AddTransition(pre, preCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	}
	AddTransition(pre, preCount, realTex, realState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	AddTransition(pre, preCount, hudlessTex, hudlessState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	AddTransition(pre, preCount, outputTex, outputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	FlushBarriers(cmdList, pre, preCount);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(rootSig_.Get());
	cmdList->SetPipelineState(psoDiffComposite_.Get());

	const uint32_t flags = (updateCounter ? 1u : 0u) | (debugMagenta ? 2u : 0u);
	const RootConstants rc{ width_, height_, 0.02f, flags };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);

	if (updateCounter)
	{
		D3D12_RESOURCE_BARRIER mid[2]{};
		UINT midCount = 0;
		mid[midCount].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		mid[midCount].UAV.pResource = outputTex;
		++midCount;
		AddTransition(mid, midCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
		FlushBarriers(cmdList, mid, midCount);
		cmdList->CopyBufferRegion(counterReadback_[readbackSlot_].Get(), 0, counterGpu_.Get(), 0, 4);
		readbackHasDiff_[readbackSlot_] = true;
		readbackDiffFromToneMatch_[readbackSlot_] = fromToneMatched;
	}
	else
	{
		UavBarrier(cmdList, outputTex);
	}

	D3D12_RESOURCE_BARRIER post[4]{};
	UINT postCount = 0;
	if (updateCounter)
		AddTransition(post, postCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	AddTransition(post, postCount, realTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, realState);
	AddTransition(post, postCount, hudlessTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, hudlessState);
	AddTransition(post, postCount, outputTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outputState);
	FlushBarriers(cmdList, post, postCount);
}

void HudTierPass::RecordStaticHudUpdate(ID3D12GraphicsCommandList* cmdList,
	ID3D12Resource* currRealTex, D3D12_RESOURCE_STATES currState,
	ID3D12Resource* prevRealTex, D3D12_RESOURCE_STATES prevState,
	bool resetHistory) noexcept
{
	if (!Available() || !cmdList || !currRealTex)
		return;
	if (!prevRealTex)
		resetHistory = true;

	const uint32_t block = AllocDescriptorBlock();
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = srvUavHeap_->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu = srvUavHeap_->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;

	D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
	srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Texture2D.MipLevels = 1;

	srv.Format = SrvFormat(currRealTex->GetDesc().Format);
	device_->CreateShaderResourceView(currRealTex, &srv, cpu);
	cpu.ptr += descSize_;

	ID3D12Resource* secondTex = prevRealTex ? prevRealTex : currRealTex;
	srv.Format = SrvFormat(secondTex->GetDesc().Format);
	device_->CreateShaderResourceView(secondTex, &srv, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavTex{};
	uavTex.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	uavTex.Format = DXGI_FORMAT_R8_UINT;
	device_->CreateUnorderedAccessView(staticCountTex_.Get(), nullptr, &uavTex, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavBuf{};
	uavBuf.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavBuf.Format = DXGI_FORMAT_UNKNOWN;
	uavBuf.Buffer.NumElements = 4;
	uavBuf.Buffer.StructureByteStride = 4;
	uavBuf.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavBuf, cpu);

	// Keep staticCountTex_ updated 100% every frame (identical HUD quality), but throttle the diagnostic
	// coverage counter readback (6 barriers + 2 buffer copies) to 1 in 16 real frames after the first 8 frames.
	if (resetHistory)
		staticReadbackFrameCounter_ = 0;
	const uint64_t staticFrame = ++staticReadbackFrameCounter_;
	const bool updateCounter = resetHistory || (staticFrame <= 8u) || ((staticFrame % 16u) == 0u);

	D3D12_RESOURCE_BARRIER pre[3]{};
	UINT preCount = 0;
	if (updateCounter)
	{
		TransitionBarrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
		cmdList->CopyBufferRegion(counterGpu_.Get(), 4, counterZeroUpload_.Get(), 4, 4);
		AddTransition(pre, preCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	}
	AddTransition(pre, preCount, currRealTex, currState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	if (prevRealTex && prevRealTex != currRealTex)
		AddTransition(pre, preCount, prevRealTex, prevState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	FlushBarriers(cmdList, pre, preCount);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(rootSig_.Get());
	cmdList->SetPipelineState(psoStaticUpdate_.Get());

	const uint32_t flags = (resetHistory ? 1u : 0u) | (updateCounter ? 2u : 0u);
	const RootConstants rc{ width_, height_, 0.008f, flags };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);

	if (updateCounter)
	{
		D3D12_RESOURCE_BARRIER mid[2]{};
		UINT midCount = 0;
		mid[midCount].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		mid[midCount].UAV.pResource = staticCountTex_.Get();
		++midCount;
		AddTransition(mid, midCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
		FlushBarriers(cmdList, mid, midCount);
		cmdList->CopyBufferRegion(counterReadback_[readbackSlot_].Get(), 4, counterGpu_.Get(), 4, 4);
		readbackHasStatic_[readbackSlot_] = true;
	}
	else
	{
		UavBarrier(cmdList, staticCountTex_.Get());
	}

	D3D12_RESOURCE_BARRIER post[3]{};
	UINT postCount = 0;
	if (updateCounter)
		AddTransition(post, postCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	AddTransition(post, postCount, currRealTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, currState);
	if (prevRealTex && prevRealTex != currRealTex)
		AddTransition(post, postCount, prevRealTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, prevState);
	FlushBarriers(cmdList, post, postCount);
}

void HudTierPass::RecordStaticHudComposite(ID3D12GraphicsCommandList* cmdList,
	ID3D12Resource* currRealTex, D3D12_RESOURCE_STATES currState,
	ID3D12Resource* outputTex, D3D12_RESOURCE_STATES outputState,
	bool debugMagenta) noexcept
{
	if (!Available() || !cmdList || !currRealTex || !outputTex)
		return;

	const DXGI_FORMAT outUavFmt = UavFormat(outputTex->GetDesc().Format);
	if (outUavFmt == DXGI_FORMAT_UNKNOWN)
		return;

	const uint32_t block = AllocDescriptorBlock();
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = srvUavHeap_->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu = srvUavHeap_->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;

	D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
	srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Texture2D.MipLevels = 1;

	srv.Format = SrvFormat(currRealTex->GetDesc().Format);
	device_->CreateShaderResourceView(currRealTex, &srv, cpu);
	cpu.ptr += descSize_;

	srv.Format = DXGI_FORMAT_R8_UINT;
	device_->CreateShaderResourceView(staticCountTex_.Get(), &srv, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavOut{};
	uavOut.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	uavOut.Format = outUavFmt;
	device_->CreateUnorderedAccessView(outputTex, nullptr, &uavOut, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavBuf{};
	uavBuf.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavBuf.Format = DXGI_FORMAT_UNKNOWN;
	uavBuf.Buffer.NumElements = 4;
	uavBuf.Buffer.StructureByteStride = 4;
	uavBuf.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavBuf, cpu);

	TransitionBarrier(cmdList, currRealTex, currState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	TransitionBarrier(cmdList, staticCountTex_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	TransitionBarrier(cmdList, outputTex, outputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(rootSig_.Get());
	cmdList->SetPipelineState(psoStaticComposite_.Get());

	const RootConstants rc{ width_, height_, 0.008f, debugMagenta ? 2u : 0u };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);

	UavBarrier(cmdList, outputTex);

	TransitionBarrier(cmdList, currRealTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, currState);
	TransitionBarrier(cmdList, staticCountTex_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	TransitionBarrier(cmdList, outputTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outputState);
}

void HudTierPass::RecordHybridHudComposite(ID3D12GraphicsCommandList* cmdList,
	ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
	ID3D12Resource* hudlessTex, D3D12_RESOURCE_STATES hudlessState,
	ID3D12Resource* outputTex, D3D12_RESOURCE_STATES outputState,
	bool updateCounter, bool debugMagenta, bool fromToneMatched) noexcept
{
	if (!Available() || !HasHybridComposite() || !cmdList || !realTex || !hudlessTex || !outputTex || !staticCountTex_)
		return;

	const DXGI_FORMAT outUavFmt = UavFormat(outputTex->GetDesc().Format);
	if (outUavFmt == DXGI_FORMAT_UNKNOWN)
		return;

	const uint32_t block = AllocDescriptorBlock();
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = srvUavHeap_->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu = srvUavHeap_->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;

	D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
	srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Texture2D.MipLevels = 1;

	srv.Format = SrvFormat(realTex->GetDesc().Format);
	device_->CreateShaderResourceView(realTex, &srv, cpu);
	cpu.ptr += descSize_;

	srv.Format = SrvFormat(hudlessTex->GetDesc().Format);
	device_->CreateShaderResourceView(hudlessTex, &srv, cpu);
	cpu.ptr += descSize_;

	srv.Format = DXGI_FORMAT_R8_UINT;
	device_->CreateShaderResourceView(staticCountTex_.Get(), &srv, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavOut{};
	uavOut.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	uavOut.Format = outUavFmt;
	device_->CreateUnorderedAccessView(outputTex, nullptr, &uavOut, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavBuf{};
	uavBuf.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavBuf.Format = DXGI_FORMAT_UNKNOWN;
	uavBuf.Buffer.NumElements = 4;
	uavBuf.Buffer.StructureByteStride = 4;
	uavBuf.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavBuf, cpu);

	D3D12_RESOURCE_BARRIER pre[5]{};
	UINT preCount = 0;
	if (updateCounter)
	{
		TransitionBarrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
		cmdList->CopyBufferRegion(counterGpu_.Get(), 0, counterZeroUpload_.Get(), 0, 4);
		AddTransition(pre, preCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	}
	AddTransition(pre, preCount, realTex, realState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	AddTransition(pre, preCount, hudlessTex, hudlessState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	AddTransition(pre, preCount, staticCountTex_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	AddTransition(pre, preCount, outputTex, outputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	FlushBarriers(cmdList, pre, preCount);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(hybridRootSig_.Get());
	cmdList->SetPipelineState(psoHybridComposite_.Get());

	const uint32_t flags = (updateCounter ? 1u : 0u) | (debugMagenta ? 2u : 0u);
	const RootConstants rc{ width_, height_, 0.02f, flags };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);

	if (updateCounter)
	{
		D3D12_RESOURCE_BARRIER mid[2]{};
		UINT midCount = 0;
		mid[midCount].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		mid[midCount].UAV.pResource = outputTex;
		++midCount;
		AddTransition(mid, midCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
		FlushBarriers(cmdList, mid, midCount);
		cmdList->CopyBufferRegion(counterReadback_[readbackSlot_].Get(), 0, counterGpu_.Get(), 0, 4);
		readbackHasDiff_[readbackSlot_] = true;
		readbackDiffFromToneMatch_[readbackSlot_] = fromToneMatched;
	}
	else
	{
		UavBarrier(cmdList, outputTex);
	}

	D3D12_RESOURCE_BARRIER post[5]{};
	UINT postCount = 0;
	if (updateCounter)
		AddTransition(post, postCount, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	AddTransition(post, postCount, realTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, realState);
	AddTransition(post, postCount, hudlessTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, hudlessState);
	AddTransition(post, postCount, staticCountTex_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	AddTransition(post, postCount, outputTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outputState);
	FlushBarriers(cmdList, post, postCount);
}

bool HudTierPass::IsToneMapMatchSupported(DXGI_FORMAT backbufferFormat) noexcept
{
	switch (backbufferFormat)
	{
	case DXGI_FORMAT_R10G10B10A2_TYPELESS:
	case DXGI_FORMAT_R10G10B10A2_UNORM:
	case DXGI_FORMAT_R8G8B8A8_TYPELESS:
	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_B8G8R8A8_TYPELESS:
	case DXGI_FORMAT_B8G8R8A8_UNORM:
		return true;
	default:
		return false;
	}
}

void HudTierPass::StopToneMatchCompile() noexcept
{
	if (toneMatchCompileJob_)
		toneMatchCompileJob_->cancel.store(true, std::memory_order_release);
	if (toneMatchCompileThread_.joinable())
	{
		try
		{
			toneMatchCompileThread_.join();
		}
		catch (...)
		{
		}
	}
	toneMatchCompileJob_.reset();
}

void HudTierPass::StopBilateralCompile() noexcept
{
	if (bilateralCompileJob_)
		bilateralCompileJob_->cancel.store(true, std::memory_order_release);
	if (bilateralCompileThread_.joinable())
	{
		try
		{
			bilateralCompileThread_.join();
		}
		catch (...)
		{
		}
	}
	bilateralCompileJob_.reset();
}

void HudTierPass::ReleaseToneMapMatchResources() noexcept
{
	if (toneMatchOutput_[0])
		D3D12VulkanWmma::ForgetResource(toneMatchOutput_[0].Get());
	if (toneMatchOutput_[1])
		D3D12VulkanWmma::ForgetResource(toneMatchOutput_[1].Get());
	DeferRelease(toneMatchOutput_[0]);
	DeferRelease(toneMatchOutput_[1]);
	DeferRelease(toneMatchHistogram_);
	DeferRelease(toneMatchLut_);
	DeferRelease(bilateralStats_);
	DeferRelease(bilateralModel_);
	DeferRelease(toneMatchHeap_);
	toneMatchDescCursor_ = 0;
	toneMatchFormat_ = DXGI_FORMAT_UNKNOWN;
	toneMatchShared_ = false;
	toneMatchResReady_ = false;
	bilateralResReady_ = false;
	bilateralHasValidModel_ = false;
	bilateralCellsX_ = 0;
	bilateralCellsY_ = 0;
	bilateralFrameCounter_ = 0;
	activeToneMapMode_ = 0;
}

void HudTierPass::RecordSplitCompare(ID3D12GraphicsCommandList* cmdList,
	ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
	ID3D12Resource* outputTex, D3D12_RESOURCE_STATES outputState,
	float splitPos) noexcept
{
	if (!Available() || !psoSplitCompare_ || !cmdList || !realTex || !outputTex)
		return;

	const DXGI_FORMAT outFmt = UavFormat(outputTex->GetDesc().Format);
	if (outFmt == DXGI_FORMAT_UNKNOWN)
		return;

	const uint32_t block = AllocDescriptorBlock();
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = srvUavHeap_->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu = srvUavHeap_->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;

	D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
	srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Texture2D.MipLevels = 1;
	srv.Format = SrvFormat(realTex->GetDesc().Format);
	device_->CreateShaderResourceView(realTex, &srv, cpu);
	cpu.ptr += descSize_;
	device_->CreateShaderResourceView(realTex, &srv, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavOut{};
	uavOut.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	uavOut.Format = outFmt;
	device_->CreateUnorderedAccessView(outputTex, nullptr, &uavOut, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavCnt{};
	uavCnt.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavCnt.Format = DXGI_FORMAT_UNKNOWN;
	uavCnt.Buffer.NumElements = 4;
	uavCnt.Buffer.StructureByteStride = 4;
	uavCnt.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavCnt, cpu);

	TransitionBarrier(cmdList, realTex, realState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	TransitionBarrier(cmdList, outputTex, outputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(rootSig_.Get());
	cmdList->SetPipelineState(psoSplitCompare_.Get());

	const float clampedPos = std::clamp(splitPos, 0.02f, 0.98f);
	const uint32_t splitX = static_cast<uint32_t>(clampedPos * static_cast<float>(width_));
	const RootConstants rc{ width_, height_, 0.0f, splitX };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);

	UavBarrier(cmdList, outputTex);
	TransitionBarrier(cmdList, realTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, realState);
	TransitionBarrier(cmdList, outputTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outputState);
}

bool HudTierPass::EnsureBilateralMatchReady(ID3D12Device* device, DXGI_FORMAT backbufferFormat, bool wantShared) noexcept
{
	const DXGI_FORMAT outFmt = UavFormat(backbufferFormat);
	if (!device || width_ == 0 || height_ == 0 || !IsToneMapMatchSupported(backbufferFormat) || outFmt == DXGI_FORMAT_UNKNOWN)
		return false;

	if (!bilateralPipeReady_)
	{
		if (!bilateralPipeAttempted_)
		{
			bilateralPipeAttempted_ = true;
			const auto compile = LoadD3DCompile();
			if (!compile)
				return false;

			// Compile all 4 3D Bilateral-Grid compute PSOs strictly asynchronously so DX12 Evaluate never stalls
			// and never hits Streamline's 500ms waitCPUFence / GPU TDR timeout.
			try
			{
				auto job = std::make_shared<BilateralCompileJob>();
				job->device = device;
				bilateralCompileJob_ = job;
				bilateralCompileThread_ = std::thread([job, compile]() noexcept {
					try
					{
						D3D12_DESCRIPTOR_RANGE ranges[2]{};
						ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
						ranges[0].NumDescriptors = 2;
						ranges[0].BaseShaderRegister = 0;
						ranges[0].OffsetInDescriptorsFromTableStart = 0;
						ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
						ranges[1].NumDescriptors = 3; // u0 (outTex), u1 (histogram/stats), u2 (lut/model)
						ranges[1].BaseShaderRegister = 0;
						ranges[1].OffsetInDescriptorsFromTableStart = 2;

						D3D12_ROOT_PARAMETER params[2]{};
						params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
						params[0].Constants = { 0, 0, 8 };
						params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
						params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
						params[1].DescriptorTable = { 2, ranges };
						params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

						D3D12_ROOT_SIGNATURE_DESC rsDesc{};
						rsDesc.NumParameters = 2;
						rsDesc.pParameters = params;

						Microsoft::WRL::ComPtr<ID3DBlob> sigBlob, errBlob;
						if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob)) || !sigBlob)
						{
							job->state.store(2, std::memory_order_release);
							return;
						}
						if (FAILED(job->device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&job->rootSig))))
						{
							job->state.store(2, std::memory_order_release);
							return;
						}

						constexpr const char* kEntries[4] = { "BilateralClearCS", "BilateralAccumCS", "BilateralSolveCS", "BilateralApplyCS" };
						for (uint32_t i = 0; i < 4; ++i)
						{
							if (job->cancel.load(std::memory_order_acquire))
							{
								job->state.store(2, std::memory_order_release);
								return;
							}
							Microsoft::WRL::ComPtr<ID3DBlob> cs;
							if (!CompileCs(compile, kBilateralToneMatchHlsl, sizeof(kBilateralToneMatchHlsl) - 1, kEntries[i], cs))
							{
								job->state.store(2, std::memory_order_release);
								return;
							}
							if (job->cancel.load(std::memory_order_acquire))
							{
								job->state.store(2, std::memory_order_release);
								return;
							}
							D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
							psoDesc.pRootSignature = job->rootSig.Get();
							psoDesc.CS.pShaderBytecode = cs->GetBufferPointer();
							psoDesc.CS.BytecodeLength = cs->GetBufferSize();
							if (FAILED(job->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&job->pso[i]))))
							{
								job->state.store(2, std::memory_order_release);
								return;
							}
						}

						Log::Info("HudTierPass: Tier 4 3D Bilateral-Grid Tone-Map Matcher pipelines compiled (async)");
						job->state.store(1, std::memory_order_release);
					}
					catch (...)
					{
						job->state.store(2, std::memory_order_release);
					}
				});
			}
			catch (...)
			{
				bilateralCompileJob_.reset();
			}
			return false;
		}

		if (!bilateralCompileJob_)
			return false;

		const uint8_t st = bilateralCompileJob_->state.load(std::memory_order_acquire);
		if (st == 0)
			return false;

		if (bilateralCompileThread_.joinable())
		{
			try { bilateralCompileThread_.join(); } catch (...) {}
		}

		if (st == 1)
		{
			bilateralRootSig_ = std::move(bilateralCompileJob_->rootSig);
			for (uint32_t i = 0; i < 4; ++i)
				bilateralPso_[i] = std::move(bilateralCompileJob_->pso[i]);
			bilateralPipeReady_ = true;
		}
		bilateralCompileJob_.reset();

		if (!bilateralPipeReady_)
			return false;
	}

	// Reuse toneMatchHeap_ & toneMatchOutput_[0..1] (allocating them if 1D LUT wasn't created first).
	if (!toneMatchResReady_ || toneMatchFormat_ != outFmt || toneMatchShared_ != wantShared)
	{
		ReleaseToneMapMatchResources();

		D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
		heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		heapDesc.NumDescriptors = kToneMatchDescSlots * 5u; // 32-slot ring * (t0, t1, u0, u1, u2)
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&toneMatchHeap_))))
			return false;
		toneMatchDescCursor_ = 0;

		D3D12_HEAP_PROPERTIES defaultHeap{ D3D12_HEAP_TYPE_DEFAULT };
		D3D12_RESOURCE_DESC texDesc{};
		texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Width = width_;
		texDesc.Height = height_;
		texDesc.DepthOrArraySize = 1;
		texDesc.MipLevels = 1;
		texDesc.Format = outFmt;
		texDesc.SampleDesc.Count = 1;
		texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		const D3D12_HEAP_FLAGS heapFlags = wantShared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE;
		for (uint32_t s = 0; s < 2; ++s)
		{
			if (FAILED(device->CreateCommittedResource(&defaultHeap, heapFlags, &texDesc,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
				nullptr, IID_PPV_ARGS(&toneMatchOutput_[s]))))
			{
				ReleaseToneMapMatchResources();
				return false;
			}
		}
		toneMatchFormat_ = outFmt;
		toneMatchShared_ = wantShared;
		toneMatchResReady_ = true;
	}

	const uint32_t cx = std::max(1u, (width_ + kBilateralCell - 1u) / kBilateralCell);
	const uint32_t cy = std::max(1u, (height_ + kBilateralCell - 1u) / kBilateralCell);
	if (bilateralResReady_ && bilateralCellsX_ == cx && bilateralCellsY_ == cy && bilateralStats_ && bilateralModel_)
		return true;

	DeferRelease(bilateralStats_);
	DeferRelease(bilateralModel_);
	bilateralResReady_ = false;
	bilateralHasValidModel_ = false;

	D3D12_HEAP_PROPERTIES defaultHeap{ D3D12_HEAP_TYPE_DEFAULT };
	auto makeBuf = [&](uint64_t bytes, Microsoft::WRL::ComPtr<ID3D12Resource>& out) -> bool {
		D3D12_RESOURCE_DESC d{};
		d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		d.Width = bytes;
		d.Height = 1;
		d.DepthOrArraySize = 1;
		d.MipLevels = 1;
		d.SampleDesc.Count = 1;
		d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		return SUCCEEDED(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &d,
			D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&out)));
	};

	const uint64_t totalCells = static_cast<uint64_t>(cx) * cy * kBilateralBins;
	if (!makeBuf(totalCells * 22ull * 4ull, bilateralStats_) ||
		!makeBuf(totalCells * 12ull * 4ull, bilateralModel_))
	{
		bilateralStats_.Reset();
		bilateralModel_.Reset();
		return false;
	}

	bilateralCellsX_ = cx;
	bilateralCellsY_ = cy;
	bilateralResReady_ = true;
	return true;
}

bool HudTierPass::EnsureToneMapMatchReady(ID3D12Device* device, DXGI_FORMAT backbufferFormat, bool wantShared) noexcept
{
	const DXGI_FORMAT outFmt = UavFormat(backbufferFormat);
	if (!device || width_ == 0 || height_ == 0 || !IsToneMapMatchSupported(backbufferFormat) || outFmt == DXGI_FORMAT_UNKNOWN)
		return false;

	if (!toneMatchPipeReady_)
	{
		if (!toneMatchPipeAttempted_)
		{
			toneMatchPipeAttempted_ = true;
			const auto compile = LoadD3DCompile();
			if (!compile)
				return false;

			// Compile the 5 Tier 4 Tone-Map Matcher compute PSOs asynchronously on a worker thread so the DX12
			// Evaluate thread never stalls >500ms and trips Streamline's waitCPUFence / GPU TDR timeout.
			try
			{
				auto job = std::make_shared<ToneMatchCompileJob>();
				job->device = device;
				toneMatchCompileJob_ = job;
				toneMatchCompileThread_ = std::thread([job, compile]() noexcept {
					try
					{
						D3D12_DESCRIPTOR_RANGE ranges[2]{};
						ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
						ranges[0].NumDescriptors = 2;
						ranges[0].BaseShaderRegister = 0;
						ranges[0].OffsetInDescriptorsFromTableStart = 0;
						ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
						ranges[1].NumDescriptors = 3; // u0 (outTex), u1 (histogram/stats), u2 (lut/model)
						ranges[1].BaseShaderRegister = 0;
						ranges[1].OffsetInDescriptorsFromTableStart = 2;

						D3D12_ROOT_PARAMETER params[2]{};
						params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
						params[0].Constants = { 0, 0, 8 };
						params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
						params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
						params[1].DescriptorTable = { 2, ranges };
						params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

						D3D12_ROOT_SIGNATURE_DESC rsDesc{};
						rsDesc.NumParameters = 2;
						rsDesc.pParameters = params;

						Microsoft::WRL::ComPtr<ID3DBlob> sigBlob, errBlob;
						if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob)) || !sigBlob)
						{
							job->state.store(2, std::memory_order_release);
							return;
						}
						if (FAILED(job->device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&job->rootSig))))
						{
							job->state.store(2, std::memory_order_release);
							return;
						}

						constexpr const char* kEntries[5] = { "clear", "accumulate", "median", "fill", "apply" };
						for (uint32_t i = 0; i < 5; ++i)
						{
							if (job->cancel.load(std::memory_order_acquire))
							{
								job->state.store(2, std::memory_order_release);
								return;
							}
							Microsoft::WRL::ComPtr<ID3DBlob> cs;
							if (!CompileCs(compile, kToneMatchHlsl, sizeof(kToneMatchHlsl) - 1, kEntries[i], cs))
							{
								job->state.store(2, std::memory_order_release);
								return;
							}
							if (job->cancel.load(std::memory_order_acquire))
							{
								job->state.store(2, std::memory_order_release);
								return;
							}
							D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
							psoDesc.pRootSignature = job->rootSig.Get();
							psoDesc.CS.pShaderBytecode = cs->GetBufferPointer();
							psoDesc.CS.BytecodeLength = cs->GetBufferSize();
							if (FAILED(job->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&job->pso[i]))))
							{
								job->state.store(2, std::memory_order_release);
								return;
							}
						}

						Log::Info("HudTierPass: Tier 4 inline 256-bin Tone-Map Matcher pipelines compiled (async)");
						job->state.store(1, std::memory_order_release);
					}
					catch (...)
					{
						job->state.store(2, std::memory_order_release);
					}
				});
			}
			catch (...)
			{
				toneMatchCompileJob_.reset();
			}
			return false;
		}

		if (!toneMatchCompileJob_)
			return false;

		const uint8_t st = toneMatchCompileJob_->state.load(std::memory_order_acquire);
		if (st == 0)
			return false;

		if (toneMatchCompileThread_.joinable())
		{
			try
			{
				toneMatchCompileThread_.join();
			}
			catch (...)
			{
			}
		}

		if (st == 1)
		{
			toneMatchRootSig_ = std::move(toneMatchCompileJob_->rootSig);
			for (uint32_t i = 0; i < 5; ++i)
				toneMatchPso_[i] = std::move(toneMatchCompileJob_->pso[i]);
			toneMatchPipeReady_ = true;
		}
		toneMatchCompileJob_.reset();

		if (!toneMatchPipeReady_)
			return false;
	}

	D3D12_HEAP_PROPERTIES defaultHeap{ D3D12_HEAP_TYPE_DEFAULT };
	auto makeBuf = [&](uint64_t bytes, Microsoft::WRL::ComPtr<ID3D12Resource>& out) -> bool {
		D3D12_RESOURCE_DESC d{};
		d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		d.Width = bytes;
		d.Height = 1;
		d.DepthOrArraySize = 1;
		d.MipLevels = 1;
		d.SampleDesc.Count = 1;
		d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		return SUCCEEDED(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &d,
			D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&out)));
	};

	if (toneMatchResReady_ && toneMatchFormat_ == outFmt && toneMatchShared_ == wantShared)
	{
		if (toneMatchHistogram_ && toneMatchLut_)
			return true;
		// Bilateral path allocated toneMatchOutput_/toneMatchHeap_ first; create only the missing 1D LUT buffers in place.
		if (!toneMatchHistogram_ && !makeBuf(uint64_t(3u) * kToneMatchBins * kToneMatchCodes * 4ull, toneMatchHistogram_))
			return false;
		if (!toneMatchLut_ && !makeBuf(uint64_t(3u) * kToneMatchBins * 4ull, toneMatchLut_))
			return false;
		return true;
	}

	ReleaseToneMapMatchResources();

	D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
	heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heapDesc.NumDescriptors = kToneMatchDescSlots * 5u; // 32-slot ring * (t0, t1, u0, u1, u2)
	heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&toneMatchHeap_))))
		return false;
	toneMatchDescCursor_ = 0;

	if (!makeBuf(uint64_t(3u) * kToneMatchBins * kToneMatchCodes * 4ull, toneMatchHistogram_) ||
		!makeBuf(uint64_t(3u) * kToneMatchBins * 4ull, toneMatchLut_))
	{
		ReleaseToneMapMatchResources();
		return false;
	}

	D3D12_RESOURCE_DESC texDesc{};
	texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Width = width_;
	texDesc.Height = height_;
	texDesc.DepthOrArraySize = 1;
	texDesc.MipLevels = 1;
	texDesc.Format = outFmt;
	texDesc.SampleDesc.Count = 1;
	texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	const D3D12_HEAP_FLAGS heapFlags = wantShared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE;
	for (uint32_t s = 0; s < 2; ++s)
	{
		if (FAILED(device->CreateCommittedResource(&defaultHeap, heapFlags, &texDesc,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
			nullptr, IID_PPV_ARGS(&toneMatchOutput_[s]))))
		{
			ReleaseToneMapMatchResources();
			return false;
		}
	}

	toneMatchFormat_ = outFmt;
	toneMatchShared_ = wantShared;
	toneMatchResReady_ = true;
	return true;
}

ID3D12Resource* HudTierPass::RecordToneMapMatch(ID3D12GraphicsCommandList* cmdList,
	uint32_t frameSlot, bool wantShared,
	ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
	ID3D12Resource* hudlessTex, D3D12_RESOURCE_STATES hudlessState,
	D3D12_RESOURCE_STATES matchedOutputState,
	int32_t toneMapMode, bool forceRefit) noexcept
{
	if (!Available() || !cmdList || !realTex || !hudlessTex)
		return nullptr;

	const auto bbDesc = realTex->GetDesc();
	const bool wantBilateral = (toneMapMode == 2) || (toneMapMode == 0 && autoPromotedBilateral_);

	if (wantBilateral && EnsureBilateralMatchReady(device_.Get(), bbDesc.Format, wantShared))
	{
		const uint32_t slot = frameSlot & 1u;
		ID3D12Resource* outTex = toneMatchOutput_[slot].Get();
		if (!outTex)
			return nullptr;

		const uint32_t descSlot = toneMatchDescCursor_;
		toneMatchDescCursor_ = (toneMatchDescCursor_ + 1u) % kToneMatchDescSlots;
		const uint32_t totalCells = bilateralCellsX_ * bilateralCellsY_ * kBilateralBins;
		const uint32_t totalStatsUints = totalCells * 22u;
		const uint32_t totalModelUints = totalCells * 12u;

		D3D12_CPU_DESCRIPTOR_HANDLE cpu = toneMatchHeap_->GetCPUDescriptorHandleForHeapStart();
		cpu.ptr += static_cast<SIZE_T>(descSlot) * 5u * descSize_;
		D3D12_GPU_DESCRIPTOR_HANDLE gpu = toneMatchHeap_->GetGPUDescriptorHandleForHeapStart();
		gpu.ptr += static_cast<SIZE_T>(descSlot) * 5u * descSize_;

		D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
		srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Texture2D.MipLevels = 1;

		srv.Format = SrvFormat(hudlessTex->GetDesc().Format);
		device_->CreateShaderResourceView(hudlessTex, &srv, cpu);
		cpu.ptr += descSize_;

		srv.Format = SrvFormat(bbDesc.Format);
		device_->CreateShaderResourceView(realTex, &srv, cpu);
		cpu.ptr += descSize_;

		D3D12_UNORDERED_ACCESS_VIEW_DESC uavOut{};
		uavOut.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		uavOut.Format = toneMatchFormat_;
		device_->CreateUnorderedAccessView(outTex, nullptr, &uavOut, cpu);
		cpu.ptr += descSize_;

		D3D12_UNORDERED_ACCESS_VIEW_DESC uavRaw{};
		uavRaw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavRaw.Format = DXGI_FORMAT_R32_TYPELESS;
		uavRaw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
		uavRaw.Buffer.NumElements = totalStatsUints;
		device_->CreateUnorderedAccessView(bilateralStats_.Get(), nullptr, &uavRaw, cpu);
		cpu.ptr += descSize_;

		uavRaw.Buffer.NumElements = totalModelUints;
		device_->CreateUnorderedAccessView(bilateralModel_.Get(), nullptr, &uavRaw, cpu);

		TransitionBarrier(cmdList, realTex, realState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		TransitionBarrier(cmdList, hudlessTex, hudlessState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		TransitionBarrier(cmdList, outTex, matchedOutputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		TransitionBarrier(cmdList, bilateralStats_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		TransitionBarrier(cmdList, bilateralModel_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

		ID3D12DescriptorHeap* heaps[] = { toneMatchHeap_.Get() };
		cmdList->SetDescriptorHeaps(1, heaps);
		cmdList->SetComputeRootSignature(bilateralRootSig_.Get());

		const BilateralConstants bc{
			width_, height_, bilateralCellsX_, bilateralCellsY_,
			kBilateralCell, kBilateralBins, kBilateralLambda, totalStatsUints
		};
		cmdList->SetComputeRoot32BitConstants(0, 8, &bc, 0);
		cmdList->SetComputeRootDescriptorTable(1, gpu);

		// Amortized Fit: run Clear+Accum+Solve on first frame, camera cut/reset, or every 4 real frames (~60-75% GPU savings).
		const bool runFit = !bilateralHasValidModel_ || forceRefit || (bilateralFrameCounter_ % kBilateralFitInterval == 0u);
		++bilateralFrameCounter_;
		if (runFit)
		{
			cmdList->SetPipelineState(bilateralPso_[0].Get());
			cmdList->Dispatch((totalStatsUints + 63u) / 64u, 1u, 1u);
			UavBarrier(cmdList, bilateralStats_.Get());

			cmdList->SetPipelineState(bilateralPso_[1].Get());
			const uint32_t sampleW = (width_ + kBilateralStride - 1u) / kBilateralStride;
			const uint32_t sampleH = (height_ + kBilateralStride - 1u) / kBilateralStride;
			cmdList->Dispatch((sampleW + 7u) / 8u, (sampleH + 7u) / 8u, 1u);
			UavBarrier(cmdList, bilateralStats_.Get());

			cmdList->SetPipelineState(bilateralPso_[2].Get());
			cmdList->Dispatch((totalCells + 63u) / 64u, 1u, 1u);
			UavBarrier(cmdList, bilateralModel_.Get());
			bilateralHasValidModel_ = true;
		}

		// Pass 4: Trilinear 3x4 affine apply (runs every real frame, ~0.04ms)
		cmdList->SetPipelineState(bilateralPso_[3].Get());
		cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);
		UavBarrier(cmdList, outTex);

		TransitionBarrier(cmdList, bilateralStats_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
		TransitionBarrier(cmdList, bilateralModel_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
		TransitionBarrier(cmdList, realTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, realState);
		TransitionBarrier(cmdList, hudlessTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, hudlessState);
		TransitionBarrier(cmdList, outTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, matchedOutputState);
		activeToneMapMode_ = 2;
		return outTex;
	}

	// Fallback to 1D 256-bin LUT while 3D Bilateral Grid compiles asynchronously (zero stall / zero 500ms timeout)
	// or when mode is Auto (before promotion) / 1D LUT only.
	if (!EnsureToneMapMatchReady(device_.Get(), bbDesc.Format, wantShared))
		return nullptr;
	activeToneMapMode_ = 1;

	const uint32_t slot = frameSlot & 1u;
	ID3D12Resource* outTex = toneMatchOutput_[slot].Get();
	if (!outTex)
		return nullptr;

	const uint32_t descSlot = toneMatchDescCursor_;
	toneMatchDescCursor_ = (toneMatchDescCursor_ + 1u) % kToneMatchDescSlots;
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = toneMatchHeap_->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += static_cast<SIZE_T>(descSlot) * 5u * descSize_;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu = toneMatchHeap_->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += static_cast<SIZE_T>(descSlot) * 5u * descSize_;

	D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
	srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Texture2D.MipLevels = 1;

	srv.Format = SrvFormat(hudlessTex->GetDesc().Format);
	device_->CreateShaderResourceView(hudlessTex, &srv, cpu);
	cpu.ptr += descSize_;

	srv.Format = SrvFormat(bbDesc.Format);
	device_->CreateShaderResourceView(realTex, &srv, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavOut{};
	uavOut.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	uavOut.Format = toneMatchFormat_;
	device_->CreateUnorderedAccessView(outTex, nullptr, &uavOut, cpu);
	cpu.ptr += descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uavRaw{};
	uavRaw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavRaw.Format = DXGI_FORMAT_R32_TYPELESS;
	uavRaw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	uavRaw.Buffer.NumElements = 3u * kToneMatchBins * kToneMatchCodes;
	device_->CreateUnorderedAccessView(toneMatchHistogram_.Get(), nullptr, &uavRaw, cpu);
	cpu.ptr += descSize_;

	uavRaw.Buffer.NumElements = 3u * kToneMatchBins;
	device_->CreateUnorderedAccessView(toneMatchLut_.Get(), nullptr, &uavRaw, cpu);

	TransitionBarrier(cmdList, realTex, realState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	TransitionBarrier(cmdList, hudlessTex, hudlessState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	TransitionBarrier(cmdList, outTex, matchedOutputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	TransitionBarrier(cmdList, toneMatchHistogram_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	TransitionBarrier(cmdList, toneMatchLut_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	ID3D12DescriptorHeap* heaps[] = { toneMatchHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(toneMatchRootSig_.Get());

	const ToneMatchConstants rc{
		width_, height_, kToneMatchLogMin, kToneMatchBinsPerOctave,
		kToneMatchBins, kToneMatchCodes, kToneMatchMinCount, 0u
	};
	cmdList->SetComputeRoot32BitConstants(0, 8, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);

	// 1. clear histogram
	cmdList->SetPipelineState(toneMatchPso_[0].Get());
	cmdList->Dispatch((3u * kToneMatchBins * kToneMatchCodes + 63u) / 64u, 1u, 1u);
	UavBarrier(cmdList, toneMatchHistogram_.Get());

	// 2. accumulate 1/16 sampled pixels
	cmdList->SetPipelineState(toneMatchPso_[1].Get());
	const uint32_t sampleW = (width_ + kToneMatchStride - 1u) / kToneMatchStride;
	const uint32_t sampleH = (height_ + kToneMatchStride - 1u) / kToneMatchStride;
	cmdList->Dispatch((sampleW + 7u) / 8u, (sampleH + 7u) / 8u, 1u);
	UavBarrier(cmdList, toneMatchHistogram_.Get());

	// 3. per-bin median
	cmdList->SetPipelineState(toneMatchPso_[2].Get());
	cmdList->Dispatch(kToneMatchBins, 3u, 1u);
	UavBarrier(cmdList, toneMatchLut_.Get());

	// 4. 5-bin outlier filter + empty-bin linear interpolation
	cmdList->SetPipelineState(toneMatchPso_[3].Get());
	cmdList->Dispatch(3u, 1u, 1u);
	UavBarrier(cmdList, toneMatchLut_.Get());

	// 5. apply LUT to full HUDless -> outTex
	cmdList->SetPipelineState(toneMatchPso_[4].Get());
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);
	UavBarrier(cmdList, outTex);

	TransitionBarrier(cmdList, toneMatchHistogram_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
	TransitionBarrier(cmdList, toneMatchLut_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
	TransitionBarrier(cmdList, realTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, realState);
	TransitionBarrier(cmdList, hudlessTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, hudlessState);
	TransitionBarrier(cmdList, outTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, matchedOutputState);
	return outTex;
}

static DXGI_FORMAT MotionVectorUavFormat(DXGI_FORMAT fmt, bool& is4Channel) noexcept
{
	switch (fmt)
	{
	case DXGI_FORMAT_R16G16_FLOAT:
	case DXGI_FORMAT_R16G16_TYPELESS:
		is4Channel = false;
		return DXGI_FORMAT_R16G16_FLOAT;
	case DXGI_FORMAT_R32G32_FLOAT:
	case DXGI_FORMAT_R32G32_TYPELESS:
		is4Channel = false;
		return DXGI_FORMAT_R32G32_FLOAT;
	case DXGI_FORMAT_R16G16B16A16_FLOAT:
	case DXGI_FORMAT_R16G16B16A16_TYPELESS:
		is4Channel = true;
		return DXGI_FORMAT_R16G16B16A16_FLOAT;
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
	case DXGI_FORMAT_R32G32B32A32_TYPELESS:
		is4Channel = true;
		return DXGI_FORMAT_R32G32B32A32_FLOAT;
	default:
		is4Channel = false;
		return DXGI_FORMAT_UNKNOWN;
	}
}

bool HudTierPass::CanUnjitterMotionVectors(DXGI_FORMAT format) const noexcept
{
	if (!Available() || !typedUavLoadSupported_ || !unjitterRootSig_)
		return false;
	bool is4Ch = false;
	const DXGI_FORMAT uavFmt = MotionVectorUavFormat(format, is4Ch);
	int fmtIdx = -1;
	switch (uavFmt)
	{
	case DXGI_FORMAT_R16G16_FLOAT:         fmtIdx = 0; break;
	case DXGI_FORMAT_R32G32_FLOAT:         fmtIdx = 1; break;
	case DXGI_FORMAT_R16G16B16A16_FLOAT:   fmtIdx = 2; break;
	case DXGI_FORMAT_R32G32B32A32_FLOAT:   fmtIdx = 3; break;
	default: return false;
	}
	if (!unjitterFormatSupported_[fmtIdx])
		return false;
	return is4Ch ? (psoUnjitterMv4_.Get() != nullptr) : (psoUnjitterMv2_.Get() != nullptr);
}

void HudTierPass::RecordUnjitterMotionVectors(ID3D12GraphicsCommandList* cmdList,
	ID3D12Resource* mvecsTex, D3D12_RESOURCE_STATES mvecsState,
	uint32_t baseX, uint32_t baseY, uint32_t sizeX, uint32_t sizeY,
	float deltaRawX, float deltaRawY) noexcept
{
	if (!Available() || !cmdList || !mvecsTex || !unjitterRootSig_ || sizeX == 0u || sizeY == 0u)
		return;
	if (std::abs(deltaRawX) <= 1e-7f && std::abs(deltaRawY) <= 1e-7f)
		return;

	const auto desc = mvecsTex->GetDesc();
	if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)
		return;

	bool is4Ch = false;
	const DXGI_FORMAT uavFmt = MotionVectorUavFormat(desc.Format, is4Ch);
	ID3D12PipelineState* pso = is4Ch ? psoUnjitterMv4_.Get() : psoUnjitterMv2_.Get();
	if (uavFmt == DXGI_FORMAT_UNKNOWN || !pso)
		return;

	const uint32_t maxW = static_cast<uint32_t>(desc.Width);
	const uint32_t maxH = desc.Height;
	if (baseX >= maxW || baseY >= maxH)
		return;
	const uint32_t clampedW = std::min(sizeX, maxW - baseX);
	const uint32_t clampedH = std::min(sizeY, maxH - baseY);
	if (clampedW == 0u || clampedH == 0u)
		return;

	const uint32_t block = AllocDescriptorBlock();
	D3D12_CPU_DESCRIPTOR_HANDLE cpu = srvUavHeap_->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu = srvUavHeap_->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += static_cast<SIZE_T>(block) * kDescriptorsPerBlock * descSize_;

	D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
	uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	uav.Format = uavFmt;
	device_->CreateUnorderedAccessView(mvecsTex, nullptr, &uav, cpu);

	TransitionBarrier(cmdList, mvecsTex, mvecsState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(unjitterRootSig_.Get());
	cmdList->SetPipelineState(pso);

	struct UnjitterConstants
	{
		uint32_t baseX, baseY, sizeX, sizeY;
		float deltaX, deltaY;
		uint32_t pad0, pad1;
	} rc{ baseX, baseY, clampedW, clampedH, deltaRawX, deltaRawY, 0u, 0u };
	cmdList->SetComputeRoot32BitConstants(0, 8, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((clampedW + 7u) / 8u, (clampedH + 7u) / 8u, 1u);

	UavBarrier(cmdList, mvecsTex);
	TransitionBarrier(cmdList, mvecsTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, mvecsState);
}
