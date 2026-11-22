#include "HudTierPass.h"
#include "Log.h"

#include <d3dcompiler.h>

#include <algorithm>
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
RWByteAddressBuffer Counter : register(u1);

// Tier 1 Guard: analyze-only 1/16 pixel sampling of |Real - Hudless| > Tolerance.
[numthreads(8, 8, 1)]
void HudDiffGuardCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;
    if ((id.x & 3u) != 0u || (id.y & 3u) != 0u) return;

    float3 d = abs(TexA[int2(id.xy)].rgb - TexB[int2(id.xy)].rgb);
    if (max(d.r, max(d.g, d.b)) > Tolerance)
    {
        uint dummy;
        Counter.InterlockedAdd(0, 1u, dummy);
    }
}

// Tier 2 Composite (MFGUnlock xmx_hud_composite style): 3x3 dilated |Real - Hudless| > Tolerance.
// Flags bit 0: update diff counter (1/16 sample). Flags bit 1: debug magenta overlay.
bool IsHudDiff(int2 p)
{
    float3 d = abs(TexA[p].rgb - TexB[p].rgb);
    return max(d.r, max(d.g, d.b)) > Tolerance;
}

[numthreads(8, 8, 1)]
void HudDiffCompositeCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;

    int2 maxCoord = int2(Width - 1u, Height - 1u);
    bool hud = false;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
    {
        [unroll] for (int dx = -1; dx <= 1; ++dx)
        {
            int2 q = clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), maxCoord);
            hud = hud || IsHudDiff(q);
        }
    }

    if ((Flags & 1u) != 0u && hud && (id.x & 3u) == 0u && (id.y & 3u) == 0u)
    {
        uint dummy;
        Counter.InterlockedAdd(0, 1u, dummy);
    }

    if (hud)
    {
        if ((Flags & 2u) != 0u)
            OutColor[id.xy] = float4(1.0f, 0.0f, 1.0f, 1.0f);
        else
            OutColor[id.xy] = TexA[int2(id.xy)];
    }
}
)";

	// Separate HLSL block for Tier 3 so u0 can be RWTexture2D<uint> in UpdateCS and RWTexture2D<float4> in CompositeCS.
	constexpr char kStaticHudUpdateHlsl[] = R"(
cbuffer Params : register(b0)
{
    uint Width;
    uint Height;
    float Tolerance;
    uint Flags; // bit 0: reset history
};

Texture2D<float4> CurrReal : register(t0);
Texture2D<float4> PrevReal : register(t1);
RWTexture2D<uint> StaticCount : register(u0);
RWByteAddressBuffer Counter : register(u1);

[numthreads(8, 8, 1)]
void StaticHudUpdateCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;

    uint count = 0u;
    if ((Flags & 1u) == 0u)
    {
        float3 d = abs(CurrReal[int2(id.xy)].rgb - PrevReal[int2(id.xy)].rgb);
        bool unchanged = max(d.r, max(d.g, d.b)) <= Tolerance;
        uint prev = StaticCount[id.xy];
        count = unchanged ? min(prev + 1u, 8u) : 0u;
    }

    StaticCount[id.xy] = count;

    // Count 1/16 sampled pixels that reached the static-HUD threshold (>= 4 consecutive real frames).
    if (count >= 4u && (id.x & 3u) == 0u && (id.y & 3u) == 0u)
    {
        uint dummy;
        Counter.InterlockedAdd(4, 1u, dummy);
    }
}
)";

	constexpr char kStaticHudCompositeHlsl[] = R"(
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
RWByteAddressBuffer Counter : register(u1);

[numthreads(8, 8, 1)]
void StaticHudCompositeCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height) return;

    if (StaticCount[int2(id.xy)] >= 4u)
    {
        if ((Flags & 2u) != 0u)
            OutColor[id.xy] = float4(1.0f, 0.0f, 1.0f, 1.0f);
        else
            OutColor[id.xy] = CurrReal[int2(id.xy)];
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
    if (!(v > 0.0)) return 0;
    return (uint)clamp((log2(v) - logMin) * binsPerOctave, 0.0, (float)(bins - 1));
}

[numthreads(64, 1, 1)]
void clear(uint3 id : SV_DispatchThreadID)
{
    if (id.x < 3 * bins * codes) histogram.Store(id.x * 4, 0);
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
        histogram.InterlockedAdd(((c * bins + BinOf(h[c])) * codes + code) * 4, 1, previous);
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
    [unroll] for (uint k = 0; k < 4; ++k) { counts[k] = histogram.Load((base + k) * 4); sum += counts[k]; }
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

[numthreads(1, 1, 1)]
void fill(uint3 group : SV_GroupID)
{
    const uint c = group.x, base = c * bins;
    float raw[256], l[256];
    for (uint r = 0; r < 256; ++r) { raw[r] = asfloat(lut.Load((base + r) * 4)); l[r] = raw[r]; }
    for (uint k = 0; k < 256; ++k)
    {
        if (raw[k] < 0.0) continue;
        float w[5];
        uint n = 0;
        for (int d = -2; d <= 2; ++d)
        {
            const int j = (int)k + d;
            if (j >= 0 && j < 256 && raw[j] >= 0.0) { w[n] = raw[j]; ++n; }
        }
        for (uint a = 1; a < n; ++a)
        {
            const float x = w[a];
            int p = (int)a - 1;
            while (p >= 0 && w[p] > x) { w[p + 1] = w[p]; --p; }
            w[p + 1] = x;
        }
        if (abs(raw[k] - w[n / 2]) > OUTLIER) l[k] = w[n / 2];
    }
    int first = -1, last = -1;
    for (uint i = 0; i < 256; ++i) if (l[i] >= 0.0) { if (first < 0) first = (int)i; last = (int)i; }
    if (first < 0) { for (uint z = 0; z < 256; ++z) lut.Store((base + z) * 4, asuint(0.0)); return; }
    int previous = first;
    for (int b = first + 1; b <= last; ++b)
    {
        if (l[b] < 0.0) continue;
        for (int g = previous + 1; g < b; ++g) l[g] = lerp(l[previous], l[b], (float)(g - previous) / (float)(b - previous));
        previous = b;
    }
    const int span = min(4, last - first);
    const float headSlope = span > 0 ? max((l[first + span] - l[first]) / (float)span, 0.0) : 0.0;
    const float tailSlope = span > 0 ? max((l[last] - l[last - span]) / (float)span, 0.0) : 0.0;
    for (int h = 0; h < first; ++h) l[h] = saturate(l[first] - headSlope * (float)(first - h));
    for (int t = last + 1; t < 256; ++t) l[t] = saturate(l[last] + tailSlope * (float)(t - last));
    for (uint o = 0; o < 256; ++o) lut.Store((base + o) * 4, asuint(l[o]));
}

[numthreads(8, 8, 1)]
void apply(uint3 id : SV_DispatchThreadID)
{
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
        o[c] = lerp(asfloat(lut.Load((c * bins + i) * 4)), asfloat(lut.Load((c * bins + i + 1) * 4)), t);
    }
    output[id.xy] = float4(o, 1.0);
}
)";

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

	void Barrier(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* res,
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
} // namespace

HudTierPass::~HudTierPass()
{
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
		counterReadback_[i].Reset();
		readbackHasDiff_[i] = false;
		readbackDiffFromToneMatch_[i] = false;
		readbackHasStatic_[i] = false;
	}
	staticCountTex_.Reset();
	counterGpu_.Reset();
	counterZeroUpload_.Reset();
	srvUavHeap_.Reset();
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
		ReleaseResources();
		for (auto& p : toneMatchPso_) p.Reset();
		toneMatchRootSig_.Reset();
		toneMatchPipeAttempted_ = false;
		toneMatchPipeReady_ = false;
		psoDiffGuard_.Reset();
		psoDiffComposite_.Reset();
		psoStaticUpdate_.Reset();
		psoStaticComposite_.Reset();
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
	descBlockCursor_ = 0;
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
	uavBuf.Format = DXGI_FORMAT_R32_TYPELESS;
	uavBuf.Buffer.NumElements = 4;
	uavBuf.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavBuf, cpu);

	// Zero counter[0] via CopyBufferRegion
	Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
	cmdList->CopyBufferRegion(counterGpu_.Get(), 0, counterZeroUpload_.Get(), 0, 4);
	Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	Barrier(cmdList, realTex, realState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	Barrier(cmdList, hudlessTex, hudlessState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(rootSig_.Get());
	cmdList->SetPipelineState(psoDiffGuard_.Get());

	const RootConstants rc{ width_, height_, 0.02f, 1u };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);

	UavBarrier(cmdList, counterGpu_.Get());
	Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	cmdList->CopyBufferRegion(counterReadback_[readbackSlot_].Get(), 0, counterGpu_.Get(), 0, 4);
	Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	readbackHasDiff_[readbackSlot_] = true;
	readbackDiffFromToneMatch_[readbackSlot_] = false;

	Barrier(cmdList, realTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, realState);
	Barrier(cmdList, hudlessTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, hudlessState);
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
	uavBuf.Format = DXGI_FORMAT_R32_TYPELESS;
	uavBuf.Buffer.NumElements = 4;
	uavBuf.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavBuf, cpu);

	if (updateCounter)
	{
		Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
		cmdList->CopyBufferRegion(counterGpu_.Get(), 0, counterZeroUpload_.Get(), 0, 4);
		Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	}

	Barrier(cmdList, realTex, realState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	Barrier(cmdList, hudlessTex, hudlessState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	Barrier(cmdList, outputTex, outputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(rootSig_.Get());
	cmdList->SetPipelineState(psoDiffComposite_.Get());

	const uint32_t flags = (updateCounter ? 1u : 0u) | (debugMagenta ? 2u : 0u);
	const RootConstants rc{ width_, height_, 0.02f, flags };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);

	UavBarrier(cmdList, outputTex);

	if (updateCounter)
	{
		UavBarrier(cmdList, counterGpu_.Get());
		Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
		cmdList->CopyBufferRegion(counterReadback_[readbackSlot_].Get(), 0, counterGpu_.Get(), 0, 4);
		Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		readbackHasDiff_[readbackSlot_] = true;
		readbackDiffFromToneMatch_[readbackSlot_] = fromToneMatched;
	}

	Barrier(cmdList, realTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, realState);
	Barrier(cmdList, hudlessTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, hudlessState);
	Barrier(cmdList, outputTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outputState);
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
	uavBuf.Format = DXGI_FORMAT_R32_TYPELESS;
	uavBuf.Buffer.NumElements = 4;
	uavBuf.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavBuf, cpu);

	// Zero counter[1] (byte offset 4) via CopyBufferRegion
	Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
	cmdList->CopyBufferRegion(counterGpu_.Get(), 4, counterZeroUpload_.Get(), 4, 4);
	Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	Barrier(cmdList, currRealTex, currState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	if (prevRealTex && prevRealTex != currRealTex)
		Barrier(cmdList, prevRealTex, prevState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(rootSig_.Get());
	cmdList->SetPipelineState(psoStaticUpdate_.Get());

	const RootConstants rc{ width_, height_, 0.008f, resetHistory ? 1u : 0u };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);

	UavBarrier(cmdList, staticCountTex_.Get());
	UavBarrier(cmdList, counterGpu_.Get());

	Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	cmdList->CopyBufferRegion(counterReadback_[readbackSlot_].Get(), 4, counterGpu_.Get(), 4, 4);
	Barrier(cmdList, counterGpu_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	readbackHasStatic_[readbackSlot_] = true;

	Barrier(cmdList, currRealTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, currState);
	if (prevRealTex && prevRealTex != currRealTex)
		Barrier(cmdList, prevRealTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, prevState);
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
	uavBuf.Format = DXGI_FORMAT_R32_TYPELESS;
	uavBuf.Buffer.NumElements = 4;
	uavBuf.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	device_->CreateUnorderedAccessView(counterGpu_.Get(), nullptr, &uavBuf, cpu);

	Barrier(cmdList, currRealTex, currState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	Barrier(cmdList, staticCountTex_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	Barrier(cmdList, outputTex, outputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	ID3D12DescriptorHeap* heaps[] = { srvUavHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(rootSig_.Get());
	cmdList->SetPipelineState(psoStaticComposite_.Get());

	const RootConstants rc{ width_, height_, 0.008f, debugMagenta ? 2u : 0u };
	cmdList->SetComputeRoot32BitConstants(0, 4, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->Dispatch((width_ + 7u) / 8u, (height_ + 7u) / 8u, 1u);

	UavBarrier(cmdList, outputTex);

	Barrier(cmdList, currRealTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, currState);
	Barrier(cmdList, staticCountTex_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	Barrier(cmdList, outputTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outputState);
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

void HudTierPass::ReleaseToneMapMatchResources() noexcept
{
	toneMatchOutput_[0].Reset();
	toneMatchOutput_[1].Reset();
	toneMatchHistogram_.Reset();
	toneMatchLut_.Reset();
	toneMatchHeap_.Reset();
	toneMatchFormat_ = DXGI_FORMAT_UNKNOWN;
	toneMatchShared_ = false;
	toneMatchResReady_ = false;
}

bool HudTierPass::EnsureToneMapMatchReady(ID3D12Device* device, DXGI_FORMAT backbufferFormat, bool wantShared) noexcept
{
	const DXGI_FORMAT outFmt = UavFormat(backbufferFormat);
	if (!device || width_ == 0 || height_ == 0 || !IsToneMapMatchSupported(backbufferFormat) || outFmt == DXGI_FORMAT_UNKNOWN)
		return false;

	if (!toneMatchPipeAttempted_)
	{
		toneMatchPipeAttempted_ = true;
		const auto compile = LoadD3DCompile();
		if (!compile)
			return false;

		D3D12_DESCRIPTOR_RANGE ranges[2]{};
		ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
		ranges[0].NumDescriptors = 2;
		ranges[0].BaseShaderRegister = 0;
		ranges[0].OffsetInDescriptorsFromTableStart = 0;
		ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
		ranges[1].NumDescriptors = 1;
		ranges[1].BaseShaderRegister = 0;
		ranges[1].OffsetInDescriptorsFromTableStart = 2;

		D3D12_ROOT_PARAMETER params[4]{};
		params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		params[0].Constants = { 0, 0, 8 };
		params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		params[1].DescriptorTable = { 2, ranges };
		params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		params[2].Descriptor = { 1, 0 };
		params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		params[3].Descriptor = { 2, 0 };
		params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

		D3D12_ROOT_SIGNATURE_DESC rsDesc{};
		rsDesc.NumParameters = 4;
		rsDesc.pParameters = params;

		Microsoft::WRL::ComPtr<ID3DBlob> sigBlob, errBlob;
		if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob)) || !sigBlob)
			return false;
		if (FAILED(device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&toneMatchRootSig_))))
			return false;

		constexpr const char* kEntries[5] = { "clear", "accumulate", "median", "fill", "apply" };
		for (uint32_t i = 0; i < 5; ++i)
		{
			Microsoft::WRL::ComPtr<ID3DBlob> cs;
			if (!CompileCs(compile, kToneMatchHlsl, sizeof(kToneMatchHlsl) - 1, kEntries[i], cs))
				return false;
			D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
			psoDesc.pRootSignature = toneMatchRootSig_.Get();
			psoDesc.CS.pShaderBytecode = cs->GetBufferPointer();
			psoDesc.CS.BytecodeLength = cs->GetBufferSize();
			if (FAILED(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&toneMatchPso_[i]))))
				return false;
		}
		toneMatchPipeReady_ = true;
		Log::Info("HudTierPass: Tier 4 inline 256-bin Tone-Map Matcher pipelines compiled");
	}

	if (!toneMatchPipeReady_)
		return false;

	if (toneMatchResReady_ && toneMatchFormat_ == outFmt && toneMatchShared_ == wantShared)
		return true;

	ReleaseToneMapMatchResources();

	D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
	heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heapDesc.NumDescriptors = 6; // 2 slots * (t0, t1, u0)
	heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&toneMatchHeap_))))
		return false;

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
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&out)));
	};

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
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
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
	D3D12_RESOURCE_STATES matchedOutputState) noexcept
{
	if (!Available() || !cmdList || !realTex || !hudlessTex)
		return nullptr;

	const auto bbDesc = realTex->GetDesc();
	if (!EnsureToneMapMatchReady(device_.Get(), bbDesc.Format, wantShared))
		return nullptr;

	const uint32_t slot = frameSlot & 1u;
	ID3D12Resource* outTex = toneMatchOutput_[slot].Get();
	if (!outTex)
		return nullptr;

	D3D12_CPU_DESCRIPTOR_HANDLE cpu = toneMatchHeap_->GetCPUDescriptorHandleForHeapStart();
	cpu.ptr += static_cast<SIZE_T>(slot) * 3u * descSize_;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu = toneMatchHeap_->GetGPUDescriptorHandleForHeapStart();
	gpu.ptr += static_cast<SIZE_T>(slot) * 3u * descSize_;

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

	Barrier(cmdList, realTex, realState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	Barrier(cmdList, hudlessTex, hudlessState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	Barrier(cmdList, outTex, matchedOutputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	ID3D12DescriptorHeap* heaps[] = { toneMatchHeap_.Get() };
	cmdList->SetDescriptorHeaps(1, heaps);
	cmdList->SetComputeRootSignature(toneMatchRootSig_.Get());

	const ToneMatchConstants rc{
		width_, height_, kToneMatchLogMin, kToneMatchBinsPerOctave,
		kToneMatchBins, kToneMatchCodes, kToneMatchMinCount, 0u
	};
	cmdList->SetComputeRoot32BitConstants(0, 8, &rc, 0);
	cmdList->SetComputeRootDescriptorTable(1, gpu);
	cmdList->SetComputeRootUnorderedAccessView(2, toneMatchHistogram_->GetGPUVirtualAddress());
	cmdList->SetComputeRootUnorderedAccessView(3, toneMatchLut_->GetGPUVirtualAddress());

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

	Barrier(cmdList, realTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, realState);
	Barrier(cmdList, hudlessTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, hudlessState);
	Barrier(cmdList, outTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, matchedOutputState);
	return outTex;
}

