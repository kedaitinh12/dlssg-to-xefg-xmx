#include "FrameGenerator.h"
#include "Log.h"
#include "MfgKernels.h"
#include "MfgPort/Cuda/cuda_route.h"
#include "MfgPort/AdapterIdentity.h"
#include "D3D12VulkanWmma.h"
#include "MfgPort/Wmma/wmma_route.h"
#include "Plugin.h"
#include "Util.h"

#include <DirectXMath.h>

#include <algorithm>
#include <cmath>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace DirectX;

namespace
{
	// The optional fatbin is built from src/MfgPort/Cuda/xefg_cuda_layers.cu when CudaRoot is set.
	// Its absence is not an error: XeFG keeps running the DXIL/DP4a programs.
	std::vector<unsigned char> LoadCudaFatbin(const std::filesystem::path& path)
	{
		std::ifstream file(path, std::ios::binary);
		if (!file)
			return {};

		std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
		if (file.bad() || bytes.empty() || bytes.size() > 16 * 1024 * 1024)
			return {};

		return bytes;
	}

	void CudaRouteLog(const char* stage, const std::string& fields) noexcept
	{
		try
		{
			Log::Info("{} {}", stage, fields);
		}
		catch (...)
		{
		}
	}
} // namespace

// States of the DLSSG resources when EvaluateFeature is called (same as dlssg-to-fsr3 assumes)
constexpr auto BackbufferState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
constexpr auto InputState = D3D12_RESOURCE_STATE_COPY_DEST; // depth, mvecs, hudless
constexpr auto OutputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

// What XeFG wants for its inputs, our copies always rest in this state
constexpr auto XeFGInputState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

// XeFG's own XMX network, shipped in libxess_fg.dll. XeFG only picks it on Intel GPUs whose driver reports XMX
constexpr int32_t XeFGXmxModel = 17;

// The raw DXGI_FORMAT number says nothing in a log, the games' formats are worth naming
static const char* FormatName(DXGI_FORMAT format)
{
	switch (format)
	{
	case DXGI_FORMAT_R8G8B8A8_UNORM:
		return "R8G8B8A8_UNORM";
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
		return "R8G8B8A8_UNORM_SRGB";
	case DXGI_FORMAT_B8G8R8A8_UNORM:
		return "B8G8R8A8_UNORM";
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
		return "B8G8R8A8_UNORM_SRGB";
	case DXGI_FORMAT_R10G10B10A2_UNORM:
		return "R10G10B10A2_UNORM";
	case DXGI_FORMAT_R16G16B16A16_FLOAT:
		return "R16G16B16A16_FLOAT";
	default:
		return "unknown format";
	}
}

// XeFG's HDR support is R10G10B10A2_UNORM in HDR10 / BT.2100 only, scRGB (FP16) is rejected. A resource doesn't
// carry its color space, so the format is the only hint there is: FP16 on a swap chain means scRGB HDR in practice
static const char* HdrDescription(DXGI_FORMAT format)
{
	switch (format)
	{
	case DXGI_FORMAT_R10G10B10A2_UNORM:
		return "HDR10 / BT.2100";
	case DXGI_FORMAT_R16G16B16A16_FLOAT:
		return "scRGB, not supported by XeFG";
	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
	case DXGI_FORMAT_B8G8R8A8_UNORM:
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
		return "SDR";
	default:
		return "unknown color space";
	}
}

// XeFG only accepts 8-bit UNORM/sRGB and HDR10 R10G10B10A2_UNORM; scRGB FP16 (R16G16B16A16_FLOAT) is rejected by libxess_fg.
static bool IsSupportedXeFGColorFormat(DXGI_FORMAT format) noexcept
{
	switch (format)
	{
	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
	case DXGI_FORMAT_B8G8R8A8_UNORM:
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
	case DXGI_FORMAT_R10G10B10A2_UNORM:
		return true;
	default:
		return false;
	}
}

// XeFG stock network. libxess_fg.dll does not ship its multi frame variant (16), which Init would switch to for
// more than one generated frame, so it gets forced. The XMX port publishes 16, and then Init loads the cheaper MFG
// model: one optical flow field for the whole batch instead of one per generated frame

FrameGenerator::Context::~Context()
{
	for (auto& out : outputs)
		D3D12VulkanWmma::ForgetResource(out.Get());
	for (auto& f : frames)
	{
		D3D12VulkanWmma::ForgetResource(f.backbuffer.Get());
		D3D12VulkanWmma::ForgetResource(f.hudless.Get());
		D3D12VulkanWmma::ForgetResource(f.depth.Get());
		D3D12VulkanWmma::ForgetResource(f.mvecs.Get());
	}
	D3D12VulkanWmma::ResetContext();
	xefg_cuda::InvalidateCache();

	if (handle)
	{
		if (auto result = XeFG::Destroy(handle); result != XEFGD_RESULT_SUCCESS)
			Log::Warn("XeFG Destroy error: {}", result);
	}
}

static void XeFGLogCallback(const char* message, xefgd_logging_level_t level, void*)
{
	switch (level)
	{
	case XEFGD_LOGGING_LEVEL_ERROR:
		Log::Error("XeFG: {}", message);
		break;
	case XEFGD_LOGGING_LEVEL_WARNING:
		Log::Warn("XeFG: {}", message);
		break;
	default:
		Log::Info("XeFG: {}", message);
		break;
	}
}

static ComPtr<ID3D12Resource> CreateTexture(ID3D12Device* device, D3D12_RESOURCE_DESC desc,
	D3D12_RESOURCE_STATES state, const wchar_t* name, const char* purpose, bool wantShared = false)
{
	ComPtr<ID3D12Resource> texture;
	D3D12_HEAP_PROPERTIES heap{ D3D12_HEAP_TYPE_DEFAULT };

	const bool tryShared = wantShared && D3D12VulkanWmma::CanShareTexture(desc);
	HRESULT hr = E_FAIL;
	if (tryShared)
	{
		D3D12_RESOURCE_DESC sharedDesc = desc;
		sharedDesc.Alignment = 0;
		hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &sharedDesc, state, nullptr,
			IID_PPV_ARGS(&texture));
	}
	if (FAILED(hr))
		hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
			IID_PPV_ARGS(&texture));
	if (FAILED(hr))
	{
		// The desc is built from the game's resource, so it has to be in the message to identify the culprit
		Log::Error("CreateCommittedResource error: {:X} for {}, {}x{}x{}, mips {}, format {}, flags {:X}, layout {}, samples {}/{}",
			(uint32_t)hr, purpose, desc.Width, desc.Height, desc.DepthOrArraySize, desc.MipLevels,
			(uint32_t)desc.Format, (uint32_t)desc.Flags, (uint32_t)desc.Layout, desc.SampleDesc.Count,
			desc.SampleDesc.Quality);
		return nullptr;
	}

	texture->SetName(name);
	return texture;
}

// Depth copies become plain color textures where the format allows, anything can read those through SRVs
static void MakeReadableDepth(D3D12_RESOURCE_DESC& desc)
{
	switch (desc.Format)
	{
	case DXGI_FORMAT_R32_TYPELESS:
	case DXGI_FORMAT_D32_FLOAT:
	case DXGI_FORMAT_R16_TYPELESS:
	case DXGI_FORMAT_D16_UNORM:
		desc.Format = TypedFormat(desc.Format);
		desc.Flags &= ~D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
		break;
	default:
		break;
	}
}

static bool IsDepth32StencilFormat(DXGI_FORMAT fmt) noexcept
{
	return fmt == DXGI_FORMAT_R32G8X24_TYPELESS ||
		fmt == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ||
		fmt == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
}

// Extracts plane 0 (32-bit float depth) from a D32_FLOAT_S8X24 / R32G8X24 depth-stencil texture into a plain
// D3D12_HEAP_FLAG_SHARED R32_FLOAT Texture2D via a DEFAULT-heap scratch buffer so the D3D12->Vulkan WMMA bridge
// can import depth without hitting D3D12's ban on sharing ALLOW_DEPTH_STENCIL resources.
static bool CaptureDepth32Plane0(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12Resource* source,
	D3D12_RESOURCE_STATES sourceState, ComPtr<ID3D12Resource>& copy, ComPtr<ID3D12Resource>& scratch, const char* purpose)
{
	const D3D12_RESOURCE_DESC rawSrcDesc = source->GetDesc();
	D3D12_RESOURCE_DESC srcDesc = rawSrcDesc;
	srcDesc.Alignment = 0;
	srcDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
	srcDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	srcDesc.MipLevels = 1;
	srcDesc.DepthOrArraySize = 1;

	D3D12_RESOURCE_DESC dstDesc = srcDesc;
	dstDesc.Format = DXGI_FORMAT_R32_FLOAT;

	D3D12_PLACED_SUBRESOURCE_FOOTPRINT srcFootprint{};
	UINT64 srcBytes = 0;
	device->GetCopyableFootprints(&srcDesc, 0, 1, 0, &srcFootprint, nullptr, nullptr, &srcBytes);

	D3D12_PLACED_SUBRESOURCE_FOOTPRINT dstFootprint{};
	UINT64 dstBytes = 0;
	device->GetCopyableFootprints(&dstDesc, 0, 1, 0, &dstFootprint, nullptr, nullptr, &dstBytes);

	if (srcBytes == 0 || srcBytes == UINT64_MAX || dstBytes == 0 || dstBytes == UINT64_MAX ||
		srcFootprint.Footprint.RowPitch != dstFootprint.Footprint.RowPitch)
	{
		static std::atomic<bool> logged{ false };
		if (!logged.exchange(true, std::memory_order_relaxed))
			Log::Warn("CaptureDepth32Plane0 skipped (srcFmt={}, srcFpFmt={}, srcPitch={}, dstPitch={}, srcBytes={}, dstBytes={}, {}x{})",
				static_cast<uint32_t>(rawSrcDesc.Format), static_cast<uint32_t>(srcFootprint.Footprint.Format),
				srcFootprint.Footprint.RowPitch, dstFootprint.Footprint.RowPitch, srcBytes, dstBytes,
				static_cast<uint64_t>(rawSrcDesc.Width), rawSrcDesc.Height);
		return false;
	}

	const UINT64 bufBytes = std::max(srcBytes, dstBytes);
	if (scratch && scratch->GetDesc().Width < bufBytes)
	{
		DeferredDestroy::Add(std::make_shared<ComPtr<ID3D12Resource>>(std::move(scratch)));
	}
	if (!scratch)
	{
		D3D12_HEAP_PROPERTIES heap{ D3D12_HEAP_TYPE_DEFAULT };
		D3D12_RESOURCE_DESC bufDesc{};
		bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		bufDesc.Width = bufBytes;
		bufDesc.Height = 1;
		bufDesc.DepthOrArraySize = 1;
		bufDesc.MipLevels = 1;
		bufDesc.Format = DXGI_FORMAT_UNKNOWN;
		bufDesc.SampleDesc.Count = 1;
		bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bufDesc,
				D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&scratch))) || !scratch)
			return false;
		scratch->SetName(L"dlssg-to-xefg depth plane0 scratch");
	}

	if (copy)
	{
		const auto copyDesc = copy->GetDesc();
		if (copyDesc.Dimension != dstDesc.Dimension || copyDesc.Width != dstDesc.Width || copyDesc.Height != dstDesc.Height ||
			copyDesc.DepthOrArraySize != dstDesc.DepthOrArraySize || copyDesc.MipLevels != dstDesc.MipLevels ||
			copyDesc.Format != dstDesc.Format || copyDesc.SampleDesc.Count != dstDesc.SampleDesc.Count ||
			copyDesc.Layout != dstDesc.Layout || copyDesc.Flags != dstDesc.Flags)
		{
			D3D12VulkanWmma::ForgetResource(copy.Get());
			DeferredDestroy::Add(std::make_shared<ComPtr<ID3D12Resource>>(std::move(copy)));
		}
	}
	if (!copy)
		copy = CreateTexture(device, dstDesc, XeFGInputState, L"dlssg-to-xefg depth plane0 copy", purpose, true);
	if (!copy)
		return false;

	D3D12_RESOURCE_BARRIER pre[2]{};
	UINT preCount = 0;
	if (sourceState != D3D12_RESOURCE_STATE_COPY_SOURCE)
	{
		pre[preCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		pre[preCount].Transition = { source, 0, sourceState, D3D12_RESOURCE_STATE_COPY_SOURCE };
		++preCount;
	}
	if (XeFGInputState != D3D12_RESOURCE_STATE_COPY_DEST)
	{
		pre[preCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		pre[preCount].Transition = { copy.Get(), 0, XeFGInputState, D3D12_RESOURCE_STATE_COPY_DEST };
		++preCount;
	}
	if (preCount)
		cmdList->ResourceBarrier(preCount, pre);

	D3D12_TEXTURE_COPY_LOCATION srcLoc{};
	srcLoc.pResource = source;
	srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	srcLoc.SubresourceIndex = 0;

	D3D12_TEXTURE_COPY_LOCATION bufSrcLoc{};
	bufSrcLoc.pResource = scratch.Get();
	bufSrcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	bufSrcLoc.PlacedFootprint = srcFootprint;
	cmdList->CopyTextureRegion(&bufSrcLoc, 0, 0, 0, &srcLoc, nullptr);

	D3D12_RESOURCE_BARRIER mid{};
	mid.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	mid.Transition = { scratch.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
		D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE };
	cmdList->ResourceBarrier(1, &mid);

	D3D12_TEXTURE_COPY_LOCATION dstLoc{};
	dstLoc.pResource = copy.Get();
	dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	dstLoc.SubresourceIndex = 0;

	D3D12_TEXTURE_COPY_LOCATION bufDstLoc{};
	bufDstLoc.pResource = scratch.Get();
	bufDstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	bufDstLoc.PlacedFootprint = dstFootprint;
	cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &bufDstLoc, nullptr);

	D3D12_RESOURCE_BARRIER post[3]{};
	UINT postCount = 0;
	for (UINT i = 0; i < preCount; ++i)
	{
		post[postCount] = pre[i];
		std::swap(post[postCount].Transition.StateBefore, post[postCount].Transition.StateAfter);
		++postCount;
	}
	post[postCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	post[postCount].Transition = { scratch.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
		D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON };
	++postCount;
	cmdList->ResourceBarrier(postCount, post);
	return true;
}

struct PendingCaptureCopy
{
	ID3D12Resource* destination{};
	ID3D12Resource* source{};
	D3D12_RESOURCE_STATES sourceState{};
};

// Prepares copy texture for source (recreated on desc change) and queues a batched CopyResource instead of
// issuing per-texture ResourceBarrier flushes (cuts 8 barriers down to 2 per real frame across CUDA/XMX/DP4a).
static bool PrepareCapture(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12Resource* source,
	D3D12_RESOURCE_STATES sourceState, ComPtr<ID3D12Resource>& copy, const char* purpose,
	PendingCaptureCopy* pending, UINT& pendingCount, bool wantShared = false,
	bool isDepth = false, ComPtr<ID3D12Resource>* depthScratch = nullptr)
{
	if (!source)
	{
		copy.Reset();
		return false;
	}

	auto desc = source->GetDesc();

	// On Unreal Engine 5 / REDengine 4 DX12 games, depth is typically D32_FLOAT_S8X24 / R32G8X24_TYPELESS.
	// Extract plane 0 into a shared R32_FLOAT texture only when the D3D12->Vulkan hardware WMMA bridge is active.
	if (wantShared && isDepth && depthScratch &&
		desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.SampleDesc.Count == 1 &&
		IsDepth32StencilFormat(desc.Format))
	{
		if (CaptureDepth32Plane0(device, cmdList, source, sourceState, copy, *depthScratch, purpose))
			return true;
	}

	// The copy only ever rests in XeFGInputState and gets read through an SRV, so none of the source's flags are
	// needed. Inheriting them can make CreateCommittedResource reject the desc: ALLOW_CROSS_ADAPTER needs a cross
	// adapter heap, ALLOW_SIMULTANEOUS_ACCESS can't be combined with a depth resource, and ALLOW_UNORDERED_ACCESS
	// can't be used with MSAA. A non committed layout (row major, 64KB swizzle) can be rejected the same way.
	desc.Flags &= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
	desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

	if (isDepth)
		MakeReadableDepth(desc);

	if (copy)
	{
		auto copyDesc = copy->GetDesc();

		// The old one can still be in use by the GPU
		if (copyDesc.Dimension != desc.Dimension || copyDesc.Width != desc.Width || copyDesc.Height != desc.Height ||
			copyDesc.DepthOrArraySize != desc.DepthOrArraySize || copyDesc.MipLevels != desc.MipLevels ||
			copyDesc.Format != desc.Format || copyDesc.SampleDesc.Count != desc.SampleDesc.Count ||
			copyDesc.SampleDesc.Quality != desc.SampleDesc.Quality || copyDesc.Layout != desc.Layout ||
			copyDesc.Flags != desc.Flags)
		{
			D3D12VulkanWmma::ForgetResource(copy.Get());
			DeferredDestroy::Add(std::make_shared<ComPtr<ID3D12Resource>>(std::move(copy)));
		}
	}

	if (!copy)
		copy = CreateTexture(device, desc, XeFGInputState, L"dlssg-to-xefg input copy", purpose, wantShared);

	if (!copy)
		return false;

	pending[pendingCount++] = { copy.Get(), source, sourceState };
	return true;
}

static void FlushCaptureCopies(ID3D12GraphicsCommandList* cmdList, const PendingCaptureCopy* pending, UINT count)
{
	if (!count)
		return;

	D3D12_RESOURCE_BARRIER before[8]{};
	UINT beforeCount = 0;
	for (UINT i = 0; i < count; ++i)
	{
		if (XeFGInputState != D3D12_RESOURCE_STATE_COPY_DEST)
		{
			before[beforeCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			before[beforeCount].Transition = { pending[i].destination, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
				XeFGInputState, D3D12_RESOURCE_STATE_COPY_DEST };
			++beforeCount;
		}
		if (pending[i].sourceState != D3D12_RESOURCE_STATE_COPY_SOURCE)
		{
			// Deduplicate if the game aliases two inputs to the same resource
			bool seen = false;
			for (UINT j = 0; j < i; ++j)
			{
				if (pending[j].source == pending[i].source) { seen = true; break; }
			}
			if (!seen)
			{
				before[beforeCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
				before[beforeCount].Transition = { pending[i].source, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
					pending[i].sourceState, D3D12_RESOURCE_STATE_COPY_SOURCE };
				++beforeCount;
			}
		}
	}
	if (beforeCount)
		cmdList->ResourceBarrier(beforeCount, before);

	for (UINT i = 0; i < count; ++i)
		cmdList->CopyResource(pending[i].destination, pending[i].source);

	if (beforeCount)
	{
		for (UINT i = 0; i < beforeCount; ++i)
			std::swap(before[i].Transition.StateBefore, before[i].Transition.StateAfter);
		cmdList->ResourceBarrier(beforeCount, before);
	}
}

struct SubrectKeys
{
	const char* baseX;
	const char* baseY;
	const char* width;
	const char* height;
};

constexpr SubrectKeys kBackbufferSubrect{
	"DLSSG.BackbufferSubrectBaseX", "DLSSG.BackbufferSubrectBaseY",
	"DLSSG.BackbufferSubrectWidth", "DLSSG.BackbufferSubrectHeight"
};
constexpr SubrectKeys kDepthSubrect{
	"DLSSG.DepthSubrectBaseX", "DLSSG.DepthSubrectBaseY",
	"DLSSG.DepthSubrectWidth", "DLSSG.DepthSubrectHeight"
};
constexpr SubrectKeys kMVecsSubrect{
	"DLSSG.MVecsSubrectBaseX", "DLSSG.MVecsSubrectBaseY",
	"DLSSG.MVecsSubrectWidth", "DLSSG.MVecsSubrectHeight"
};
constexpr SubrectKeys kHudlessSubrect{
	"DLSSG.HUDLessSubrectBaseX", "DLSSG.HUDLessSubrectBaseY",
	"DLSSG.HUDLessSubrectWidth", "DLSSG.HUDLessSubrectHeight"
};

static xefgd_resource_t Subrect(const NGXParameter& params, const SubrectKeys& keys, ID3D12Resource* resource)
{
	xefgd_resource_t subrect{};
	const auto desc = resource->GetDesc();
	const uint32_t resW = static_cast<uint32_t>(desc.Width);
	const uint32_t resH = desc.Height;
	subrect.base = { params.GetOr<uint32_t>(keys.baseX, 0), params.GetOr<uint32_t>(keys.baseY, 0) };
	subrect.size = { params.GetOr<uint32_t>(keys.width, 0), params.GetOr<uint32_t>(keys.height, 0) };

	if (subrect.size.x == 0 || subrect.size.y == 0 || subrect.base.x >= resW || subrect.base.y >= resH)
	{
		subrect.base = {};
		subrect.size = { resW, resH };
	}
	else
	{
		subrect.size.x = std::min(subrect.size.x, resW - subrect.base.x);
		subrect.size.y = std::min(subrect.size.y, resH - subrect.base.y);
	}

	return subrect;
}

static bool IsFiniteVec3(const XMFLOAT3& v) noexcept
{
	return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static XMFLOAT3 GetVector(const NGXParameter& params, const char* keyX, const char* keyY, const char* keyZ)
{
	const XMFLOAT3 v{ params.GetOr(keyX, 0.0f), params.GetOr(keyY, 0.0f), params.GetOr(keyZ, 0.0f) };
	return IsFiniteVec3(v) ? v : XMFLOAT3{ 0.0f, 0.0f, 0.0f };
}

// XeFG's scene change detection uses the forward axis
static void GetViewMatrix(const NGXParameter& params, float* out)
{
	const XMFLOAT3 pos = GetVector(params, "DLSSG.CameraPosX", "DLSSG.CameraPosY", "DLSSG.CameraPosZ");
	const XMFLOAT3 right = GetVector(params, "DLSSG.CameraRightX", "DLSSG.CameraRightY", "DLSSG.CameraRightZ");
	const XMFLOAT3 up = GetVector(params, "DLSSG.CameraUpX", "DLSSG.CameraUpY", "DLSSG.CameraUpZ");
	const XMFLOAT3 forward = GetVector(params, "DLSSG.CameraFwdX", "DLSSG.CameraFwdY", "DLSSG.CameraFwdZ");

	// All zero (or sanitized non-finite) disables XeFG's scene change detection
	if (forward.x == 0.0f && forward.y == 0.0f && forward.z == 0.0f)
		return;

	XMVECTOR p = XMLoadFloat3(&pos);
	XMMATRIX view = { XMVectorSet(right.x, up.x, forward.x, 0.0f), XMVectorSet(right.y, up.y, forward.y, 0.0f),
					  XMVectorSet(right.z, up.z, forward.z, 0.0f),
					  XMVectorSet(-XMVectorGetX(XMVector3Dot(p, XMLoadFloat3(&right))),
								  -XMVectorGetX(XMVector3Dot(p, XMLoadFloat3(&up))),
								  -XMVectorGetX(XMVector3Dot(p, XMLoadFloat3(&forward))), 1.0f) };

	memcpy(out, view.r, sizeof(view));
}

// Near/far/fov from a perspective view to clip matrix, dlssg-to-fsr3's approach. Near and far come out in depth
// buffer order
static bool CameraFromViewToClip(const NGXParameter& params, float& cameraNear, float& cameraFar, float& fov)
{
	float(*viewToClip)[4] = nullptr;
	params.Get("DLSSG.CameraViewToClip", reinterpret_cast<void**>(&viewToClip));

	if (params.GetOr<uint32_t>("DLSSG.OrthoProjection", 0) || !viewToClip)
		return false;

	// Some games (RTX Remix ones) pass zero or identity
	float m[4][4] = {};
	if (memcmp(viewToClip, m, sizeof(m)) == 0)
		return false;

	m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f;
	if (memcmp(viewToClip, m, sizeof(m)) == 0)
		return false;

	const double b = viewToClip[1][1];
	const double c = viewToClip[2][2];
	const double d = viewToClip[3][2];
	const double e = viewToClip[2][3];

	if (e < 0.0)
	{
		cameraNear = (float)(c == 0.0 ? 0.0 : d / c);
		cameraFar = (float)(d / (c + 1.0));
	}
	else
	{
		cameraNear = (float)(c == 0.0 ? 0.0 : -d / c);
		cameraFar = (float)(-d / (c - 1.0));
	}

	fov = (float)(2.0 * std::atan(1.0 / b));

	return std::isfinite(cameraNear) && std::isfinite(cameraFar) && std::isfinite(fov) && cameraNear >= 0.0f && cameraFar >= 0.0f && fov > 0.0f;
}

// Unreal's executables live in Binaries\<platform> and end with -Shipping
static bool IsUnrealEngine()
{
	static const bool isUnreal = []
		{
			wchar_t path[MAX_PATH]{};
			GetModuleFileNameW(nullptr, path, MAX_PATH);
			_wcslwr_s(path);

			std::wstring_view exe = path;
			return exe.find(L"\\binaries\\win64\\") != exe.npos || exe.find(L"\\binaries\\wingdk\\") != exe.npos ||
				exe.ends_with(L"-shipping.exe");
		}();

	return isUnreal;
}

// Left handed perspective matrix (row vectors) matching the depth buffer: inverted depth swaps near and far,
// cameraFar == 0 means an infinite far plane
static void GetProjectionMatrix(const NGXParameter& params, bool depthInverted, float aspectRatio, float* out)
{
	float cameraNear = 0.0f;
	float cameraFar = 0.0f;
	float fov = 0.0f;

	// Unreal doesn't pass a correct CameraViewToClip, its separate camera values are fine
	if (!IsUnrealEngine() && CameraFromViewToClip(params, cameraNear, cameraFar, fov))
	{
		if (depthInverted)
			std::swap(cameraNear, cameraFar);
	}
	else
	{
		cameraNear = params.GetOr("DLSSG.CameraNear", 0.0f);
		cameraFar = params.GetOr("DLSSG.CameraFar", 0.0f);
		fov = params.GetOr("DLSSG.CameraFOV", 0.0f);

		// Some games pass degrees, some radians
		if (fov > 10.0f)
			fov *= std::numbers::pi_v<float> / 180.0f;
	}

	if (float paramAspectRatio = params.GetOr("DLSSG.CameraAspectRatio", 0.0f); paramAspectRatio > 0.0f)
		aspectRatio = paramAspectRatio;

	if (cameraNear <= 0.0f || cameraFar < 0.0f || fov <= 0.0f || aspectRatio <= 0.0f)
		return;

	const bool infinite = cameraFar == 0.0f;
	const float h = 1.0f / std::tan(fov * 0.5f);
	const float w = h / aspectRatio;

	// z_ndc = (a * z + b) / z
	float a;
	float b;

	if (infinite)
	{
		a = depthInverted ? 0.0f : 1.0f;
		b = depthInverted ? cameraNear : -cameraNear;
	}
	else
	{
		if (XMScalarNearEqual(cameraNear, cameraFar, 0.00001f))
			cameraFar++;

		const float range = cameraFar - cameraNear;
		a = depthInverted ? -cameraNear / range : cameraFar / range;
		b = depthInverted ? cameraNear * cameraFar / range : -cameraNear * cameraFar / range;
	}

	const float projection[16] = { w, 0, 0, 0, 0, h, 0, 0, 0, 0, a, 1, 0, 0, b, 0 };
	memcpy(out, projection, sizeof(projection));
}

// The ported XMX MFG model is XeFG's XMX network. Without the program set XeFG keeps its default DP4a model.
static bool PortAvailable()
{
	static const bool available = []
		{
			const bool installed = MfgKernels::Install(XeFG::Module());
			Log::Info("Ported XMX MFG model available: {}", installed);
			return installed;
		}();

	return available;
}

// XeFG's own XMX check plus real physical adapter verification (bypassing DXGI spoofing):
// XMX (native) is strictly restricted to real Intel GPUs (vendor 0x8086) whose driver supports XMX.
static int32_t NativeModel(ID3D12Device* device)
{
	static const int32_t model = [device]
		{
			const auto probe = device ? adapter_identity::Probe(device->GetAdapterLuid()) : adapter_identity::ProbeResult{};
			const bool isRealIntel = (probe.vendorId == 0x8086);
			const int32_t picked = XeFG::AutoModel(device);
			const bool nativeXmxAllowed = isRealIntel && (picked == XeFGXmxModel);

			Plugin::xmxSupport = nativeXmxAllowed ? Plugin::XmxSupport::Supported : Plugin::XmxSupport::Unsupported;
			if (picked == XeFGXmxModel && !isRealIntel)
			{
				Log::Warn("XeFG reported XMX model {} on non-Intel GPU (vendor 0x{:04X}, {}); restricting XMX (native) to Intel GPUs with XMX",
					picked, probe.vendorId, adapter_identity::WideToAsciiJson(probe.description));
			}
			Log::Info("XeFG model: {} (real vendor 0x{:04X}, native XMX: {})",
				picked, probe.vendorId, nativeXmxAllowed ? "supported" : "unsupported");
			return nativeXmxAllowed ? XeFGXmxModel : MfgKernels::Dp4aModel;
		}();

	return model;
}

// Probes the GPU once, as early as a device is available: XeFG's own XMX network first, then the ported model and its
// CUDA tensor-core and AMD WMMA INT8 routes. The menu uses the flags to grey out paths that would fail, and the first
// context no longer runs these checks. A later different device keeps the first device's answer.
void FrameGenerator::ProbeSupport(ID3D12Device* device)
{
	if (!device)
		return;

	static std::once_flag once;
	std::call_once(once, [device]
		{
			const int32_t native = NativeModel(device);

			// XeFG's own XMX network works here; the port and its CUDA/WMMA routes are only needed when it doesn't.
			if (native == XeFGXmxModel)
			{
				Plugin::cudaSupport = Plugin::CudaSupport::Unsupported;
				Plugin::wmmaSupport = Plugin::WmmaSupport::Unsupported;
				Log::Info("XeFG native XMX network available, CUDA/WMMA routes not applicable");
				return;
			}

			// Publishing the ported model patches libxess_fg.dll process wide. It's installed even when another path
			// is selected, so the menu knows whether CUDA/WMMA are available and switching paths needs no new context.
			const bool mfgModel = MfgKernels::Install(XeFG::Module());
			Plugin::cudaSupport = Plugin::CudaSupport::Unsupported;
			Plugin::wmmaSupport = Plugin::WmmaSupport::Unsupported;

			if (!mfgModel)
			{
				Log::Info("Hardware matrix routes skipped: the ported MFG model is not available");
				return;
			}

			static std::once_flag routeLogOnce;
			std::call_once(routeLogOnce, [] {
				xefg_cuda::SetLog(CudaRouteLog);
				xefg_wmma::SetLog(CudaRouteLog);
			});

			wchar_t tensorCores[16]{};
			const DWORD tensorCoresLength = GetEnvironmentVariableW(L"XEFG_TENSOR_CORES", tensorCores, 16);
			const bool tensorOff = tensorCoresLength && tensorCoresLength < 16 && (tensorCores[0] == L'0'
				|| _wcsicmp(tensorCores, L"off") == 0 || _wcsicmp(tensorCores, L"false") == 0);

			if (tensorOff)
			{
				Log::Info("Tensor-core route disabled by XEFG_TENSOR_CORES");
			}
			else
			{
				const auto fatbin = LoadCudaFatbin(MfgKernels::ProgramsDirectory() / L"xefg_cuda_layers.fatbin");
				std::string reason;
				if (xefg_cuda::Install(device, XeFG::Module(), fatbin, reason))
				{
					Plugin::cudaSupport = Plugin::CudaSupport::Supported;
					// The route's hooks stay installed, but only a context that selects CUDA turns Dispatch replacement on
					xefg_cuda::SetEnabled(false);
				}
				else
				{
					Log::Info("Tensor-core route inactive: {}", reason);
				}
			}

			std::string wmmaReason;
			if (xefg_wmma::Install(device, XeFG::Module(), MfgKernels::ProgramsDirectory() / L"wmma", wmmaReason))
			{
				Plugin::wmmaSupport = Plugin::WmmaSupport::Supported;
				xefg_wmma::SetEnabled(false);
				const auto st = xefg_wmma::Snapshot();
				const char* archName = st.arch == xefg_wmma::Arch::Rdna4 ? "RDNA 4 (gfx12)" : "RDNA 3 (gfx11)";
				if (st.dxgiSpoofed)
				{
					Log::Info("WMMA route ready: {} ({}, device 0x{:04X}, WaveMMA tier {}, {}/{} shader overrides, source: {}, bypassed DXGI spoof 0x{:04X}:0x{:04X})",
						st.gpuName.empty() ? "AMD GPU" : st.gpuName, archName, st.deviceId, st.waveMmaTier,
						st.overrideLayers, st.layers, st.probeSource, st.dxgiVendorId, st.dxgiDeviceId);
				}
				else
				{
					Log::Info("WMMA route ready: {} ({}, device 0x{:04X}, WaveMMA tier {}, {}/{} shader overrides, source: {})",
						st.gpuName.empty() ? "AMD GPU" : st.gpuName, archName, st.deviceId, st.waveMmaTier,
						st.overrideLayers, st.layers, st.probeSource);
				}
				std::string vkWmmaReason;
				if (D3D12VulkanWmma::Install(device, vkWmmaReason))
				{
					const auto vkSt = D3D12VulkanWmma::Snapshot();
					Log::Info("WMMA hardware bridge active: D3D12 -> Vulkan VK_KHR_cooperative_matrix ({}x{}x{} s8->s32, {} hardware WMMA shaders)",
						vkSt.tileM, vkSt.tileN, vkSt.tileK, vkSt.coopLayers);
				}
				else
				{
					Log::Info("WMMA D3D12->Vulkan hardware bridge unavailable ({}); falling back to D3D12 cs_6_6 overrides",
						vkWmmaReason.empty() ? "unknown" : vkWmmaReason);
				}
				if (st.overrideLayers == 0 && !D3D12VulkanWmma::Active())
				{
					Log::Warn("WMMA route active, but 0/{} compiled .dxbc shaders were found in {}; layers will run on XMX DP4a until wmma/*.dxbc shaders are installed",
						st.layers, (MfgKernels::ProgramsDirectory() / L"wmma").string());
				}
			}
			else
			{
				const auto st = xefg_wmma::Snapshot();
				Log::Info("WMMA route inactive: {} (real GPU: {}, vendor 0x{:04X}, device 0x{:04X}, source: {}, DXGI spoofed: {}, WaveMMA tier: {})",
					wmmaReason, st.gpuName.empty() ? "unknown" : st.gpuName, st.vendorId, st.deviceId,
					st.probeSource.empty() ? "none" : st.probeSource, st.dxgiSpoofed ? "true" : "false", st.waveMmaTier);
			}
		});
}

static int32_t PickModel(ID3D12Device* device)
{
	const int32_t native = NativeModel(device);

	switch (Plugin::options.path.load())
	{
	case Plugin::Path::Fallback:
		return MfgKernels::Dp4aModel;
	case Plugin::Path::XmxNative:
		if (native == XeFGXmxModel)
			return XeFGXmxModel;

		// Unsupported, the CUDA / WMMA / ported defaults below
		break;
	case Plugin::Path::Cuda:
	case Plugin::Path::Wmma:
	case Plugin::Path::XmxPort:
		break;
	}

	return PortAvailable() ? MfgKernels::MfgModel : MfgKernels::Dp4aModel;
}

// The ported model runs on the CUDA tensor-core route when that's the selected path, otherwise on WMMA or DXIL. XeFG's own
// XMX network doesn't need the port at all.
static bool UseCuda(int32_t model)
{
	if (model != MfgKernels::MfgModel)
		return false;

	const auto path = Plugin::options.path.load();
	return path == Plugin::Path::Cuda || (path == Plugin::Path::XmxNative && Plugin::cudaSupport.load() == Plugin::CudaSupport::Supported);
}

// The ported model runs on the AMD WMMA INT8 route only when WMMA is explicitly selected.
// XMX (native) is strictly reserved for Intel GPUs with XMX support and falls back to XMX Port (or CUDA) elsewhere.
static bool UseWmma(int32_t model)
{
	if (model != MfgKernels::MfgModel || !xefg_wmma::Supported())
		return false;

	return Plugin::options.path.load() == Plugin::Path::Wmma;
}

static Plugin::Network CurrentNetwork(int32_t model)
{
	if (model == XeFGXmxModel)
		return Plugin::Network::XmxNative;

	if (model != MfgKernels::MfgModel)
		return Plugin::Network::Fallback;

	if (xefg_cuda::Enabled())
		return Plugin::Network::Cuda;

	if (xefg_wmma::Enabled())
		return Plugin::Network::Wmma;

	return Plugin::Network::XmxPort;
}

bool FrameGenerator::CreateContext()
{
	// Usually already done at NGX Init. Kept as a fallback when the game never called Init with a device.
	ProbeSupport(device.Get());

	auto newContext = std::make_shared<Context>();

	// Publishing the ported model patches libxess_fg.dll process wide, so it runs before the first context. The probe
	// already installed it when XeFG's own XMX network isn't available; this also covers a port path chosen later.
	const bool mfgModel = settings.model != XeFGXmxModel && MfgKernels::Install(XeFG::Module());

	if (!mfgModel && settings.model != XeFGXmxModel)
	{
		// The route runs the ported model, so without the port there's nothing for it to launch
		Log::Info("Hardware matrix route skipped: the ported MFG model is not available");
	}

	xefg_cuda::SetEnabled(UseCuda(settings.model));
	xefg_wmma::SetEnabled(UseWmma(settings.model));

	// Process wide, they also affect other XeFG contexts. Must happen before the context picks its model
	XeFG::SetMaxInterpolatedFrames(MaxFrames);
	XeFG::OverrideModel(settings.model);

	if (auto result = XeFG::CreateContext(device.Get(), &newContext->handle); result != XEFGD_RESULT_SUCCESS)
	{
		Log::Error("XeFG CreateContext error: {}", result);
		return false;
	}

	XeFG::SetLoggingCallback(newContext->handle, XEFGD_LOGGING_LEVEL_INFO, XeFGLogCallback);

	xefgd_init_params_t initParams{};
	initParams.outputResolution = { settings.width, settings.height };
	initParams.backBufferFormat = settings.format;
	initParams.initFlags = settings.initFlags;
	initParams.maxInterpolatedFrames = MaxFrames;
	initParams.uiMode = settings.uiMode;

	Log::Info("Creating XeFG context: {}x{}, format: {}, flags: {:X}, ui mode: {}, model: {}", settings.width,
		settings.height, (uint32_t)settings.format, settings.initFlags, settings.uiMode, settings.model);

	if (auto result = XeFG::Init(newContext->handle, initParams); result != XEFGD_RESULT_SUCCESS)
	{
		Log::Error("XeFG Init error: {}", result);
		return false;
	}

	newContext->sharedTexturesMode = UseWmma(settings.model) && D3D12VulkanWmma::WantsSharedTextures();
	context = std::move(newContext);

	// Warm up the Vulkan WMMA bridge context asynchronously on a background thread so CreateContext
	// returns immediately after D3D12 XeFG::Init and never stalls Streamline's 500ms waitCPUFence deadline
	// (first 1-2 frames run seamlessly on D3D12 cs_6_6 overrides until Vulkan context init completes).
	if (UseWmma(settings.model) && D3D12VulkanWmma::Active())
	{
		const D3D12VulkanWmma::ContextSettings vkSettings{
			settings.width, settings.height, settings.format, settings.initFlags, settings.uiMode, settings.model
		};
		D3D12VulkanWmma::WarmupContextAsync(vkSettings);
	}

	routeLaunches = UseWmma(settings.model) ? xefg_wmma::Snapshot().launches : xefg_cuda::Snapshot().launches;
	routeFrames = 0;
	routeReported = false;
	return true;
}

bool FrameGenerator::PollVramGuard() noexcept
{
	if (!Plugin::options.vramGuard.load(std::memory_order_relaxed))
	{
		if (vramGuardActive)
		{
			vramGuardActive = false;
			Log::Info("Adaptive VRAM guard disabled by user, restoring full MFG multiplier");
		}
		return false;
	}

	if (!vramAdapterChecked)
	{
		vramAdapterChecked = true;
		ComPtr<IDXGIFactory4> factory;
		if (device && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
		{
			const LUID luid = device->GetAdapterLuid();
			ComPtr<IDXGIAdapter1> adapter;
			if (SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) && adapter)
				adapter.As(&vramAdapter);
		}
	}

	if (!vramAdapter)
		return false;

	// Poll every 8 real frames to keep QueryVideoMemoryInfo overhead negligible
	if ((vramPollCounter++ & 7u) != 0)
		return vramGuardActive;

	DXGI_QUERY_VIDEO_MEMORY_INFO memInfo{};
	if (FAILED(vramAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memInfo)) || memInfo.Budget == 0)
		return vramGuardActive;

	constexpr uint64_t kSafetyMarginBytes = 200ull * 1024ull * 1024ull; // 200 MiB safety margin
	constexpr uint64_t kReleaseMarginBytes = 400ull * 1024ull * 1024ull; // 400 MiB hysteresis to avoid oscillation

	if (!vramGuardActive)
	{
		if (memInfo.Budget > kSafetyMarginBytes && memInfo.CurrentUsage + kSafetyMarginBytes >= memInfo.Budget)
		{
			vramGuardActive = true;
			Log::Warn("Adaptive VRAM guard engaged ({} MiB / {} MiB budget): clamping MFG temporarily to 2x to prevent swap stutter",
				memInfo.CurrentUsage / (1024ull * 1024ull), memInfo.Budget / (1024ull * 1024ull));
		}
	}
	else
	{
		if (memInfo.Budget > kReleaseMarginBytes && memInfo.CurrentUsage + kReleaseMarginBytes < memInfo.Budget)
		{
			vramGuardActive = false;
			Log::Info("Adaptive VRAM guard released ({} MiB / {} MiB budget): restoring full MFG multiplier",
				memInfo.CurrentUsage / (1024ull * 1024ull), memInfo.Budget / (1024ull * 1024ull));
		}
	}

	return vramGuardActive;
}

void FrameGenerator::CaptureRealFrame(ID3D12GraphicsCommandList* cmdList, const NGXParameter& params)
{
	static bool warnedCameraMotion = false;
	if (params.GetOr<uint32_t>("DLSSG.CameraMotionIncluded", 1) == 0 && !std::exchange(warnedCameraMotion, true))
		Log::Warn("DLSSG.CameraMotionIncluded is 0, the camera's motion is missing from the motion vectors");

	auto backbuffer = params.GetOr<ID3D12Resource*>("DLSSG.Backbuffer", nullptr);
	auto hudless = Plugin::options.hudlessUi ? params.GetOr<ID3D12Resource*>("DLSSG.HUDLess", nullptr) : nullptr;
	auto depth = params.GetOr<ID3D12Resource*>("DLSSG.Depth", nullptr);
	auto mvecs = params.GetOr<ID3D12Resource*>("DLSSG.MVecs", nullptr);
	auto output = params.GetOr<ID3D12Resource*>("DLSSG.OutputInterpolated", nullptr);

	const bool depthInverted = XeFgEnabler::ApplyTriStateOverride(
		Plugin::options.depthInvertedOverride.load(std::memory_order_relaxed),
		params.GetOr<uint32_t>("DLSSG.DepthInverted", 0) != 0);
	const bool mvecJittered = XeFgEnabler::ApplyTriStateOverride(
		Plugin::options.jitteredMvOverride.load(std::memory_order_relaxed),
		params.GetOr<uint32_t>("DLSSG.MvecJittered", 0) != 0);

	const auto bbDesc = backbuffer->GetDesc();
	const auto backbufferFormat = TypedFormat(bbDesc.Format);
	auto hudlessFormat = hudless ? TypedFormat(hudless->GetDesc().Format) : DXGI_FORMAT_UNKNOWN;

	// Guard against mismatched HUD-less textures (different size, format, or MSAA) which cause XeFG errors or HDR glitches.
	if (hudless)
	{
		const auto hlDesc = hudless->GetDesc();
		if (hlDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || hlDesc.SampleDesc.Count > 1 ||
			hlDesc.Width != bbDesc.Width || hlDesc.Height != bbDesc.Height || hudlessFormat != backbufferFormat)
		{
			static bool warnedHudlessMismatch = false;
			if (!std::exchange(warnedHudlessMismatch, true))
				Log::Warn("Ignoring HUD-less input due to mismatch with back buffer (HUD-less {}x{} {}, back buffer {}x{} {}); falling back to UI Auto",
					static_cast<uint64_t>(hlDesc.Width), hlDesc.Height, FormatName(hudlessFormat),
					static_cast<uint64_t>(bbDesc.Width), bbDesc.Height, FormatName(backbufferFormat));
			hudless = nullptr;
			hudlessFormat = DXGI_FORMAT_UNKNOWN;
		}
	}

	Settings wanted;
	auto outputDesc = output->GetDesc();
	wanted.width = (uint32_t)outputDesc.Width;
	wanted.height = outputDesc.Height;
	wanted.format = TypedFormat(outputDesc.Format);
	wanted.initFlags = (depthInverted ? XEFGD_INIT_FLAG_INVERTED_DEPTH : 0) |
		(mvecJittered ? XEFGD_INIT_FLAG_JITTERED_MV : 0);
	wanted.uiMode = hudless ? XEFGD_UI_MODE_BACKBUFFER_HUDLESS : XEFGD_UI_MODE_AUTO;
	wanted.model = PickModel(device.Get());

	// The menu path can change without a context change: XMX Port, CUDA, and WMMA all run the ported model (16), only the
	// route that executes its programs differs. Log the switch, and say whether it took effect below
	const auto path = Plugin::options.path.load();
	const bool pathChanged = (int32_t)path != loggedPath;

	if (pathChanged)
	{
		loggedPath = (int32_t)path;
		Log::Info("Network path: {}, model {}", Plugin::PathName(path), wanted.model);

		// A switch can retry a context that failed to start
		createFailed = false;
	}

	if (wanted != settings)
	{
		// Logged whenever the formats change, they decide whether XeFG's HDR path can work at all: HDR10 is
		// R10G10B10A2_UNORM, scRGB is FP16 and unsupported, and the HDR inputs must all share one format
		const auto describe = [](const char* name, DXGI_FORMAT format)
			{
				if (format == DXGI_FORMAT_UNKNOWN)
					return std::format("{}: none", name);

				return std::format("{}: {} ({}, {})", name, FormatName(format), (uint32_t)format, HdrDescription(format));
			};

		Log::Info("HDR formats, {}; {}; {}", describe("back buffer", backbufferFormat), describe("HUD-less", hudlessFormat),
			describe("output", wanted.format));

		if (backbufferFormat == DXGI_FORMAT_R16G16B16A16_FLOAT)
			Log::Warn("FP16 (R16G16B16A16_FLOAT) is scRGB HDR, which XeFG doesn't support. Use HDR10 "
				"(R10G10B10A2_UNORM) or turn HDR off, otherwise frame generation can fail or show wrong colors");
		else if (hudlessFormat != DXGI_FORMAT_UNKNOWN && hudlessFormat != backbufferFormat)
			Log::Warn("HUD-less color is {}, but the back buffer is {}, XeFG's HDR path needs every input in the same "
				"HDR10 format", FormatName(hudlessFormat), FormatName(backbufferFormat));

		settings = wanted;
		createFailed = false;

		if (context)
			DeferredDestroy::Add(std::move(context));
	}

	if (!IsSupportedXeFGColorFormat(settings.format) || !IsSupportedXeFGColorFormat(backbufferFormat) ||
		settings.width == 0 || settings.height == 0)
	{
		createFailed = true;
		return;
	}

	if (!context && !createFailed)
		createFailed = !CreateContext();

	if (!context)
		return;

	xefg_cuda::SetEnabled(UseCuda(settings.model));
	xefg_wmma::SetEnabled(UseWmma(settings.model));
	const bool debugLines = Plugin::options.debugLines.load(std::memory_order_relaxed);
	static std::atomic<int> sLastDebugLines{ -1 };
	if (sLastDebugLines.exchange(debugLines ? 1 : 0, std::memory_order_relaxed) != (debugLines ? 1 : 0))
		XeFG::SetDebugLines(debugLines);

	// After CreateContext: before it, the route has not been installed yet
	if (pathChanged && path == Plugin::Path::Cuda)
	{
		const auto route = xefg_cuda::Snapshot();

		if (route.state != xefg_cuda::State::Active)
			Log::Warn("CUDA path selected, but the tensor-core route is unavailable ({}), the DXIL port runs instead",
				route.reason.empty() ? std::string("not installed") : route.reason);
	}
	else if (pathChanged && path == Plugin::Path::Wmma)
	{
		const auto route = xefg_wmma::Snapshot();

		if (route.state != xefg_wmma::State::Active)
		{
			Log::Warn("WMMA path selected, but the WMMA INT8 route is unavailable ({}; GPU: {}, vendor 0x{:04X}, device 0x{:04X}, source: {}, DXGI spoofed: {}), the XMX DP4a INT8 port runs instead",
				route.reason.empty() ? std::string("not installed") : route.reason,
				route.gpuName.empty() ? std::string("unknown") : route.gpuName,
				route.vendorId, route.deviceId,
				route.probeSource.empty() ? std::string("none") : route.probeSource,
				route.dxgiSpoofed ? "true" : "false");
		}
		else if (route.overrideLayers == 0)
		{
			Log::Warn("WMMA path selected on {} (0x{:04X}), but 0/{} WMMA .dxbc overrides are loaded; layers fall back to XMX DP4a INT8",
				route.gpuName.empty() ? std::string("AMD GPU") : route.gpuName, route.deviceId, route.layers);
		}
	}
	else if (pathChanged && !UseWmma(settings.model))
	{
		const auto route = xefg_wmma::Snapshot();
		if (route.vendorId == 0x1002)
		{
			if (route.state != xefg_wmma::State::Active)
			{
				Log::Info("WMMA not selected on AMD GPU {} (0x{:04X}): route unavailable ({}, WaveMMA tier {}, source: {})",
					route.gpuName.empty() ? std::string("unknown") : route.gpuName, route.deviceId,
					route.reason.empty() ? std::string("not installed") : route.reason,
					route.waveMmaTier, route.probeSource.empty() ? std::string("none") : route.probeSource);
			}
			else if (settings.model != MfgKernels::MfgModel)
			{
				Log::Info("WMMA not selected on AMD GPU {} (0x{:04X}): active model is {} (requires ported model {})",
					route.gpuName.empty() ? std::string("unknown") : route.gpuName, route.deviceId,
					settings.model, MfgKernels::MfgModel);
			}
			else
			{
				Log::Info("WMMA available on AMD GPU {} (0x{:04X}) but not active because Network path is '{}' (switch to WMMA to enable)",
					route.gpuName.empty() ? std::string("unknown") : route.gpuName, route.deviceId, Plugin::PathName(path));
			}
		}
	}

	if (pathChanged && (UseCuda(settings.model) || UseWmma(settings.model)))
	{
		routeLaunches = UseWmma(settings.model) ? xefg_wmma::Snapshot().launches : xefg_cuda::Snapshot().launches;
		routeFrames = 0;
		routeReported = false;
	}

	const bool wantShared = UseWmma(settings.model) && D3D12VulkanWmma::WantsSharedTextures();
	if (context->sharedTexturesMode != wantShared)
	{
		auto releaseTex = [](ComPtr<ID3D12Resource>& res) {
			if (res)
			{
				D3D12VulkanWmma::ForgetResource(res.Get());
				DeferredDestroy::Add(std::make_shared<ComPtr<ID3D12Resource>>(std::move(res)));
			}
		};
		for (auto& out : context->outputs)
			releaseTex(out);
		for (auto& f : context->frames)
		{
			releaseTex(f.backbuffer);
			releaseTex(f.hudless);
			releaseTex(f.depth);
			releaseTex(f.depthScratch);
			releaseTex(f.mvecs);
			f.valid = false;
		}
		context->pairValid = false;
		context->batchGeneratedThrough = 0;
		context->sharedTexturesMode = wantShared;
	}

	context->current ^= 1;
	auto& frame = context->frames[context->current];
	auto& previous = context->frames[context->current ^ 1];

	PendingCaptureCopy pendingCopies[4]{};
	UINT pendingCopyCount = 0;
	bool captured = PrepareCapture(device.Get(), cmdList, backbuffer, BackbufferState, frame.backbuffer, "Backbuffer", pendingCopies, pendingCopyCount, wantShared);
	captured &= PrepareCapture(device.Get(), cmdList, depth, InputState, frame.depth, "Depth copy", pendingCopies, pendingCopyCount, wantShared, true, &frame.depthScratch);
	captured &= PrepareCapture(device.Get(), cmdList, mvecs, InputState, frame.mvecs, "MVecs copy", pendingCopies, pendingCopyCount, wantShared);
	PrepareCapture(device.Get(), cmdList, hudless, InputState, frame.hudless, "HUDLess copy", pendingCopies, pendingCopyCount, wantShared);
	FlushCaptureCopies(cmdList, pendingCopies, pendingCopyCount);

	auto& data = frame.data;
	data = {};

	if (captured)
	{
		data.backBuffer = Subrect(params, kBackbufferSubrect, backbuffer);
		data.depth = Subrect(params, kDepthSubrect, depth);
		data.motionVectors = Subrect(params, kMVecsSubrect, mvecs);

		if (hudless)
			data.hudlessColor = Subrect(params, kHudlessSubrect, hudless);
	}

	// The outputs are full size, XeFG writes its region at the back buffer's offset to keep letterboxing aligned
	std::ranges::fill(context->outputBases, data.backBuffer.base);

	GetViewMatrix(params, data.viewMatrix);
	GetProjectionMatrix(params, depthInverted, (float)settings.width / (float)settings.height,
		data.projectionMatrix);

	data.outputResolution[0] = (float)settings.width;
	data.outputResolution[1] = (float)settings.height;
	const float jx = params.GetOr("DLSSG.JitterOffsetX", 0.0f);
	const float jy = params.GetOr("DLSSG.JitterOffsetY", 0.0f);
	data.jitterOffset[0] = std::isfinite(jx) ? jx : 0.0f;
	data.jitterOffset[1] = std::isfinite(jy) ? jy : 0.0f;

	// Unlike Streamline's constant, the NGX parameter already converts to pixels
	const float mvsx = params.GetOr("DLSSG.MvecScaleX", 1.0f);
	const float mvsy = params.GetOr("DLSSG.MvecScaleY", 1.0f);
	data.motionVectorScale[0] = (std::isfinite(mvsx) && mvsx != 0.0f) ? mvsx : 1.0f;
	data.motionVectorScale[1] = (std::isfinite(mvsy) && mvsy != 0.0f) ? mvsy : 1.0f;

	const bool ignoreGameReset = Plugin::options.ignoreGameReset.load(std::memory_order_relaxed);
	bool reset = !ignoreGameReset && (params.GetOr<uint32_t>("DLSSG.Reset", 0) != 0);

	// Dynamic Resolution / Subrect Change Detector: reset XeFG history if active render/viewport rect changed
	if (captured && previous.valid && !reset)
	{
		const auto same2d = [](const xefgd_2d_t& a, const xefgd_2d_t& b) noexcept {
			return a.x == b.x && a.y == b.y;
		};
		if (!same2d(data.backBuffer.base, previous.data.backBuffer.base) ||
			!same2d(data.backBuffer.size, previous.data.backBuffer.size) ||
			!same2d(data.depth.size, previous.data.depth.size) ||
			!same2d(data.motionVectors.size, previous.data.motionVectors.size))
		{
			reset = true;
		}
	}

	// Camera Cut & Teleport Detector + GhostBuster anti-ghosting
	const bool camCutOn = Plugin::options.cameraCutDetection.load(std::memory_order_relaxed);
	const bool ghostBusterOn = Plugin::options.ghostBuster.load(std::memory_order_relaxed);
	if (camCutOn || ghostBusterOn)
	{
		const XMFLOAT3 curPos = GetVector(params, "DLSSG.CameraPosX", "DLSSG.CameraPosY", "DLSSG.CameraPosZ");
		const XMFLOAT3 curFwd = GetVector(params, "DLSSG.CameraFwdX", "DLSSG.CameraFwdY", "DLSSG.CameraFwdZ");
		const float fwdLenSq = curFwd.x * curFwd.x + curFwd.y * curFwd.y + curFwd.z * curFwd.z;
		const bool hasCamera = fwdLenSq > 1e-6f;
		float curFov = params.GetOr("DLSSG.CameraFOV", 0.0f);
		if (!std::isfinite(curFov))
			curFov = 0.0f;
		if (curFov > 10.0f)
			curFov *= std::numbers::pi_v<float> / 180.0f;

		float distSq = 0.0f;
		float fwdDot = 1.0f;
		if (hasCamera && context->hasPrevCamera)
		{
			const float dx = curPos.x - context->prevCameraPos[0];
			const float dy = curPos.y - context->prevCameraPos[1];
			const float dz = curPos.z - context->prevCameraPos[2];
			distSq = dx * dx + dy * dy + dz * dz;

			const float invCur = 1.0f / std::sqrt(fwdLenSq);
			const float prevLenSq = context->prevCameraFwd[0] * context->prevCameraFwd[0] +
				context->prevCameraFwd[1] * context->prevCameraFwd[1] +
				context->prevCameraFwd[2] * context->prevCameraFwd[2];
			fwdDot = (prevLenSq > 1e-6f)
				? ((curFwd.x * context->prevCameraFwd[0] + curFwd.y * context->prevCameraFwd[1] +
				    curFwd.z * context->prevCameraFwd[2]) * invCur / std::sqrt(prevLenSq))
				: 1.0f;

			constexpr float kCutThresholdSq = 4.5f * 4.5f;
			if (camCutOn && !reset && (distSq > kCutThresholdSq || fwdDot < -0.5f))
				reset = true;
		}

		if (ghostBusterOn && !reset)
		{
			const bool camMvIncluded = params.GetOr<uint32_t>("DLSSG.CameraMotionIncluded", 1) != 0;
			if (XeFgEnabler::ShouldGhostBusterReset(true, hasCamera, context->hasPrevCamera, fwdDot,
				curFov, context->prevCameraFov, context->hasPrevFov, camMvIncluded, distSq))
			{
				reset = true;
			}
		}

		if (hasCamera)
		{
			context->prevCameraPos[0] = curPos.x;
			context->prevCameraPos[1] = curPos.y;
			context->prevCameraPos[2] = curPos.z;
			context->prevCameraFwd[0] = curFwd.x;
			context->prevCameraFwd[1] = curFwd.y;
			context->prevCameraFwd[2] = curFwd.z;
			context->hasPrevCamera = true;
		}
		if (curFov > 0.01f)
		{
			context->prevCameraFov = curFov;
			context->hasPrevFov = true;
		}
	}

	// Force history reset + 10-frame UI transition cooldown if HUD-less presence toggles (DLSS Enabler XeFG_Dx12)
	if (captured && previous.valid && (frame.hudless != nullptr) != (previous.hudless != nullptr))
	{
		reset = true;
		if (Plugin::options.uiTransitionCooldown.load(std::memory_order_relaxed))
		{
			context->uiTransitionCooldown = 10u;
			Log::Info("HUD-less state changed ({} -> {}), skipping interpolation for 10 frames",
				previous.hudless != nullptr, frame.hudless != nullptr);
		}
	}

	bool inUiCooldown = false;
	if (context->uiTransitionCooldown > 0u)
	{
		--context->uiTransitionCooldown;
		inUiCooldown = true;
		reset = true;
	}

	if (reset)
	{
		context->consecutiveExecuteFailures = 0;
		context->stallCooldownEvaluations = 0;
	}

	data.resetHistory = reset;

	// XeFG needs both frames to have hudless or neither
	context->pairValid = captured && previous.valid && !reset && !inUiCooldown &&
		(frame.hudless != nullptr) == (previous.hudless != nullptr);
	context->batchGeneratedThrough = 0;
	context->directOutput = false;
	frame.valid = captured;
}

FrameGenerator::InterpolateResult FrameGenerator::Interpolate(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* output,
	D3D12_RESOURCE_DESC outputDesc, uint32_t frameIndex, uint32_t frameCount, float interpolationTime)
{
	if (!context || !context->pairValid)
		return InterpolateResult::Skipped;

	if (context->stallCooldownEvaluations > 0)
	{
		--context->stallCooldownEvaluations;
		return InterpolateResult::Skipped;
	}

	const bool customInterpTime = interpolationTime > 0.0f && interpolationTime < 1.0f;
	// Fractional FlipMeter TargetFps timestamps (DLSSG.InterpolationTime) evaluate one custom-t frame per call
	// through the patched XeFG::Execute step rather than the fixed-step multi-frame Vulkan batch path.
	const bool useVulkanWmma = !customInterpTime && UseWmma(settings.model) && D3D12VulkanWmma::Active();

	// Multi-frame batch reuse (D3D12->Vulkan WMMA bridge only): when ScheduleBatch already generated 1..frameCount
	// at frameIndex == 1 to keep a single D3D12<->Vulkan fence bounce, subsequent calls copy the pre-generated output.
	if (useVulkanWmma && frameIndex > 1 && context->batchGeneratedThrough >= frameIndex &&
		outputDesc.Width == settings.width && outputDesc.Height == settings.height &&
		TypedFormat(outputDesc.Format) == settings.format &&
		context->outputs[frameIndex - 1] != nullptr)
	{
		if (useVulkanWmma &&
			D3D12VulkanWmma::AppendOutputCopy(cmdList, context->outputs[frameIndex - 1].Get(), output, OutputState))
		{
			context->directOutput = true;
		}
		else
		{
			context->directOutput = false;
		}
		return InterpolateResult::Interpolated;
	}

	// Zero-copy in-place UAV fast-path for 2x FG (frameCount == 1) when the game's output texture is directly UAV-compatible
	const bool canWriteDirect = !useVulkanWmma && frameCount == 1 && frameIndex == 1 && output != nullptr &&
		outputDesc.Format == settings.format &&
		(outputDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0 &&
		outputDesc.SampleDesc.Count == 1 &&
		outputDesc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN;
	context->directOutput = canWriteDirect;

	if (!canWriteDirect)
	{
		outputDesc.Format = settings.format;
		outputDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		// The game's output can live in a layout a committed UAV texture can't use
		outputDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

		for (uint32_t i = 0; i < frameCount; i++)
		{
			if (!context->outputs[i])
				context->outputs[i] = CreateTexture(device.Get(), outputDesc, OutputState, L"dlssg-to-xefg output",
					"XeFG output", useVulkanWmma);

			if (!context->outputs[i])
				return InterpolateResult::Failed;
		}
	}

	auto fill = [this](const Frame& frame, xefgd_frame_t& out)
		{
			out = frame.data;
			out.backBuffer.pResource = frame.backbuffer.Get();
			out.depth.pResource = frame.depth.Get();
			out.motionVectors.pResource = frame.mvecs.Get();
			out.hudlessColor.pResource = frame.hudless.Get();
		};

	xefgd_exec_params_t params{};
	fill(context->frames[context->current ^ 1], params.prev);
	fill(context->frames[context->current], params.next);

	// XeFG wants the same array for every frame of a pair
	static_assert(sizeof(ComPtr<ID3D12Resource>) == sizeof(ID3D12Resource*));
	ID3D12Resource* directOutputs[1]{ output };
	params.ppOutputs = canWriteDirect ? directOutputs : reinterpret_cast<ID3D12Resource* const*>(context->outputs);
	params.pOutputBases = context->outputBases;
	params.numInterpolatedFrames = (uint8_t)frameCount;
	params.uiCompositionEnabled = settings.uiMode != XEFGD_UI_MODE_AUTO;

	// Execute one interpolated frame per Evaluate call on native D3D12 so GPU work stays pipelined with
	// Streamline/V-Sync/XeLL presents and the game's next real frame, avoiding front-loaded batch stalls.
	params.firstFrameIndex = frameIndex - 1;
	params.frameCount = 1;

	// When WMMA is selected and the D3D12->Vulkan VK_KHR_cooperative_matrix bridge is active, run the full XeFG
	// batch on Vulkan hardware WMMA (16x16x16 SINT8->SINT32) with 1 fence bounce instead of D3D12 cs_6_6 DP4a.
	if (useVulkanWmma && frameIndex == 1 && output != nullptr)
	{
		const Frame& prevFrame = context->frames[context->current ^ 1];
		const Frame& nextFrame = context->frames[context->current];
		D3D12VulkanWmma::ContextSettings vkSettings{
			settings.width, settings.height, settings.format, settings.initFlags, settings.uiMode, settings.model
		};
		D3D12VulkanWmma::FrameInputs vkPrev{
			prevFrame.backbuffer.Get(), prevFrame.hudless.Get(), prevFrame.depth.Get(), prevFrame.mvecs.Get(), prevFrame.data
		};
		D3D12VulkanWmma::FrameInputs vkNext{
			nextFrame.backbuffer.Get(), nextFrame.hudless.Get(), nextFrame.depth.Get(), nextFrame.mvecs.Get(), nextFrame.data
		};
		if (D3D12VulkanWmma::ScheduleBatch(cmdList, vkSettings, vkPrev, vkNext,
			reinterpret_cast<ID3D12Resource* const*>(context->outputs), context->outputBases, frameCount,
			output, OutputState))
		{
			context->batchGeneratedThrough = frameCount;
			context->directOutput = true;
			return InterpolateResult::Interpolated;
		}
	}

	// Execute records the provider's dispatches into the caller's list; make sure that list's table is hooked too
	if (UseCuda(settings.model))
		xefg_cuda::Adopt(cmdList);
	else if (UseWmma(settings.model))
		xefg_wmma::Adopt(cmdList);

	auto result = XeFG::Execute(context->handle, cmdList, params, interpolationTime);

	if (result == XEFGD_RESULT_SUCCESS)
	{
		context->consecutiveExecuteFailures = 0;
		return InterpolateResult::Interpolated;
	}

	if (result > 0)
	{
		// Warnings, like prev and next areas differing after a dynamic resolution change
		Log::Debug("XeFG Execute warning: {}, frame {}/{}", result, frameIndex, frameCount);
		return InterpolateResult::Skipped;
	}

	Log::Error("XeFG Execute error: {}, frame {}/{}", result, frameIndex, frameCount);
	context->pairValid = false;
	if (++context->consecutiveExecuteFailures >= 3u)
	{
		context->stallCooldownEvaluations = 60u;
		context->consecutiveExecuteFailures = 0;
		Log::Warn("XeFG stalled after 3 consecutive Execute errors; bypassing interpolation for 60 evaluations");
	}
	return InterpolateResult::Failed;
}

NGXResult FrameGenerator::Evaluate(ID3D12GraphicsCommandList* cmdList, const NGXParameter& params)
{
	auto backbuffer = params.GetOr<ID3D12Resource*>("DLSSG.Backbuffer", nullptr);
	auto output = params.GetOr<ID3D12Resource*>("DLSSG.OutputInterpolated", nullptr);
	auto outputReal = params.GetOr<ID3D12Resource*>("DLSSG.OutputReal", nullptr);

	float interpolationTime = params.GetOr<float>("DLSSG.InterpolationTime", 0.0f);
	if (!(interpolationTime > 0.0f && interpolationTime < 1.0f) || !XeFG::SupportsInterpolationTime())
		interpolationTime = 0.0f;

	if (!backbuffer || (!output && interpolationTime == 0.0f))
	{
		Log::Error("Missing backbuffer or output");
		return NGX_FAIL_INVALID_PARAMETER;
	}

	if (outputReal)
		Copy(cmdList, outputReal, OutputState, backbuffer, BackbufferState);

	if (!params.GetOr<uint32_t>("DLSSG.EnableInterp", 1) ||
		params.GetOr<uint32_t>("DLSSG.OutputDisableInterpolation", 0) != 0)
	{
		if (context)
		{
			context->pairValid = false;
			context->consecutiveExecuteFailures = 0;
			context->stallCooldownEvaluations = 0;
			context->uiTransitionCooldown = 0;
		}

		if (output)
			Copy(cmdList, output, OutputState, backbuffer, BackbufferState);
		return NGX_SUCCESS;
	}

	if (!params.GetOr<ID3D12Resource*>("DLSSG.Depth", nullptr) ||
		!params.GetOr<ID3D12Resource*>("DLSSG.MVecs", nullptr))
	{
		Log::Error("Missing depth or motion vectors");
		return NGX_FAIL_INVALID_PARAMETER;
	}

	// 1 based index of the generated frame and the number of generated frames between two real frames
	const uint32_t requestedCount = std::clamp(params.GetOr<uint32_t>("DLSSG.MultiFrameCount", 1), 1u, MaxFrames);
	const uint32_t frameIndex = std::clamp(params.GetOr<uint32_t>("DLSSG.MultiFrameIndex", 1), 1u, requestedCount);

	bool vramGuarded = vramGuardActive;
	if (frameIndex == 1)
	{
		vramGuarded = PollVramGuard();
		CaptureRealFrame(cmdList, params);
	}

	const uint32_t effectiveCount = (vramGuarded && requestedCount > 1u) ? 1u : requestedCount;

	if (frameIndex == 1)
	{
		Plugin::Report({ settings.width, settings.height, effectiveCount, settings.uiMode != XEFGD_UI_MODE_AUTO,
			CurrentNetwork(settings.model), context == nullptr, settings.format == DXGI_FORMAT_R16G16B16A16_FLOAT });
	}

	const auto targetDesc = (output ? output : backbuffer)->GetDesc();

	// When VRAM guard clamps 3x-6x down to 2x, skip extra generated frames gracefully by showing the real frame
	const auto result = (frameIndex <= effectiveCount)
		? Interpolate(cmdList, output, targetDesc, frameIndex, effectiveCount, interpolationTime)
		: InterpolateResult::Skipped;

	// A few frames after the path or context changed, say whether the hardware matrix route is actually the one running
	if (result == InterpolateResult::Interpolated && !routeReported && ++routeFrames >= 8)
	{
		if (UseCuda(settings.model))
		{
			routeReported = true;
			const auto route = xefg_cuda::Snapshot();

			if (route.launches > routeLaunches)
				Log::Info("CUDA route: {} tensor-core kernels launched ({} layer binds, {} layers recognized)",
					route.launches - routeLaunches, route.binds, route.layers);
			else
				Log::Warn("CUDA route: no tensor-core kernel launched ({} layer binds, {} layers recognized), the DXIL port is running",
					route.binds, route.layers);
		}
		else if (UseWmma(settings.model))
		{
			routeReported = true;
			const auto vkWmma = D3D12VulkanWmma::Snapshot();
			const auto route = xefg_wmma::Snapshot();

			if (vkWmma.active && vkWmma.batches > 0)
				Log::Info("WMMA hardware route: {} Vulkan VK_KHR_cooperative_matrix ({}x{}x{}) batches executed ({} hardware WMMA shaders, {} fallbacks)",
					vkWmma.batches, vkWmma.tileM, vkWmma.tileN, vkWmma.tileK, vkWmma.coopLayers, vkWmma.fallbacks);
			else if (route.launches > routeLaunches)
				Log::Info("WMMA D3D12 fallback route: {} cs_6_6 INT8 dispatches accelerated ({} layer binds, {} layers recognized)",
					route.launches - routeLaunches, route.binds, route.layers);
		}
	}

	// Without a usable pair (first frame, reset) the real frame is shown again, that's not an error
	if (output)
	{
		if (result == InterpolateResult::Interpolated)
		{
			if (!context->directOutput)
				Copy(cmdList, output, OutputState, context->outputs[frameIndex - 1].Get(), OutputState);
		}
		else
		{
			Copy(cmdList, output, OutputState, backbuffer, BackbufferState);
		}
	}

	return result == InterpolateResult::Failed ? NGX_FAIL : NGX_SUCCESS;
}
