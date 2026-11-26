#include "FrameGenerator.h"
#include "ContextLifecycle.h"
#include "D3D12QueueFence.h"
#include "HangWatchdog.h"
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

static bool IsHdrFormat(DXGI_FORMAT format) noexcept
{
	return format == DXGI_FORMAT_R10G10B10A2_UNORM || format == DXGI_FORMAT_R16G16B16A16_FLOAT;
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

static ContextOwnershipCoordinator& ContextOwner() noexcept
{
	static ContextOwnershipCoordinator* coordinator = new ContextOwnershipCoordinator;
	return *coordinator;
}

std::vector<std::shared_ptr<DeferredDestroy::FenceSignal>> FrameGenerator::ActiveGpuSignals() const
{
	std::lock_guard lock(activeSignalsMutex);
	return activeGpuSignals;
}

FrameGenerator::Context::~Context()
{
	for (auto& out : outputs)
		D3D12VulkanWmma::ForgetResource(out.Get());
	for (auto& f : frames)
	{
		D3D12VulkanWmma::ForgetResource(f.backbuffer.Get());
		D3D12VulkanWmma::ForgetResource(f.hudless.Get());
		D3D12VulkanWmma::ForgetResource(f.depth.Get());
		D3D12VulkanWmma::ForgetResource(f.spareDepth.Get());
		D3D12VulkanWmma::ForgetResource(f.mvecs.Get());
		D3D12VulkanWmma::ForgetResource(f.spareMotion.Get());
	}
	ContextOwner().ResetIfOwner(generation, retiredForReplacement, []() noexcept {
		D3D12VulkanWmma::ResetContext();
		xefg_cuda::InvalidateCache();
	});

	if (handle)
	{
		if (auto result = XeFG::Destroy(handle); result != XEFGD_RESULT_SUCCESS)
			Log::Warn("XeFG Destroy error: {}", result);
	}
	// cudaheld.1: prune now-unreferenced provider PSOs so repeated FG restarts never exhaust gHeld (256-slot cap).
	xefg_cuda::ContextRetired();
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
	case DXGI_FORMAT_R32_FLOAT:
	case DXGI_FORMAT_R16_FLOAT:
		// Alan Wake 2 (Northlight) and other engines submitting R16_FLOAT / R32_FLOAT depth:
		// strip ALLOW_DEPTH_STENCIL so the copy is SRV-readable and shareable with Vulkan WMMA.
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

	const bool validPlane0Fmt =
		srcFootprint.Footprint.Format == DXGI_FORMAT_R32_TYPELESS ||
		srcFootprint.Footprint.Format == DXGI_FORMAT_R32_FLOAT ||
		srcFootprint.Footprint.Format == DXGI_FORMAT_D32_FLOAT;
	const bool sameDims =
		srcFootprint.Footprint.Width == dstFootprint.Footprint.Width &&
		srcFootprint.Footprint.Height == dstFootprint.Footprint.Height &&
		srcFootprint.Footprint.Depth == dstFootprint.Footprint.Depth &&
		dstFootprint.Footprint.Width > 0u && dstFootprint.Footprint.Height > 0u;
	const UINT commonPitch = std::max(srcFootprint.Footprint.RowPitch, dstFootprint.Footprint.RowPitch);
	const UINT64 rowBytes = static_cast<UINT64>(dstFootprint.Footprint.Width) * 4u;

	if (srcBytes == 0 || srcBytes == UINT64_MAX || dstBytes == 0 || dstBytes == UINT64_MAX ||
		!validPlane0Fmt || !sameDims || commonPitch < rowBytes ||
		(commonPitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) != 0)
	{
		static std::atomic<bool> logged{ false };
		if (!logged.exchange(true, std::memory_order_relaxed))
			Log::Warn("CaptureDepth32Plane0 skipped (srcFmt={}, srcFpFmt={}, srcPitch={}, dstPitch={}, srcBytes={}, dstBytes={}, {}x{})",
				static_cast<uint32_t>(rawSrcDesc.Format), static_cast<uint32_t>(srcFootprint.Footprint.Format),
				srcFootprint.Footprint.RowPitch, dstFootprint.Footprint.RowPitch, srcBytes, dstBytes,
				static_cast<uint64_t>(rawSrcDesc.Width), rawSrcDesc.Height);
		return false;
	}

	// AMD RDNA drivers can report a wider RowPitch for R32G8X24 plane 0 than for plain R32_FLOAT
	// (e.g., 7168 vs 6912 for 1708x960). D3D12 CopyTextureRegion allows any RowPitch >= minimum
	// that stays aligned to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256), so share the larger pitch.
	srcFootprint.Footprint.RowPitch = commonPitch;
	dstFootprint.Footprint.RowPitch = commonPitch;

	const UINT64 requiredBytes =
		static_cast<UINT64>(commonPitch) * (dstFootprint.Footprint.Height - 1u) + rowBytes;
	const UINT64 bufBytes = std::max(std::max(srcBytes, dstBytes), requiredBytes);
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
	bool copyMip0Only{ false };
};

// Prepares copy texture for source (recreated on desc change) and queues a batched CopyResource instead of
// issuing per-texture ResourceBarrier flushes (cuts 8 barriers down to 2 per real frame across CUDA/XMX/DP4a).
static bool SameTextureDesc(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b) noexcept
{
	return a.Dimension == b.Dimension && a.Width == b.Width && a.Height == b.Height &&
		a.DepthOrArraySize == b.DepthOrArraySize && a.MipLevels == b.MipLevels &&
		a.Format == b.Format && a.SampleDesc.Count == b.SampleDesc.Count &&
		a.SampleDesc.Quality == b.SampleDesc.Quality && a.Layout == b.Layout &&
		a.Flags == b.Flags;
}

static bool PrepareCapture(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12Resource* source,
	D3D12_RESOURCE_STATES sourceState, ComPtr<ID3D12Resource>& copy, const char* purpose,
	PendingCaptureCopy* pending, UINT& pendingCount, bool wantShared = false,
	bool isDepth = false, ComPtr<ID3D12Resource>* depthScratch = nullptr,
	ComPtr<ID3D12Resource>* spare = nullptr, bool allowUav = false)
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
	// Mip-chain guard (MSFS 2024 submits R32_FLOAT depth with 11 mips): for single-plane 2D textures with
	// MipLevels > 1, allocate a 1-mip internal copy and use CopyTextureRegion + TransitionMip0 on subresource 0
	// so game mip 1..N states remain untouched and VRAM/copy bandwidth are saved.
	const bool isMultiPlaneDepth = IsDepth32StencilFormat(desc.Format) ||
		desc.Format == DXGI_FORMAT_R24G8_TYPELESS || desc.Format == DXGI_FORMAT_D24_UNORM_S8_UINT;
	const bool copyMip0Only = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
		desc.MipLevels > 1 && desc.SampleDesc.Count == 1 && desc.DepthOrArraySize == 1 && !isMultiPlaneDepth;
	if (copyMip0Only)
		desc.MipLevels = 1;

	desc.Flags &= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
	if (allowUav && !isDepth && desc.SampleDesc.Count == 1)
		desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

	if (isDepth)
		MakeReadableDepth(desc);

	if (copy)
	{
		const auto copyDesc = copy->GetDesc();
		if (!SameTextureDesc(copyDesc, desc))
		{
			// provkey.1: Dying Light 2 & The Beast alternate depth/motion formats every frame; keep one spare copy
			// per slot so alternating formats swap in place with zero VRAM reallocation or context recreation.
			if (spare && *spare && SameTextureDesc((*spare)->GetDesc(), desc))
			{
				copy.Swap(*spare);
			}
			else if (spare)
			{
				if (*spare)
				{
					D3D12VulkanWmma::ForgetResource((*spare).Get());
					DeferredDestroy::Add(std::make_shared<ComPtr<ID3D12Resource>>(std::move(*spare)));
				}
				*spare = std::move(copy);
			}
			else
			{
				D3D12VulkanWmma::ForgetResource(copy.Get());
				DeferredDestroy::Add(std::make_shared<ComPtr<ID3D12Resource>>(std::move(copy)));
			}
		}
	}

	if (!copy)
		copy = CreateTexture(device, desc, XeFGInputState, L"dlssg-to-xefg input copy", purpose, wantShared);

	if (!copy)
		return false;

	pending[pendingCount++] = { copy.Get(), source, sourceState, copyMip0Only };
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
		const UINT subresource = pending[i].copyMip0Only ? 0u : D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		if (XeFGInputState != D3D12_RESOURCE_STATE_COPY_DEST)
		{
			before[beforeCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			before[beforeCount].Transition = { pending[i].destination, subresource,
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
				before[beforeCount].Transition = { pending[i].source, subresource,
					pending[i].sourceState, D3D12_RESOURCE_STATE_COPY_SOURCE };
				++beforeCount;
			}
		}
	}
	if (beforeCount)
		cmdList->ResourceBarrier(beforeCount, before);

	for (UINT i = 0; i < count; ++i)
	{
		if (pending[i].copyMip0Only)
		{
			D3D12_TEXTURE_COPY_LOCATION dstLoc{};
			dstLoc.pResource = pending[i].destination;
			dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			dstLoc.SubresourceIndex = 0;
			D3D12_TEXTURE_COPY_LOCATION srcLoc{};
			srcLoc.pResource = pending[i].source;
			srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			srcLoc.SubresourceIndex = 0;
			cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
		}
		else
		{
			cmdList->CopyResource(pending[i].destination, pending[i].source);
		}
	}

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
			if (probe.vendorId != 0)
				Plugin::gpuVendorId.store(probe.vendorId, std::memory_order_relaxed);
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

			// Enable lightweight DX12 compute PSO blob cache + disk persistence on AMD/Intel/non-CUDA too
			if (Plugin::cudaSupport != Plugin::CudaSupport::Supported)
			{
				if (xefg_cuda::InstallBlobCacheOnly(device, XeFG::Module()))
					Log::Info("D3D12 compute PSO blob cache active (disk-backed)");
			}

			std::string wmmaReason;
			if (xefg_wmma::Install(device, XeFG::Module(), MfgKernels::ProgramsDirectory() / L"wmma", wmmaReason))
			{
				xefg_wmma::SetEnabled(false);
				const auto st = xefg_wmma::Snapshot();
				const char* archName = st.arch == xefg_wmma::Arch::Rdna4 ? "RDNA 4 (gfx12)" : "RDNA 3 (gfx11)";
				if (st.dxgiSpoofed)
				{
					Log::Info("WMMA hardware probe ready: {} ({}, device 0x{:04X}, WaveMMA tier {}, source: {}, bypassed DXGI spoof 0x{:04X}:0x{:04X})",
						st.gpuName.empty() ? "AMD GPU" : st.gpuName, archName, st.deviceId, st.waveMmaTier,
						st.probeSource, st.dxgiVendorId, st.dxgiDeviceId);
				}
				else
				{
					Log::Info("WMMA hardware probe ready: {} ({}, device 0x{:04X}, WaveMMA tier {}, source: {})",
						st.gpuName.empty() ? "AMD GPU" : st.gpuName, archName, st.deviceId, st.waveMmaTier,
						st.probeSource);
				}
				std::string vkWmmaReason;
				if (D3D12VulkanWmma::Install(device, vkWmmaReason))
				{
					Plugin::wmmaSupport = Plugin::WmmaSupport::Supported;
					const auto vkSt = D3D12VulkanWmma::Snapshot();
					Log::Info("WMMA hardware bridge active: D3D12 -> Vulkan VK_KHR_cooperative_matrix ({}x{}x{} s8->s32, {} hardware WMMA shaders)",
						vkSt.tileM, vkSt.tileN, vkSt.tileK, vkSt.coopLayers);
				}
				else
				{
					Log::Info("WMMA D3D12->Vulkan hardware bridge unavailable ({}); XMX Port runs stock D3D12 DXIL directly",
						vkWmmaReason.empty() ? "unknown" : vkWmmaReason);
				}
			}
			else
			{
				const auto st = xefg_wmma::Snapshot();
				Log::Info("WMMA route inactive: {} (real GPU: {}, vendor 0x{:04X}, device 0x{:04X}, source: {}, DXGI spoofed: {}, WaveMMA tier: {})",
					wmmaReason, st.gpuName.empty() ? "unknown" : st.gpuName, st.vendorId, st.deviceId,
					st.probeSource.empty() ? "none" : st.probeSource, st.dxgiSpoofed ? "true" : "false", st.waveMmaTier);
			}

			// Normalize unsupported default/saved path (e.g. XMX (native) on AMD/NVIDIA) to the GPU's preferred path.
			const auto curPath = Plugin::options.path.load(std::memory_order_relaxed);
			const auto resolved = Plugin::ResolveEffectivePath(
				curPath,
				Plugin::gpuVendorId.load(std::memory_order_relaxed),
				Plugin::xmxSupport.load(std::memory_order_relaxed) == Plugin::XmxSupport::Supported,
				Plugin::cudaSupport.load(std::memory_order_relaxed) == Plugin::CudaSupport::Supported,
				Plugin::wmmaSupport.load(std::memory_order_relaxed) == Plugin::WmmaSupport::Supported,
				PortAvailable());
			if (resolved != curPath)
			{
				Plugin::RecordPathFallback(curPath, resolved);
				Plugin::options.path.store(resolved, std::memory_order_relaxed);
				Log::Warn("Model '{}' is not supported on GPU vendor 0x{:04X}; automatically switched to '{}'",
					Plugin::PathName(curPath), Plugin::gpuVendorId.load(std::memory_order_relaxed), Plugin::PathName(resolved));
			}
		});
}

static Plugin::Path EffectiveD3D12Path(bool xmxReady)
{
	return Plugin::ResolveEffectivePath(
		Plugin::options.path.load(std::memory_order_relaxed),
		Plugin::gpuVendorId.load(std::memory_order_relaxed),
		xmxReady,
		Plugin::cudaSupport.load(std::memory_order_relaxed) == Plugin::CudaSupport::Supported,
		Plugin::wmmaSupport.load(std::memory_order_relaxed) == Plugin::WmmaSupport::Supported && xefg_wmma::Supported(),
		PortAvailable());
}

static int32_t PickModel(ID3D12Device* device)
{
	const int32_t native = NativeModel(device);
	const auto eff = EffectiveD3D12Path(native == XeFGXmxModel);

	switch (eff)
	{
	case Plugin::Path::Fallback:
		return MfgKernels::Dp4aModel;
	case Plugin::Path::XmxNative:
		return (native == XeFGXmxModel) ? XeFGXmxModel : (PortAvailable() ? MfgKernels::MfgModel : MfgKernels::Dp4aModel);
	case Plugin::Path::Cuda:
	case Plugin::Path::Wmma:
	case Plugin::Path::XmxPort:
	default:
		return PortAvailable() ? MfgKernels::MfgModel : MfgKernels::Dp4aModel;
	}
}

// The ported model runs on the CUDA tensor-core route when NVIDIA resolves to CUDA.
static bool UseCuda(int32_t model)
{
	if (model != MfgKernels::MfgModel)
		return false;

	return EffectiveD3D12Path(Plugin::xmxSupport.load(std::memory_order_relaxed) == Plugin::XmxSupport::Supported) == Plugin::Path::Cuda;
}

// The ported model runs on the AMD WMMA INT8 route when AMD resolves to WMMA (including safety fallback from XMX (native)).
static bool UseWmma(int32_t model)
{
	if (model != MfgKernels::MfgModel || !xefg_wmma::Supported())
		return false;

	return EffectiveD3D12Path(Plugin::xmxSupport.load(std::memory_order_relaxed) == Plugin::XmxSupport::Supported) == Plugin::Path::Wmma;
}

static Plugin::Network CurrentNetwork(int32_t model)
{
	if (model == XeFGXmxModel)
		return Plugin::Network::XmxNative;

	if (model != MfgKernels::MfgModel)
		return Plugin::Network::Fallback;

	if (xefg_cuda::Enabled())
		return Plugin::Network::Cuda;

	if (UseWmma(model))
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
		const HRESULT removed = device ? device->GetDeviceRemovedReason() : S_OK;
		if (FAILED(removed))
		{
			Log::Error("XeFG CreateContext error: {} (D3D12 device removed: {} {:#010x})",
				result, Log::DeviceRemovedName(static_cast<uint32_t>(removed)), static_cast<uint32_t>(removed));
			Log::DumpDred(device.Get());
		}
		else
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
		const HRESULT removed = device ? device->GetDeviceRemovedReason() : S_OK;
		if (FAILED(removed))
		{
			Log::Error("XeFG Init error: {} (D3D12 device removed: {} {:#010x})",
				result, Log::DeviceRemovedName(static_cast<uint32_t>(removed)), static_cast<uint32_t>(removed));
			Log::DumpDred(device.Get());
			return false;
		}
		Log::Error("XeFG Init error: {}", result);
		if (settings.model != MfgKernels::Dp4aModel)
		{
			Log::Warn("XeFG Init failed for model {} (error {}); automatically falling back to DP4a model {}",
				settings.model, result, MfgKernels::Dp4aModel);
			portedModelInitFailed = true;
			newContext.reset();
			settings.model = MfgKernels::Dp4aModel;
			return CreateContext();
		}
		return false;
	}

	newContext->sharedTexturesMode = UseWmma(settings.model) && D3D12VulkanWmma::WantsSharedTextures();
	// Transfer global bridge/cache ownership ONLY AFTER XeFG::CreateContext() and XeFG::Init() succeeded,
	// serialized against any deferred ~Context() so an old destructor can never reset the new live bridge.
	std::shared_ptr<Context> retiredOld;
	ContextOwner().Commit([&](uint64_t gen) noexcept {
		newContext->generation = gen;
		if (context)
		{
			context->retiredForReplacement = true;
			retiredOld = std::move(context);
		}
		context = std::move(newContext);
		if (UseWmma(settings.model) && D3D12VulkanWmma::Active())
		{
			const D3D12VulkanWmma::ContextSettings vkSettings{
				settings.width, settings.height, settings.format, settings.initFlags, settings.uiMode, settings.model
			};
			D3D12VulkanWmma::WarmupContextAsync(vkSettings);
		}
	});
	if (retiredOld)
		DeferredDestroy::Add(std::move(retiredOld), /*heavy=*/true, ActiveGpuSignals());

	routeLaunches = UseWmma(settings.model) ? xefg_wmma::Snapshot().launches : xefg_cuda::Snapshot().launches;
	routeFrames = 0;
	routeReported = false;
	lastExecRoute = ActiveExecRoute::None;
	currentExecRoute = ActiveExecRoute::None;
	currentExecRouteReason = "";
	return true;
}

bool FrameGenerator::PollVramGuard(uint32_t requestedCount) noexcept
{
	if (!Plugin::options.vramGuard.load(std::memory_order_relaxed))
	{
		if (vramGuardActive || vramGuardStepDown > 0u)
		{
			vramGuardActive = false;
			vramGuardStepDown = 0u;
			vramReleasePending = false;
			vramTightStreak = 0u;
			Log::Info("Adaptive VRAM guard disabled by user, restoring full MFG multiplier");
		}
		return false;
	}

	if (!vramAdapterChecked)
	{
		vramAdapterChecked = true;
		ComPtr<IDXGIFactory4> factory;
		if (device && SUCCEEDED(adapter_identity::CreateDxgiFactory4Safe(&factory)))
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

	vramUsageMiB = static_cast<uint32_t>(memInfo.CurrentUsage / (1024ull * 1024ull));
	vramBudgetMiB = static_cast<uint32_t>(memInfo.Budget / (1024ull * 1024ull));

	constexpr uint64_t kCriticalMarginBytes = 100ull * 1024ull * 1024ull; // 100 MiB critical -> clamp straight to 2x
	constexpr uint64_t kSafetyMarginBytes = 200ull * 1024ull * 1024ull;   // 200 MiB safety -> step down gradually
	constexpr uint64_t kReleaseMarginBytes = 400ull * 1024ull * 1024ull;  // 400 MiB hysteresis + 3s dwell before stepping up
	constexpr auto kReleaseDwell = std::chrono::seconds(3);
	// Toggling libxess_fg.dll between 1 (2x) and 2 (3x) generated frames triggers a ~112 ms pipeline rebuild,
	// whereas 2 <-> 3+ is 0 ms; require stronger hysteresis across the 1 <-> 2 boundary.
	constexpr auto kCrossOneToTwoReleaseDwell = std::chrono::seconds(10);

	const bool critical = memInfo.Budget > kCriticalMarginBytes && memInfo.CurrentUsage + kCriticalMarginBytes >= memInfo.Budget;
	const bool tight = memInfo.Budget > kSafetyMarginBytes && memInfo.CurrentUsage + kSafetyMarginBytes >= memInfo.Budget;
	const bool comfortable = memInfo.Budget > kReleaseMarginBytes && memInfo.CurrentUsage + kReleaseMarginBytes < memInfo.Budget;

	auto effectiveForStep = [&](uint32_t step) noexcept -> uint32_t {
		if (requestedCount <= 1u || step == 0u) return requestedCount;
		return requestedCount > step ? (requestedCount - step) : 1u;
	};
	const uint32_t curEffective = effectiveForStep(vramGuardActive ? vramGuardStepDown : 0u);

	if (critical || tight)
	{
		vramReleasePending = false;
		const uint32_t nextStep = critical ? 4u : std::min(vramGuardStepDown + 1u, 4u);
		const uint32_t nextEffective = effectiveForStep(nextStep);
		const bool crossesTwoToOne = !critical && curEffective >= 2u && nextEffective == 1u;
		if (crossesTwoToOne && ++vramTightStreak < 2u)
			return vramGuardActive;
		vramTightStreak = 0u;
		if (!vramGuardActive || nextStep != vramGuardStepDown)
		{
			vramGuardActive = true;
			vramGuardStepDown = nextStep;
			Log::Warn("Adaptive VRAM guard engaged ({} MiB / {} MiB budget): stepping down MFG by {} to prevent swap stutter",
				memInfo.CurrentUsage / (1024ull * 1024ull), memInfo.Budget / (1024ull * 1024ull), vramGuardStepDown);
		}
	}
	else if (vramGuardActive && comfortable)
	{
		vramTightStreak = 0u;
		const uint32_t nextStep = (vramGuardStepDown > 1u) ? (vramGuardStepDown - 1u) : 0u;
		const uint32_t nextEffective = effectiveForStep(nextStep);
		const auto requiredDwell = (curEffective == 1u && nextEffective >= 2u) ? kCrossOneToTwoReleaseDwell : kReleaseDwell;
		const auto now = std::chrono::steady_clock::now();
		if (!vramReleasePending)
		{
			vramReleasePending = true;
			vramReleaseSince = now;
		}
		else if ((now - vramReleaseSince) >= requiredDwell)
		{
			vramReleasePending = false;
			if (vramGuardStepDown > 1u)
			{
				--vramGuardStepDown;
				Log::Info("Adaptive VRAM guard easing ({} MiB / {} MiB budget): step-down now {}",
					memInfo.CurrentUsage / (1024ull * 1024ull), memInfo.Budget / (1024ull * 1024ull), vramGuardStepDown);
			}
			else
			{
				vramGuardActive = false;
				vramGuardStepDown = 0u;
				Log::Info("Adaptive VRAM guard released ({} MiB / {} MiB budget): restoring full MFG multiplier",
					memInfo.CurrentUsage / (1024ull * 1024ull), memInfo.Budget / (1024ull * 1024ull));
			}
		}
	}
	else
	{
		vramTightStreak = 0u;
		vramReleasePending = false;
	}

	return vramGuardActive;
}

void FrameGenerator::CaptureRealFrame(ID3D12GraphicsCommandList* cmdList, const NGXParameter& params)
{
	static bool warnedCameraMotion = false;
	if (params.GetOr<uint32_t>("DLSSG.CameraMotionIncluded", 1) == 0 && !std::exchange(warnedCameraMotion, true))
		Log::Warn("DLSSG.CameraMotionIncluded is 0, the camera's motion is missing from the motion vectors");

	Plugin::ApplyBrokenHudlessQuirkOnce();

	auto backbuffer = params.GetOr<ID3D12Resource*>("DLSSG.Backbuffer", nullptr);
	const auto bbDesc = backbuffer->GetDesc();
	const auto backbufferFormat = TypedFormat(bbDesc.Format);
	auto rawHudless = params.GetOr<ID3D12Resource*>("DLSSG.HUDLess", nullptr);
	if (!rawHudless && !IsHdrFormat(backbufferFormat))
	{
		if (auto optiHudless = params.GetOr<ID3D12Resource*>("Opti.Hudless", nullptr);
			optiHudless && optiHudless != backbuffer)
		{
			rawHudless = optiHudless;
			static bool loggedOptiHudless = false;
			if (!std::exchange(loggedOptiHudless, true))
				Log::Info("UnityUpscalerBridge: using Opti.Hudless as HUD-less source (SDR)");
		}
	}
	auto hudless = Plugin::options.hudlessUi ? rawHudless : nullptr;
	auto depth = params.GetOr<ID3D12Resource*>("DLSSG.Depth", nullptr);
	auto mvecs = params.GetOr<ID3D12Resource*>("DLSSG.MVecs", nullptr);
	auto output = params.GetOr<ID3D12Resource*>("DLSSG.OutputInterpolated", nullptr);

	const bool depthInverted = XeFgEnabler::ApplyTriStateOverride(
		Plugin::options.depthInvertedOverride.load(std::memory_order_relaxed),
		params.GetOr<uint32_t>("DLSSG.DepthInverted", 0) != 0);
	const bool mvecJittered = XeFgEnabler::ApplyTriStateOverride(
		Plugin::options.jitteredMvOverride.load(std::memory_order_relaxed),
		params.GetOr<uint32_t>("DLSSG.MvecJittered", 0) != 0);
	auto hudlessFormat = hudless ? TypedFormat(hudless->GetDesc().Format) : DXGI_FORMAT_UNKNOWN;

	// Guard against aliased or mismatched HUD-less textures (size, format, MSAA, or subrect) which break XeFG UI extraction.
	bool hudlessFormatMismatch = false;
	if (hudless)
	{
		const auto hlDesc = hudless->GetDesc();
		const auto bbSubrect = Subrect(params, kBackbufferSubrect, backbuffer);
		const auto hlSubrect = Subrect(params, kHudlessSubrect, hudless);
		const bool subrectMismatch =
			hlSubrect.base.x != bbSubrect.base.x || hlSubrect.base.y != bbSubrect.base.y ||
			hlSubrect.size.x != bbSubrect.size.x || hlSubrect.size.y != bbSubrect.size.y;
		const bool geometryBad =
			hlDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || hlDesc.SampleDesc.Count > 1 ||
			hlDesc.Width != bbDesc.Width || hlDesc.Height != bbDesc.Height || subrectMismatch;

		if (hudless == backbuffer)
		{
			static bool warnedHudlessAlias = false;
			if (!std::exchange(warnedHudlessAlias, true))
				Log::Warn("Ignoring HUD-less input because it aliases back buffer; falling back to UI Auto");
			hudless = nullptr;
			hudlessFormat = DXGI_FORMAT_UNKNOWN;
		}
		else if (geometryBad)
		{
			static bool warnedHudlessMismatch = false;
			if (!std::exchange(warnedHudlessMismatch, true))
				Log::Warn("Ignoring HUD-less input due to geometry/subrect mismatch with back buffer (HUD-less {}x{} {} rect [{},{}+{}x{}], back buffer {}x{} {} rect [{},{}+{}x{}]); falling back to UI Auto",
					static_cast<uint64_t>(hlDesc.Width), hlDesc.Height, FormatName(hudlessFormat),
					hlSubrect.base.x, hlSubrect.base.y, hlSubrect.size.x, hlSubrect.size.y,
					static_cast<uint64_t>(bbDesc.Width), bbDesc.Height, FormatName(backbufferFormat),
					bbSubrect.base.x, bbSubrect.base.y, bbSubrect.size.x, bbSubrect.size.y);
			hudless = nullptr;
			hudlessFormat = DXGI_FORMAT_UNKNOWN;
		}
		else if (hudlessFormat != backbufferFormat)
		{
			if (HudTierPass::IsToneMapMatchSupported(backbufferFormat))
			{
				hudlessFormatMismatch = true;
			}
			else
			{
				static bool warnedHudlessFmtMismatch = false;
				if (!std::exchange(warnedHudlessFmtMismatch, true))
					Log::Warn("Ignoring HUD-less input due to format mismatch with back buffer (HUD-less {}, back buffer {}); falling back to UI Auto",
						FormatName(hudlessFormat), FormatName(backbufferFormat));
				hudless = nullptr;
				hudlessFormat = DXGI_FORMAT_UNKNOWN;
			}
		}
	}

	Settings wanted;
	ID3D12Resource* const targetForDesc = output ? output : backbuffer;
	auto outputDesc = targetForDesc->GetDesc();
	wanted.width = (uint32_t)outputDesc.Width;
	wanted.height = outputDesc.Height;
	wanted.format = TypedFormat(outputDesc.Format);
	wanted.initFlags = (depthInverted ? XEFGD_INIT_FLAG_INVERTED_DEPTH : 0) |
		(mvecJittered ? XEFGD_INIT_FLAG_JITTERED_MV : 0);
	const int32_t curPath = static_cast<int32_t>(Plugin::options.path.load(std::memory_order_relaxed));
	if (loggedPath != -1 && curPath != loggedPath)
		portedModelInitFailed = false;
	wanted.model = PickModel(device.Get());
	if (portedModelInitFailed && wanted.model != MfgKernels::Dp4aModel)
		wanted.model = MfgKernels::Dp4aModel;

	// 3-Tier Adaptive HUD Protection (Tier 3 Hybrid = Tone-Map Match + Static HUD Mask): poll readbacks & evaluate.
	const auto requestedHudMode = static_cast<HudTier::Mode>(
		std::clamp(Plugin::options.hudTierMode.load(std::memory_order_relaxed), 0, 4));
	const bool needHudCompute = (requestedHudMode != HudTier::Mode::Off) ||
		Plugin::options.hudSplitCompare.load(std::memory_order_relaxed) ||
		(mvecJittered && wanted.model == MfgKernels::MfgModel);
	const bool hudComputeReady = needHudCompute && hudTierPass.EnsureInitialized(device.Get(), wanted.width, wanted.height);
	if (hudComputeReady && requestedHudMode != HudTier::Mode::Off)
	{
		float diffFrac = 0.0f;
		bool fromToneMatched = false;
		if (hudTierPass.PollDiffFraction(diffFrac, fromToneMatched))
			hudTierState.FeedHudDiffFraction(diffFrac, fromToneMatched);
		float staticFrac = 0.0f;
		if (hudTierPass.PollStaticFraction(staticFrac))
			hudTierState.FeedStaticHudFraction(staticFrac);
	}

	HudTier::Inputs hudInputs{};
	hudInputs.mode = requestedHudMode;
	hudInputs.hudlessUiEnabled = Plugin::options.hudlessUi.load(std::memory_order_relaxed);
	hudInputs.hasGameHudless = rawHudless != nullptr;
	hudInputs.hudlessStructurallyValid = hudless != nullptr && !hudlessFormatMismatch;
	hudInputs.hudlessFormatMismatch = hudless != nullptr && hudlessFormatMismatch;
	hudInputs.hasSeparateUiPlane = false;
	hudInputs.gpuComputeAvailable = hudComputeReady;
	hudInputs.toneMapMatchAvailable = hudComputeReady && HudTierPass::IsToneMapMatchSupported(backbufferFormat);
	hudDecision = hudTierState.Evaluate(hudInputs);

	if (!hasLoggedHudTier || hudDecision.activeTier != loggedHudTier)
	{
		hasLoggedHudTier = true;
		loggedHudTier = hudDecision.activeTier;
		Log::Info("HUD protection tier: {} (reason: {}, diff={:.2f}%, static={:.2f}%)",
			HudTier::ActiveTierShortName(hudDecision.activeTier), hudDecision.reason,
			hudTierState.LastDiffFraction() * 100.0f, hudTierState.LastStaticFraction() * 100.0f);
	}

	// Only keep HUD-less capture when Tier 1 Native, Tier 2 Diff-Composite, Tier 4 Tone-Map Match, or diff-guard needs it.
	if (!hudDecision.useXeFgNativeUi && !hudDecision.useHudlessAsSceneSource && !hudDecision.runHudDiffGuard && !hudDecision.runToneMapMatch)
	{
		hudless = nullptr;
		hudlessFormat = DXGI_FORMAT_UNKNOWN;
	}

	wanted.uiMode = hudDecision.useXeFgNativeUi ? XEFGD_UI_MODE_BACKBUFFER_HUDLESS : XEFGD_UI_MODE_AUTO;

	// XMX Port / CUDA / WMMA (model 16) payload omits JitteredMV reconstruction shader variants (e.g. 35112441167526893749578).
	// Unjitter motion vectors in-place on our private single-sampled UAV mvecs copy and clear XEFGD_INIT_FLAG_JITTERED_MV.
	// If the motion vector resource cannot be unjittered safely, fall back to DP4a (model 18) with JITTERED_MV intact.
	if (mvecJittered && wanted.model == MfgKernels::MfgModel)
	{
		const auto mvDesc = mvecs ? mvecs->GetDesc() : D3D12_RESOURCE_DESC{};
		const bool canUnjitter = hudComputeReady && mvecs &&
			mvDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
			mvDesc.SampleDesc.Count == 1 &&
			hudTierPass.CanUnjitterMotionVectors(mvDesc.Format);
		if (canUnjitter)
		{
			wanted.initFlags &= ~XEFGD_INIT_FLAG_JITTERED_MV;
			wanted.softwareUnjitterMvecs = true;
		}
		else
		{
			wanted.model = MfgKernels::Dp4aModel;
		}
	}

	// The menu path can change without a context change: XMX Port, CUDA, and WMMA all run the ported model (16), only the
	// route that executes its programs differs. Log the switch, and say whether it took effect below
	const auto path = EffectiveD3D12Path(Plugin::xmxSupport.load(std::memory_order_relaxed) == Plugin::XmxSupport::Supported);
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

		if (wanted.softwareUnjitterMvecs && !settings.softwareUnjitterMvecs)
			Log::Info("XMX Port: software motion-vector unjitter active (clearing XEFGD_INIT_FLAG_JITTERED_MV for model 16)");

		settings = wanted;
		createFailed = false;

		if (context)
		{
			// Reset global WMMA/CUDA context state now under the ownership lock (before new context creation)
			// so the old context's deferred destructor never races with or wipes the new active bridge/cache.
			ContextOwner().ResetBeforeReplacement(context->generation, []() noexcept {
				D3D12VulkanWmma::ResetContext();
				xefg_cuda::InvalidateCache();
			});
			context->retiredForReplacement = true;
			DeferredDestroy::Add(std::move(context), /*heavy=*/true, ActiveGpuSignals());
		}
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
		lastExecRoute = ActiveExecRoute::None;
		currentExecRoute = ActiveExecRoute::None;
		currentExecRouteReason = "";
	}

	const bool wantShared = UseWmma(settings.model) && D3D12VulkanWmma::WantsSharedTextures();
	const bool wmmaReadyRefresh = wantShared && context->wmmaFallbackTouchedShared && D3D12VulkanWmma::ContextReady();
	if (context->sharedTexturesMode != wantShared || wmmaReadyRefresh)
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
			f.matchedHudless = nullptr;
			releaseTex(f.depth);
			releaseTex(f.spareDepth);
			releaseTex(f.depthScratch);
			releaseTex(f.mvecs);
			releaseTex(f.spareMotion);
			f.valid = false;
		}
		context->pairValid = false;
		context->batchGeneratedThrough = 0;
		context->sharedTexturesMode = wantShared;
		context->wmmaFallbackTouchedShared = false;
		if (wmmaReadyRefresh)
			Log::Info("WMMA Vulkan bridge ready: refreshed shared textures cleanly after D3D12 init fallback");
	}

	context->current ^= 1;
	auto& frame = context->frames[context->current];
	auto& previous = context->frames[context->current ^ 1];

	PendingCaptureCopy pendingCopies[4]{};
	UINT pendingCopyCount = 0;
	bool captured = PrepareCapture(device.Get(), cmdList, backbuffer, BackbufferState, frame.backbuffer, "Backbuffer", pendingCopies, pendingCopyCount, wantShared);
	captured &= PrepareCapture(device.Get(), cmdList, depth, InputState, frame.depth, "Depth copy", pendingCopies, pendingCopyCount, wantShared, true, &frame.depthScratch, &frame.spareDepth);
	captured &= PrepareCapture(device.Get(), cmdList, mvecs, InputState, frame.mvecs, "MVecs copy", pendingCopies, pendingCopyCount, wantShared, false, nullptr, &frame.spareMotion, settings.softwareUnjitterMvecs);
	const bool hudlessCaptured = hudless &&
		PrepareCapture(device.Get(), cmdList, hudless, InputState, frame.hudless, "HUDLess copy", pendingCopies, pendingCopyCount, wantShared);
	if (!hudlessCaptured && frame.hudless)
	{
		D3D12VulkanWmma::ForgetResource(frame.hudless.Get());
		DeferredDestroy::Add(std::make_shared<ComPtr<ID3D12Resource>>(std::move(frame.hudless)));
	}
	FlushCaptureCopies(cmdList, pendingCopies, pendingCopyCount);

	frame.matchedHudless = nullptr;
	if (hudComputeReady && captured)
	{
		hudTierPass.BeginRealFrame();
		// Tier 4 Tone-Map Match (conditional inline): runs ONLY when HUDLess color/format mismatches Backbuffer.
		if (hudDecision.runToneMapMatch && hudlessCaptured && frame.hudless)
		{
			const int32_t tmMode = std::clamp(Plugin::options.toneMapMatchMode.load(std::memory_order_relaxed), 0, 2);
			const bool forceRefit = !previous.valid || !previous.backbuffer || !context->pairValid;
			frame.matchedHudless = hudTierPass.RecordToneMapMatch(cmdList, context->current, wantShared,
				frame.backbuffer.Get(), XeFGInputState, frame.hudless.Get(), XeFGInputState, XeFGInputState,
				tmMode, forceRefit);
		}
		// Tier 1 Native (or Tier 3 recovery monitoring): analyze-only 1/16-pixel Real vs Hudless diff guard.
		// Probe every real frame for the first 8 frames (fast mismatch detection), then 1 in 16 real frames (>93% less GPU cost).
		if (hudDecision.runHudDiffGuard && !hudDecision.runHudDiffComposite && hudlessCaptured && frame.hudless)
		{
			const uint64_t guardFrame = ++context->hudDiffGuardFrameCounter;
			if (guardFrame <= 8u || (guardFrame % 16u) == 0u)
			{
				hudTierPass.RecordHudDiffGuard(cmdList, frame.backbuffer.Get(), XeFGInputState,
					frame.hudless.Get(), XeFGInputState);
			}
		}
		// Tier 3 Static HUD Mask: update per-pixel temporal stability map across consecutive real frames.
		if (hudDecision.runStaticHudUpdate)
		{
			const bool staticReset = !previous.valid || !previous.backbuffer;
			hudTierPass.RecordStaticHudUpdate(cmdList, frame.backbuffer.Get(), XeFGInputState,
				previous.valid ? previous.backbuffer.Get() : nullptr, XeFGInputState, staticReset);
		}
	}

	auto& data = frame.data;
	data = {};

	if (captured)
	{
		data.backBuffer = Subrect(params, kBackbufferSubrect, backbuffer);
		data.depth = Subrect(params, kDepthSubrect, depth);
		data.motionVectors = Subrect(params, kMVecsSubrect, mvecs);

		if (hudlessCaptured && frame.hudless)
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

	// Long-pause / Alt+Tab / window-minimize history reset: if >500ms elapsed between consecutive real frames,
	// flush stale history so the first resumed frame never smears (1 GetTickCount64 subtraction/frame, ~0 overhead).
	const uint64_t nowTickMs = GetTickCount64();
	if (captured && previous.valid && !reset && context->lastRealFrameTickMs != 0 &&
		nowTickMs > context->lastRealFrameTickMs && (nowTickMs - context->lastRealFrameTickMs) > 500ull)
	{
		reset = true;
	}
	if (captured)
		context->lastRealFrameTickMs = nowTickMs;

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

	// Camera Cut & Teleport Detector + GhostBuster anti-ghosting (internal safety resets still apply when
	// ignoreGameReset is on; adaptive camera-step scale prevents meter-threshold false positives in UE cm-scale games)
	constexpr uint32_t kSyntheticResetQuietFrames = 4u;
	if (reset)
		context->syntheticResetQuietFrames = kSyntheticResetQuietFrames;
	else if (context->syntheticResetQuietFrames > 0u)
		--context->syntheticResetQuietFrames;

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

			const float cutThresholdSq = context->cameraMotionScale.CutThresholdSq();
			const float hardTeleportSq = context->cameraMotionScale.HardTeleportThresholdSq();
			const bool quietActive = context->syntheticResetQuietFrames > 0u;
			// GhostBuster already checks sharper rotation snaps (fwdDot < 0.707); only check 120-deg flip when GhostBuster is off.
			if (camCutOn && !reset && (distSq > hardTeleportSq || (!quietActive && (distSq > cutThresholdSq || (!ghostBusterOn && fwdDot < -0.5f)))))
			{
				reset = true;
				context->syntheticResetQuietFrames = kSyntheticResetQuietFrames;
			}
		}

		if (ghostBusterOn && !reset && context->syntheticResetQuietFrames == 0u)
		{
			const bool camMvIncluded = params.GetOr<uint32_t>("DLSSG.CameraMotionIncluded", 1) != 0;
			if (XeFgEnabler::ShouldGhostBusterReset(true, hasCamera, context->hasPrevCamera, fwdDot,
				curFov, context->prevCameraFov, context->hasPrevFov, camMvIncluded, distSq,
				context->cameraMotionScale.GhostBusterMoveThresholdSq()))
			{
				reset = true;
				context->syntheticResetQuietFrames = kSyntheticResetQuietFrames;
			}
		}

		if (hasCamera && context->hasPrevCamera)
			context->cameraMotionScale.ObserveCandidate(std::sqrt(distSq), reset);

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

	// Force history reset + 10-frame UI transition cooldown only when the game's raw HUD-less presence toggles
	// (not when HudTierPolicy internally switches between Tier 1 Native and Tier 3 Static Mask).
	const bool curRawHudless = (rawHudless != nullptr);
	if (captured && context->hasPrevRawHudless && curRawHudless != context->prevRawHudless)
	{
		if (!ignoreGameReset)
			reset = true;
		if (Plugin::options.uiTransitionCooldown.load(std::memory_order_relaxed))
		{
			context->uiTransitionCooldown = 10u;
			Log::Info("HUD-less state changed ({} -> {}), skipping interpolation for 10 frames",
				context->prevRawHudless, curRawHudless);
		}
	}
	if (captured)
	{
		context->prevRawHudless = curRawHudless;
		context->hasPrevRawHudless = true;
	}

	bool inUiCooldown = false;
	if (context->uiTransitionCooldown > 0u)
	{
		--context->uiTransitionCooldown;
		inUiCooldown = true;
		reset = true;
	}

	if (captured && settings.softwareUnjitterMvecs && frame.mvecs && previous.valid && !reset)
	{
		const float dpx = previous.data.jitterOffset[0] - data.jitterOffset[0];
		const float dpy = previous.data.jitterOffset[1] - data.jitterOffset[1];
		const float deltaRawX = dpx / data.motionVectorScale[0];
		const float deltaRawY = dpy / data.motionVectorScale[1];
		if (std::isfinite(deltaRawX) && std::isfinite(deltaRawY))
		{
			hudTierPass.RecordUnjitterMotionVectors(cmdList, frame.mvecs.Get(), XeFGInputState,
				data.motionVectors.base.x, data.motionVectors.base.y,
				data.motionVectors.size.x, data.motionVectors.size.y,
				deltaRawX, deltaRawY);
		}
	}

	if (reset)
	{
		context->consecutiveExecuteFailures = 0;
		context->stallCooldownEvaluations = 0;
		context->stallCooldownUntilMs = 0;
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
	D3D12_RESOURCE_DESC outputDesc, uint32_t frameIndex, uint32_t frameCount, float interpolationTime, bool callerRecording)
{
	if (!context || !context->pairValid || context->deviceLostLatched)
		return InterpolateResult::Skipped;

	if (context->stallCooldownEvaluations > 0 || context->stallCooldownUntilMs != 0)
	{
		if (context->stallCooldownEvaluations > 0)
			--context->stallCooldownEvaluations;
		if (context->stallCooldownEvaluations > 0 || GetTickCount64() < context->stallCooldownUntilMs)
			return InterpolateResult::Skipped;
		context->stallCooldownUntilMs = 0;
	}

	const bool customInterpTime = interpolationTime > 0.0f && interpolationTime < 1.0f;
	// Fractional FlipMeter TargetFps timestamps (DLSSG.InterpolationTime) evaluate one custom-t frame per call
	// through the patched XeFG::Execute step rather than the fixed-step multi-frame Vulkan batch path.
	if (!customInterpTime && frameIndex == 1 && UseWmma(settings.model) && !D3D12VulkanWmma::Active())
	{
		const D3D12VulkanWmma::ContextSettings vkSettings{
			settings.width, settings.height, settings.format, settings.initFlags, settings.uiMode, settings.model
		};
		D3D12VulkanWmma::TryRecoverAsync(vkSettings);
	}

	const bool needsPostHudComposite = hudTierPass.Available() &&
		(hudDecision.runHudDiffComposite || hudDecision.runStaticHudComposite);
	const bool needsSplitCompare = hudTierPass.Available() &&
		Plugin::options.hudSplitCompare.load(std::memory_order_relaxed);
	// WMMA bridge runs Vulkan + output copy AFTER the game's cmdList executes, which would overwrite any
	// post-HUD composite or split-screen pass recorded on cmdList; use native DX12 when either is active.
	const bool useVulkanWmma = !customInterpTime && !needsPostHudComposite && !needsSplitCompare &&
		UseWmma(settings.model) && D3D12VulkanWmma::Active();

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
		(!needsPostHudComposite || HudTierPass::IsUavCompatibleFormat(outputDesc.Format)) &&
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
					"XeFG output", context->sharedTexturesMode || useVulkanWmma);

			if (!context->outputs[i])
				return InterpolateResult::Failed;
		}
	}

	auto fill = [this](const Frame& frame, xefgd_frame_t& out)
		{
			out = frame.data;
			// Tier 2 / Tier 4: interpolate the clean (or tone-matched) HUD-less scene in UI_MODE_AUTO, then composite HUD on top.
			ID3D12Resource* effectiveHudless = (hudDecision.runToneMapMatch && frame.matchedHudless)
				? frame.matchedHudless
				: frame.hudless.Get();
			const bool useHudlessScene = hudDecision.useHudlessAsSceneSource && effectiveHudless != nullptr;
			out.backBuffer.pResource = useHudlessScene ? effectiveHudless : frame.backbuffer.Get();
			out.depth.pResource = frame.depth.Get();
			out.motionVectors.pResource = frame.mvecs.Get();
			out.hudlessColor.pResource = hudDecision.useXeFgNativeUi ? effectiveHudless : nullptr;
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
		ID3D12Resource* prevEffectiveHudless = (hudDecision.runToneMapMatch && prevFrame.matchedHudless)
			? prevFrame.matchedHudless : prevFrame.hudless.Get();
		ID3D12Resource* nextEffectiveHudless = (hudDecision.runToneMapMatch && nextFrame.matchedHudless)
			? nextFrame.matchedHudless : nextFrame.hudless.Get();
		const bool useHudlessScene = hudDecision.useHudlessAsSceneSource && prevEffectiveHudless != nullptr && nextEffectiveHudless != nullptr;
		D3D12VulkanWmma::FrameInputs vkPrev{
			useHudlessScene ? prevEffectiveHudless : prevFrame.backbuffer.Get(),
			hudDecision.useXeFgNativeUi ? prevEffectiveHudless : nullptr,
			prevFrame.depth.Get(), prevFrame.mvecs.Get(), prevFrame.data
		};
		D3D12VulkanWmma::FrameInputs vkNext{
			useHudlessScene ? nextEffectiveHudless : nextFrame.backbuffer.Get(),
			hudDecision.useXeFgNativeUi ? nextEffectiveHudless : nullptr,
			nextFrame.depth.Get(), nextFrame.mvecs.Get(), nextFrame.data,
			nextFrame.backbuffer.Get(), XeFGInputState
		};
		if (D3D12VulkanWmma::ScheduleBatch(cmdList, vkSettings, vkPrev, vkNext,
			reinterpret_cast<ID3D12Resource* const*>(context->outputs), context->outputBases, frameCount,
			output, OutputState, callerRecording))
		{
			currentExecRoute = ActiveExecRoute::VulkanWmma;
			currentExecRouteReason = "hardware_wmma";
			context->batchGeneratedThrough = frameCount;
			context->directOutput = true;
			return InterpolateResult::Interpolated;
		}
		context->wmmaFallbackTouchedShared = true;
	}

	// Record the actual execution route for this frame so HUD composite / split-screen / fallback switches log truthfully.
	if (UseCuda(settings.model))
	{
		currentExecRoute = ActiveExecRoute::Cuda;
		currentExecRouteReason = "cuda";
		xefg_cuda::Adopt(cmdList);
	}
	else if (UseWmma(settings.model))
	{
		// Fallback to stock D3D12 DXIL when Vulkan hardware WMMA cannot run this frame
		// (avoids slower software cs_6_6 shared-tile DP4a overrides).
		currentExecRoute = ActiveExecRoute::D3D12WmmaFallback;
		currentExecRouteReason = needsSplitCompare ? "hud_split_compare"
			: (needsPostHudComposite ? "hud_post_composite"
			: (customInterpTime ? "custom_interp_time" : "vk_bridge_fallback"));
	}
	else
	{
		currentExecRoute = ActiveExecRoute::D3D12Dxil;
		currentExecRouteReason = "d3d12_dxil";
	}

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

	const HRESULT removed = device ? device->GetDeviceRemovedReason() : S_OK;
	if (FAILED(removed))
	{
		Log::Error("XeFG Execute error: {}, frame {}/{} (D3D12 device removed: {} {:#010x}); latching pass-through until device recreation",
			result, frameIndex, frameCount, Log::DeviceRemovedName(static_cast<uint32_t>(removed)), static_cast<uint32_t>(removed));
		Log::DumpDred(device.Get());
		context->deviceLostLatched = true;
		context->pairValid = false;
		return InterpolateResult::Failed;
	}
	Log::Error("XeFG Execute error: {}, frame {}/{}", result, frameIndex, frameCount);
	context->pairValid = false;
	if (++context->consecutiveExecuteFailures >= 3u)
	{
		context->stallCooldownEvaluations = 8u;
		context->stallCooldownUntilMs = GetTickCount64() + 300ull;
		context->consecutiveExecuteFailures = 0;
		Log::Warn("XeFG stalled after 3 consecutive Execute errors; bypassing interpolation for >=8 evaluations and >=300ms");
	}
	return InterpolateResult::Failed;
}

bool FrameGenerator::GpuProfiler::EnsureReady(ID3D12Device* dev) noexcept
{
	if (ready)
		return true;
	if (initAttempted || !dev)
		return false;
	initAttempted = true;

	D3D12_COMMAND_QUEUE_DESC qd{};
	qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	ComPtr<ID3D12CommandQueue> tmpQueue;
	if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&tmpQueue))) || !tmpQueue)
		return false;
	if (FAILED(tmpQueue->GetTimestampFrequency(&freqHz)) || freqHz == 0)
		return false;

	D3D12_QUERY_HEAP_DESC hd{};
	hd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
	hd.Count = kSlots * kQueriesPerSlot;
	if (FAILED(dev->CreateQueryHeap(&hd, IID_PPV_ARGS(&queryHeap))) || !queryHeap)
		return false;

	D3D12_HEAP_PROPERTIES rbHeap{ D3D12_HEAP_TYPE_READBACK };
	D3D12_RESOURCE_DESC rbDesc{};
	rbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	rbDesc.Width = static_cast<uint64_t>(kSlots) * kQueriesPerSlot * sizeof(uint64_t);
	rbDesc.Height = 1;
	rbDesc.DepthOrArraySize = 1;
	rbDesc.MipLevels = 1;
	rbDesc.SampleDesc.Count = 1;
	rbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	if (FAILED(dev->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
		D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))) || !readback)
	{
		queryHeap.Reset();
		return false;
	}

	void* ptr = nullptr;
	if (FAILED(readback->Map(0, nullptr, &ptr)) || !ptr)
	{
		readback.Reset();
		queryHeap.Reset();
		return false;
	}
	mapped = static_cast<const uint64_t*>(ptr);
	ready = true;
	return true;
}

void FrameGenerator::GpuProfiler::PollAndAdvance() noexcept
{
	if (!ready || !mapped || freqHz == 0)
		return;
	slot = (slot + 1u) % kSlots;
	if (framesRecorded >= kSlots && slotValid[slot])
	{
		const uint64_t* q = mapped + static_cast<size_t>(slot) * kQueriesPerSlot;
		// Verify the GPU actually finished resolving this 4-frame-old slot (new monotonic end timestamp > start)
		if (q[5] == 0 || q[5] == lastConsumedEnd[slot] || q[5] <= q[0])
			return;
		lastConsumedEnd[slot] = q[5];
		const double tickToMs = 1000.0 / static_cast<double>(freqHz);
		auto spanMs = [tickToMs](uint64_t a, uint64_t b) noexcept -> float {
			return (b > a) ? static_cast<float>(static_cast<double>(b - a) * tickToMs) : 0.0f;
		};
		const float curTotal = spanMs(q[0], q[5]);
		const float curInterp = spanMs(q[2], q[3]);
		const float curHud = spanMs(q[0], q[1]) + spanMs(q[4], q[5]);
		if (curTotal > 0.0f && curTotal < 100.0f)
		{
			constexpr float kAlpha = 0.15f;
			if (!hasReading)
			{
				totalMs = curTotal;
				interpMs = curInterp;
				hudMs = curHud;
				hasReading = true;
			}
			else
			{
				totalMs += kAlpha * (curTotal - totalMs);
				interpMs += kAlpha * (curInterp - interpMs);
				hudMs += kAlpha * (curHud - hudMs);
			}
		}
	}
	slotValid[slot] = false;
}

void FrameGenerator::GpuProfiler::Mark(ID3D12GraphicsCommandList* cmdList, uint32_t idx) noexcept
{
	if (!ready || !cmdList || idx >= kQueriesPerSlot)
		return;
	cmdList->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * kQueriesPerSlot + idx);
}

void FrameGenerator::GpuProfiler::ResolveSlot(ID3D12GraphicsCommandList* cmdList) noexcept
{
	if (!ready || !cmdList)
		return;
	const uint32_t base = slot * kQueriesPerSlot;
	cmdList->ResolveQueryData(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, base, kQueriesPerSlot,
		readback.Get(), static_cast<uint64_t>(base) * sizeof(uint64_t));
	slotValid[slot] = true;
	++framesRecorded;
}

NGXResult FrameGenerator::Evaluate(ID3D12GraphicsCommandList* cmdList, const NGXParameter& params)
{
	D3D12QueueFence::Install(device.Get());
	DeferredDestroy::ScopedCommandList deferredScope(cmdList);
	std::vector<std::shared_ptr<DeferredDestroy::FenceSignal>> evalSignals;
	if (deferredScope.current)
	{
		std::lock_guard lock(activeSignalsMutex);
		activeGpuSignals.erase(
			std::remove_if(activeGpuSignals.begin(), activeGpuSignals.end(),
				[](const std::shared_ptr<DeferredDestroy::FenceSignal>& s) noexcept {
					return !s || s->IsGpuComplete();
				}),
			activeGpuSignals.end());
		if (std::find(activeGpuSignals.begin(), activeGpuSignals.end(), deferredScope.current) == activeGpuSignals.end())
			activeGpuSignals.push_back(deferredScope.current);
		evalSignals = activeGpuSignals;
	}
	else
	{
		evalSignals = ActiveGpuSignals();
	}
	// Bind ALL in-flight queue signals to every DeferredDestroy::Add inside Evaluate (including depth/motion/HUD
	// texture replacements) so a faster Compute queue never frees a texture still read by a busy Direct queue.
	DeferredDestroy::ScopedSignals deferredSignalsScope(std::move(evalSignals));

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
			context->stallCooldownUntilMs = 0;
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

	const bool profileThisFrame = (frameIndex == 1) && Plugin::options.gpuProfiler.load(std::memory_order_relaxed) &&
		gpuProfiler.EnsureReady(device.Get());
	if (profileThisFrame)
	{
		gpuProfiler.PollAndAdvance();
		gpuProfiler.Mark(cmdList, 0);
	}

	bool vramGuarded = vramGuardActive;
	if (frameIndex == 1)
	{
		HangWatchdog::Beat("DX12::CaptureRealFrame", true);
		vramGuarded = PollVramGuard(requestedCount);
		CaptureRealFrame(cmdList, params);
	}
	else
	{
		HangWatchdog::Beat("DX12::Interpolate");
	}
	if (profileThisFrame)
		gpuProfiler.Mark(cmdList, 1);

	const bool vramClamped = vramGuarded && vramGuardStepDown > 0u && requestedCount > 1u;
	const uint32_t effectiveCount = vramClamped
		? (requestedCount > vramGuardStepDown ? (requestedCount - vramGuardStepDown) : 1u)
		: requestedCount;

	if (frameIndex == 1)
	{
		const bool profOn = Plugin::options.gpuProfiler.load(std::memory_order_relaxed) && gpuProfiler.hasReading;
		Plugin::Report({ settings.width, settings.height, effectiveCount,
			hudDecision.activeTier != HudTier::ActiveTier::Interpolated,
			static_cast<int32_t>(hudDecision.activeTier),
			CurrentNetwork(settings.model), context == nullptr, settings.format == DXGI_FORMAT_R16G16B16A16_FLOAT,
			requestedCount, vramClamped, vramUsageMiB, vramBudgetMiB,
			hudDecision.runToneMapMatch ? hudTierPass.ActiveToneMapMode() : 0,
			profOn, gpuProfiler.totalMs, gpuProfiler.interpMs, gpuProfiler.hudMs });
	}

	const auto targetDesc = (output ? output : backbuffer)->GetDesc();

	if (profileThisFrame)
		gpuProfiler.Mark(cmdList, 2);

	// OptiScaler FlipMeter sets DLSSG.IsRecording=1 when recording generated frames on the present thread;
	// skip the bridge's 50ms CPU paceWait there so FlipMeter's _averageFrameQpc pacing stays undistorted.
	const bool callerRecording = params.GetOr<uint32_t>("DLSSG.IsRecording", 0u) != 0u;

	// When VRAM guard steps down 3x-6x, skip extra generated frames gracefully by showing the real frame
	const auto result = (frameIndex <= effectiveCount)
		? Interpolate(cmdList, output, targetDesc, frameIndex, effectiveCount, interpolationTime, callerRecording)
		: InterpolateResult::Skipped;
	if (frameIndex == 1 && currentExecRoute == ActiveExecRoute::D3D12WmmaFallback && D3D12VulkanWmma::NeedsRecovery())
	{
		if (result == InterpolateResult::Interpolated)
			D3D12VulkanWmma::RecordCleanFallbackFrame();
		else if (result == InterpolateResult::Failed)
			D3D12VulkanWmma::ResetCleanFallbackFrames();
	}
	HangWatchdog::EndFrame();

	if (profileThisFrame)
	{
		gpuProfiler.Mark(cmdList, 3);
		gpuProfiler.Mark(cmdList, 4);
	}

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

			if (currentExecRoute == ActiveExecRoute::VulkanWmma)
				Log::Info("WMMA hardware route: {} Vulkan VK_KHR_cooperative_matrix ({}x{}x{}) batches executed ({} hardware WMMA shaders, {} fallbacks, paceWait {:.2f} ms, slotWait {:.2f} ms, skipPace={})",
					vkWmma.batches, vkWmma.tileM, vkWmma.tileN, vkWmma.tileK, vkWmma.coopLayers, vkWmma.fallbacks,
					vkWmma.avgCpuWaitMs, vkWmma.avgSlotWaitMs, vkWmma.lastSkipCpuPaceWait ? "true" : "false");
			else
				Log::Info("WMMA D3D12 fallback ({}): running stock D3D12 DXIL", currentExecRouteReason);
		}
	}

	// Log whenever the actual per-frame execution route changes (e.g. Vulkan WMMA <-> D3D12 fallback when HUD composite toggles).
	if (result == InterpolateResult::Interpolated && frameIndex == 1 && currentExecRoute != ActiveExecRoute::None &&
		currentExecRoute != lastExecRoute)
	{
		if (lastExecRoute != ActiveExecRoute::None)
		{
			auto routeLabel = [](ActiveExecRoute r) -> const char* {
				switch (r)
				{
				case ActiveExecRoute::VulkanWmma: return "Vulkan hardware WMMA";
				case ActiveExecRoute::D3D12WmmaFallback: return "D3D12 DXIL fallback";
				case ActiveExecRoute::D3D12Dxil: return "D3D12 DXIL";
				case ActiveExecRoute::Cuda: return "CUDA tensor cores";
				default: return "none";
				}
			};
			Log::Info("Active FG execution route changed: {} -> {} (reason: {})",
				routeLabel(lastExecRoute), routeLabel(currentExecRoute), currentExecRouteReason);
		}
		lastExecRoute = currentExecRoute;
	}

	// Without a usable pair (first frame, reset) the real frame is shown again, that's not an error
	if (output)
	{
		if (result == InterpolateResult::Interpolated)
		{
			// Tier 2 (HUD-Diff Composite) or Tier 3 (Temporal Static-HUD Mask) post-interpolation composite pass.
			const bool runTier2 = hudTierPass.Available() && hudDecision.runHudDiffComposite;
			const bool runTier3 = hudTierPass.Available() && hudDecision.runStaticHudComposite;
			if (runTier2 || runTier3)
			{
				const Frame& curFrame = context->frames[context->current];
				const bool debugMagenta = Plugin::options.hudDebugMagenta.load(std::memory_order_relaxed);
				// When directOutput is active, XeFG wrote directly into `output` — composite in-place (zero extra CopyResource).
				ID3D12Resource* compositeTarget = context->directOutput
					? (HudTierPass::IsUavCompatibleFormat(output->GetDesc().Format) ? output : nullptr)
					: (context->outputs[frameIndex - 1] != nullptr
						? context->outputs[frameIndex - 1].Get()
						: (HudTierPass::IsUavCompatibleFormat(output->GetDesc().Format) ? output : nullptr));
				if (compositeTarget && curFrame.backbuffer)
				{
					ID3D12Resource* compositeHudless = (hudDecision.runToneMapMatch && curFrame.matchedHudless)
						? curFrame.matchedHudless : curFrame.hudless.Get();
					if (runTier2 && runTier3 && compositeHudless && hudTierPass.HasHybridComposite())
					{
						// Single fused compute pass for Tier 3 Hybrid (1/2 VRAM bandwidth, 0 intermediate UAV barrier).
						hudTierPass.RecordHybridHudComposite(cmdList, curFrame.backbuffer.Get(), XeFGInputState,
							compositeHudless, XeFGInputState, compositeTarget, OutputState,
							frameIndex == 1, debugMagenta, hudDecision.runToneMapMatch && curFrame.matchedHudless != nullptr);
					}
					else
					{
						if (runTier2 && compositeHudless)
						{
							hudTierPass.RecordHudDiffComposite(cmdList, curFrame.backbuffer.Get(), XeFGInputState,
								compositeHudless, XeFGInputState, compositeTarget, OutputState,
								frameIndex == 1, debugMagenta, hudDecision.runToneMapMatch && curFrame.matchedHudless != nullptr);
						}
						// In Tier 3 Hybrid, Static HUD Mask runs alongside Tone-Map Match to lock fixed UI (crosshairs/text)
						// sharp, and takes over completely when HUDLess is missing or Tone-Map Match is unavailable.
						if (runTier3)
						{
							hudTierPass.RecordStaticHudComposite(cmdList, curFrame.backbuffer.Get(), XeFGInputState,
								compositeTarget, OutputState, debugMagenta);
						}
					}
					if (compositeTarget != output)
						context->directOutput = false;
				}
			}

			// Opt-in split-screen debug comparison (0% overhead when disabled)
			if (Plugin::options.hudSplitCompare.load(std::memory_order_relaxed) && hudTierPass.Available() && context)
			{
				const Frame& curFrame = context->frames[context->current];
				ID3D12Resource* splitTarget = context->directOutput
					? (HudTierPass::IsUavCompatibleFormat(output->GetDesc().Format) ? output : nullptr)
					: (context->outputs[frameIndex - 1] != nullptr
						? context->outputs[frameIndex - 1].Get()
						: (HudTierPass::IsUavCompatibleFormat(output->GetDesc().Format) ? output : nullptr));
				if (splitTarget && curFrame.backbuffer)
				{
					hudTierPass.RecordSplitCompare(cmdList, curFrame.backbuffer.Get(), XeFGInputState,
						splitTarget, OutputState, Plugin::options.hudSplitPos.load(std::memory_order_relaxed));
					if (splitTarget != output)
						context->directOutput = false;
				}
			}

			if (!context->directOutput && context->outputs[frameIndex - 1])
				Copy(cmdList, output, OutputState, context->outputs[frameIndex - 1].Get(), OutputState);
		}
		else
		{
			Copy(cmdList, output, OutputState, backbuffer, BackbufferState);
		}
	}

	if (profileThisFrame)
	{
		gpuProfiler.Mark(cmdList, 5);
		gpuProfiler.ResolveSlot(cmdList);
	}

	return result == InterpolateResult::Failed ? NGX_FAIL : NGX_SUCCESS;
}
