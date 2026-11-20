// D3D12 -> Vulkan VK_KHR_cooperative_matrix (hardware WMMA INT8) bridge for AMD RDNA 4 / RDNA 3.
// See D3D12VulkanWmma.h for the architecture overview.
#include "../D3D12VulkanWmma.h"

#include "../FrameGenerator.h"
#include "../Log.h"
#include "../MfgPort/Cuda/cuda_route.h"
#include "../VulkanPrograms.h"
#include "../XeFG.h"
#include "xvk.h"
#include "VulkanCoopMat.h"
#include "VulkanHost.h"
#include "VulkanPolicy.h"

#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace D3D12VulkanWmma
{
namespace
{
using Microsoft::WRL::ComPtr;

constexpr uint32_t kRingSize = 4;
constexpr uint32_t kMaxFrames = FrameGenerator::MaxFrames;
constexpr int kExecuteCommandListsSlot = 10;

HMODULE SelfModule() noexcept
{
	HMODULE mod = nullptr;
	GetModuleHandleExW(
		GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		reinterpret_cast<LPCWSTR>(&SelfModule), &mod);
	return mod;
}

std::wstring EnvString(const wchar_t* name)
{
	wchar_t value[64]{};
	const DWORD len = GetEnvironmentVariableW(name, value, static_cast<DWORD>(std::size(value)));
	return (len && len < std::size(value)) ? std::wstring(value, len) : std::wstring();
}

VkFormat DxgiToVkFormat(DXGI_FORMAT fmt) noexcept
{
	switch (fmt)
	{
	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_TYPELESS:
		return VK_FORMAT_R8G8B8A8_UNORM;
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
		return VK_FORMAT_R8G8B8A8_SRGB;
	case DXGI_FORMAT_B8G8R8A8_UNORM:
	case DXGI_FORMAT_B8G8R8A8_TYPELESS:
		return VK_FORMAT_B8G8R8A8_UNORM;
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
		return VK_FORMAT_B8G8R8A8_SRGB;
	case DXGI_FORMAT_R10G10B10A2_UNORM:
	case DXGI_FORMAT_R10G10B10A2_TYPELESS:
		return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
	case DXGI_FORMAT_R16G16B16A16_FLOAT:
	case DXGI_FORMAT_R16G16B16A16_TYPELESS:
		return VK_FORMAT_R16G16B16A16_SFLOAT;
	case DXGI_FORMAT_R11G11B10_FLOAT:
		return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
	case DXGI_FORMAT_R32_FLOAT:
	case DXGI_FORMAT_R32_TYPELESS:
		return VK_FORMAT_R32_SFLOAT;
	case DXGI_FORMAT_R16_FLOAT:
		return VK_FORMAT_R16_SFLOAT;
	case DXGI_FORMAT_R16_UNORM:
	case DXGI_FORMAT_R16_TYPELESS:
		return VK_FORMAT_R16_UNORM;
	case DXGI_FORMAT_R16G16_FLOAT:
	case DXGI_FORMAT_R16G16_TYPELESS:
		return VK_FORMAT_R16G16_SFLOAT;
	case DXGI_FORMAT_R32G32_FLOAT:
	case DXGI_FORMAT_R32G32_TYPELESS:
		return VK_FORMAT_R32G32_SFLOAT;
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
	case DXGI_FORMAT_R32G32B32A32_TYPELESS:
		return VK_FORMAT_R32G32B32A32_SFLOAT;
	default:
		return VK_FORMAT_UNDEFINED;
	}
}

struct SharedImport
{
	VkImage vkImage = VK_NULL_HANDLE;
	VkDeviceMemory vkMemory = VK_NULL_HANDLE;
	ComPtr<ID3D12Resource> xvkResource;
	VkFormat format = VK_FORMAT_UNDEFINED;
	uint32_t width = 0;
	uint32_t height = 0;
	bool isOutput = false;
};

struct OutputCopy
{
	ComPtr<ID3D12Resource> sharedOutput;
	ComPtr<ID3D12Resource> gameOutput;
	D3D12_RESOURCE_STATES outputState = D3D12_RESOURCE_STATE_COMMON;
};

struct PendingBatch
{
	ID3D12GraphicsCommandList* cmdList = nullptr;
	ContextSettings settings{};
	ComPtr<ID3D12Resource> prevBackbuffer, prevHudless, prevDepth, prevMvecs;
	ComPtr<ID3D12Resource> nextBackbuffer, nextHudless, nextDepth, nextMvecs;
	xefgd_frame_t prevData{};
	xefgd_frame_t nextData{};
	ComPtr<ID3D12Resource> sharedOutputs[kMaxFrames];
	xefgd_2d_t outputBases[kMaxFrames]{};
	uint32_t frameCount = 1;
	std::vector<OutputCopy> copies;
};

struct PostCopyRing
{
	ComPtr<ID3D12CommandAllocator> alloc[kRingSize];
	ComPtr<ID3D12GraphicsCommandList> list[kRingSize];
	uint64_t doneFence[kRingSize]{};
	uint32_t cursor = 0;
	bool ready = false;
};

#define BRIDGE_VK_DEV_FNS(X)                     \
	X(vkDestroyDevice)                           \
	X(vkGetDeviceQueue)                          \
	X(vkCreateCommandPool)                       \
	X(vkDestroyCommandPool)                      \
	X(vkAllocateCommandBuffers)                  \
	X(vkResetCommandBuffer)                      \
	X(vkBeginCommandBuffer)                      \
	X(vkEndCommandBuffer)                        \
	X(vkQueueSubmit)                             \
	X(vkQueueWaitIdle)                           \
	X(vkCreateImage)                             \
	X(vkDestroyImage)                            \
	X(vkGetImageMemoryRequirements)              \
	X(vkAllocateMemory)                          \
	X(vkFreeMemory)                              \
	X(vkBindImageMemory)                         \
	X(vkGetMemoryWin32HandlePropertiesKHR)       \
	X(vkCreateSemaphore)                         \
	X(vkDestroySemaphore)                        \
	X(vkImportSemaphoreWin32HandleKHR)           \
	X(vkWaitSemaphores)                          \
	X(vkCmdCopyImage)                            \
	X(vkCmdPipelineBarrier)

struct Bridge
{
	std::mutex mutex;
	std::atomic<bool> active{ false };
	bool attempted = false;
	std::string reason = "not_attempted";

	// D3D12 side
	ComPtr<ID3D12Device> d3dDevice;
	ComPtr<ID3D12Device> nativeD3DDevice;
	ComPtr<ID3D12Fence> sharedFence;
	ComPtr<ID3D12Fence> nativeSharedFence;
	HANDLE fenceEvent = nullptr;
	std::atomic<uint64_t> fenceValue{ 0 };
	PostCopyRing directRing;
	PostCopyRing computeRing;
	PostCopyRing nativeDirectRing;
	PostCopyRing nativeComputeRing;

	// Vulkan side
	HMODULE vulkanDll = nullptr;
	PFN_vkGetInstanceProcAddr gipa = nullptr;
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t queueFamily = 0;
	VkPhysicalDeviceMemoryProperties memProps{};
	VkCommandPool cmdPool = VK_NULL_HANDLE;
	VkCommandBuffer cmdBufs[kRingSize]{};
	uint64_t vkSlotDoneFence[kRingSize]{};
	uint32_t vkCursor = 0;
	VkSemaphore sharedSemaphore = VK_NULL_HANDLE;

#define BRIDGE_FIELD(name) PFN_##name name = nullptr;
	BRIDGE_VK_DEV_FNS(BRIDGE_FIELD)
#undef BRIDGE_FIELD

	// xvk + XeFG Vulkan context
	ComPtr<ID3D12Device> xvkDevice;
	ComPtr<ID3D12CommandAllocator> xvkAlloc;
	ComPtr<ID3D12GraphicsCommandList> xvkList;
	xefgd_handle_t xefgContext = nullptr;
	ContextSettings contextSettings{};
	bool contextFailed = false;
	ContextSettings failedContextSettings{};
	uint32_t tileM = 0;
	uint32_t tileN = 0;
	uint32_t tileK = 0;
	uint32_t coopLayers = 0;

	// Cached D3D12 -> Vulkan shared imports keyed by ID3D12Resource*
	std::unordered_map<ID3D12Resource*, SharedImport> imports;

	// Pending batches keyed by ID3D12GraphicsCommandList*
	std::mutex pendingMutex;
	std::atomic<uint32_t> pendingCount{ 0 };
	std::vector<PendingBatch> pending;

	// Telemetry
	std::atomic<uint64_t> batches{ 0 };
	std::atomic<uint64_t> fallbacks{ 0 };
};

// Kept alive for process lifetime (same policy as VulkanRuntime.cpp: never call into a torn-down GPU driver in CRT exit).
Bridge& gBridge = *new Bridge;

// ---- ID3D12CommandQueue::ExecuteCommandLists vtable hook ----------------------------------------------
using ExecuteCommandListsFn = void (STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);

struct QueueHookEntry
{
	void** vtable = nullptr;
	ExecuteCommandListsFn original = nullptr;
};

std::mutex gQueueHookMutex;
std::array<QueueHookEntry, 8> gQueueHooks{};
std::atomic<size_t> gQueueHookCount{ 0 };

ExecuteCommandListsFn FindOriginalExecute(ID3D12CommandQueue* queue) noexcept
{
	void** vt = *reinterpret_cast<void***>(queue);
	const size_t n = gQueueHookCount.load(std::memory_order_acquire);
	for (size_t i = 0; i < n; ++i)
	{
		if (gQueueHooks[i].vtable == vt)
			return gQueueHooks[i].original;
	}
	return nullptr;
}

bool PatchQueueVtable(ID3D12CommandQueue* queue, ExecuteCommandListsFn hookFn) noexcept
{
	if (!queue)
		return false;
	void** vt = *reinterpret_cast<void***>(queue);
	if (!vt || !vt[kExecuteCommandListsSlot])
		return false;

	std::lock_guard lock(gQueueHookMutex);
	const size_t n = gQueueHookCount.load(std::memory_order_relaxed);
	for (size_t i = 0; i < n; ++i)
	{
		if (gQueueHooks[i].vtable == vt)
			return true;
	}
	if (n >= gQueueHooks.size())
		return false;

	void* orig = vt[kExecuteCommandListsSlot];
	if (orig == reinterpret_cast<void*>(hookFn))
		return true;

	// Publish the original pointer before patching the vtable slot so concurrent ExecuteCommandLists callers
	// on other threads never observe a null FindOriginalExecute result and drop GPU work.
	gQueueHooks[n] = { vt, reinterpret_cast<ExecuteCommandListsFn>(orig) };
	gQueueHookCount.store(n + 1, std::memory_order_release);

	DWORD oldProtect = 0;
	if (!VirtualProtect(&vt[kExecuteCommandListsSlot], sizeof(void*), PAGE_EXECUTE_READWRITE, &oldProtect))
	{
		gQueueHookCount.store(n, std::memory_order_release);
		return false;
	}

	InterlockedExchangePointer(&vt[kExecuteCommandListsSlot], reinterpret_cast<void*>(hookFn));
	DWORD ignored = 0;
	VirtualProtect(&vt[kExecuteCommandListsSlot], sizeof(void*), oldProtect, &ignored);
	return true;
}

// Forward declaration
void SubmitBatchOnQueue(ID3D12CommandQueue* queue, ExecuteCommandListsFn origExec, PendingBatch& batch) noexcept;

void STDMETHODCALLTYPE HookExecuteCommandLists(
	ID3D12CommandQueue* self, UINT numCommandLists, ID3D12CommandList* const* ppCommandLists)
{
	ExecuteCommandListsFn orig = FindOriginalExecute(self);
	if (!orig)
		return;

	if (gBridge.pendingCount.load(std::memory_order_acquire) == 0 || !numCommandLists || !ppCommandLists)
	{
		orig(self, numCommandLists, ppCommandLists);
		return;
	}

	std::vector<PendingBatch> matched;
	{
		std::lock_guard lock(gBridge.pendingMutex);
		if (!gBridge.pending.empty())
		{
			for (UINT i = 0; i < numCommandLists; ++i)
			{
				for (auto it = gBridge.pending.begin(); it != gBridge.pending.end();)
				{
					if (static_cast<ID3D12CommandList*>(it->cmdList) == ppCommandLists[i])
					{
						matched.push_back(std::move(*it));
						it = gBridge.pending.erase(it);
					}
					else
					{
						++it;
					}
				}
			}
			gBridge.pendingCount.store(static_cast<uint32_t>(gBridge.pending.size()), std::memory_order_release);
		}
	}

	// Execute the game's command list(s) first so CaptureRealFrame copies land in our shared D3D12 input textures.
	orig(self, numCommandLists, ppCommandLists);

	for (auto& batch : matched)
		SubmitBatchOnQueue(self, orig, batch);
}

bool EnsurePostCopyRing(ID3D12Device* dev, D3D12_COMMAND_LIST_TYPE type, PostCopyRing& ring) noexcept
{
	if (ring.ready)
		return true;

	for (uint32_t i = 0; i < kRingSize; ++i)
	{
		if (FAILED(dev->CreateCommandAllocator(type, IID_PPV_ARGS(&ring.alloc[i]))))
			return false;
		if (FAILED(dev->CreateCommandList(0, type, ring.alloc[i].Get(), nullptr, IID_PPV_ARGS(&ring.list[i]))))
			return false;
		ring.list[i]->Close();
		ring.doneFence[i] = 0;
	}
	ring.cursor = 0;
	ring.ready = true;
	return true;
}

void WaitFenceCpu(ID3D12Fence* fence, HANDLE ev, uint64_t target) noexcept
{
	if (!target || !fence || fence->GetCompletedValue() >= target)
		return;
	if (SUCCEEDED(fence->SetEventOnCompletion(target, ev)))
		WaitForSingleObject(ev, 5000);
}

uint32_t FindMemoryType(const VkPhysicalDeviceMemoryProperties& props, uint32_t typeBits, VkMemoryPropertyFlags required) noexcept
{
	for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
	{
		if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & required) == required)
			return i;
	}
	for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
	{
		if (typeBits & (1u << i))
			return i;
	}
	return UINT32_MAX;
}

void DestroyImport(Bridge& b, SharedImport& imp) noexcept
{
	imp.xvkResource.Reset();
	if (imp.vkImage)
	{
		b.vkDestroyImage(b.device, imp.vkImage, nullptr);
		imp.vkImage = VK_NULL_HANDLE;
	}
	if (imp.vkMemory)
	{
		b.vkFreeMemory(b.device, imp.vkMemory, nullptr);
		imp.vkMemory = VK_NULL_HANDLE;
	}
}

// Imports a D3D12_HEAP_FLAG_SHARED texture into Vulkan via VK_KHR_external_memory_win32 and creates
// its companion xvk texture on b.xvkDevice. Caller holds b.mutex.
SharedImport* GetOrImportLocked(Bridge& b, ID3D12Resource* d3dRes, bool isOutput) noexcept
{
	if (!d3dRes)
		return nullptr;

	auto it = b.imports.find(d3dRes);
	if (it != b.imports.end())
		return &it->second;

	const D3D12_RESOURCE_DESC desc = d3dRes->GetDesc();
	if (!CanShareTexture(desc))
		return nullptr;

	const VkFormat vkFmt = DxgiToVkFormat(desc.Format);
	if (vkFmt == VK_FORMAT_UNDEFINED)
		return nullptr;

	HANDLE ntHandle = nullptr;
	if (FAILED(b.d3dDevice->CreateSharedHandle(d3dRes, nullptr, GENERIC_ALL, nullptr, &ntHandle)) || !ntHandle)
		return nullptr;

	VkExternalMemoryImageCreateInfo extInfo{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
	extInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;

	VkImageCreateInfo imgInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &extInfo };
	imgInfo.imageType = VK_IMAGE_TYPE_2D;
	imgInfo.format = vkFmt;
	imgInfo.extent = { static_cast<uint32_t>(desc.Width), desc.Height, 1u };
	imgInfo.mipLevels = desc.MipLevels ? desc.MipLevels : 1u;
	imgInfo.arrayLayers = desc.DepthOrArraySize ? desc.DepthOrArraySize : 1u;
	imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
		(isOutput ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
	imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	SharedImport imp{};
	imp.format = vkFmt;
	imp.width = static_cast<uint32_t>(desc.Width);
	imp.height = desc.Height;
	imp.isOutput = isOutput;

	if (b.vkCreateImage(b.device, &imgInfo, nullptr, &imp.vkImage) != VK_SUCCESS)
	{
		CloseHandle(ntHandle);
		return nullptr;
	}

	VkMemoryWin32HandlePropertiesKHR handleProps{ VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
	if (b.vkGetMemoryWin32HandlePropertiesKHR(
			b.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, ntHandle, &handleProps) != VK_SUCCESS)
	{
		CloseHandle(ntHandle);
		DestroyImport(b, imp);
		return nullptr;
	}

	VkMemoryRequirements memReq{};
	b.vkGetImageMemoryRequirements(b.device, imp.vkImage, &memReq);

	const uint32_t memType = FindMemoryType(
		b.memProps, memReq.memoryTypeBits & handleProps.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (memType == UINT32_MAX)
	{
		CloseHandle(ntHandle);
		DestroyImport(b, imp);
		return nullptr;
	}

	VkMemoryDedicatedAllocateInfo dedInfo{ VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
	dedInfo.image = imp.vkImage;

	VkImportMemoryWin32HandleInfoKHR importInfo{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR, &dedInfo };
	importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
	importInfo.handle = ntHandle;

	VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &importInfo };
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = memType;

	const VkResult allocRes = b.vkAllocateMemory(b.device, &allocInfo, nullptr, &imp.vkMemory);
	// D3D12_RESOURCE_BIT duplicates the handle in the driver; close our NT handle regardless of outcome.
	CloseHandle(ntHandle);

	if (allocRes != VK_SUCCESS || b.vkBindImageMemory(b.device, imp.vkImage, imp.vkMemory, 0) != VK_SUCCESS)
	{
		DestroyImport(b, imp);
		return nullptr;
	}

	// Create matching xvk texture so XeFG's Vulkan runtime sees native GENERAL-layout xvk images.
	D3D12_RESOURCE_DESC xvkDesc = desc;
	xvkDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	xvkDesc.Flags = isOutput ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
	const D3D12_HEAP_PROPERTIES heap{ D3D12_HEAP_TYPE_DEFAULT };
	const D3D12_RESOURCE_STATES xvkState = isOutput
		? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
		: D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

	if (FAILED(b.xvkDevice->CreateCommittedResource(
			&heap, D3D12_HEAP_FLAG_NONE, &xvkDesc, xvkState, nullptr, IID_PPV_ARGS(&imp.xvkResource))) ||
		!imp.xvkResource || !xvk::ImageOf(imp.xvkResource.Get()))
	{
		DestroyImport(b, imp);
		return nullptr;
	}

	auto [inserted, _] = b.imports.emplace(d3dRes, std::move(imp));
	return &inserted->second;
}

void RecordImageBarrier(const Bridge& b, VkCommandBuffer cmd, VkImage img,
	VkImageLayout oldLayout, VkImageLayout newLayout,
	VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
	VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) noexcept
{
	VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
	barrier.srcAccessMask = srcAccess;
	barrier.dstAccessMask = dstAccess;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = img;
	barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	b.vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void RecordCopyImage(const Bridge& b, VkCommandBuffer cmd, VkImage src, VkImage dst, uint32_t w, uint32_t h) noexcept
{
	VkImageCopy region{};
	region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	region.extent = { w, h, 1 };
	b.vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

void RecordD3D12Copy(ID3D12GraphicsCommandList* list, ID3D12Resource* dst, D3D12_RESOURCE_STATES dstState,
	ID3D12Resource* src, D3D12_RESOURCE_STATES srcState) noexcept
{
	D3D12_RESOURCE_BARRIER barriers[2]{};
	UINT count = 0;
	if (dstState != D3D12_RESOURCE_STATE_COPY_DEST)
	{
		barriers[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barriers[count].Transition = { dst, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, dstState, D3D12_RESOURCE_STATE_COPY_DEST };
		++count;
	}
	if (srcState != D3D12_RESOURCE_STATE_COPY_SOURCE)
	{
		barriers[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barriers[count].Transition = { src, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, srcState, D3D12_RESOURCE_STATE_COPY_SOURCE };
		++count;
	}
	if (count)
		list->ResourceBarrier(count, barriers);

	list->CopyResource(dst, src);

	for (UINT i = 0; i < count; ++i)
		std::swap(barriers[i].Transition.StateBefore, barriers[i].Transition.StateAfter);
	if (count)
		list->ResourceBarrier(count, barriers);
}

void SubmitBatchOnQueue(ID3D12CommandQueue* queue, ExecuteCommandListsFn origExec, PendingBatch& batch) noexcept
{
	std::lock_guard lock(gBridge.mutex);
	if (!gBridge.active.load(std::memory_order_acquire) || !gBridge.xefgContext)
		return;

	auto importOrNull = [&](ID3D12Resource* r, bool isOut) -> SharedImport* {
		return r ? GetOrImportLocked(gBridge, r, isOut) : nullptr;
	};

	SharedImport* prevBB = importOrNull(batch.prevBackbuffer.Get(), false);
	SharedImport* prevD  = importOrNull(batch.prevDepth.Get(), false);
	SharedImport* prevMV = importOrNull(batch.prevMvecs.Get(), false);
	SharedImport* prevHL = importOrNull(batch.prevHudless.Get(), false);

	SharedImport* nextBB = importOrNull(batch.nextBackbuffer.Get(), false);
	SharedImport* nextD  = importOrNull(batch.nextDepth.Get(), false);
	SharedImport* nextMV = importOrNull(batch.nextMvecs.Get(), false);
	SharedImport* nextHL = importOrNull(batch.nextHudless.Get(), false);

	if (!prevBB || !prevD || !prevMV || !nextBB || !nextD || !nextMV)
	{
		gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
		return;
	}

	SharedImport* outImports[kMaxFrames]{};
	ID3D12Resource* xvkOutPtrs[kMaxFrames]{};
	for (uint32_t i = 0; i < batch.frameCount; ++i)
	{
		outImports[i] = importOrNull(batch.sharedOutputs[i].Get(), true);
		if (!outImports[i])
		{
			gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		xvkOutPtrs[i] = outImports[i]->xvkResource.Get();
	}

	// Select Vulkan command buffer slot and ensure previous submission on this slot finished.
	const uint32_t vkSlot = gBridge.vkCursor++ % kRingSize;
	if (gBridge.vkSlotDoneFence[vkSlot])
	{
		VkSemaphoreWaitInfo waitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
		waitInfo.semaphoreCount = 1;
		waitInfo.pSemaphores = &gBridge.sharedSemaphore;
		waitInfo.pValues = &gBridge.vkSlotDoneFence[vkSlot];
		gBridge.vkWaitSemaphores(gBridge.device, &waitInfo, 5'000'000'000ull);
	}

	VkCommandBuffer cmd = gBridge.cmdBufs[vkSlot];
	gBridge.vkResetCommandBuffer(cmd, 0);
	VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (gBridge.vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
	{
		gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
		return;
	}

	// Open xvk recording BEFORE copyInput so any pendingInit_ (UNDEFINED -> GENERAL)
	// layout transitions on newly created xvk resources flush before we copy frame pixels into them.
	bool execOk = false;
	if (xvk::BeginRecording(gBridge.xvkDevice.Get(), cmd))
	{
		// Copy D3D12 shared inputs into xvk's GENERAL-layout textures.
		auto copyInput = [&](SharedImport* imp) {
			if (!imp)
				return;
			VkImage xvkImg = xvk::ImageOf(imp->xvkResource.Get());
			RecordImageBarrier(gBridge, cmd, imp->vkImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
			RecordImageBarrier(gBridge, cmd, xvkImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
			RecordCopyImage(gBridge, cmd, imp->vkImage, xvkImg, imp->width, imp->height);
			RecordImageBarrier(gBridge, cmd, xvkImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
		};

		copyInput(prevBB);
		copyInput(prevD);
		copyInput(prevMV);
		copyInput(prevHL);
		copyInput(nextBB);
		copyInput(nextD);
		copyInput(nextMV);
		copyInput(nextHL);

		// Record the entire XeFG batch (all 28 layers, with 17 INT8 layers on hardware VK_KHR_cooperative_matrix).
		xefgd_exec_params_t vkParams{};
		vkParams.prev = batch.prevData;
		vkParams.prev.backBuffer.pResource = prevBB->xvkResource.Get();
		vkParams.prev.depth.pResource = prevD->xvkResource.Get();
		vkParams.prev.motionVectors.pResource = prevMV->xvkResource.Get();
		vkParams.prev.hudlessColor.pResource = prevHL ? prevHL->xvkResource.Get() : nullptr;

		vkParams.next = batch.nextData;
		vkParams.next.backBuffer.pResource = nextBB->xvkResource.Get();
		vkParams.next.depth.pResource = nextD->xvkResource.Get();
		vkParams.next.motionVectors.pResource = nextMV->xvkResource.Get();
		vkParams.next.hudlessColor.pResource = nextHL ? nextHL->xvkResource.Get() : nullptr;

		vkParams.ppOutputs = xvkOutPtrs;
		vkParams.pOutputBases = batch.outputBases;
		vkParams.numInterpolatedFrames = static_cast<uint8_t>(batch.frameCount);
		vkParams.firstFrameIndex = 0;
		vkParams.frameCount = static_cast<uint8_t>(batch.frameCount);
		vkParams.uiCompositionEnabled = batch.settings.uiMode != XEFGD_UI_MODE_AUTO;

		bool listReady = false;
		if (!gBridge.xvkList)
		{
			listReady = SUCCEEDED(gBridge.xvkDevice->CreateCommandList(
				0, D3D12_COMMAND_LIST_TYPE_DIRECT, gBridge.xvkAlloc.Get(), nullptr, IID_PPV_ARGS(&gBridge.xvkList))) &&
				gBridge.xvkList != nullptr;
		}
		else
		{
			listReady = SUCCEEDED(gBridge.xvkAlloc->Reset()) &&
				SUCCEEDED(gBridge.xvkList->Reset(gBridge.xvkAlloc.Get(), nullptr));
		}

		if (listReady)
		{
			const xefgd_result_t res = XeFG::Execute(gBridge.xefgContext, gBridge.xvkList.Get(), vkParams);
			gBridge.xvkList->Close();
			execOk = (res == XEFGD_RESULT_SUCCESS);
		}
		xvk::EndRecording(gBridge.xvkDevice.Get());
	}

	if (!execOk)
	{
		gBridge.vkEndCommandBuffer(cmd);
		gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
		return;
	}

	// Copy xvk outputs back into the shared D3D12 output textures.
	for (uint32_t i = 0; i < batch.frameCount; ++i)
	{
		VkImage xvkOut = xvk::ImageOf(outImports[i]->xvkResource.Get());
		RecordImageBarrier(gBridge, cmd, xvkOut, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
		RecordImageBarrier(gBridge, cmd, outImports[i]->vkImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
		RecordCopyImage(gBridge, cmd, xvkOut, outImports[i]->vkImage, outImports[i]->width, outImports[i]->height);
		RecordImageBarrier(gBridge, cmd, outImports[i]->vkImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
	}

	if (gBridge.vkEndCommandBuffer(cmd) != VK_SUCCESS)
	{
		gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
		return;
	}

	// Match fence and post-copy command list device layer (wrapper vs unwrapped native) to the queue.
	ComPtr<ID3D12Device> queueDev;
	const bool isNativeQueue = SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&queueDev))) && queueDev &&
		gBridge.nativeD3DDevice && queueDev.Get() == gBridge.nativeD3DDevice.Get() &&
		gBridge.nativeD3DDevice.Get() != gBridge.d3dDevice.Get();
	ID3D12Fence* useFence = (isNativeQueue && gBridge.nativeSharedFence)
		? gBridge.nativeSharedFence.Get()
		: gBridge.sharedFence.Get();
	ID3D12Device* ringDev = (isNativeQueue && gBridge.nativeD3DDevice)
		? gBridge.nativeD3DDevice.Get()
		: gBridge.d3dDevice.Get();

	// 1) D3D12 signals Vulkan that CaptureRealFrame copies are complete.
	const uint64_t waitVal = gBridge.fenceValue.fetch_add(1, std::memory_order_acq_rel) + 1;
	const uint64_t signalVal = gBridge.fenceValue.fetch_add(1, std::memory_order_acq_rel) + 1;
	if (FAILED(queue->Signal(useFence, waitVal)))
	{
		gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
		return;
	}

	// 2) Vulkan runs the entire XeFG batch (all 17 hardware WMMA INT8 layers) in one submission.
	const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	VkTimelineSemaphoreSubmitInfo timelineInfo{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
	timelineInfo.waitSemaphoreValueCount = 1;
	timelineInfo.pWaitSemaphoreValues = &waitVal;
	timelineInfo.signalSemaphoreValueCount = 1;
	timelineInfo.pSignalSemaphoreValues = &signalVal;

	VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO, &timelineInfo };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &gBridge.sharedSemaphore;
	submitInfo.pWaitDstStageMask = &waitStage;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &gBridge.sharedSemaphore;

	if (gBridge.vkQueueSubmit(gBridge.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
	{
		gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	gBridge.vkSlotDoneFence[vkSlot] = signalVal;

	// 3) D3D12 waits on Vulkan completion (GPU-side wait, zero CPU stall).
	queue->Wait(useFence, signalVal);

	// 4) Copy generated shared outputs into the game's output texture(s) on D3D12.
	if (!batch.copies.empty())
	{
		const D3D12_COMMAND_LIST_TYPE qType = queue->GetDesc().Type;
		const D3D12_COMMAND_LIST_TYPE useType = (qType == D3D12_COMMAND_LIST_TYPE_COMPUTE)
			? D3D12_COMMAND_LIST_TYPE_COMPUTE
			: D3D12_COMMAND_LIST_TYPE_DIRECT;
		PostCopyRing& ring = isNativeQueue
			? ((useType == D3D12_COMMAND_LIST_TYPE_COMPUTE) ? gBridge.nativeComputeRing : gBridge.nativeDirectRing)
			: ((useType == D3D12_COMMAND_LIST_TYPE_COMPUTE) ? gBridge.computeRing : gBridge.directRing);

		if (EnsurePostCopyRing(ringDev, useType, ring))
		{
			const uint32_t slot = ring.cursor++ % kRingSize;
			WaitFenceCpu(useFence, gBridge.fenceEvent, ring.doneFence[slot]);
			if (SUCCEEDED(ring.alloc[slot]->Reset()) &&
				SUCCEEDED(ring.list[slot]->Reset(ring.alloc[slot].Get(), nullptr)))
			{
				for (const auto& cp : batch.copies)
					RecordD3D12Copy(ring.list[slot].Get(), cp.gameOutput.Get(), cp.outputState, cp.sharedOutput.Get(), cp.outputState);
				ring.list[slot]->Close();

				ID3D12CommandList* postLists[1] = { ring.list[slot].Get() };
				origExec(queue, 1, postLists);

				const uint64_t doneVal = gBridge.fenceValue.fetch_add(1, std::memory_order_acq_rel) + 1;
				queue->Signal(useFence, doneVal);
				ring.doneFence[slot] = doneVal;
			}
		}
	}

	gBridge.batches.fetch_add(1, std::memory_order_relaxed);
}

ComPtr<ID3D12Device> UnwrapNativeDevice(ID3D12Device* device) noexcept
{
	ComPtr<ID3D12Fence> probeFence;
	ComPtr<ID3D12Device> native;
	if (device && SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&probeFence))) &&
		SUCCEEDED(probeFence->GetDevice(IID_PPV_ARGS(&native))) && native)
	{
		return native;
	}
	return device;
}

bool HasExtension(const std::vector<VkExtensionProperties>& exts, const char* name, uint32_t* specOut = nullptr) noexcept
{
	for (const auto& e : exts)
	{
		if (std::strcmp(e.extensionName, name) == 0)
		{
			if (specOut)
				*specOut = e.specVersion;
			return true;
		}
	}
	return false;
}
} // namespace

bool CanShareTexture(const D3D12_RESOURCE_DESC& desc) noexcept
{
	if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
		return false;
	if (desc.SampleDesc.Count > 1)
		return false;
	if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
		return false;
	return DxgiToVkFormat(desc.Format) != VK_FORMAT_UNDEFINED;
}

bool Active() noexcept
{
	return gBridge.active.load(std::memory_order_acquire);
}

bool WantsSharedTextures() noexcept
{
	return gBridge.active.load(std::memory_order_acquire);
}

bool Install(ID3D12Device* d3dDevice, std::string& reason) noexcept
{
	reason.clear();
	std::lock_guard lock(gBridge.mutex);
	if (gBridge.attempted)
	{
		reason = gBridge.reason;
		return gBridge.active.load(std::memory_order_acquire);
	}
	gBridge.attempted = true;

	auto fail = [&](const char* code) -> bool {
		gBridge.reason = code;
		reason = code;
		gBridge.active.store(false, std::memory_order_release);
		return false;
	};

	if (VulkanPolicy::IsOffWord(EnvString(L"XEFG_D3D12_VULKAN_WMMA")))
		return fail("environment_off");

	if (!d3dDevice)
		return fail("no_d3d12_device");

	if (!VulkanPrograms::Installed().Usable())
		return fail("no_vulkan_models");

	struct ScopedVulkanHostBypass
	{
		ScopedVulkanHostBypass() noexcept { VulkanHost::SetInternalBridgeBypass(true); }
		~ScopedVulkanHostBypass() noexcept { VulkanHost::SetInternalBridgeBypass(false); }
	} bypassHost;

	// Load vulkan-1.dll dynamically so we have zero link-time dependency on Vulkan.
	gBridge.vulkanDll = LoadLibraryW(L"vulkan-1.dll");
	if (!gBridge.vulkanDll)
		return fail("no_vulkan_loader");

	gBridge.gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(gBridge.vulkanDll, "vkGetInstanceProcAddr"));
	if (!gBridge.gipa)
		return fail("no_gipa");

	const auto createInstance = reinterpret_cast<PFN_vkCreateInstance>(gBridge.gipa(VK_NULL_HANDLE, "vkCreateInstance"));
	if (!createInstance)
		return fail("no_create_instance");

	VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
	appInfo.pApplicationName = "dlssg-to-xefg-wmma";
	appInfo.apiVersion = VK_API_VERSION_1_3;

	VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	instInfo.pApplicationInfo = &appInfo;

	if (createInstance(&instInfo, nullptr, &gBridge.instance) != VK_SUCCESS || !gBridge.instance)
		return fail("create_instance_failed");

	const auto enumPhys = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gBridge.gipa(gBridge.instance, "vkEnumeratePhysicalDevices"));
	const auto getProps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gBridge.gipa(gBridge.instance, "vkGetPhysicalDeviceProperties2"));
	const auto getFeats2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(gBridge.gipa(gBridge.instance, "vkGetPhysicalDeviceFeatures2"));
	const auto getMemProps = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(gBridge.gipa(gBridge.instance, "vkGetPhysicalDeviceMemoryProperties"));
	const auto getQueueFams = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(gBridge.gipa(gBridge.instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
	const auto enumExts = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(gBridge.gipa(gBridge.instance, "vkEnumerateDeviceExtensionProperties"));
	const auto createDevice = reinterpret_cast<PFN_vkCreateDevice>(gBridge.gipa(gBridge.instance, "vkCreateDevice"));
	const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gBridge.gipa(gBridge.instance, "vkGetDeviceProcAddr"));

	if (!enumPhys || !getProps2 || !getFeats2 || !getMemProps || !getQueueFams || !enumExts || !createDevice || !gdpa)
		return fail("instance_functions");

	uint32_t physCount = 0;
	if (enumPhys(gBridge.instance, &physCount, nullptr) != VK_SUCCESS || !physCount)
		return fail("no_physical_device");

	std::vector<VkPhysicalDevice> physList(physCount);
	if (enumPhys(gBridge.instance, &physCount, physList.data()) != VK_SUCCESS)
		return fail("no_physical_device");

	// Match exact physical GPU by adapter LUID (immune to DXGI spoofing).
	const LUID d3dLuid = d3dDevice->GetAdapterLuid();
	VkPhysicalDevice matchedPhys = VK_NULL_HANDLE;
	uint32_t matchedVendor = 0;
	for (VkPhysicalDevice pd : physList)
	{
		VkPhysicalDeviceIDProperties idProps{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
		VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idProps };
		getProps2(pd, &props2);

		if (props2.properties.apiVersion < VK_API_VERSION_1_3)
			continue;

		if (idProps.deviceLUIDValid == VK_TRUE &&
			std::memcmp(idProps.deviceLUID, &d3dLuid, sizeof(LUID)) == 0)
		{
			matchedPhys = pd;
			matchedVendor = props2.properties.vendorID;
			break;
		}
	}

	if (!matchedPhys)
		return fail("no_luid_match");

	// Verify required device extensions for xvk + VK_KHR_cooperative_matrix + D3D12 Win32 interop.
	uint32_t extCount = 0;
	if (enumExts(matchedPhys, nullptr, &extCount, nullptr) != VK_SUCCESS || !extCount)
		return fail("no_extensions");
	std::vector<VkExtensionProperties> exts(extCount);
	if (enumExts(matchedPhys, nullptr, &extCount, exts.data()) != VK_SUCCESS)
		return fail("no_extensions");

	uint32_t coopSpec = 0;
	if (!HasExtension(exts, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) ||
		!HasExtension(exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, &coopSpec) ||
		!HasExtension(exts, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) ||
		!HasExtension(exts, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME))
	{
		return fail("missing_interop_or_coopmat_ext");
	}

	// Query supported features so we only enable features the driver actually advertises.
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeats{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
	VkPhysicalDeviceVulkan13Features vk13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &coopFeats };
	VkPhysicalDeviceVulkan12Features vk12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &vk13 };
	VkPhysicalDeviceVulkan11Features vk11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &vk12 };
	VkPhysicalDeviceFeatures2 feats2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vk11 };
	getFeats2(matchedPhys, &feats2);

	if (!coopFeats.cooperativeMatrix || !vk12.timelineSemaphore || !vk12.shaderInt8 ||
		!vk12.storageBuffer8BitAccess || !vk12.vulkanMemoryModel || !vk12.vulkanMemoryModelDeviceScope ||
		!vk13.synchronization2 || !vk13.maintenance4 || !vk13.shaderIntegerDotProduct ||
		!feats2.features.shaderStorageImageExtendedFormats)
	{
		return fail("missing_vulkan13_features");
	}

	// Pick a compute + transfer queue family (prefer family 0 / graphics+compute on AMD).
	uint32_t famCount = 0;
	getQueueFams(matchedPhys, &famCount, nullptr);
	if (!famCount)
		return fail("no_queue_family");
	std::vector<VkQueueFamilyProperties> fams(famCount);
	getQueueFams(matchedPhys, &famCount, fams.data());

	uint32_t chosenFamily = UINT32_MAX;
	for (uint32_t i = 0; i < famCount; ++i)
	{
		if ((fams[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) ==
			(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT))
		{
			chosenFamily = i;
			if (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
				break;
		}
	}
	if (chosenFamily == UINT32_MAX)
		return fail("no_compute_queue");

	// Build minimal enabled feature chain for private VkDevice.
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR enableCoop{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
	enableCoop.cooperativeMatrix = VK_TRUE;

	VkPhysicalDeviceVulkan13Features enable13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &enableCoop };
	enable13.synchronization2 = VK_TRUE;
	enable13.maintenance4 = VK_TRUE;
	enable13.shaderIntegerDotProduct = VK_TRUE;
	enable13.subgroupSizeControl = vk13.subgroupSizeControl;
	enable13.computeFullSubgroups = vk13.computeFullSubgroups;

	VkPhysicalDeviceVulkan12Features enable12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &enable13 };
	enable12.timelineSemaphore = VK_TRUE;
	enable12.shaderInt8 = VK_TRUE;
	enable12.storageBuffer8BitAccess = VK_TRUE;
	enable12.uniformAndStorageBuffer8BitAccess = vk12.uniformAndStorageBuffer8BitAccess;
	enable12.shaderFloat16 = vk12.shaderFloat16;
	enable12.vulkanMemoryModel = VK_TRUE;
	enable12.vulkanMemoryModelDeviceScope = VK_TRUE;
	enable12.scalarBlockLayout = vk12.scalarBlockLayout;

	VkPhysicalDeviceVulkan11Features enable11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &enable12 };
	enable11.storageBuffer16BitAccess = vk11.storageBuffer16BitAccess;

	VkPhysicalDeviceFeatures2 enableFeats{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &enable11 };
	enableFeats.features.shaderStorageImageExtendedFormats = VK_TRUE;
	enableFeats.features.shaderInt16 = feats2.features.shaderInt16;

	const char* enabledExts[] = {
		VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
	};

	const float priority = 1.0f;
	VkDeviceQueueCreateInfo qInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	qInfo.queueFamilyIndex = chosenFamily;
	qInfo.queueCount = 1;
	qInfo.pQueuePriorities = &priority;

	VkDeviceCreateInfo devInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enableFeats };
	devInfo.queueCreateInfoCount = 1;
	devInfo.pQueueCreateInfos = &qInfo;
	devInfo.enabledExtensionCount = static_cast<uint32_t>(std::size(enabledExts));
	devInfo.ppEnabledExtensionNames = enabledExts;

	if (createDevice(matchedPhys, &devInfo, nullptr, &gBridge.device) != VK_SUCCESS || !gBridge.device)
		return fail("create_device_failed");

	gBridge.physical = matchedPhys;
	gBridge.queueFamily = chosenFamily;
	getMemProps(matchedPhys, &gBridge.memProps);

	bool fnsOk = true;
#define LOAD_BRIDGE_FN(name)                                                       \
	gBridge.name = reinterpret_cast<PFN_##name>(gdpa(gBridge.device, #name));      \
	fnsOk &= (gBridge.name != nullptr);
	BRIDGE_VK_DEV_FNS(LOAD_BRIDGE_FN)
#undef LOAD_BRIDGE_FN
	if (!fnsOk)
		return fail("device_functions");

	gBridge.vkGetDeviceQueue(gBridge.device, chosenFamily, 0, &gBridge.queue);
	if (!gBridge.queue)
		return fail("no_queue");

	// Create Vulkan command pool and ring command buffers.
	VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = chosenFamily;
	if (gBridge.vkCreateCommandPool(gBridge.device, &poolInfo, nullptr, &gBridge.cmdPool) != VK_SUCCESS)
		return fail("command_pool_failed");

	VkCommandBufferAllocateInfo cbInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	cbInfo.commandPool = gBridge.cmdPool;
	cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbInfo.commandBufferCount = kRingSize;
	if (gBridge.vkAllocateCommandBuffers(gBridge.device, &cbInfo, gBridge.cmdBufs) != VK_SUCCESS)
		return fail("command_buffers_failed");

	// Create D3D12 shared fence and import it into Vulkan as a timeline semaphore.
	if (FAILED(d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&gBridge.sharedFence))))
		return fail("shared_fence_failed");

	HANDLE fenceHandle = nullptr;
	if (FAILED(d3dDevice->CreateSharedHandle(gBridge.sharedFence.Get(), nullptr, GENERIC_ALL, nullptr, &fenceHandle)) || !fenceHandle)
		return fail("shared_fence_handle_failed");

	VkSemaphoreTypeCreateInfo semType{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
	semType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	semType.initialValue = 0;

	VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &semType };
	if (gBridge.vkCreateSemaphore(gBridge.device, &semInfo, nullptr, &gBridge.sharedSemaphore) != VK_SUCCESS)
	{
		CloseHandle(fenceHandle);
		return fail("vulkan_semaphore_failed");
	}

	VkImportSemaphoreWin32HandleInfoKHR semImport{ VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR };
	semImport.semaphore = gBridge.sharedSemaphore;
	semImport.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
	semImport.handle = fenceHandle;

	const VkResult semRes = gBridge.vkImportSemaphoreWin32HandleKHR(gBridge.device, &semImport);
	ComPtr<ID3D12Device> nativeDev = UnwrapNativeDevice(d3dDevice);
	if (semRes == VK_SUCCESS && nativeDev && nativeDev.Get() != d3dDevice)
		nativeDev->OpenSharedHandle(fenceHandle, IID_PPV_ARGS(&gBridge.nativeSharedFence));
	CloseHandle(fenceHandle);
	if (semRes != VK_SUCCESS)
		return fail("semaphore_import_failed");

	gBridge.fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	if (!gBridge.fenceEvent)
		return fail("fence_event_failed");

	// Mount xvk (D3D12-on-Vulkan recorded device) and install hardware VK_KHR_cooperative_matrix shaders.
	xvk::DeviceDesc xvkDesc{};
	xvkDesc.instance = gBridge.instance;
	xvkDesc.physicalDevice = gBridge.physical;
	xvkDesc.device = gBridge.device;
	xvkDesc.getInstanceProcAddr = gBridge.gipa;
	xvkDesc.programModule = SelfModule();
	xvkDesc.recorded = true;
	xvkDesc.recordedFamily = chosenFamily;
	xvkDesc.coopMatrixExtensions = true;
	xvkDesc.coopMatrixSpec = coopSpec;

	if (FAILED(xvk::CreateDevice(xvkDesc, IID_PPV_ARGS(&gBridge.xvkDevice))) || !gBridge.xvkDevice)
		return fail("xvk_device_failed");

	// Recorded xvk devices require an active xvk::BeginRecording(cmd) before
	// CreateCommandList can open recording, so create xvkList lazily on first execute.
	if (FAILED(gBridge.xvkDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&gBridge.xvkAlloc))) || !gBridge.xvkAlloc)
	{
		return fail("xvk_alloc_failed");
	}

	std::string coopReason;
	if (!VulkanCoopMat::Install(gBridge.xvkDevice.Get(), coopReason))
		return fail(coopReason.empty() ? "coopmat_install_failed" : coopReason.c_str());

	const auto coopStatus = xefg_cuda::CoopMatrixSnapshot();
	if (coopStatus.layers == 0)
		return fail("no_coopmat_modules");

	xvk::SetCoopMatrixEnabled(gBridge.xvkDevice.Get(), true);

	// Hook ID3D12CommandQueue::ExecuteCommandLists on both the caller's device and the native runtime device.
	gBridge.d3dDevice = d3dDevice;
	gBridge.nativeD3DDevice = UnwrapNativeDevice(d3dDevice);

	bool hookedQueue = false;
	auto tryHookQueue = [&](ID3D12Device* dev, D3D12_COMMAND_LIST_TYPE type) {
		if (!dev)
			return;
		D3D12_COMMAND_QUEUE_DESC qd{};
		qd.Type = type;
		ComPtr<ID3D12CommandQueue> probeQ;
		if (SUCCEEDED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&probeQ))) && probeQ)
			hookedQueue |= PatchQueueVtable(probeQ.Get(), &HookExecuteCommandLists);
	};
	tryHookQueue(d3dDevice, D3D12_COMMAND_LIST_TYPE_DIRECT);
	tryHookQueue(d3dDevice, D3D12_COMMAND_LIST_TYPE_COMPUTE);
	if (gBridge.nativeD3DDevice.Get() != d3dDevice)
	{
		tryHookQueue(gBridge.nativeD3DDevice.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
		tryHookQueue(gBridge.nativeD3DDevice.Get(), D3D12_COMMAND_LIST_TYPE_COMPUTE);
	}
	if (!hookedQueue)
		return fail("queue_hook_failed");

	gBridge.tileM = static_cast<uint32_t>(coopStatus.major);
	gBridge.tileN = static_cast<uint32_t>((coopStatus.minor >> 8) & 0xFF);
	gBridge.tileK = static_cast<uint32_t>(coopStatus.minor & 0xFF);
	gBridge.coopLayers = coopStatus.layers;
	gBridge.reason.clear();
	gBridge.active.store(true, std::memory_order_release);

	Log::Info("D3D12->Vulkan WMMA bridge ready (vendor {:#06x}, queue family {}, VK_KHR_cooperative_matrix {}x{}x{} s8->s32, {} hardware WMMA shaders)",
		matchedVendor, chosenFamily, gBridge.tileM, gBridge.tileN, gBridge.tileK, gBridge.coopLayers);
	return true;
}

bool SameContextSettings(const ContextSettings& a, const ContextSettings& b) noexcept
{
	return a.width == b.width &&
		a.height == b.height &&
		a.format == b.format &&
		a.initFlags == b.initFlags &&
		a.uiMode == b.uiMode &&
		a.model == b.model;
}

bool EnsureContext(const ContextSettings& settings) noexcept
{
	std::lock_guard lock(gBridge.mutex);
	if (!gBridge.active.load(std::memory_order_acquire) || !gBridge.xvkDevice)
		return false;

	if (gBridge.xefgContext && SameContextSettings(gBridge.contextSettings, settings))
		return true;

	if (gBridge.contextFailed && SameContextSettings(gBridge.failedContextSettings, settings))
		return false;

	auto markFailed = [&]() noexcept {
		gBridge.contextFailed = true;
		gBridge.failedContextSettings = settings;
		return false;
	};

	// Drain in-flight Vulkan work so destroying any prior context and reusing cmdBufs[0]
	// for XeFG::Init weight uploads is completely safe.
	if (gBridge.queue && gBridge.vkQueueWaitIdle)
		gBridge.vkQueueWaitIdle(gBridge.queue);
	for (uint32_t i = 0; i < kRingSize; ++i)
		gBridge.vkSlotDoneFence[i] = 0;

	if (gBridge.xefgContext)
	{
		XeFG::Destroy(gBridge.xefgContext);
		gBridge.xefgContext = nullptr;
	}

	XeFG::OverrideModel(settings.model);

	// Recorded xvk devices require an open VkCommandBuffer + xvk::BeginRecording()
	// during XeFG::Init because libxess_fg creates internal command lists to upload
	// neural network weights and initialize resources.
	VkCommandBuffer initCmd = gBridge.cmdBufs[0];
	gBridge.vkResetCommandBuffer(initCmd, 0);
	VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (gBridge.vkBeginCommandBuffer(initCmd, &beginInfo) != VK_SUCCESS)
	{
		Log::Warn("D3D12->Vulkan WMMA bridge: vkBeginCommandBuffer failed for XeFG::Init, falling back to D3D12");
		return markFailed();
	}

	if (!xvk::BeginRecording(gBridge.xvkDevice.Get(), initCmd))
	{
		gBridge.vkEndCommandBuffer(initCmd);
		Log::Warn("D3D12->Vulkan WMMA bridge: xvk::BeginRecording failed for XeFG::Init, falling back to D3D12");
		return markFailed();
	}

	xefgd_handle_t ctx = nullptr;
	const xefgd_result_t createRes = XeFG::CreateContext(gBridge.xvkDevice.Get(), &ctx);
	if (createRes != XEFGD_RESULT_SUCCESS || !ctx)
	{
		xvk::EndRecording(gBridge.xvkDevice.Get());
		gBridge.vkEndCommandBuffer(initCmd);
		Log::Warn("D3D12->Vulkan WMMA bridge: XeFG::CreateContext failed (result {}), falling back to D3D12",
			static_cast<int>(createRes));
		return markFailed();
	}

	xefgd_init_params_t init{};
	init.outputResolution = { settings.width, settings.height };
	init.backBufferFormat = static_cast<uint32_t>(settings.format);
	init.initFlags = settings.initFlags;
	init.maxInterpolatedFrames = kMaxFrames;
	init.creationNodeMask = 1;
	init.visibleNodeMask = 1;
	init.uiMode = settings.uiMode;

	const xefgd_result_t initRes = XeFG::Init(ctx, init);
	xvk::EndRecording(gBridge.xvkDevice.Get());
	const VkResult endRes = gBridge.vkEndCommandBuffer(initCmd);

	if (initRes != XEFGD_RESULT_SUCCESS || endRes != VK_SUCCESS)
	{
		XeFG::Destroy(ctx);
		Log::Warn("D3D12->Vulkan WMMA bridge: XeFG::Init failed (result {}, vkEnd {}), falling back to D3D12",
			static_cast<int>(initRes), static_cast<int>(endRes));
		return markFailed();
	}

	VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &initCmd;
	if (gBridge.vkQueueSubmit(gBridge.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
	{
		XeFG::Destroy(ctx);
		Log::Warn("D3D12->Vulkan WMMA bridge: XeFG::Init vkQueueSubmit failed, falling back to D3D12");
		return markFailed();
	}
	gBridge.vkQueueWaitIdle(gBridge.queue);

	gBridge.xefgContext = ctx;
	gBridge.contextSettings = settings;
	gBridge.contextFailed = false;
	Log::Info("D3D12->Vulkan WMMA bridge context ready: {}x{}, format {}, model {}, {}x{}x{} hardware WMMA",
		settings.width, settings.height, static_cast<uint32_t>(settings.format), settings.model,
		gBridge.tileM, gBridge.tileN, gBridge.tileK);
	return true;
}

void ResetContext() noexcept
{
	std::lock_guard lock(gBridge.mutex);
	gBridge.contextFailed = false;
	if (gBridge.xefgContext)
	{
		if (gBridge.queue && gBridge.vkQueueWaitIdle)
			gBridge.vkQueueWaitIdle(gBridge.queue);
		XeFG::Destroy(gBridge.xefgContext);
		gBridge.xefgContext = nullptr;
	}
}

bool ScheduleBatch(
	ID3D12GraphicsCommandList* cmdList,
	const ContextSettings& settings,
	const FrameInputs& prev,
	const FrameInputs& next,
	ID3D12Resource* const* sharedOutputs,
	const xefgd_2d_t* outputBases,
	uint32_t frameCount,
	ID3D12Resource* firstGameOutput,
	D3D12_RESOURCE_STATES outputState) noexcept
{
	if (!Active() || !cmdList || !sharedOutputs || !outputBases || !firstGameOutput || frameCount == 0 || frameCount > kMaxFrames)
		return false;
	if (!prev.backbuffer || !prev.depth || !prev.mvecs || !next.backbuffer || !next.depth || !next.mvecs)
		return false;
	if (!EnsureContext(settings))
		return false;

	// Verify all participating textures can be imported into Vulkan before deferring D3D12 execution.
	{
		std::lock_guard lock(gBridge.mutex);
		if (!GetOrImportLocked(gBridge, prev.backbuffer, false) ||
			!GetOrImportLocked(gBridge, prev.depth, false) ||
			!GetOrImportLocked(gBridge, prev.mvecs, false) ||
			(prev.hudless && !GetOrImportLocked(gBridge, prev.hudless, false)) ||
			!GetOrImportLocked(gBridge, next.backbuffer, false) ||
			!GetOrImportLocked(gBridge, next.depth, false) ||
			!GetOrImportLocked(gBridge, next.mvecs, false) ||
			(next.hudless && !GetOrImportLocked(gBridge, next.hudless, false)))
		{
			gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		for (uint32_t i = 0; i < frameCount; ++i)
		{
			if (!sharedOutputs[i] || !GetOrImportLocked(gBridge, sharedOutputs[i], true))
			{
				gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
		}
	}

	PendingBatch batch{};
	batch.cmdList = cmdList;
	batch.settings = settings;
	batch.prevBackbuffer = prev.backbuffer;
	batch.prevHudless = prev.hudless;
	batch.prevDepth = prev.depth;
	batch.prevMvecs = prev.mvecs;
	batch.prevData = prev.data;
	batch.nextBackbuffer = next.backbuffer;
	batch.nextHudless = next.hudless;
	batch.nextDepth = next.depth;
	batch.nextMvecs = next.mvecs;
	batch.nextData = next.data;
	batch.frameCount = frameCount;
	for (uint32_t i = 0; i < frameCount; ++i)
	{
		batch.sharedOutputs[i] = sharedOutputs[i];
		batch.outputBases[i] = outputBases[i];
	}
	batch.copies.push_back({ sharedOutputs[0], firstGameOutput, outputState });

	{
		std::lock_guard lock(gBridge.pendingMutex);
		// Drop any stale entry for the same command list if the game discarded it without executing.
		gBridge.pending.erase(
			std::remove_if(gBridge.pending.begin(), gBridge.pending.end(),
				[cmdList](const PendingBatch& b) { return b.cmdList == cmdList; }),
			gBridge.pending.end());
		gBridge.pending.push_back(std::move(batch));
		gBridge.pendingCount.store(static_cast<uint32_t>(gBridge.pending.size()), std::memory_order_release);
	}
	return true;
}

bool AppendOutputCopy(
	ID3D12GraphicsCommandList* cmdList,
	ID3D12Resource* sharedOutput,
	ID3D12Resource* gameOutput,
	D3D12_RESOURCE_STATES outputState) noexcept
{
	if (!Active() || !cmdList || !sharedOutput || !gameOutput)
		return false;

	std::lock_guard lock(gBridge.pendingMutex);
	for (auto& b : gBridge.pending)
	{
		if (b.cmdList == cmdList)
		{
			b.copies.push_back({ sharedOutput, gameOutput, outputState });
			return true;
		}
	}
	return false;
}

void ForgetResource(ID3D12Resource* resource) noexcept
{
	if (!resource)
		return;
	std::lock_guard lock(gBridge.mutex);
	auto it = gBridge.imports.find(resource);
	if (it != gBridge.imports.end())
	{
		if (gBridge.queue && gBridge.vkQueueWaitIdle)
			gBridge.vkQueueWaitIdle(gBridge.queue);
		DestroyImport(gBridge, it->second);
		gBridge.imports.erase(it);
	}
}

Status Snapshot() noexcept
{
	Status s;
	s.built = true;
	s.active = gBridge.active.load(std::memory_order_acquire);
	s.tileM = gBridge.tileM;
	s.tileN = gBridge.tileN;
	s.tileK = gBridge.tileK;
	s.coopLayers = gBridge.coopLayers;
	s.batches = gBridge.batches.load(std::memory_order_relaxed);
	s.fallbacks = gBridge.fallbacks.load(std::memory_order_relaxed);
	{
		std::lock_guard lock(gBridge.mutex);
		s.reason = gBridge.reason;
	}
	return s;
}
} // namespace D3D12VulkanWmma
