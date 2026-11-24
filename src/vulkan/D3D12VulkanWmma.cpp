// D3D12 -> Vulkan VK_KHR_cooperative_matrix (hardware WMMA INT8) bridge for AMD RDNA 4 / RDNA 3.
// See D3D12VulkanWmma.h for the architecture overview.
#include "../D3D12VulkanWmma.h"

#include "../FrameGenerator.h"
#include "../Log.h"
#include "../Plugin.h"
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
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace D3D12VulkanWmma
{
namespace
{
using Microsoft::WRL::ComPtr;

constexpr uint32_t kRingSize = 8;
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
	case DXGI_FORMAT_R16G16_SNORM:
		return VK_FORMAT_R16G16_SNORM;
	case DXGI_FORMAT_R16G16_UNORM:
		return VK_FORMAT_R16G16_UNORM;
	case DXGI_FORMAT_R16G16B16A16_UNORM:
		return VK_FORMAT_R16G16B16A16_UNORM;
	case DXGI_FORMAT_R32_UINT:
		return VK_FORMAT_R32_UINT;
	case DXGI_FORMAT_R16_UINT:
		return VK_FORMAT_R16_UINT;
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
	bool zeroCopy = false;
	bool evicted = false;
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
	ComPtr<ID3D12Resource> directFirstGameOutput;
	D3D12_RESOURCE_STATES directFirstOutputState = D3D12_RESOURCE_STATE_COMMON;
	ComPtr<ID3D12Resource> fallbackRealFrame;
	D3D12_RESOURCE_STATES fallbackRealFrameState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	// Holds shared ownership of every Vulkan SharedImport referenced by the pre-recorded command buffer
	// across both out-of-pending race windows (post-record/pre-push and post-pop/pre-submit).
	std::vector<std::shared_ptr<SharedImport>> keptImports;
	bool preRecorded = false;
	bool anyOutputCopy = false;
	uint32_t vkSlot = 0;
	uint64_t vkSlotToken = 0;
	uint64_t contextEpoch = 0;
};

struct PostCopyRing
{
	ComPtr<ID3D12CommandAllocator> alloc[kRingSize];
	ComPtr<ID3D12GraphicsCommandList> list[kRingSize];
	uint64_t doneFence[kRingSize]{};
	std::vector<ComPtr<ID3D12CommandAllocator>> parkedAllocs; // busy allocators parked on fence timeout (never reset in-flight)
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
	X(vkQueueSubmit2)                            \
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
	X(vkGetSemaphoreCounterValue)                \
	X(vkCreateFence)                             \
	X(vkDestroyFence)                            \
	X(vkWaitForFences)                           \
	X(vkCmdCopyImage)                            \
	X(vkCmdPipelineBarrier)

struct Bridge
{
	std::mutex mutex;
	std::mutex reasonMutex;
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
	uint64_t vkSlotOwner[kRingSize]{};
	uint64_t nextSlotToken = 1;
	std::atomic<uint64_t> lastSubmittedVkFence{ 0 };
	std::atomic<uint64_t> prevSubmittedVkFence{ 0 };
	std::atomic<uint64_t> lastSignaledBridgeFence{ 0 };
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
	std::atomic<bool> contextReady{ false };
	std::atomic<bool> initInFlight{ false };
	std::atomic<bool> resetRequested{ false };
	bool contextFailed = false;
	uint32_t contextFailCount = 0;
	std::chrono::steady_clock::time_point contextFailedAt{};
	ContextSettings failedContextSettings{};
	uint32_t tileM = 0;
	uint32_t tileN = 0;
	uint32_t tileK = 0;
	uint32_t coopLayers = 0;

	uint64_t contextEpoch = 1;

	struct RetiredImport
	{
		uint64_t retireFence = 0;
		std::chrono::steady_clock::time_point retireAfter{};
		std::shared_ptr<SharedImport> imp;
	};
	struct RetiredContext
	{
		uint64_t retireFence = 0;
		xefgd_handle_t handle = nullptr;
	};

	// Cached D3D12 -> Vulkan shared imports keyed by ID3D12Resource*
	std::unordered_map<ID3D12Resource*, std::shared_ptr<SharedImport>> imports;
	std::unordered_map<ID3D12Resource*, bool> directOutEligible;
	// Deferred retirement queues for contexts/imports whose 500ms drain timed out (never destroy while GPU may still read them)
	std::vector<RetiredImport> retiredImports;
	std::vector<RetiredContext> retiredContexts;
	std::atomic<size_t> importCount{ 0 };
	std::atomic<uint64_t> directZeroCopyOuts{ 0 };

	// Pending batches keyed by ID3D12GraphicsCommandList*
	std::mutex pendingMutex;
	std::atomic<uint32_t> pendingCount{ 0 };
	std::vector<PendingBatch> pending;

	// Bounded auto-recovery after transient VK_ERROR_DEVICE_LOST / GPU stall while D3D12 stays healthy
	std::atomic<bool> recoveryNeeded{ false };
	std::atomic<bool> recoveryInFlight{ false };
	uint32_t recoveryAttempts = 0;
	std::chrono::steady_clock::time_point lastFaultAt{};

	// Telemetry
	std::atomic<uint64_t> batches{ 0 };
	std::atomic<uint64_t> fallbacks{ 0 };
	std::atomic<uint64_t> totalCpuWaitUs{ 0 };
	std::atomic<uint64_t> cpuWaitSamples{ 0 };
	std::atomic<uint64_t> totalSlotWaitUs{ 0 };
	std::atomic<uint64_t> slotWaitSamples{ 0 };
	std::atomic<uint64_t> paceTimeouts{ 0 };
	std::atomic<bool> lastSkipCpuPaceWait{ false };
	std::atomic<uint32_t> paceWaitActive{ 0 };
};

constexpr uint32_t kMaxRecoveryAttempts = 3u;
constexpr auto kRecoveryCooldown = std::chrono::seconds(5);

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
bool SameContextSettings(const ContextSettings& a, const ContextSettings& b) noexcept
{
	return a.width == b.width &&
		a.height == b.height &&
		a.format == b.format &&
		a.initFlags == b.initFlags &&
		a.uiMode == b.uiMode &&
		a.model == b.model;
}

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

bool WaitFenceCpu(ID3D12Fence* fence, HANDLE ev, uint64_t target, DWORD timeoutMs = 500u) noexcept
{
	if (!target || !fence || fence->GetCompletedValue() >= target)
		return true;
	if (SUCCEEDED(fence->SetEventOnCompletion(target, ev)))
	{
		WaitForSingleObject(ev, timeoutMs);
		return fence->GetCompletedValue() >= target;
	}
	return false;
}

// Bounded timeline-semaphore drain (replaces unbounded vkQueueWaitIdle so a stalled queue can never freeze the game).
bool DrainVulkanQueueLocked(uint64_t timeoutNs) noexcept
{
	const uint64_t target = gBridge.lastSubmittedVkFence.load(std::memory_order_acquire);
	if (!target || !gBridge.device || !gBridge.sharedSemaphore || !gBridge.vkWaitSemaphores)
		return true;
	VkSemaphoreWaitInfo waitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
	waitInfo.semaphoreCount = 1;
	waitInfo.pSemaphores = &gBridge.sharedSemaphore;
	waitInfo.pValues = &target;
	return gBridge.vkWaitSemaphores(gBridge.device, &waitInfo, timeoutNs) == VK_SUCCESS;
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

void ResetPostCopyRing(PostCopyRing& ring) noexcept
{
	for (uint32_t i = 0; i < kRingSize; ++i)
	{
		ring.list[i].Reset();
		ring.alloc[i].Reset();
		ring.doneFence[i] = 0;
	}
	ring.parkedAllocs.clear();
	ring.cursor = 0;
	ring.ready = false;
}

void MarkBridgeFaultLocked(const char* stage, int vkResult) noexcept
{
	gBridge.active.store(false, std::memory_order_release);
	gBridge.contextReady.store(false, std::memory_order_release);
	gBridge.lastFaultAt = std::chrono::steady_clock::now();

	const bool d3dAlive = gBridge.d3dDevice && gBridge.d3dDevice->GetDeviceRemovedReason() == S_OK;
	if (d3dAlive && gBridge.recoveryAttempts < kMaxRecoveryAttempts)
	{
		gBridge.recoveryNeeded.store(true, std::memory_order_release);
		Log::Warn("D3D12->Vulkan WMMA bridge: {} timed out/failed (result {}), temporarily falling back to D3D12 (recovery {}/{} in 5s)",
			stage, vkResult, gBridge.recoveryAttempts + 1u, kMaxRecoveryAttempts);
	}
	else
	{
		gBridge.recoveryNeeded.store(false, std::memory_order_release);
		Log::Warn("D3D12->Vulkan WMMA bridge: {} timed out/failed (result {}), disabling bridge and falling back to D3D12",
			stage, vkResult);
	}
}

void DestroyImport(Bridge& b, SharedImport& imp) noexcept
{
	const bool hadWrapped = imp.zeroCopy && imp.xvkResource != nullptr;
	imp.xvkResource.Reset();
	if (hadWrapped && b.xvkDevice)
		xvk::WaitRecordedIdle(b.xvkDevice.Get(), 50'000'000ull);
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

bool FenceValueCompletedLocked(uint64_t value) noexcept
{
	if (value == 0 || !gBridge.device || !gBridge.sharedSemaphore || !gBridge.vkWaitSemaphores)
		return true;
	VkSemaphoreWaitInfo waitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
	waitInfo.semaphoreCount = 1;
	waitInfo.pSemaphores = &gBridge.sharedSemaphore;
	waitInfo.pValues = &value;
	return gBridge.vkWaitSemaphores(gBridge.device, &waitInfo, 0ull) == VK_SUCCESS;
}

void ReapRetiredLocked(bool forceAll = false) noexcept
{
	if (!gBridge.retiredImports.empty())
	{
		auto it = gBridge.retiredImports.begin();
		const auto now = std::chrono::steady_clock::now();
		while (it != gBridge.retiredImports.end())
		{
			if (!it->imp)
			{
				it = gBridge.retiredImports.erase(it);
				continue;
			}
			if (forceAll || (it->imp.use_count() == 1 && now >= it->retireAfter && FenceValueCompletedLocked(it->retireFence)))
			{
				DestroyImport(gBridge, *it->imp);
				it = gBridge.retiredImports.erase(it);
			}
			else
			{
				++it;
			}
		}
	}
	if (!gBridge.retiredContexts.empty())
	{
		auto it = gBridge.retiredContexts.begin();
		while (it != gBridge.retiredContexts.end())
		{
			if (forceAll || FenceValueCompletedLocked(it->retireFence))
			{
				if (it->handle)
					XeFG::Destroy(it->handle);
				it = gBridge.retiredContexts.erase(it);
			}
			else
			{
				++it;
			}
		}
	}
}

// Imports a D3D12_HEAP_FLAG_SHARED texture into Vulkan via VK_KHR_external_memory_win32 and creates
// its companion xvk texture on b.xvkDevice. Caller holds b.mutex.
std::shared_ptr<SharedImport> GetOrImportLocked(Bridge& b, ID3D12Resource* d3dRes, bool isOutput) noexcept
{
	if (!d3dRes)
		return nullptr;

	auto it = b.imports.find(d3dRes);
	if (it != b.imports.end())
		return it->second;

	auto logImportFail = [](const char* step, const D3D12_RESOURCE_DESC& d, int64_t code) noexcept {
		static std::atomic<bool> logged{ false };
		if (!logged.exchange(true, std::memory_order_relaxed))
			Log::Warn("D3D12->Vulkan WMMA bridge: resource import failed at {} (fmt={}, {}x{}, flags={:#x}, code={:#x}), falling back to D3D12",
				step, static_cast<uint32_t>(d.Format), static_cast<uint64_t>(d.Width), d.Height,
				static_cast<uint32_t>(d.Flags), static_cast<uint64_t>(code));
	};

	const D3D12_RESOURCE_DESC desc = d3dRes->GetDesc();
	if (!CanShareTexture(desc))
	{
		logImportFail("CanShareTexture", desc, 0);
		return nullptr;
	}

	const VkFormat vkFmt = DxgiToVkFormat(desc.Format);
	if (vkFmt == VK_FORMAT_UNDEFINED)
	{
		logImportFail("DxgiToVkFormat", desc, static_cast<int64_t>(desc.Format));
		return nullptr;
	}

	HANDLE ntHandle = nullptr;
	const HRESULT shareHr = b.d3dDevice->CreateSharedHandle(d3dRes, nullptr, GENERIC_ALL, nullptr, &ntHandle);
	if (FAILED(shareHr) || !ntHandle)
	{
		logImportFail("CreateSharedHandle", desc, static_cast<int64_t>(shareHr));
		return nullptr;
	}

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
	imgInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
	// Mirror D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS onto VK_IMAGE_USAGE_STORAGE_BIT even for inputs
	// (e.g. Tier 4 Tone-Map Matcher outputs and software-unjittered motion vectors). On AMD RDNA 3,
	// ALLOW_UNORDERED_ACCESS changes surface tiling/DCC metadata; omitting STORAGE_BIT causes a
	// D3D12<->Vulkan layout mismatch and DXGI_ERROR_DEVICE_HUNG on the first Vulkan WMMA batch.
	const bool needsStorageBit = isOutput || ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0);
	imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
		(needsStorageBit ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
	imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	SharedImport imp{};
	imp.format = vkFmt;
	imp.width = static_cast<uint32_t>(desc.Width);
	imp.height = desc.Height;
	imp.isOutput = isOutput;

	bool mutableOk = (b.vkCreateImage(b.device, &imgInfo, nullptr, &imp.vkImage) == VK_SUCCESS);
	if (!mutableOk)
	{
		imgInfo.flags = 0;
		if (b.vkCreateImage(b.device, &imgInfo, nullptr, &imp.vkImage) != VK_SUCCESS)
		{
			CloseHandle(ntHandle);
			return nullptr;
		}
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

	// Zero-copy: wrap the imported D3D12 shared VkImage directly as an xvk ID3D12Resource so XeFG
	// samples inputs and writes outputs directly without per-frame full-resolution vkCmdCopyImage copies.
	D3D12_RESOURCE_DESC xvkDesc = desc;
	xvkDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	xvkDesc.Flags = isOutput ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;

	if (mutableOk &&
		SUCCEEDED(xvk::WrapExternalImage(b.xvkDevice.Get(), xvkDesc, imp.vkImage, vkFmt, IID_PPV_ARGS(&imp.xvkResource))) &&
		imp.xvkResource && xvk::ImageOf(imp.xvkResource.Get()) == imp.vkImage)
	{
		imp.zeroCopy = true;
	}
	else
	{
		// Fallback: allocate a separate xvk texture and copy per frame if mutable external views are unsupported.
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
		imp.zeroCopy = false;
	}

	auto sp = std::make_shared<SharedImport>(std::move(imp));
	auto [inserted, _] = b.imports.emplace(d3dRes, std::move(sp));
	b.importCount.store(b.imports.size(), std::memory_order_release);
	return inserted->second;
}

VkImageMemoryBarrier MakeImageBarrier(VkImage img,
	VkImageLayout oldLayout, VkImageLayout newLayout,
	VkAccessFlags srcAccess, VkAccessFlags dstAccess) noexcept
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
	return barrier;
}

void RecordCopyImage(const Bridge& b, VkCommandBuffer cmd, VkImage src, VkImage dst, uint32_t w, uint32_t h) noexcept
{
	VkImageCopy region{};
	region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	region.extent = { w, h, 1 };
	b.vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

void RecordD3D12CopiesBatched(ID3D12GraphicsCommandList* list, const std::vector<OutputCopy>& copies) noexcept
{
	D3D12_RESOURCE_BARRIER barriers[kMaxFrames * 2]{};
	UINT count = 0;
	const size_t n = std::min<size_t>(copies.size(), kMaxFrames);
	for (size_t i = 0; i < n; ++i)
	{
		const auto& cp = copies[i];
		if (cp.outputState != D3D12_RESOURCE_STATE_COPY_DEST)
		{
			barriers[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barriers[count].Transition = { cp.gameOutput.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, cp.outputState, D3D12_RESOURCE_STATE_COPY_DEST };
			++count;
		}
		if (cp.outputState != D3D12_RESOURCE_STATE_COPY_SOURCE)
		{
			barriers[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barriers[count].Transition = { cp.sharedOutput.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, cp.outputState, D3D12_RESOURCE_STATE_COPY_SOURCE };
			++count;
		}
	}
	if (count)
		list->ResourceBarrier(count, barriers);

	for (size_t i = 0; i < n; ++i)
		list->CopyResource(copies[i].gameOutput.Get(), copies[i].sharedOutput.Get());

	for (UINT i = 0; i < count; ++i)
		std::swap(barriers[i].Transition.StateBefore, barriers[i].Transition.StateAfter);
	if (count)
		list->ResourceBarrier(count, barriers);
}

void RecordD3D12RealFrameFallbackCopies(ID3D12GraphicsCommandList* list, ID3D12Resource* realFrame,
	D3D12_RESOURCE_STATES realFrameState, const std::vector<OutputCopy>& copies) noexcept
{
	if (!realFrame || copies.empty())
		return;
	D3D12_RESOURCE_BARRIER barriers[kMaxFrames + 1]{};
	UINT count = 0;
	// Transition the real backbuffer source ONCE from its tracked capture state (XeFGInputState) to COPY_SOURCE.
	if (realFrameState != D3D12_RESOURCE_STATE_COPY_SOURCE)
	{
		barriers[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barriers[count].Transition = { realFrame, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, realFrameState, D3D12_RESOURCE_STATE_COPY_SOURCE };
		++count;
	}
	const size_t n = std::min<size_t>(copies.size(), kMaxFrames);
	for (size_t i = 0; i < n; ++i)
	{
		const auto& cp = copies[i];
		if (cp.outputState != D3D12_RESOURCE_STATE_COPY_DEST)
		{
			barriers[count].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barriers[count].Transition = { cp.gameOutput.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, cp.outputState, D3D12_RESOURCE_STATE_COPY_DEST };
			++count;
		}
	}
	if (count)
		list->ResourceBarrier(count, barriers);
	for (size_t i = 0; i < n; ++i)
		list->CopyResource(copies[i].gameOutput.Get(), realFrame);
	for (UINT i = 0; i < count; ++i)
		std::swap(barriers[i].Transition.StateBefore, barriers[i].Transition.StateAfter);
	if (count)
		list->ResourceBarrier(count, barriers);
}

void RunRealFrameFallbackLocked(ID3D12CommandQueue* queue, ExecuteCommandListsFn origExec, const PendingBatch& batch) noexcept
{
	ID3D12Resource* realFrame = batch.fallbackRealFrame ? batch.fallbackRealFrame.Get() : batch.nextBackbuffer.Get();
	if (!queue || !origExec || !realFrame)
		return;
	const D3D12_RESOURCE_DESC srcDesc = realFrame->GetDesc();
	std::vector<OutputCopy> fallbackCopies;
	auto tryPush = [&](ID3D12Resource* dst, D3D12_RESOURCE_STATES dstState) {
		if (!dst || dst == realFrame) return;
		for (const auto& existing : fallbackCopies)
			if (existing.gameOutput.Get() == dst) return;
		const D3D12_RESOURCE_DESC d = dst->GetDesc();
		if (d.Dimension == srcDesc.Dimension && d.Width == srcDesc.Width && d.Height == srcDesc.Height &&
			d.Format == srcDesc.Format && d.SampleDesc.Count == srcDesc.SampleDesc.Count)
		{
			fallbackCopies.push_back({ realFrame, dst, dstState });
		}
	};
	tryPush(batch.directFirstGameOutput.Get(), batch.directFirstOutputState);
	for (const auto& cp : batch.copies)
		tryPush(cp.gameOutput.Get(), cp.outputState);
	if (fallbackCopies.empty())
		return;
	ComPtr<ID3D12Device> queueDev;
	const bool isNativeQueue = SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&queueDev))) && queueDev &&
		gBridge.nativeD3DDevice && queueDev.Get() == gBridge.nativeD3DDevice.Get() &&
		gBridge.nativeD3DDevice.Get() != gBridge.d3dDevice.Get();
	ID3D12Fence* useFence = (isNativeQueue && gBridge.nativeSharedFence) ? gBridge.nativeSharedFence.Get() : gBridge.sharedFence.Get();
	ID3D12Device* ringDev = (isNativeQueue && gBridge.nativeD3DDevice) ? gBridge.nativeD3DDevice.Get() : gBridge.d3dDevice.Get();
	const D3D12_COMMAND_LIST_TYPE qType = queue->GetDesc().Type;
	const D3D12_COMMAND_LIST_TYPE useType = (qType == D3D12_COMMAND_LIST_TYPE_COMPUTE) ? D3D12_COMMAND_LIST_TYPE_COMPUTE : D3D12_COMMAND_LIST_TYPE_DIRECT;
	PostCopyRing& ring = isNativeQueue
		? ((useType == D3D12_COMMAND_LIST_TYPE_COMPUTE) ? gBridge.nativeComputeRing : gBridge.nativeDirectRing)
		: ((useType == D3D12_COMMAND_LIST_TYPE_COMPUTE) ? gBridge.computeRing : gBridge.directRing);
	if (!useFence || !ringDev || !EnsurePostCopyRing(ringDev, useType, ring))
		return;
	const uint32_t slot = ring.cursor++ % kRingSize;
	if (WaitFenceCpu(useFence, gBridge.fenceEvent, ring.doneFence[slot], 500u) &&
		SUCCEEDED(ring.alloc[slot]->Reset()) &&
		SUCCEEDED(ring.list[slot]->Reset(ring.alloc[slot].Get(), nullptr)))
	{
		RecordD3D12RealFrameFallbackCopies(ring.list[slot].Get(), realFrame, batch.fallbackRealFrameState, fallbackCopies);
		ring.list[slot]->Close();
		ID3D12CommandList* postLists[1] = { ring.list[slot].Get() };
		origExec(queue, 1, postLists);
		const uint64_t doneVal = gBridge.fenceValue.fetch_add(1, std::memory_order_acq_rel) + 1;
		if (SUCCEEDED(queue->Signal(useFence, doneVal)))
			gBridge.lastSignaledBridgeFence.store(doneVal, std::memory_order_release);
		ring.doneFence[slot] = doneVal;
	}
}

// Strict, cached-once qualification for writing directly into the game/OptiScaler output texture (zero-copy).
// Runs at most once per unique ID3D12Resource* (cached true/false in b.directOutEligible) so hot-path overhead is zero.
bool IsDirectGameOutputEligibleLocked(Bridge& b, ID3D12Resource* gameOut, D3D12_RESOURCE_STATES outputState) noexcept
{
	if (!gameOut || outputState != D3D12_RESOURCE_STATE_COMMON)
		return false;
	if (auto it = b.directOutEligible.find(gameOut); it != b.directOutEligible.end())
		return it->second;

	bool eligible = false;
	const D3D12_RESOURCE_DESC desc = gameOut->GetDesc();
	const bool shapeOk =
		desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
		desc.MipLevels == 1 &&
		desc.DepthOrArraySize == 1 &&
		desc.SampleDesc.Count == 1 &&
		(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0 &&
		(desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) == 0 &&
		DxgiToVkFormat(desc.Format) != VK_FORMAT_UNDEFINED;

	if (shapeOk)
	{
		D3D12_HEAP_PROPERTIES heapProps{};
		D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE;
		if (SUCCEEDED(gameOut->GetHeapProperties(&heapProps, &heapFlags)) &&
			heapProps.Type == D3D12_HEAP_TYPE_DEFAULT &&
			(heapFlags & D3D12_HEAP_FLAG_SHARED) != 0)
		{
			auto imp = GetOrImportLocked(b, gameOut, true);
			eligible = (imp != nullptr && imp->zeroCopy);
		}
	}

	if (b.directOutEligible.size() >= 256u)
		b.directOutEligible.clear();
	b.directOutEligible.emplace(gameOut, eligible);
	return eligible;
}

// Records the entire Vulkan XeFG batch into a ring VkCommandBuffer while holding gBridge.mutex.
// Called early from ScheduleBatch so HookExecuteCommandLists only signals, submits, and waits (low RenderSubmit latency).
bool RecordVulkanBatchLocked(PendingBatch& batch) noexcept
{
	if (!gBridge.active.load(std::memory_order_acquire) || !gBridge.xefgContext)
		return false;

	if (!gBridge.retiredImports.empty() || !gBridge.retiredContexts.empty())
		ReapRetiredLocked();

	std::vector<std::shared_ptr<SharedImport>> batchImports;
	batchImports.reserve(8u + batch.frameCount);
	auto importOrNull = [&](ID3D12Resource* r, bool isOut) -> SharedImport* {
		if (!r)
			return nullptr;
		auto sp = GetOrImportLocked(gBridge, r, isOut);
		if (!sp)
			return nullptr;
		batchImports.push_back(sp);
		return sp.get();
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
		return false;

	SharedImport* outImports[kMaxFrames]{};
	ID3D12Resource* xvkOutPtrs[kMaxFrames]{};
	for (uint32_t i = 0; i < batch.frameCount; ++i)
	{
		outImports[i] = importOrNull(batch.sharedOutputs[i].Get(), true);
		if (!outImports[i])
			return false;
		xvkOutPtrs[i] = outImports[i]->xvkResource.Get();
	}

	// Query completed timeline value once, prefer an unowned slot that is ALREADY completed (0 ms wait),
	// and only fall back to waiting on the earliest-finishing unowned slot if all slots are still in flight.
	uint64_t completedVal = 0;
	bool haveCompletedVal = false;
	if (gBridge.device && gBridge.sharedSemaphore && gBridge.vkGetSemaphoreCounterValue)
	{
		haveCompletedVal = (gBridge.vkGetSemaphoreCounterValue(gBridge.device, gBridge.sharedSemaphore, &completedVal) == VK_SUCCESS);
	}

	uint32_t vkSlot = kRingSize;
	if (haveCompletedVal)
	{
		for (uint32_t probe = 0; probe < kRingSize; ++probe)
		{
			const uint32_t candidate = (gBridge.vkCursor + probe) % kRingSize;
			if (gBridge.vkSlotOwner[candidate] == 0 && gBridge.vkSlotDoneFence[candidate] <= completedVal)
			{
				vkSlot = candidate;
				break;
			}
		}
	}
	if (vkSlot >= kRingSize)
	{
		uint64_t bestFence = UINT64_MAX;
		for (uint32_t probe = 0; probe < kRingSize; ++probe)
		{
			const uint32_t candidate = (gBridge.vkCursor + probe) % kRingSize;
			if (gBridge.vkSlotOwner[candidate] == 0 && gBridge.vkSlotDoneFence[candidate] < bestFence)
			{
				bestFence = gBridge.vkSlotDoneFence[candidate];
				vkSlot = candidate;
			}
		}
	}
	if (vkSlot >= kRingSize)
		return false;
	gBridge.vkCursor = (vkSlot + 1u) % kRingSize;
	if (gBridge.vkSlotDoneFence[vkSlot] && (!haveCompletedVal || gBridge.vkSlotDoneFence[vkSlot] > completedVal))
	{
		VkSemaphoreWaitInfo waitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
		waitInfo.semaphoreCount = 1;
		waitInfo.pSemaphores = &gBridge.sharedSemaphore;
		waitInfo.pValues = &gBridge.vkSlotDoneFence[vkSlot];
		const auto slotWaitStart = std::chrono::steady_clock::now();
		const VkResult waitRes = gBridge.vkWaitSemaphores(gBridge.device, &waitInfo, 250'000'000ull);
		const auto slotWaitUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::steady_clock::now() - slotWaitStart).count());
		gBridge.totalSlotWaitUs.fetch_add(slotWaitUs, std::memory_order_relaxed);
		gBridge.slotWaitSamples.fetch_add(1u, std::memory_order_relaxed);
		if (waitRes == VK_ERROR_DEVICE_LOST)
		{
			MarkBridgeFaultLocked("vkWaitSemaphores", static_cast<int>(waitRes));
			return false;
		}
		if (waitRes != VK_SUCCESS)
		{
			// Transient VK_TIMEOUT: keep the live Vulkan device and in-flight submissions intact;
			// simply fall back to D3D12 for this frame instead of tearing down an active device.
			static std::atomic<uint64_t> lastTimeoutLogMs{ 0 };
			const uint64_t nowMs = GetTickCount64();
			uint64_t prevMs = lastTimeoutLogMs.load(std::memory_order_relaxed);
			if (nowMs - prevMs >= 5000u && lastTimeoutLogMs.compare_exchange_strong(prevMs, nowMs, std::memory_order_relaxed))
				Log::Warn("D3D12->Vulkan WMMA bridge: slot fence wait timed out (result {}), falling back to D3D12 for this frame without device teardown",
					static_cast<int>(waitRes));
			return false;
		}
	}

	VkCommandBuffer cmd = gBridge.cmdBufs[vkSlot];
	gBridge.vkResetCommandBuffer(cmd, 0);
	VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (gBridge.vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
		return false;

	// Open xvk recording BEFORE copyInput so any pendingInit_ (UNDEFINED -> GENERAL)
	// layout transitions on newly created xvk resources flush before we copy frame pixels into them.
	bool execOk = false;
	if (xvk::BeginRecording(gBridge.xvkDevice.Get(), cmd))
	{
		// Prepare D3D12 shared inputs & outputs for XeFG compute. Zero-copy imports only need a single
		// batched acquire barrier (zero vkCmdCopyImage); non-zero-copy fallbacks copy into xvk textures.
		SharedImport* const inputs[8] = { prevBB, prevD, prevMV, prevHL, nextBB, nextD, nextMV, nextHL };
		VkImageMemoryBarrier copyPreIn[16]{};
		VkImageMemoryBarrier computeReady[8 + kMaxFrames]{};
		VkImage xvkInImgs[8]{};
		uint32_t copyPreInCount = 0;
		uint32_t computeReadyCount = 0;
		bool anyInputCopy = false;
		for (uint32_t i = 0; i < 8; ++i)
		{
			if (!inputs[i])
				continue;
			VkImage xvkImg = xvk::ImageOf(inputs[i]->xvkResource.Get());
			xvkInImgs[i] = xvkImg;
			if (inputs[i]->zeroCopy)
			{
				computeReady[computeReadyCount++] = MakeImageBarrier(inputs[i]->vkImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
					0, VK_ACCESS_SHADER_READ_BIT);
			}
			else
			{
				anyInputCopy = true;
				copyPreIn[copyPreInCount++] = MakeImageBarrier(inputs[i]->vkImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
					0, VK_ACCESS_TRANSFER_READ_BIT);
				copyPreIn[copyPreInCount++] = MakeImageBarrier(xvkImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
					VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
				computeReady[computeReadyCount++] = MakeImageBarrier(xvkImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
					VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
			}
		}
		for (uint32_t i = 0; i < batch.frameCount; ++i)
		{
			if (outImports[i]->zeroCopy)
			{
				computeReady[computeReadyCount++] = MakeImageBarrier(outImports[i]->vkImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
					0, VK_ACCESS_SHADER_WRITE_BIT);
			}
		}
		if (copyPreInCount)
			gBridge.vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, copyPreInCount, copyPreIn);
		if (anyInputCopy)
		{
			for (uint32_t i = 0; i < 8; ++i)
			{
				if (inputs[i] && !inputs[i]->zeroCopy)
					RecordCopyImage(gBridge, cmd, inputs[i]->vkImage, xvkInImgs[i], inputs[i]->width, inputs[i]->height);
			}
		}
		if (computeReadyCount)
			gBridge.vkCmdPipelineBarrier(cmd,
				anyInputCopy ? (VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT) : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0, 0, nullptr, 0, nullptr, computeReadyCount, computeReady);

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
		return false;
	}

	// Publish generated outputs back to D3D12. Zero-copy outputs were written directly by XeFG compute
	// and only need a single release barrier; non-zero-copy fallbacks copy xvkOut -> shared vkImage first.
	VkImageMemoryBarrier preOut[kMaxFrames * 2]{};
	VkImageMemoryBarrier postOut[kMaxFrames]{};
	VkImage xvkOutImgs[kMaxFrames]{};
	uint32_t preOutCount = 0;
	uint32_t postOutCount = 0;
	bool anyOutputCopy = false;
	for (uint32_t i = 0; i < batch.frameCount; ++i)
	{
		if (outImports[i]->zeroCopy)
		{
			postOut[postOutCount++] = MakeImageBarrier(outImports[i]->vkImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_ACCESS_SHADER_WRITE_BIT, 0);
		}
		else
		{
			anyOutputCopy = true;
			VkImage xvkOut = xvk::ImageOf(outImports[i]->xvkResource.Get());
			xvkOutImgs[i] = xvkOut;
			preOut[preOutCount++] = MakeImageBarrier(xvkOut, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
			preOut[preOutCount++] = MakeImageBarrier(outImports[i]->vkImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				0, VK_ACCESS_TRANSFER_WRITE_BIT);
			postOut[postOutCount++] = MakeImageBarrier(outImports[i]->vkImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
				VK_ACCESS_TRANSFER_WRITE_BIT, 0);
		}
	}
	if (preOutCount)
		gBridge.vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, preOutCount, preOut);
	if (anyOutputCopy)
	{
		for (uint32_t i = 0; i < batch.frameCount; ++i)
		{
			if (!outImports[i]->zeroCopy)
				RecordCopyImage(gBridge, cmd, xvkOutImgs[i], outImports[i]->vkImage, outImports[i]->width, outImports[i]->height);
		}
	}
	if (postOutCount)
		gBridge.vkCmdPipelineBarrier(cmd,
			anyOutputCopy ? (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT) : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0, 0, nullptr, 0, nullptr, postOutCount, postOut);

	if (gBridge.vkEndCommandBuffer(cmd) != VK_SUCCESS)
		return false;

	uint64_t slotToken = gBridge.nextSlotToken++;
	if (slotToken == 0)
		slotToken = gBridge.nextSlotToken++;
	gBridge.vkSlotOwner[vkSlot] = slotToken;
	batch.keptImports = std::move(batchImports);
	batch.preRecorded = true;
	batch.anyOutputCopy = anyOutputCopy;
	batch.vkSlot = vkSlot;
	batch.vkSlotToken = slotToken;
	batch.contextEpoch = gBridge.contextEpoch;
	return true;
}

void SubmitBatchOnQueue(ID3D12CommandQueue* queue, ExecuteCommandListsFn origExec, PendingBatch& batch) noexcept
{
	std::lock_guard lock(gBridge.mutex);
	struct SlotReleaseGuard
	{
		uint32_t slot = kRingSize;
		uint64_t token = 0;
		void ReleaseNow() noexcept
		{
			if (slot < kRingSize && token != 0 && gBridge.vkSlotOwner[slot] == token)
				gBridge.vkSlotOwner[slot] = 0;
			slot = kRingSize;
			token = 0;
		}
		~SlotReleaseGuard() { ReleaseNow(); }
	} slotGuard{ batch.preRecorded ? batch.vkSlot : kRingSize, batch.preRecorded ? batch.vkSlotToken : 0 };

	// If the bridge is temporarily inactive/rebuilding or a concurrent context change invalidated this batch,
	// never execute or re-record stale frame data onto a mismatched context; copy the real frame (nextBackbuffer)
	// into every matching output texture via PostCopyRing so the game always presents a valid frame.
	if (!gBridge.active.load(std::memory_order_acquire) || !gBridge.xefgContext ||
		batch.contextEpoch != gBridge.contextEpoch || !SameContextSettings(gBridge.contextSettings, batch.settings))
	{
		slotGuard.ReleaseNow();
		gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
		RunRealFrameFallbackLocked(queue, origExec, batch);
		return;
	}
	const bool validOwnership = batch.preRecorded &&
		batch.vkSlot < kRingSize && batch.vkSlotToken != 0 && gBridge.vkSlotOwner[batch.vkSlot] == batch.vkSlotToken;
	if (!validOwnership)
	{
		slotGuard.ReleaseNow();
		if (!RecordVulkanBatchLocked(batch))
		{
			gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		slotGuard.slot = batch.vkSlot;
		slotGuard.token = batch.vkSlotToken;
	}
	const uint32_t vkSlot = batch.vkSlot;
	const bool anyOutputCopy = batch.anyOutputCopy;
	VkCommandBuffer cmd = gBridge.cmdBufs[vkSlot];

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
	// Use Vulkan 1.3 vkQueueSubmit2 (synchronization2) to avoid OptiScaler's Vulkan_wDx12::hk_vkQueueSubmit hook.
	VkSemaphoreSubmitInfo waitSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	waitSemInfo.semaphore = gBridge.sharedSemaphore;
	waitSemInfo.value = waitVal;
	waitSemInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

	VkCommandBufferSubmitInfo cmdInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	cmdInfo.commandBuffer = cmd;

	VkSemaphoreSubmitInfo signalSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
	signalSemInfo.semaphore = gBridge.sharedSemaphore;
	signalSemInfo.value = signalVal;
	signalSemInfo.stageMask = anyOutputCopy ? VK_PIPELINE_STAGE_2_COPY_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

	VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	submitInfo.waitSemaphoreInfoCount = 1;
	submitInfo.pWaitSemaphoreInfos = &waitSemInfo;
	submitInfo.commandBufferInfoCount = 1;
	submitInfo.pCommandBufferInfos = &cmdInfo;
	submitInfo.signalSemaphoreInfoCount = 1;
	submitInfo.pSignalSemaphoreInfos = &signalSemInfo;

	const VkResult submitRes = gBridge.vkQueueSubmit2(gBridge.queue, 1, &submitInfo, VK_NULL_HANDLE);
	if (submitRes != VK_SUCCESS)
	{
		gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
		if (submitRes == VK_ERROR_DEVICE_LOST)
			MarkBridgeFaultLocked("vkQueueSubmit2", static_cast<int>(submitRes));
		return;
	}
	gBridge.vkSlotDoneFence[vkSlot] = signalVal;
	const uint64_t prevSubmitted = gBridge.lastSubmittedVkFence.exchange(signalVal, std::memory_order_acq_rel);
	gBridge.prevSubmittedVkFence.store(prevSubmitted, std::memory_order_release);
	gBridge.lastSignaledBridgeFence.store(signalVal, std::memory_order_release);

	// 3) D3D12 waits on Vulkan completion (GPU-side wait, zero CPU stall).
	queue->Wait(useFence, signalVal);

	uint64_t batchDoneFence = signalVal;

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
			bool slotReady = WaitFenceCpu(useFence, gBridge.fenceEvent, ring.doneFence[slot], 500u);
			if (!slotReady)
			{
				// Never reset a command allocator that the GPU may still be reading (f5low-fg ROBUSTNESS.md):
				// park the busy allocator and swap in a fresh spare allocator to avoid TDR / device removal.
				ComPtr<ID3D12CommandAllocator> spare;
				if (ring.parkedAllocs.size() < 16u &&
					SUCCEEDED(ringDev->CreateCommandAllocator(useType, IID_PPV_ARGS(&spare))) && spare)
				{
					ring.parkedAllocs.push_back(std::move(ring.alloc[slot]));
					ring.alloc[slot] = std::move(spare);
					ring.doneFence[slot] = 0;
					slotReady = true;
					Log::Warn("D3D12->Vulkan WMMA bridge: post-copy fence timed out (500 ms), parked busy allocator (parked={})",
						ring.parkedAllocs.size());
				}
			}
			if (slotReady &&
				SUCCEEDED(ring.alloc[slot]->Reset()) &&
				SUCCEEDED(ring.list[slot]->Reset(ring.alloc[slot].Get(), nullptr)))
			{
				RecordD3D12CopiesBatched(ring.list[slot].Get(), batch.copies);
				ring.list[slot]->Close();

				ID3D12CommandList* postLists[1] = { ring.list[slot].Get() };
				origExec(queue, 1, postLists);

				const uint64_t doneVal = gBridge.fenceValue.fetch_add(1, std::memory_order_acq_rel) + 1;
				if (SUCCEEDED(queue->Signal(useFence, doneVal)))
				{
					gBridge.lastSignaledBridgeFence.store(doneVal, std::memory_order_release);
					batchDoneFence = doneVal;
				}
				ring.doneFence[slot] = doneVal;
			}
		}
	}

	// If ForgetResource evicted any import while this batch was outside gBridge.pending (between record and push,
	// or between pop and submit), bind its deferred destruction to THIS batch's exact GPU completion fence.
	if (!batch.keptImports.empty())
	{
		const auto retireAfter = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
		for (auto& imp : batch.keptImports)
		{
			if (!imp || !imp->evicted)
				continue;
			bool updated = false;
			for (auto& r : gBridge.retiredImports)
			{
				if (r.imp.get() == imp.get())
				{
					if (batchDoneFence > r.retireFence)
						r.retireFence = batchDoneFence;
					if (retireAfter > r.retireAfter)
						r.retireAfter = retireAfter;
					updated = true;
					break;
				}
			}
			if (!updated)
				gBridge.retiredImports.push_back({ batchDoneFence, retireAfter, imp });
		}
		batch.keptImports.clear();
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

bool ContextReady() noexcept
{
	return gBridge.active.load(std::memory_order_acquire) && gBridge.contextReady.load(std::memory_order_acquire);
}

bool WantsSharedTextures() noexcept
{
	return gBridge.active.load(std::memory_order_acquire) ||
		gBridge.recoveryNeeded.load(std::memory_order_acquire) ||
		gBridge.recoveryInFlight.load(std::memory_order_acquire);
}

void TeardownVulkanLocked() noexcept
{
	gBridge.contextReady.store(false, std::memory_order_release);
	gBridge.active.store(false, std::memory_order_release);

	// Wait (bounded) for any ScheduleBatch thread currently inside outside-mutex vkWaitSemaphores
	// before destroying gBridge.sharedSemaphore or gBridge.device.
	for (uint32_t i = 0; i < 150u && gBridge.paceWaitActive.load(std::memory_order_acquire) > 0u; ++i)
		Sleep(1);

	// Drain in-flight Vulkan submissions (bounded) before destroying imports, command pool, and device.
	(void)DrainVulkanQueueLocked(500'000'000ull);

	{
		std::lock_guard plock(gBridge.pendingMutex);
		gBridge.pending.clear();
		gBridge.pendingCount.store(0, std::memory_order_release);
	}

	for (auto& kv : gBridge.imports)
	{
		if (kv.second)
			DestroyImport(gBridge, *kv.second);
	}
	gBridge.imports.clear();
	ReapRetiredLocked(true);
	gBridge.directOutEligible.clear();
	gBridge.importCount.store(0, std::memory_order_release);

	if (gBridge.xefgContext)
	{
		XeFG::Destroy(gBridge.xefgContext);
		gBridge.xefgContext = nullptr;
	}
	gBridge.xvkList.Reset();
	gBridge.xvkAlloc.Reset();
	gBridge.xvkDevice.Reset();

	if (gBridge.device)
	{
		if (gBridge.sharedSemaphore && gBridge.vkDestroySemaphore)
			gBridge.vkDestroySemaphore(gBridge.device, gBridge.sharedSemaphore, nullptr);
		if (gBridge.cmdPool && gBridge.vkDestroyCommandPool)
			gBridge.vkDestroyCommandPool(gBridge.device, gBridge.cmdPool, nullptr);
		if (gBridge.vkDestroyDevice)
			gBridge.vkDestroyDevice(gBridge.device, nullptr);
	}
	gBridge.sharedSemaphore = VK_NULL_HANDLE;
	gBridge.cmdPool = VK_NULL_HANDLE;
	for (uint32_t i = 0; i < kRingSize; ++i)
	{
		gBridge.cmdBufs[i] = VK_NULL_HANDLE;
		gBridge.vkSlotDoneFence[i] = 0;
		gBridge.vkSlotOwner[i] = 0;
	}
	gBridge.device = VK_NULL_HANDLE;
	gBridge.queue = VK_NULL_HANDLE;
	gBridge.physical = VK_NULL_HANDLE;

	if (gBridge.instance && gBridge.gipa)
	{
		if (const auto destroyInst = reinterpret_cast<PFN_vkDestroyInstance>(gBridge.gipa(gBridge.instance, "vkDestroyInstance")))
			destroyInst(gBridge.instance, nullptr);
	}
	gBridge.instance = VK_NULL_HANDLE;

	gBridge.sharedFence.Reset();
	gBridge.nativeSharedFence.Reset();
	if (gBridge.fenceEvent)
	{
		CloseHandle(gBridge.fenceEvent);
		gBridge.fenceEvent = nullptr;
	}
	gBridge.fenceValue.store(0, std::memory_order_release);
	gBridge.lastSubmittedVkFence.store(0, std::memory_order_release);
	gBridge.prevSubmittedVkFence.store(0, std::memory_order_release);
	gBridge.lastSignaledBridgeFence.store(0, std::memory_order_release);
	gBridge.vkCursor = 0;

	ResetPostCopyRing(gBridge.directRing);
	ResetPostCopyRing(gBridge.computeRing);
	ResetPostCopyRing(gBridge.nativeDirectRing);
	ResetPostCopyRing(gBridge.nativeComputeRing);

	gBridge.contextFailed = false;
	gBridge.contextFailCount = 0;
	++gBridge.contextEpoch;
}

bool InstallLocked(ID3D12Device* d3dDevice, std::string& reason) noexcept
{
	reason.clear();
	if (gBridge.attempted)
	{
		reason = gBridge.reason;
		return gBridge.active.load(std::memory_order_acquire);
	}
	gBridge.attempted = true;

	auto fail = [&](const char* code) -> bool {
		{
			std::lock_guard rlock(gBridge.reasonMutex);
			gBridge.reason = code;
		}
		reason = code;
		gBridge.active.store(false, std::memory_order_release);
		return false;
	};

	const std::wstring envBridge = EnvString(L"XEFG_D3D12_VULKAN_WMMA");
	if (VulkanPolicy::IsOffWord(envBridge))
		return fail("environment_off");
	if (!VulkanPolicy::IsOnWord(envBridge) && !Plugin::options.d3d12VulkanWmmaBridge.load(std::memory_order_acquire))
		return fail("disabled_in_options");

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
	if (!gBridge.vulkanDll)
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

	// Optional diagnostic query: VK_AMD_shader_core_properties2 (active compute units & core features)
	uint32_t amdActiveCUs = 0;
	uint32_t amdCoreFeatures = 0;
	const bool hasAmdShaderCoreProps2 = HasExtension(exts, "VK_AMD_shader_core_properties2");
	if (hasAmdShaderCoreProps2)
	{
		struct LocalAmdShaderCoreProps2
		{
			VkStructureType sType;
			void* pNext;
			VkFlags shaderCoreFeatures;
			uint32_t activeComputeUnitCount;
		} amdCore2{ static_cast<VkStructureType>(1000227000), nullptr, 0, 0 };
		VkPhysicalDeviceProperties2 coreProps2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &amdCore2 };
		getProps2(matchedPhys, &coreProps2);
		amdCoreFeatures = amdCore2.shaderCoreFeatures;
		amdActiveCUs = amdCore2.activeComputeUnitCount;
	}

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

	// Pick a compute + transfer queue family. Prefer a dedicated Async Compute family (COMPUTE|TRANSFER without
	// GRAPHICS) so Vulkan XeFG init/compute does not collide with the D3D12 game's GFX ring on AMD RDNA 3/4,
	// unless XEFG_D3D12_VULKAN_WMMA_GFX_QUEUE=1 explicitly requests the graphics queue family.
	uint32_t famCount = 0;
	getQueueFams(matchedPhys, &famCount, nullptr);
	if (!famCount)
		return fail("no_queue_family");
	std::vector<VkQueueFamilyProperties> fams(famCount);
	getQueueFams(matchedPhys, &famCount, fams.data());

	const bool preferGfxQueue = VulkanPolicy::IsOnWord(EnvString(L"XEFG_D3D12_VULKAN_WMMA_GFX_QUEUE"));
	uint32_t chosenFamily = UINT32_MAX;
	for (uint32_t i = 0; i < famCount; ++i)
	{
		if ((fams[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) ==
			(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT))
		{
			const bool hasGraphics = (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
			if (chosenFamily == UINT32_MAX || (preferGfxQueue ? hasGraphics : !hasGraphics))
				chosenFamily = i;
			if (preferGfxQueue ? hasGraphics : !hasGraphics)
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
	{
		std::lock_guard rlock(gBridge.reasonMutex);
		gBridge.reason.clear();
	}
	gBridge.active.store(true, std::memory_order_release);

	Log::Info("D3D12->Vulkan WMMA bridge ready (vendor {:#06x}, queue family {}, VK_KHR_cooperative_matrix {}x{}x{} s8->s32, {} hardware WMMA shaders)",
		matchedVendor, chosenFamily, gBridge.tileM, gBridge.tileN, gBridge.tileK, gBridge.coopLayers);
	if (hasAmdShaderCoreProps2)
		Log::Info("D3D12->Vulkan WMMA bridge AMD core props: activeCUs={}, shaderCoreFeatures={:#x}",
			amdActiveCUs, amdCoreFeatures);
	return true;
}

bool Install(ID3D12Device* d3dDevice, std::string& reason) noexcept
{
	std::lock_guard lock(gBridge.mutex);
	return InstallLocked(d3dDevice, reason);
}


bool EnsureContext(const ContextSettings& settings) noexcept
{
	const auto tStart = std::chrono::steady_clock::now();
	auto elapsedMs = [&]() noexcept -> long long {
		return std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - tStart).count();
	};

	auto markFailed = [&]() noexcept {
		std::lock_guard lock(gBridge.mutex);
		if (!gBridge.contextFailed || !SameContextSettings(gBridge.failedContextSettings, settings))
			gBridge.contextFailCount = 1;
		else if (gBridge.contextFailCount < 3u)
			++gBridge.contextFailCount;
		gBridge.contextFailed = true;
		gBridge.contextFailedAt = std::chrono::steady_clock::now();
		gBridge.failedContextSettings = settings;
		return false;
	};

	// Never hold gBridge.mutex across XeFG::CreateContext / XeFG::Init / vkQueueSubmit2 / vkWaitForFences
	// so UI/render threads calling Snapshot() or ForgetResource() can never stall on async init.
	xefgd_handle_t oldCtx = nullptr;
	VkCommandBuffer initCmd = VK_NULL_HANDLE;
	{
		std::lock_guard lock(gBridge.mutex);
		if (!gBridge.active.load(std::memory_order_acquire) || !gBridge.xvkDevice)
			return false;

		if (gBridge.xefgContext && SameContextSettings(gBridge.contextSettings, settings))
			return true;

		const auto canRetryFailed = [&]() noexcept {
			return gBridge.contextFailCount < 3u &&
				(std::chrono::steady_clock::now() - gBridge.contextFailedAt) >= std::chrono::seconds(10);
		};
		if (gBridge.contextFailed && SameContextSettings(gBridge.failedContextSettings, settings) && !canRetryFailed())
			return false;

		// Drain in-flight Vulkan work with a bounded timeout so destroying any prior context and
		// reusing cmdBufs[0] for XeFG::Init weight uploads can never hang indefinitely.
		gBridge.contextReady.store(false, std::memory_order_release);
		if (!DrainVulkanQueueLocked(500'000'000ull))
		{
			MarkBridgeFaultLocked("DrainVulkanQueue", static_cast<int>(VK_TIMEOUT));
			gBridge.contextFailed = true;
			gBridge.failedContextSettings = settings;
			return false;
		}
		++gBridge.contextEpoch;
		{
			std::lock_guard plock(gBridge.pendingMutex);
			for (auto& b : gBridge.pending)
			{
				b.preRecorded = false;
				b.vkSlotToken = 0;
				b.contextEpoch = 0;
			}
		}
		for (uint32_t i = 0; i < kRingSize; ++i)
		{
			gBridge.vkSlotDoneFence[i] = 0;
			gBridge.vkSlotOwner[i] = 0;
		}

		ReapRetiredLocked();
		oldCtx = gBridge.xefgContext;
		gBridge.xefgContext = nullptr;
		initCmd = gBridge.cmdBufs[0];
	}

	if (oldCtx)
		XeFG::Destroy(oldCtx);

	XeFG::OverrideModel(settings.model);

	// Recorded xvk devices require an open VkCommandBuffer + xvk::BeginRecording()
	// during XeFG::Init because libxess_fg creates internal command lists to upload
	// neural network weights and initialize resources.
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

	Log::Info("D3D12->Vulkan WMMA bridge: async init start ({}x{}, format {}, model {})",
		settings.width, settings.height, static_cast<uint32_t>(settings.format), settings.model);

	xefgd_handle_t ctx = nullptr;
	const xefgd_result_t createRes = XeFG::CreateContext(gBridge.xvkDevice.Get(), &ctx);
	if (createRes != XEFGD_RESULT_SUCCESS || !ctx)
	{
		xvk::EndRecording(gBridge.xvkDevice.Get());
		gBridge.vkEndCommandBuffer(initCmd);
		Log::Warn("D3D12->Vulkan WMMA bridge: XeFG::CreateContext failed (result {}, {} ms), falling back to D3D12",
			static_cast<int>(createRes), elapsedMs());
		return markFailed();
	}
	Log::Info("D3D12->Vulkan WMMA bridge: XeFG::CreateContext ok ({} ms), starting XeFG::Init", elapsedMs());

	xefgd_init_params_t init{};
	init.outputResolution = { settings.width, settings.height };
	init.backBufferFormat = static_cast<uint32_t>(settings.format);
	init.initFlags = settings.initFlags;
	init.maxInterpolatedFrames = kMaxFrames;
	init.creationNodeMask = 1;
	init.visibleNodeMask = 1;
	init.uiMode = settings.uiMode;

	const xefgd_result_t initRes = XeFG::Init(ctx, init);
	Log::Info("D3D12->Vulkan WMMA bridge: XeFG::Init returned {} ({} ms)", static_cast<int>(initRes), elapsedMs());
	xvk::EndRecording(gBridge.xvkDevice.Get());
	const VkResult endRes = gBridge.vkEndCommandBuffer(initCmd);

	if (initRes != XEFGD_RESULT_SUCCESS || endRes != VK_SUCCESS)
	{
		XeFG::Destroy(ctx);
		Log::Warn("D3D12->Vulkan WMMA bridge: XeFG::Init failed (result {}, vkEnd {}, {} ms), falling back to D3D12",
			static_cast<int>(initRes), static_cast<int>(endRes), elapsedMs());
		return markFailed();
	}

	// Submit XeFG::Init weight uploads via Vulkan 1.3 vkQueueSubmit2 (bypassing OptiScaler's
	// Vulkan_wDx12::hk_vkQueueSubmit hook) with a private local VkFence instead of the imported
	// D3D12 WDDM sharedSemaphore, so init never touches cross-API D3D12/Vulkan sync while the game renders.
	VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VkFence initFence = VK_NULL_HANDLE;
	if (gBridge.vkCreateFence(gBridge.device, &fenceInfo, nullptr, &initFence) != VK_SUCCESS || !initFence)
	{
		XeFG::Destroy(ctx);
		Log::Warn("D3D12->Vulkan WMMA bridge: vkCreateFence failed for XeFG::Init, falling back to D3D12");
		return markFailed();
	}

	VkCommandBufferSubmitInfo cmdInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
	cmdInfo.commandBuffer = initCmd;

	VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
	submitInfo.commandBufferInfoCount = 1;
	submitInfo.pCommandBufferInfos = &cmdInfo;

	Log::Info("D3D12->Vulkan WMMA bridge: submitting XeFG::Init weight upload on queue family {} ({} ms)",
		gBridge.queueFamily, elapsedMs());
	const VkResult subRes = gBridge.vkQueueSubmit2(gBridge.queue, 1, &submitInfo, initFence);
	if (subRes != VK_SUCCESS)
	{
		gBridge.vkDestroyFence(gBridge.device, initFence, nullptr);
		XeFG::Destroy(ctx);
		Log::Warn("D3D12->Vulkan WMMA bridge: XeFG::Init vkQueueSubmit2 failed (result {}, {} ms), falling back to D3D12",
			static_cast<int>(subRes), elapsedMs());
		return markFailed();
	}

	const VkResult waitRes = gBridge.vkWaitForFences(gBridge.device, 1, &initFence, VK_TRUE, 2'000'000'000ull);
	if (waitRes != VK_SUCCESS)
	{
		// Do not destroy initFence while the GPU submission may still be in flight.
		{
			std::lock_guard lock(gBridge.mutex);
			MarkBridgeFaultLocked("XeFG::Init vkWaitForFences", static_cast<int>(waitRes));
		}
		return markFailed();
	}
	gBridge.vkDestroyFence(gBridge.device, initFence, nullptr);

	if (gBridge.resetRequested.exchange(false, std::memory_order_acq_rel))
	{
		XeFG::Destroy(ctx);
		return false;
	}

	{
		std::lock_guard lock(gBridge.mutex);
		gBridge.xefgContext = ctx;
		gBridge.contextSettings = settings;
		gBridge.contextFailed = false;
		gBridge.contextFailCount = 0;
	}
	xvk::FlushCoopMatrixCache(gBridge.xvkDevice.Get());
	gBridge.contextReady.store(true, std::memory_order_release);
	Log::Info("D3D12->Vulkan WMMA bridge context ready ({} ms): {}x{}, format {}, model {}, {}x{}x{} hardware WMMA",
		elapsedMs(), settings.width, settings.height, static_cast<uint32_t>(settings.format), settings.model,
		gBridge.tileM, gBridge.tileN, gBridge.tileK);
	return true;
}

void WarmupContextAsync(const ContextSettings& settings) noexcept
{
	if (!Active())
		return;
	if (gBridge.initInFlight.exchange(true, std::memory_order_acq_rel))
		return;

	gBridge.resetRequested.store(false, std::memory_order_release);
	try
	{
		std::thread([settings]() noexcept {
			EnsureContext(settings);
			gBridge.initInFlight.store(false, std::memory_order_release);
		}).detach();
	}
	catch (...)
	{
		gBridge.initInFlight.store(false, std::memory_order_release);
	}
}

void ResetContext() noexcept
{
	gBridge.contextReady.store(false, std::memory_order_release);
	if (gBridge.initInFlight.load(std::memory_order_acquire))
	{
		gBridge.resetRequested.store(true, std::memory_order_release);
		return;
	}
	std::lock_guard lock(gBridge.mutex);
	gBridge.contextFailed = false;
	gBridge.contextFailCount = 0;
	if (gBridge.xefgContext)
	{
		if (DrainVulkanQueueLocked(500'000'000ull))
		{
			XeFG::Destroy(gBridge.xefgContext);
			ReapRetiredLocked();
		}
		else
		{
			const uint64_t fence = gBridge.lastSubmittedVkFence.load(std::memory_order_acquire);
			gBridge.retiredContexts.push_back({ fence, gBridge.xefgContext });
			Log::Warn("D3D12->Vulkan WMMA bridge: ResetContext drain timed out; deferring XeFG context destroy until fence {} completes", fence);
		}
		gBridge.xefgContext = nullptr;
		++gBridge.contextEpoch;
		gBridge.directOutEligible.clear();
	}
}

bool TryRecoverAsync(const ContextSettings& settings) noexcept
{
	if (!gBridge.recoveryNeeded.load(std::memory_order_acquire) ||
		gBridge.recoveryInFlight.load(std::memory_order_acquire) ||
		gBridge.initInFlight.load(std::memory_order_acquire))
		return false;

	ComPtr<ID3D12Device> d3dDev;
	uint32_t attempt = 0;
	{
		std::lock_guard lock(gBridge.mutex);
		if (!gBridge.recoveryNeeded.load(std::memory_order_acquire) ||
			gBridge.active.load(std::memory_order_acquire) ||
			gBridge.recoveryAttempts >= kMaxRecoveryAttempts)
			return false;
		if ((std::chrono::steady_clock::now() - gBridge.lastFaultAt) < kRecoveryCooldown)
			return false;
		if (!gBridge.d3dDevice || gBridge.d3dDevice->GetDeviceRemovedReason() != S_OK)
		{
			gBridge.recoveryNeeded.store(false, std::memory_order_release);
			return false;
		}
		d3dDev = gBridge.d3dDevice;
		attempt = ++gBridge.recoveryAttempts;
		gBridge.recoveryNeeded.store(false, std::memory_order_release);
		gBridge.recoveryInFlight.store(true, std::memory_order_release);
	}

	try
	{
		std::thread([d3dDev, settings, attempt]() noexcept {
			std::string reason;
			bool installed = false;
			{
				std::lock_guard lock(gBridge.mutex);
				TeardownVulkanLocked();
				gBridge.attempted = false;
				installed = InstallLocked(d3dDev.Get(), reason);
			}
			if (installed)
			{
				const bool ctxOk = EnsureContext(settings);
				if (ctxOk)
				{
					Log::Info("D3D12->Vulkan WMMA bridge: recovery attempt {}/{} succeeded", attempt, kMaxRecoveryAttempts);
				}
				else
				{
					Log::Warn("D3D12->Vulkan WMMA bridge: recovery attempt {}/{} reinstalled device but XeFG context init failed",
						attempt, kMaxRecoveryAttempts);
				}
			}
			else
			{
				Log::Warn("D3D12->Vulkan WMMA bridge: recovery attempt {}/{} failed ({})",
					attempt, kMaxRecoveryAttempts, reason.empty() ? "unknown" : reason);
			}
			gBridge.recoveryInFlight.store(false, std::memory_order_release);
		}).detach();
		return true;
	}
	catch (...)
	{
		gBridge.recoveryInFlight.store(false, std::memory_order_release);
		return false;
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
	D3D12_RESOURCE_STATES outputState,
	bool skipCpuPaceWait) noexcept
{
	if (!Active() || !cmdList || !sharedOutputs || !outputBases || !firstGameOutput || frameCount == 0 || frameCount > kMaxFrames)
		return false;
	if (!prev.backbuffer || !prev.depth || !prev.mvecs || !next.backbuffer || !next.depth || !next.mvecs)
		return false;

	// Never block the game/Streamline render thread while Vulkan context init is running asynchronously.
	if (gBridge.initInFlight.load(std::memory_order_acquire))
		return false;

	if (!gBridge.contextReady.load(std::memory_order_acquire))
	{
		{
			std::lock_guard lock(gBridge.mutex);
			if (gBridge.contextFailed && SameContextSettings(gBridge.failedContextSettings, settings))
			{
				const bool retryReady = gBridge.contextFailCount < 3u &&
					(std::chrono::steady_clock::now() - gBridge.contextFailedAt) >= std::chrono::seconds(10);
				if (!retryReady)
					return false;
			}
		}
		WarmupContextAsync(settings);
		return false;
	}

	// Snapshot the N-1 Vulkan batch fence (2 batches ago) rather than N (currently executing on GPU)
	// so normal 1-frame-in-flight CPU/GPU overlap stays intact (0 ms RenderSubmit stall, no -2..3 FPS drop
	// or base-FPS pacing clumping) while still capping Vulkan queue depth if CPU ever runs >2 frames ahead.
	const uint64_t prevVkFence = !skipCpuPaceWait
		? gBridge.prevSubmittedVkFence.load(std::memory_order_acquire)
		: 0ull;

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
	batch.fallbackRealFrame = next.realBackbuffer ? next.realBackbuffer : next.backbuffer;
	batch.fallbackRealFrameState = next.realBackbufferState;
	batch.frameCount = frameCount;
	for (uint32_t i = 0; i < frameCount; ++i)
	{
		batch.sharedOutputs[i] = sharedOutputs[i];
		batch.outputBases[i] = outputBases[i];
	}

	// Drop any stale unsubmitted entry for the same command list first and free its owned slot only if token matches.
	struct StaleSlotRef { uint32_t slot = 0; uint64_t token = 0; };
	StaleSlotRef staleSlots[kRingSize]{};
	uint32_t staleSlotCount = 0;
	{
		std::lock_guard plock(gBridge.pendingMutex);
		for (auto it = gBridge.pending.begin(); it != gBridge.pending.end();)
		{
			if (it->cmdList == cmdList)
			{
				if (it->preRecorded && it->vkSlot < kRingSize && it->vkSlotToken != 0 && staleSlotCount < kRingSize)
					staleSlots[staleSlotCount++] = { it->vkSlot, it->vkSlotToken };
				it = gBridge.pending.erase(it);
			}
			else
			{
				++it;
			}
		}
		gBridge.pendingCount.store(static_cast<uint32_t>(gBridge.pending.size()), std::memory_order_release);
	}

	// Verify context settings, check cached zero-copy eligibility for firstGameOutput, and pre-record
	// the Vulkan command buffer now so HookExecuteCommandLists never stalls on CPU XeFG::Execute recording.
	{
		std::lock_guard lock(gBridge.mutex);
		for (uint32_t i = 0; i < staleSlotCount; ++i)
		{
			if (gBridge.vkSlotOwner[staleSlots[i].slot] == staleSlots[i].token)
				gBridge.vkSlotOwner[staleSlots[i].slot] = 0;
		}
		if (!gBridge.xefgContext || !SameContextSettings(gBridge.contextSettings, settings))
		{
			const bool alreadyFailed = gBridge.contextFailed && SameContextSettings(gBridge.failedContextSettings, settings) &&
				!(gBridge.contextFailCount < 3u && (std::chrono::steady_clock::now() - gBridge.contextFailedAt) >= std::chrono::seconds(10));
			gBridge.contextReady.store(false, std::memory_order_release);
			if (!alreadyFailed)
				WarmupContextAsync(settings);
			return false;
		}

		const bool directFirstOut = IsDirectGameOutputEligibleLocked(gBridge, firstGameOutput, outputState);
		if (directFirstOut)
		{
			batch.sharedOutputs[0] = firstGameOutput;
			batch.directFirstGameOutput = firstGameOutput;
			batch.directFirstOutputState = outputState;
			gBridge.directZeroCopyOuts.fetch_add(1, std::memory_order_relaxed);
		}
		else
		{
			batch.copies.push_back({ sharedOutputs[0], firstGameOutput, outputState });
		}

		if (!RecordVulkanBatchLocked(batch))
		{
			gBridge.fallbacks.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
	}

	gBridge.lastSkipCpuPaceWait.store(skipCpuPaceWait, std::memory_order_relaxed);

	// Overlap-preserving 1-frame-in-flight pacing (outside gBridge.mutex): wait for Batch N-1 after
	// Batch N+1 is already recorded. Bracket with paceWaitActive so TeardownVulkanLocked never destroys
	// the Vulkan device or timeline semaphore while this wait is executing.
	if (prevVkFence)
	{
		VkDevice waitDev = VK_NULL_HANDLE;
		VkSemaphore waitSem = VK_NULL_HANDLE;
		PFN_vkWaitSemaphores waitFn = nullptr;
		{
			std::lock_guard lock(gBridge.mutex);
			if (gBridge.active.load(std::memory_order_acquire) && gBridge.device && gBridge.sharedSemaphore && gBridge.vkWaitSemaphores)
			{
				waitDev = gBridge.device;
				waitSem = gBridge.sharedSemaphore;
				waitFn = gBridge.vkWaitSemaphores;
				gBridge.paceWaitActive.fetch_add(1u, std::memory_order_acq_rel);
			}
		}
		if (waitFn)
		{
			struct PaceWaitGuard { ~PaceWaitGuard() { gBridge.paceWaitActive.fetch_sub(1u, std::memory_order_release); } } guard;
			VkSemaphoreWaitInfo paceWait{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
			paceWait.semaphoreCount = 1;
			paceWait.pSemaphores = &waitSem;
			paceWait.pValues = &prevVkFence;
			const auto waitStart = std::chrono::steady_clock::now();
			const VkResult paceRes = waitFn(waitDev, &paceWait, 50'000'000ull);
			const auto waitUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - waitStart).count());
			gBridge.totalCpuWaitUs.fetch_add(waitUs, std::memory_order_relaxed);
			gBridge.cpuWaitSamples.fetch_add(1u, std::memory_order_relaxed);
			if (paceRes == VK_TIMEOUT)
				gBridge.paceTimeouts.fetch_add(1u, std::memory_order_relaxed);
		}
	}

	{
		std::lock_guard lock(gBridge.pendingMutex);
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
	if (!resource || gBridge.importCount.load(std::memory_order_acquire) == 0)
		return;

	// 1) Invalidate any unsubmitted PendingBatch referencing this resource BEFORE retiring its VkImage/VkDeviceMemory
	//    so HookExecuteCommandLists never submits a pre-recorded VkCommandBuffer with destroyed Vulkan handles
	//    (fixes 0x887A0005 DXGI_ERROR_DEVICE_REMOVED when switching WMMA -> XMX Port).
	struct StaleSlotRef { uint32_t slot = 0; uint64_t token = 0; };
	StaleSlotRef staleSlots[kRingSize]{};
	uint32_t staleSlotCount = 0;
	{
		std::lock_guard plock(gBridge.pendingMutex);
		for (auto& b : gBridge.pending)
		{
			bool references = (b.prevBackbuffer.Get() == resource || b.prevHudless.Get() == resource ||
				b.prevDepth.Get() == resource || b.prevMvecs.Get() == resource ||
				b.nextBackbuffer.Get() == resource || b.nextHudless.Get() == resource ||
				b.nextDepth.Get() == resource || b.nextMvecs.Get() == resource ||
				b.directFirstGameOutput.Get() == resource);
			if (!references)
			{
				for (uint32_t i = 0; i < b.frameCount && i < kMaxFrames; ++i)
					if (b.sharedOutputs[i].Get() == resource) { references = true; break; }
			}
			if (!references)
			{
				for (const auto& cp : b.copies)
					if (cp.sharedOutput.Get() == resource || cp.gameOutput.Get() == resource) { references = true; break; }
			}
			if (references)
			{
				if (b.preRecorded && b.vkSlot < kRingSize && b.vkSlotToken != 0 && staleSlotCount < kRingSize)
					staleSlots[staleSlotCount++] = { b.vkSlot, b.vkSlotToken };
				b.preRecorded = false;
				b.vkSlotToken = 0;
				b.contextEpoch = 0;
			}
		}
	}

	// 2) Always defer DestroyImport until BOTH the latest signaled D3D12+Vulkan bridge fence (including PostCopyRing)
	//    AND a 150 ms grace period have elapsed, avoiding synchronous vkFreeMemory while D3D12 still reads the shared texture.
	std::lock_guard lock(gBridge.mutex);
	for (uint32_t i = 0; i < staleSlotCount; ++i)
	{
		if (gBridge.vkSlotOwner[staleSlots[i].slot] == staleSlots[i].token)
			gBridge.vkSlotOwner[staleSlots[i].slot] = 0;
	}
	gBridge.directOutEligible.erase(resource);
	auto it = gBridge.imports.find(resource);
	if (it != gBridge.imports.end())
	{
		const uint64_t fence = std::max(
			gBridge.lastSubmittedVkFence.load(std::memory_order_acquire),
			gBridge.lastSignaledBridgeFence.load(std::memory_order_acquire));
		const auto retireAfter = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
		if (it->second)
		{
			it->second->evicted = true;
			gBridge.retiredImports.push_back({ fence, retireAfter, std::move(it->second) });
		}
		gBridge.imports.erase(it);
		gBridge.importCount.store(gBridge.imports.size(), std::memory_order_release);
	}
	ReapRetiredLocked();
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
	const uint64_t waitSamples = gBridge.cpuWaitSamples.load(std::memory_order_relaxed);
	s.avgCpuWaitMs = waitSamples ? (static_cast<float>(gBridge.totalCpuWaitUs.load(std::memory_order_relaxed)) / (1000.0f * static_cast<float>(waitSamples))) : 0.0f;
	const uint64_t slotSamples = gBridge.slotWaitSamples.load(std::memory_order_relaxed);
	s.avgSlotWaitMs = slotSamples ? (static_cast<float>(gBridge.totalSlotWaitUs.load(std::memory_order_relaxed)) / (1000.0f * static_cast<float>(slotSamples))) : 0.0f;
	s.paceTimeouts = gBridge.paceTimeouts.load(std::memory_order_relaxed);
	s.lastSkipCpuPaceWait = gBridge.lastSkipCpuPaceWait.load(std::memory_order_relaxed);
	{
		std::lock_guard lock(gBridge.reasonMutex);
		s.reason = gBridge.reason;
	}
	return s;
}
} // namespace D3D12VulkanWmma
