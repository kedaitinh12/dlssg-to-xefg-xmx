// The Vulkan runtime built into this dll, see VulkanRuntime.h: XeFG on a game's VkDevice.
//
// This is MFG-Unlocker's xess/vulkan/vk_provider.cpp (MIT, Copyright (c) 2026 Michael Robles, see NOTICE.md) ported to
// this repository: XeFG is driven through XeFG.h, the ported model comes from MfgKernels, the model follows the menu's
// network path, and the programs are the ones in xefg_mfg_programs\vulkan (VulkanPrograms.h). The tensor-core route of an
// NVIDIA GPU is in (VulkanTensor.h), selected by the CUDA network path. The cooperative-matrix route of an AMD RDNA3/RDNA4
// or Intel Arc GPU is in (VulkanCoopMat.h), selected by the WMMA, XMX Port and XMX (native) network paths. Left out: its GPU
// timing and its registration as an OptiScaler provider.
//
// Every evaluation records into the VkCommandBuffer the game gives NVSDK_NGX_VULKAN_EvaluateFeature. The game's images
// are copied into images of the layer's own device, XeFG records its dispatches on a D3D12 command list of that device,
// the layer turns them into Vulkan commands in the same command buffer, and the generated frame is copied into the
// game's output. Nothing is submitted here: the game submits its command buffer.
#include "../VulkanRuntime.h"

#include "../FrameGenerator.h"
#include "../Log.h"
#include "../MfgKernels.h"
#include "../MfgPort/Cuda/cuda_route.h"
#include "../NGX.h"
#include "../Plugin.h"
#include "../XeFgEnabler.h"
#include "../VulkanPrograms.h"
#include "../XeFG.h"
#include "VulkanCoopMat.h"
#include "VulkanHost.h"
#include "VulkanInputCopy.h"
#include "VulkanPolicy.h"
#include "../HudTierPolicy.h"
#include "VulkanTensor.h"
#include "xvk_extra.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <array>
#include <cmath>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace
{
	using Microsoft::WRL::ComPtr;

	constexpr uint32_t MaxGenerated = VulkanPolicy::MaxGenerated;
	static_assert(MaxGenerated == FrameGenerator::MaxFrames, "the D3D12 and the Vulkan route generate the same 6x");

	// The layout the game's images are in while it evaluates, as MFG-Unlocker reads and writes them
	constexpr VkImageLayout GameLayout = VK_IMAGE_LAYOUT_GENERAL;

	// A call into XeFG that threw
	constexpr xefgd_result_t XeFGThrew = -1000;

	// VulkanPolicy.h spells the formats out so it needs neither header, these are where they are checked
	static_assert(VulkanPolicy::Vk::R8G8B8A8Unorm == static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_UNORM));
	static_assert(VulkanPolicy::Vk::R8G8B8A8Srgb == static_cast<uint32_t>(VK_FORMAT_R8G8B8A8_SRGB));
	static_assert(VulkanPolicy::Vk::B8G8R8A8Unorm == static_cast<uint32_t>(VK_FORMAT_B8G8R8A8_UNORM));
	static_assert(VulkanPolicy::Vk::B8G8R8A8Srgb == static_cast<uint32_t>(VK_FORMAT_B8G8R8A8_SRGB));
	static_assert(VulkanPolicy::Vk::A2B10G10R10UnormPack32 == static_cast<uint32_t>(VK_FORMAT_A2B10G10R10_UNORM_PACK32));
	static_assert(VulkanPolicy::Vk::R16G16Sfloat == static_cast<uint32_t>(VK_FORMAT_R16G16_SFLOAT));
	static_assert(VulkanPolicy::Vk::R16G16B16A16Sfloat == static_cast<uint32_t>(VK_FORMAT_R16G16B16A16_SFLOAT));
	static_assert(VulkanPolicy::Vk::R32G32Sfloat == static_cast<uint32_t>(VK_FORMAT_R32G32_SFLOAT));
	static_assert(VulkanPolicy::Vk::R32G32B32A32Sfloat == static_cast<uint32_t>(VK_FORMAT_R32G32B32A32_SFLOAT));
	static_assert(VulkanPolicy::Vk::B10G11R11UfloatPack32 == static_cast<uint32_t>(VK_FORMAT_B10G11R11_UFLOAT_PACK32));

	static_assert(VulkanPolicy::Dxgi::Unknown == static_cast<uint32_t>(DXGI_FORMAT_UNKNOWN));
	static_assert(VulkanPolicy::Dxgi::R16G16B16A16Float == static_cast<uint32_t>(DXGI_FORMAT_R16G16B16A16_FLOAT));
	static_assert(VulkanPolicy::Dxgi::R32G32Float == static_cast<uint32_t>(DXGI_FORMAT_R32G32_FLOAT));
	static_assert(VulkanPolicy::Dxgi::R10G10B10A2Unorm == static_cast<uint32_t>(DXGI_FORMAT_R10G10B10A2_UNORM));
	static_assert(VulkanPolicy::Dxgi::R11G11B10Float == static_cast<uint32_t>(DXGI_FORMAT_R11G11B10_FLOAT));
	static_assert(VulkanPolicy::Dxgi::R8G8B8A8Unorm == static_cast<uint32_t>(DXGI_FORMAT_R8G8B8A8_UNORM));
	static_assert(VulkanPolicy::Dxgi::R16G16Float == static_cast<uint32_t>(DXGI_FORMAT_R16G16_FLOAT));
	static_assert(VulkanPolicy::Dxgi::R32Float == static_cast<uint32_t>(DXGI_FORMAT_R32_FLOAT));
	static_assert(VulkanPolicy::Dxgi::B8G8R8A8Unorm == static_cast<uint32_t>(DXGI_FORMAT_B8G8R8A8_UNORM));

	// NVSDK_NGX_Resource_VK as DLSS-G hands an image over: the view, the image, its subresource range, its format and
	// size, then which member of NVIDIA's union it is (0, an image view) and the read/write flag. Declared here because
	// NVIDIA's header isn't GPL compatible. MFG-Unlocker checks the same layout against that header
	struct NgxResource
	{
		VkImageView imageView;
		VkImage image;
		VkImageSubresourceRange range;
		VkFormat format;
		uint32_t width;
		uint32_t height;
		uint32_t type;
		bool readWrite;
	};

	static_assert(offsetof(NgxResource, image) == 8 && offsetof(NgxResource, range) == 16 &&
		offsetof(NgxResource, format) == 36 && offsetof(NgxResource, width) == 40 && offsetof(NgxResource, height) == 44 &&
		offsetof(NgxResource, type) == 48 && sizeof(NgxResource) == 56);

	constexpr uint32_t NgxResourceImageView = 0;

	// The device functions the runtime calls itself
#define VULKAN_RUNTIME_FNS(X)                                                                                         \
	X(vkCmdCopyImage) X(vkCmdPipelineBarrier) X(vkCmdBindPipeline) X(vkCmdPushDescriptorSetKHR) X(vkCmdPushConstants) \
	X(vkCmdDispatch) X(vkCreateSampler) X(vkDestroySampler) X(vkCreateDescriptorSetLayout)                            \
	X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateShaderModule)      \
	X(vkDestroyShaderModule) X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCmdCopyBuffer) X(vkCmdFillBuffer)

	struct Device
	{
		std::mutex mutex;
		bool initialized = false; // NVSDK_NGX_VULKAN_Init gave a device
		bool ready = false;       // the layer's device, XeFG and the copy pipelines exist
		// Why it isn't ready, a VulkanPolicy::Describe code. A refusal is final for the device and, the game having
		// one device, for the process
		std::string reason = "not_initialized";
		VkInstance instance = VK_NULL_HANDLE;
		VkPhysicalDevice physical = VK_NULL_HANDLE;
		VkDevice device = VK_NULL_HANDLE;
		struct RetiredFeature
		{
			VkQueryPool profPool = VK_NULL_HANDLE;
			xefgd_handle_t context = nullptr;
		};
		std::vector<RetiredFeature> retiredFeatures;
		PFN_vkGetInstanceProcAddr gipa = nullptr;
		uint32_t family = 0;
		uint32_t vendor = 0;
		ComPtr<ID3D12Device> d3d; // the layer's recorded device on `device`
		bool portInstalled = false;
		// The tensor-core route (VulkanTensor.h) is installed on the layer's device. When it isn't, tensorReason is the
		// VulkanPolicy::DescribeTensor code
		bool tensorInstalled = false;
		std::string tensorReason = "not_attempted";
		// The cooperative-matrix route (VulkanCoopMat.h) is installed on the layer's device. When it isn't, coopReason
		// is the VulkanPolicy::DescribeCoopMatrix code
		bool coopInstalled = false;
		std::string coopReason = "not_attempted";

#define VULKAN_RUNTIME_FIELD(name) PFN_##name name = nullptr;
		VULKAN_RUNTIME_FNS(VULKAN_RUNTIME_FIELD)
#undef VULKAN_RUNTIME_FIELD

		VkSampler sampler = VK_NULL_HANDLE;
		VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipeline pipelines[VulkanInputCopy::Count] = {}; // 0 r32f (depth), 1 rg16f, 2 rg32f (motion vectors)

		// 3-Tier Adaptive HUD Protection compute pipelines (4 modes x 2 color formats: 0=rgba8/bgra8, 1=rgb10a2)
		VkDescriptorSetLayout hudSetLayout = VK_NULL_HANDLE;
		VkPipelineLayout hudPipelineLayout = VK_NULL_HANDLE;
		VkPipeline hudPipelines[4][2] = {};
		bool hudPipelinesReady = false;
		VkDescriptorSetLayout hudHybridSetLayout = VK_NULL_HANDLE;
		VkPipelineLayout hudHybridPipelineLayout = VK_NULL_HANDLE;
		VkPipeline hudHybridPipelines[2] = {};
		bool hudHybridReady = false;

		// Tier 3 Hybrid Tone-Map Matcher pipelines (clear, accumulate, median, fill, apply[rgba8/rgb10a2])
		VkDescriptorSetLayout hudTmSetLayout = VK_NULL_HANDLE;
		VkPipelineLayout hudTmPipelineLayout = VK_NULL_HANDLE;
		VkPipeline hudTmPipelines[4] = {};      // 0=clear, 1=accumulate, 2=median, 3=fill
		VkPipeline hudTmApplyPipelines[2] = {}; // 0=rgba8, 1=rgb10a2
		bool hudToneMatchReady = false;
		VkPipeline hudBgPipelines[3] = {};      // 0=clear, 1=accum, 2=solve
		VkPipeline hudBgApplyPipelines[2] = {}; // 0=rgba8, 1=rgb10a2
		bool hudBgReady = false;

		// Optional per-stage GPU timestamp profiler (zero overhead when off)
		PFN_vkCreateQueryPool vkCreateQueryPool = nullptr;
		PFN_vkDestroyQueryPool vkDestroyQueryPool = nullptr;
		PFN_vkCmdResetQueryPool vkCmdResetQueryPool = nullptr;
		PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp = nullptr;
		PFN_vkGetQueryPoolResults vkGetQueryPoolResults = nullptr;
		float timestampPeriodNs = 0.0f;
	};

	// The game owns the VkDevice and destroys it before the CRT is torn down. The layer's device is kept for the life of
	// the process: its destructor must not call into a dead VkDevice
	Device& gDevice = *new Device;

	// What the menu and supported() read, without waiting for an evaluation that holds gDevice.mutex
	std::mutex gStatusMutex;
	std::string gDeviceReason = "not_initialized";
	std::string gModelName;
	std::string gTensorReason = "not_attempted";
	std::string gCoopReason = "not_attempted";

	// The route is replacing layers: installed, and the CUDA path selected with the ported model (ApplyTensorGate)
	std::atomic_bool gTensorGate = false;
	std::atomic_bool gCoopGate = false;

	// One real frame's inputs, kept for the next batch as its previous frame
	struct Slot
	{
		ComPtr<ID3D12Resource> backbuffer, hudless, matchedHudless, ui, depth, motion;
		VkImageView bbView = VK_NULL_HANDLE;
		VkImageView hudlessView = VK_NULL_HANDLE;
		VkImageView matchedHudlessView = VK_NULL_HANDLE;
		VkImageView depthView = VK_NULL_HANDLE;
		VkImageView motionView = VK_NULL_HANDLE;
		bool hasHudless = false;
		bool hasUi = false;
		bool reset = false;
		float view[16]{};
		float projection[16]{};
		xefgd_2d_t bbBase{}, bbSize{};
		xefgd_2d_t depthBase{}, depthSize{};
		xefgd_2d_t motionBase{}, motionSize{};
		xefgd_2d_t hudlessBase{}, hudlessSize{};
		float jitterX = 0, jitterY = 0, motionScaleX = 0, motionScaleY = 0;
	};

	struct Feature
	{
		xefgd_handle_t context = nullptr;
		ComPtr<ID3D12CommandAllocator> allocator;
		ComPtr<ID3D12GraphicsCommandList> list;

		// What the context was created for, it is recreated when any of it changes
		uint32_t width = 0, height = 0;           // back buffer
		DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN; // back buffer (the UNORM view of an sRGB format)
		uint32_t depthWidth = 0, depthHeight = 0;
		DXGI_FORMAT motionFormat = DXGI_FORMAT_UNKNOWN;
		DXGI_FORMAT hudlessFormat = DXGI_FORMAT_UNKNOWN;
		DXGI_FORMAT uiFormat = DXGI_FORMAT_UNKNOWN;
		uint32_t uiMode = XEFGD_UI_MODE_AUTO;
		bool inverted = false;
		bool jittered = false;
		int32_t model = 0;
		int32_t loggedPath = -1;
		bool createFailed = false; // retried when settings or network path (including WMMA) change

		std::array<Slot, 2> slots;
		uint32_t current = 0;
		bool havePrevious = false;
		std::array<ComPtr<ID3D12Resource>, MaxGenerated> outputs;
		std::array<VkImageView, MaxGenerated> outputViews{};

		// 3-Tier Adaptive HUD Protection (zero-PCL-overhead inline compute + 2-slot non-blocking readback)
		HudTier::StateMachine hudTierState;
		HudTier::Decision hudDecision{};
		HudTier::ActiveTier loggedHudTier = HudTier::ActiveTier::Interpolated;
		bool hasLoggedHudTier = false;
		ComPtr<ID3D12Resource> staticCountTex;
		VkImageView staticCountView = VK_NULL_HANDLE;
		ComPtr<ID3D12Resource> hudCounterGpu;
		VkBuffer hudCounterGpuBuf = VK_NULL_HANDLE;
		ComPtr<ID3D12Resource> hudCounterReadback[2];
		VkBuffer hudCounterReadbackBuf[2]{};
		const uint32_t* hudMappedReadback[2]{};
		bool hudReadbackHasDiff[2]{};
		bool hudReadbackDiffFromToneMatch[2]{};
		bool hudReadbackHasStatic[2]{};
		uint64_t hudRealFrameCount = 0;
		uint32_t hudReadbackSlot = 0;
		bool hudResourcesReady = false;
		uint32_t activeBatchSlot = 0;

		// Tier 3/4 Tone-Map Matcher lazy buffers (1D LUT + 3D Bilateral Grid; 0 VRAM allocated unless active)
		ComPtr<ID3D12Resource> hudTmHistogram;
		VkBuffer hudTmHistogramBuf = VK_NULL_HANDLE;
		ComPtr<ID3D12Resource> hudTmLut;
		VkBuffer hudTmLutBuf = VK_NULL_HANDLE;
		bool hudTmResourcesReady = false;
		ComPtr<ID3D12Resource> hudBgStats;
		VkBuffer hudBgStatsBuf = VK_NULL_HANDLE;
		ComPtr<ID3D12Resource> hudBgModel;
		VkBuffer hudBgModelBuf = VK_NULL_HANDLE;
		uint32_t hudBgCellsX = 0, hudBgCellsY = 0;
		uint32_t hudBgFitCountdown = 0;
		bool hudBgResourcesReady = false;
		bool hudBgHasModel = false;
		int32_t activeToneMapMode = 0; // 0=off, 1=1D LUT, 2=3D Bilateral Grid

		// Optional per-stage GPU profiler (4-slot non-blocking ring, 4 timestamps per slot; lazy-allocated only when ON)
		static constexpr uint32_t kProfRing = 4u;
		VkQueryPool profPool = VK_NULL_HANDLE;
		uint32_t profSlot = 0;
		bool profWritten[kProfRing]{};
		bool profHasReading = false;
		float profTotalMs = 0.0f;
		float profInterpMs = 0.0f;
		float profHudMs = 0.0f;

		// Adaptive VRAM Guard (DXGI budget via physical device LUID, identical hysteresis to DX12)
		ComPtr<IDXGIAdapter3> vramAdapter;
		bool vramAdapterChecked = false;
		bool vramGuardActive = false;
		uint32_t vramPollCounter = 0;
		uint32_t vramUsageMiB = 0;
		uint32_t vramBudgetMiB = 0;

		uint32_t lastIndex = 0; // the previous evaluation's DLSSG.MultiFrameIndex
		uint32_t batchCount = 0;
		uint32_t requestedBatchCount = 0;
		bool batchReady = false;
		bool cameraLogged = false;
		uint64_t batches = 0, failures = 0, evaluations = 0, realCopies = 0, paused = 0;
		VkImage lastCopiedRealImage = VK_NULL_HANDLE;
		std::string fault;
		std::string initFault; // permanent missing-program failure for the current input configuration

		// Reference-mod stability guards: camera cut/teleport detector, DRS/subrect shift detector,
		// HUD-less mismatch one-shot warning, and consecutive execute-failure stall circuit breaker.
		float prevCameraPos[3]{};
		float prevCameraFwd[3]{};
		float prevCameraFov = 0.0f;
		bool hasPrevCamera = false;
		bool hasPrevFov = false;
		uint32_t syntheticResetQuietFrames = 0;
		XeFgEnabler::CameraMotionScale cameraMotionScale;
		bool hasPrevUiState = false;
		bool prevUiState = false;
		uint32_t uiTransitionCooldown = 0;
		std::array<uint32_t, 12> prevSubrects{};
		bool hasPrevSubrects = false;
		std::chrono::steady_clock::time_point lastRealFrameTime{};
		bool hasLastRealFrameTime = false;
		bool hudlessMismatchLogged = false;
		uint32_t consecutiveFailures = 0;
		uint32_t stallCooldownBatches = 0;

		// The batch's XeFG call, kept for its later evaluations, which each record their own output. XeFG takes the same
		// output array for every call of a pair
		xefgd_exec_params_t exec{};
		std::array<ID3D12Resource*, MaxGenerated> execOutputs{};
		std::array<xefgd_2d_t, MaxGenerated> execBases{};
		uint32_t generatedThrough = 0; // the batch's outputs recorded so far
	};

	struct Inputs
	{
		const NgxResource* backbuffer = nullptr;
		const NgxResource* depth = nullptr;
		const NgxResource* motion = nullptr;
		const NgxResource* hudless = nullptr;
		const NgxResource* ui = nullptr;
	};

	// ----------------------------------------------------------------------------------------------- helpers

	HMODULE SelfModule()
	{
		HMODULE module = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			reinterpret_cast<LPCWSTR>(&SelfModule), &module);

		return module;
	}

	bool EnvironmentOff(const wchar_t* name)
	{
		wchar_t value[16]{};
		const DWORD length = GetEnvironmentVariableW(name, value, 16);

		return length && length < 16 && (value[0] == L'0' || _wcsicmp(value, L"off") == 0 || _wcsicmp(value, L"false") == 0);
	}

	// XeFG's validators can throw out of the library, and a game's thread must not see it
	template <class Fn> xefgd_result_t CallXeFG(const char* what, Fn&& call)
	{
		try
		{
			return call();
		}
		catch (const std::exception& error)
		{
			Log::Error("XeFG {} threw: {}", what, error.what());
		}
		catch (...)
		{
			Log::Error("XeFG {} threw", what);
		}

		return XeFGThrew;
	}

	void XeFGLog(const char* message, xefgd_logging_level_t level, void*)
	{
		if (!message)
			return;

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

	const char* ModelText(int32_t model)
	{
		return model != MfgKernels::MfgModel ? "DP4a"
		     : (Plugin::options.path.load() == Plugin::Path::Wmma && gCoopGate.load() ? "WMMA" : "port");
	}

	// With gDevice.mutex held
	void Publish(const Device& d)
	{
		std::scoped_lock lock(gStatusMutex);
		gDeviceReason = d.reason;
		gTensorReason = d.tensorReason;
		gCoopReason = d.coopReason;
	}

	void SetModelName(int32_t model)
	{
		std::scoped_lock lock(gStatusMutex);
		gModelName = ModelText(model);
	}

	// A game's image the runtime can read: an image view with an image and a size
	const NgxResource* ImageParameter(const NGXParameter& parameters, const char* name)
	{
		void* value = nullptr;

		if (parameters.Get(name, &value) != NGX_SUCCESS || !value)
			return nullptr;

		const auto* resource = static_cast<const NgxResource*>(value);

		if (resource->type != NgxResourceImageView || !resource->image || !resource->width || !resource->height ||
			!resource->range.levelCount || resource->range.layerCount != 1)
			return nullptr;

		return resource;
	}

	bool SameSubresource(const NgxResource& a, const NgxResource& b)
	{
		return a.image == b.image && a.range.baseMipLevel == b.range.baseMipLevel &&
			a.range.baseArrayLayer == b.range.baseArrayLayer;
	}

	// Why a device the game created has no record: the reason the hooks gave, else what state they are in
	std::string NotPreparedReason(VkPhysicalDevice physical)
	{
		if (auto refusal = VulkanHost::RefusalFor(physical); !refusal.empty())
			return refusal;

		const auto host = VulkanHost::Snapshot();

		if (!host.enabled)
			return "hooks_off";

		if (host.hooks == VulkanRuntime::Hooks::Failed)
			return "hooks_failed";

		return "device_not_prepared";
	}

	// ------------------------------------------------------------------------------------------ recording

	void StageBarrier(const Device& d, VkCommandBuffer command,
		VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
		VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
	{
		VkMemoryBarrier memory{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, srcAccess, dstAccess };
		d.vkCmdPipelineBarrier(command, srcStage, dstStage, 0, 1, &memory, 0, nullptr, 0, nullptr);
	}

	void CopyImage(const Device& d, VkCommandBuffer command, VkImage source, VkImageLayout sourceLayout, VkImage target,
		VkImageLayout targetLayout, uint32_t width, uint32_t height, const NgxResource* sourceResource = nullptr,
		const NgxResource* targetResource = nullptr)
	{
		VkImageCopy region{};
		region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		if (sourceResource)
		{
			region.srcSubresource.mipLevel = sourceResource->range.baseMipLevel;
			region.srcSubresource.baseArrayLayer = sourceResource->range.baseArrayLayer;
		}
		if (targetResource)
		{
			region.dstSubresource.mipLevel = targetResource->range.baseMipLevel;
			region.dstSubresource.baseArrayLayer = targetResource->range.baseArrayLayer;
		}
		region.extent = { width, height, 1 };

		d.vkCmdCopyImage(command, source, sourceLayout, target, targetLayout, 1, &region);
	}

	// The target's width x height from the source's top left sourceWidth x sourceHeight, point sampled when they differ
	void RecordSampledCopy(const Device& d, VkCommandBuffer command, VkPipeline pipeline, VkImageView source,
		VkImageView target, uint32_t width, uint32_t height, uint32_t sourceWidth, uint32_t sourceHeight)
	{
		d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

		VkDescriptorImageInfo sourceInfo{ d.sampler, source, GameLayout };
		VkDescriptorImageInfo targetInfo{ VK_NULL_HANDLE, target, VK_IMAGE_LAYOUT_GENERAL };
		VkWriteDescriptorSet writes[2] = {
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1,
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sourceInfo },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				&targetInfo },
		};

		d.vkCmdPushDescriptorSetKHR(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipelineLayout, 0, 2, writes);

		const uint32_t size[4] = { width, height, sourceWidth, sourceHeight };
		d.vkCmdPushConstants(command, d.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(size), size);
		d.vkCmdDispatch(command, (width + 7) / 8, (height + 7) / 8, 1);
	}

	struct VkHudPushConstants
	{
		uint32_t width;
		uint32_t height;
		float tolerance;
		uint32_t flags;
	};

	void RecordHudDispatch(const Device& d, VkCommandBuffer command, VkPipeline pipeline,
		VkImageView view0, VkImageView view1, VkImageView view2, VkBuffer counterBuf,
		uint32_t width, uint32_t height, float tolerance, uint32_t flags, uint32_t sampleStride = 1u) noexcept
	{
		if (!pipeline || !view0 || !view1 || !view2 || !counterBuf)
			return;

		d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

		VkDescriptorImageInfo img0{ VK_NULL_HANDLE, view0, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorImageInfo img1{ VK_NULL_HANDLE, view1, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorImageInfo img2{ VK_NULL_HANDLE, view2, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorBufferInfo buf3{ counterBuf, 0, 16 };

		VkWriteDescriptorSet writes[4] = {
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img0, nullptr, nullptr },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img1, nullptr, nullptr },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 2, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img2, nullptr, nullptr },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 3, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf3, nullptr },
		};

		d.vkCmdPushDescriptorSetKHR(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudPipelineLayout, 0, 4, writes);
		const VkHudPushConstants pc{ width, height, tolerance, flags };
		d.vkCmdPushConstants(command, d.hudPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
		const uint32_t stride = sampleStride ? sampleStride : 1u;
		const uint32_t gridW = (width + stride - 1u) / stride;
		const uint32_t gridH = (height + stride - 1u) / stride;
		d.vkCmdDispatch(command, (gridW + 7u) / 8u, (gridH + 7u) / 8u, 1u);
	}

	void RecordHudHybridDispatch(const Device& d, VkCommandBuffer command, VkPipeline pipeline,
		VkImageView realView, VkImageView hudlessView, VkImageView outView, VkBuffer counterBuf,
		VkImageView staticCountView, uint32_t width, uint32_t height, float tolerance, uint32_t flags) noexcept
	{
		if (!pipeline || !realView || !hudlessView || !outView || !counterBuf || !staticCountView)
			return;

		d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

		VkDescriptorImageInfo img0{ VK_NULL_HANDLE, realView, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorImageInfo img1{ VK_NULL_HANDLE, hudlessView, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorImageInfo img2{ VK_NULL_HANDLE, outView, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorBufferInfo buf3{ counterBuf, 0, 16 };
		VkDescriptorImageInfo img4{ VK_NULL_HANDLE, staticCountView, VK_IMAGE_LAYOUT_GENERAL };

		VkWriteDescriptorSet writes[5] = {
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img0, nullptr, nullptr },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img1, nullptr, nullptr },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 2, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img2, nullptr, nullptr },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 3, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf3, nullptr },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 4, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img4, nullptr, nullptr },
		};

		d.vkCmdPushDescriptorSetKHR(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudHybridPipelineLayout, 0, 5, writes);
		const VkHudPushConstants pc{ width, height, tolerance, flags };
		d.vkCmdPushConstants(command, d.hudHybridPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
		d.vkCmdDispatch(command, (width + 7u) / 8u, (height + 7u) / 8u, 1u);
	}

	bool LoadSpirvFileBytes(const std::filesystem::path& path, std::vector<uint32_t>& outWords) noexcept
	{
		try
		{
			std::error_code ec;
			const auto bytes = std::filesystem::file_size(path, ec);
			if (ec || bytes < 20 || bytes > (1024u * 1024u) || (bytes % 4u) != 0)
				return false;
			std::ifstream file(path, std::ios::binary);
			if (!file)
				return false;
			outWords.resize(static_cast<size_t>(bytes / 4u));
			file.read(reinterpret_cast<char*>(outWords.data()), static_cast<std::streamsize>(bytes));
			return file.good() && outWords[0] == 0x07230203u;
		}
		catch (...)
		{
			return false;
		}
	}

	void CreateHudPipelines(Device& d) noexcept
	{
		d.hudPipelinesReady = false;
		if (!d.vkCmdFillBuffer || !d.vkCmdCopyBuffer)
			return;

		wchar_t modPath[MAX_PATH]{};
		if (!GetModuleFileNameW(SelfModule(), modPath, MAX_PATH))
			return;
		const std::filesystem::path baseDir = std::filesystem::path(modPath).parent_path();
		std::filesystem::path hudDir = baseDir / L"xefg_mfg_programs" / L"vulkan" / L"coopmat" / L"hud";
		std::error_code ec;
		if (!std::filesystem::is_directory(hudDir, ec))
			hudDir = baseDir / L"coopmat" / L"hud";
		if (!std::filesystem::is_directory(hudDir, ec))
			return;

		VkDescriptorSetLayoutBinding bindings[4] = {
			{ 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
		};
		VkDescriptorSetLayoutCreateInfo setInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
			VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, 4, bindings };
		VkPushConstantRange push{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkHudPushConstants) };

		if (d.vkCreateDescriptorSetLayout(d.device, &setInfo, nullptr, &d.hudSetLayout) != VK_SUCCESS)
			return;

		VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1,
			&d.hudSetLayout, 1, &push };
		if (d.vkCreatePipelineLayout(d.device, &layoutInfo, nullptr, &d.hudPipelineLayout) != VK_SUCCESS)
			return;

		static constexpr const char* kNames[4][2] = {
			{ "hud_diff_guard_rgba8.spv", "hud_diff_guard_rgb10a2.spv" },
			{ "hud_diff_composite_rgba8.spv", "hud_diff_composite_rgb10a2.spv" },
			{ "hud_static_update_rgba8.spv", "hud_static_update_rgb10a2.spv" },
			{ "hud_static_composite_rgba8.spv", "hud_static_composite_rgb10a2.spv" },
		};

		std::vector<uint32_t> words;
		for (int m = 0; m < 4; ++m)
		{
			for (int f = 0; f < 2; ++f)
			{
				if (!LoadSpirvFileBytes(hudDir / kNames[m][f], words))
					return;
				VkShaderModuleCreateInfo modInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
					words.size() * sizeof(uint32_t), words.data() };
				VkShaderModule mod = VK_NULL_HANDLE;
				if (d.vkCreateShaderModule(d.device, &modInfo, nullptr, &mod) != VK_SUCCESS)
					return;
				VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
				pipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
					VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };
				pipeInfo.layout = d.hudPipelineLayout;
				const VkResult res = d.vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
					&d.hudPipelines[m][f]);
				d.vkDestroyShaderModule(d.device, mod, nullptr);
				if (res != VK_SUCCESS)
					return;
			}
		}

		d.hudPipelinesReady = true;
		Log::Info("Vulkan 3-tier HUD compute pipelines ready (8/8 SPIR-V from {})", hudDir.string());

		// Optional fused Tier 3 Hybrid Composite pipelines (2 SPIR-V files in hud/ directory)
		d.hudHybridReady = false;
		VkDescriptorSetLayoutBinding hybridBindings[5] = {
			{ 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
		};
		VkDescriptorSetLayoutCreateInfo hybridSetInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
			VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, 5, hybridBindings };
		if (d.vkCreateDescriptorSetLayout(d.device, &hybridSetInfo, nullptr, &d.hudHybridSetLayout) == VK_SUCCESS)
		{
			VkPipelineLayoutCreateInfo hybridLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1,
				&d.hudHybridSetLayout, 1, &push };
			if (d.vkCreatePipelineLayout(d.device, &hybridLayoutInfo, nullptr, &d.hudHybridPipelineLayout) == VK_SUCCESS)
			{
				static constexpr const char* kHybridNames[2] = {
					"hud_hybrid_composite_rgba8.spv", "hud_hybrid_composite_rgb10a2.spv"
				};
				bool hybridOk = true;
				for (int f = 0; f < 2 && hybridOk; ++f)
				{
					if (!LoadSpirvFileBytes(hudDir / kHybridNames[f], words))
					{
						hybridOk = false;
						break;
					}
					VkShaderModuleCreateInfo modInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
						words.size() * sizeof(uint32_t), words.data() };
					VkShaderModule mod = VK_NULL_HANDLE;
					if (d.vkCreateShaderModule(d.device, &modInfo, nullptr, &mod) != VK_SUCCESS)
					{
						hybridOk = false;
						break;
					}
					VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
					pipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
						VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };
					pipeInfo.layout = d.hudHybridPipelineLayout;
					const VkResult res = d.vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
						&d.hudHybridPipelines[f]);
					d.vkDestroyShaderModule(d.device, mod, nullptr);
					if (res != VK_SUCCESS)
						hybridOk = false;
				}
				d.hudHybridReady = hybridOk;
			}
		}

		// Optional Tier 3 Hybrid Tone-Map Matcher pipelines (6 SPIR-V files in the same hud/ directory).
		struct VkToneMatchPushConstants {
			uint32_t width, height;
			float logMin, binsPerOctave;
			uint32_t bins, codes, minimumCount, unused;
		};
		VkDescriptorSetLayoutBinding tmBindings[5] = {
			{ 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
		};
		VkDescriptorSetLayoutCreateInfo tmSetInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
			VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, 5, tmBindings };
		VkPushConstantRange tmPush{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkToneMatchPushConstants) };
		if (d.vkCreateDescriptorSetLayout(d.device, &tmSetInfo, nullptr, &d.hudTmSetLayout) != VK_SUCCESS)
			return;
		VkPipelineLayoutCreateInfo tmLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1,
			&d.hudTmSetLayout, 1, &tmPush };
		if (d.vkCreatePipelineLayout(d.device, &tmLayoutInfo, nullptr, &d.hudTmPipelineLayout) != VK_SUCCESS)
			return;

		auto loadTmPipe = [&](const char* fname, VkPipeline& outPipe) -> bool {
			if (!LoadSpirvFileBytes(hudDir / fname, words))
				return false;
			VkShaderModuleCreateInfo modInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
				words.size() * sizeof(uint32_t), words.data() };
			VkShaderModule mod = VK_NULL_HANDLE;
			if (d.vkCreateShaderModule(d.device, &modInfo, nullptr, &mod) != VK_SUCCESS)
				return false;
			VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
			pipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
				VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };
			pipeInfo.layout = d.hudTmPipelineLayout;
			const VkResult res = d.vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &outPipe);
			d.vkDestroyShaderModule(d.device, mod, nullptr);
			return res == VK_SUCCESS;
		};

		if (loadTmPipe("tonemap_clear.spv", d.hudTmPipelines[0]) &&
			loadTmPipe("tonemap_accumulate.spv", d.hudTmPipelines[1]) &&
			loadTmPipe("tonemap_median.spv", d.hudTmPipelines[2]) &&
			loadTmPipe("tonemap_fill.spv", d.hudTmPipelines[3]) &&
			loadTmPipe("tonemap_apply_rgba8.spv", d.hudTmApplyPipelines[0]) &&
			loadTmPipe("tonemap_apply_rgb10a2.spv", d.hudTmApplyPipelines[1]))
		{
			d.hudToneMatchReady = true;
			Log::Info("Vulkan Tier 3 Hybrid Tone-Map Matcher pipelines ready (6/6 SPIR-V)");
		}

		if (loadTmPipe("tonemap_bg_clear.spv", d.hudBgPipelines[0]) &&
			loadTmPipe("tonemap_bg_accum.spv", d.hudBgPipelines[1]) &&
			loadTmPipe("tonemap_bg_solve.spv", d.hudBgPipelines[2]) &&
			loadTmPipe("tonemap_bg_apply_rgba8.spv", d.hudBgApplyPipelines[0]) &&
			loadTmPipe("tonemap_bg_apply_rgb10a2.spv", d.hudBgApplyPipelines[1]))
		{
			d.hudBgReady = true;
			Log::Info("Vulkan Tier 4 3D Bilateral-Grid Tone-Map Matcher pipelines ready (5/5 SPIR-V)");
		}
	}

	bool IsVkToneMapSupported(DXGI_FORMAT bbFormat) noexcept
	{
		return bbFormat == DXGI_FORMAT_R10G10B10A2_UNORM ||
			bbFormat == DXGI_FORMAT_R8G8B8A8_UNORM ||
			bbFormat == DXGI_FORMAT_B8G8R8A8_UNORM;
	}

	bool PollVkVramGuard(const Device& d, Feature& f) noexcept
	{
		if (!Plugin::options.vramGuard.load(std::memory_order_relaxed))
		{
			if (f.vramGuardActive)
			{
				f.vramGuardActive = false;
				Log::Info("Vulkan: Adaptive VRAM guard disabled by user, restoring full MFG multiplier");
			}
			return false;
		}

		if (!f.vramAdapterChecked)
		{
			f.vramAdapterChecked = true;
			ComPtr<IDXGIFactory4> factory;
			if (d.d3d && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
			{
				const LUID luid = d.d3d->GetAdapterLuid();
				ComPtr<IDXGIAdapter1> adapter;
				if (SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) && adapter)
					adapter->QueryInterface(IID_PPV_ARGS(&f.vramAdapter));
			}
		}

		if (!f.vramAdapter)
			return false;

		if ((f.vramPollCounter++ & 7u) != 0)
			return f.vramGuardActive;

		DXGI_QUERY_VIDEO_MEMORY_INFO memInfo{};
		if (FAILED(f.vramAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memInfo)) || memInfo.Budget == 0)
			return f.vramGuardActive;

		f.vramUsageMiB = static_cast<uint32_t>(memInfo.CurrentUsage / (1024ull * 1024ull));
		f.vramBudgetMiB = static_cast<uint32_t>(memInfo.Budget / (1024ull * 1024ull));

		constexpr uint64_t kSafetyMarginBytes = 200ull * 1024ull * 1024ull;
		constexpr uint64_t kReleaseMarginBytes = 400ull * 1024ull * 1024ull;

		if (!f.vramGuardActive)
		{
			if (memInfo.Budget > kSafetyMarginBytes && memInfo.CurrentUsage + kSafetyMarginBytes >= memInfo.Budget)
			{
				f.vramGuardActive = true;
				Log::Warn("Vulkan: Adaptive VRAM guard engaged ({} MiB / {} MiB budget): clamping MFG temporarily to 2x",
					f.vramUsageMiB, f.vramBudgetMiB);
			}
		}
		else
		{
			if (memInfo.Budget > kReleaseMarginBytes && memInfo.CurrentUsage + kReleaseMarginBytes < memInfo.Budget)
			{
				f.vramGuardActive = false;
				Log::Info("Vulkan: Adaptive VRAM guard released ({} MiB / {} MiB budget): restoring full MFG multiplier",
					f.vramUsageMiB, f.vramBudgetMiB);
			}
		}
		return f.vramGuardActive;
	}

	// ------------------------------------------------------------------------------------------ the device

	bool CreatePipelines(Device& d)
	{
		VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
		samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
		samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW =
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

		VkDescriptorSetLayoutBinding bindings[2] = {
			{ 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
			{ 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
		};
		VkDescriptorSetLayoutCreateInfo setInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
			VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, 2, bindings };
		VkPushConstantRange push{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };

		if (d.vkCreateSampler(d.device, &samplerInfo, nullptr, &d.sampler) != VK_SUCCESS ||
			d.vkCreateDescriptorSetLayout(d.device, &setInfo, nullptr, &d.setLayout) != VK_SUCCESS)
			return false;

		VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &d.setLayout,
			1, &push };

		if (d.vkCreatePipelineLayout(d.device, &layoutInfo, nullptr, &d.pipelineLayout) != VK_SUCCESS)
			return false;

		for (size_t i = 0; i < VulkanInputCopy::Count; ++i)
		{
			VkShaderModuleCreateInfo moduleInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
				VulkanInputCopy::Bytes[i], VulkanInputCopy::Modules[i] };
			VkShaderModule module = VK_NULL_HANDLE;

			if (d.vkCreateShaderModule(d.device, &moduleInfo, nullptr, &module) != VK_SUCCESS)
				return false;

			VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
			pipelineInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
				VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr };
			pipelineInfo.layout = d.pipelineLayout;

			const VkResult created = d.vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
				&d.pipelines[i]);
			d.vkDestroyShaderModule(d.device, module, nullptr);

			if (created != VK_SUCCESS)
				return false;
		}

		CreateHudPipelines(d);
		return true;
	}

	// The tensor-core route on the layer's device, on an NVIDIA GPU the hooks gave the NVX extensions: tried once per device,
	// and no failure here is one of the device's, the SPIR-V programs run the layers. Menu: CUDA greyed out when it can't
	// run, as the D3D12 probe does (a Vulkan game never gets that probe). With gDevice.mutex held
	void InstallTensorRoute(Device& d, const VulkanHost::DeviceRecord& record)
	{
		std::string reason;

		if (!d.portInstalled)
		{
			// The layers are the ported model's, XeFG's own DP4a model has none
			reason = "no_port";
			xefg_cuda::ReportVulkan(xefg_cuda::State::Unavailable, 0, 0, 0, reason.c_str());
			Log::Info("Vulkan tensor cores: unavailable, {}", VulkanPolicy::DescribeTensor(reason));
		}
		else if (record.vendor != 0x10DE)
		{
			// Not an NVIDIA GPU: the route was never an option, nothing to report
			reason = "not_nvidia";
		}
		else
		{
			d.tensorInstalled = VulkanTensor::Install(d.d3d.Get(), reason);
		}

		d.tensorReason = d.tensorInstalled ? std::string() : reason;

		auto unknown = Plugin::CudaSupport::Unknown;
		Plugin::cudaSupport.compare_exchange_strong(unknown,
			d.tensorInstalled ? Plugin::CudaSupport::Supported : Plugin::CudaSupport::Unsupported);
	}

	// The cooperative-matrix route on the layer's device, on an AMD RDNA3/RDNA4 (WMMA) or Intel Arc/Battlemage (XMX) GPU
	// the hooks gave VK_KHR_cooperative_matrix: tried once per device, with DP4a SPIR-V fallback on any layer. With gDevice.mutex held
	void InstallCoopMatrixRoute(Device& d, const VulkanHost::DeviceRecord& record)
	{
		std::string coopStatusReason;

		if (!d.portInstalled)
		{
			coopStatusReason = "no_port";
			xefg_cuda::ReportCoopMatrix(xefg_cuda::State::Unavailable, 0, 0, 0, 0, coopStatusReason.c_str());
			Log::Info("Vulkan cooperative matrix: unavailable, {}", VulkanPolicy::DescribeCoopMatrix(coopStatusReason));
		}
		else if (!record.coopMatrixExtensions && record.vendor == 0x10DE)
		{
			// NVIDIA GPUs use the native CUDA cubin tensor-core route (VK_NVX_binary_import) unless cooperative matrix is enabled
			coopStatusReason = "no_extensions";
		}
		else
		{
			// Native Vulkan keeps the standard accumulator layout on every driver.
			d.coopInstalled = VulkanCoopMat::Install(d.d3d.Get(), coopStatusReason);
		}

		d.coopReason = d.coopInstalled ? std::string() : coopStatusReason;

		auto unknown = Plugin::WmmaSupport::Unknown;
		Plugin::wmmaSupport.compare_exchange_strong(unknown,
			d.coopInstalled ? Plugin::WmmaSupport::Supported : Plugin::WmmaSupport::Unsupported);
	}

	// The layer's device on the game's, XeFG with its model, and the copy pipelines. With gDevice.mutex held
	bool PrepareDevice(Device& d)
	{
		VulkanHost::DeviceRecord record;

		if (!VulkanHost::Find(d.device, record))
		{
			d.reason = NotPreparedReason(d.physical);
			return false;
		}

		d.physical = record.physical;
		d.instance = record.instance;
		d.gipa = record.gipa;
		d.family = record.family;
		d.vendor = record.vendor;

		auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(d.gipa(d.instance, "vkGetDeviceProcAddr"));

		if (!gdpa)
		{
			d.reason = "device_functions";
			return false;
		}

		bool complete = true;

#define VULKAN_RUNTIME_LOAD(name)                                          \
	d.name = reinterpret_cast<PFN_##name>(gdpa(d.device, #name));          \
	complete &= d.name != nullptr;
		VULKAN_RUNTIME_FNS(VULKAN_RUNTIME_LOAD)
#undef VULKAN_RUNTIME_LOAD

		if (!complete)
		{
			d.reason = "device_functions";
			return false;
		}

		// Optional timestamp query support for per-stage GPU profiler (never blocks init if unavailable)
		d.vkCreateQueryPool = reinterpret_cast<PFN_vkCreateQueryPool>(gdpa(d.device, "vkCreateQueryPool"));
		d.vkDestroyQueryPool = reinterpret_cast<PFN_vkDestroyQueryPool>(gdpa(d.device, "vkDestroyQueryPool"));
		d.vkCmdResetQueryPool = reinterpret_cast<PFN_vkCmdResetQueryPool>(gdpa(d.device, "vkCmdResetQueryPool"));
		d.vkCmdWriteTimestamp = reinterpret_cast<PFN_vkCmdWriteTimestamp>(gdpa(d.device, "vkCmdWriteTimestamp"));
		d.vkGetQueryPoolResults = reinterpret_cast<PFN_vkGetQueryPoolResults>(gdpa(d.device, "vkGetQueryPoolResults"));
		if (const auto getProps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
			d.gipa(d.instance, "vkGetPhysicalDeviceProperties2")))
		{
			VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
			getProps2(d.physical, &p2);
			d.timestampPeriodNs = p2.properties.limits.timestampPeriod;
		}

		if (!VulkanPrograms::Installed().Usable())
		{
			d.reason = "no_vulkan_models";
			return false;
		}

		if (!XeFG::Load())
		{
			d.reason = "no_xefg";
			return false;
		}

		xvk::DeviceDesc desc;
		desc.instance = d.instance;
		desc.physicalDevice = d.physical;
		desc.device = d.device;
		desc.getInstanceProcAddr = d.gipa;
		desc.programModule = SelfModule();
		desc.recorded = true;
		desc.recordedFamily = d.family;
		// What the hooks enabled for the tensor-core route, NVIDIA GPUs only
		desc.tensorExtensions = record.tensorExtensions;
		desc.binaryImportSpec = record.binaryImportSpec;
		desc.imageViewHandleSpec = record.imageViewHandleSpec;
		// What the hooks enabled for the VK_KHR_cooperative_matrix route (AMD WMMA / Intel XMX)
		desc.coopMatrixExtensions = record.coopMatrixExtensions;
		desc.coopMatrixSpec = record.coopMatrixSpec;

		if (FAILED(xvk::CreateDevice(desc, IID_PPV_ARGS(&d.d3d))) || !d.d3d)
		{
			d.reason = "xvk_device";
			return false;
		}

		// Publishing the ported model patches libxess_fg.dll process wide, so it runs before the first context. It is
		// installed even when the menu selects the DP4a model, so switching the path needs no new process
		d.portInstalled = MfgKernels::Install(XeFG::Module());

		if (d.vendor != 0)
			Plugin::gpuVendorId.store(d.vendor, std::memory_order_relaxed);

		InstallTensorRoute(d, record);
		InstallCoopMatrixRoute(d, record);

		{
			const auto curPath = Plugin::options.path.load(std::memory_order_relaxed);
			const auto resolved = Plugin::ResolveEffectivePath(
				curPath, d.vendor, false, d.tensorInstalled, d.coopInstalled, d.portInstalled);
			if (resolved != curPath)
			{
				Plugin::RecordPathFallback(curPath, resolved);
				Plugin::options.path.store(resolved, std::memory_order_relaxed);
				Log::Warn("Vulkan: model '{}' is not supported on GPU vendor {:#06x}; automatically switched to '{}'",
					Plugin::PathName(curPath), d.vendor, Plugin::PathName(resolved));
			}
		}

		if (!CreatePipelines(d))
		{
			d.reason = "copy_pipelines";
			return false;
		}

		d.ready = true;
		d.reason.clear();

		Log::Info("Vulkan XeMFG ready on queue family {}, vendor {:#06x}: {} programs, ported model {}, tensor cores {}, coopmat {}, up to {}x",
			d.family, d.vendor, VulkanPrograms::Installed().programs.size(), d.portInstalled ? "installed" : "not available",
			d.tensorInstalled ? "ready" : "not available", d.coopInstalled ? "ready" : "not available", MaxGenerated + 1);

		return true;
	}

	// A refusal is final for the device. With gDevice.mutex held
	bool EnsureDevice(Device& d)
	{
		if (d.ready)
			return true;

		if (!d.initialized)
		{
			d.reason = "not_initialized";
			Publish(d);
			return false;
		}

		if (d.reason != "not_initialized" && d.reason != "pending")
			return false;

		const bool prepared = PrepareDevice(d);
		Publish(d);

		return prepared;
	}

	// ------------------------------------------------------------------------------------------- the images

	HRESULT CreateTexture(ID3D12Device* d3d, uint32_t width, uint32_t height, DXGI_FORMAT format,
		ComPtr<ID3D12Resource>& out)
	{
		out.Reset();

		D3D12_HEAP_PROPERTIES heap{ D3D12_HEAP_TYPE_DEFAULT };
		D3D12_RESOURCE_DESC desc{ D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, format, { 1, 0 },
			D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS };

		return d3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&out));
	}

	DXGI_FORMAT ColorFormat(const NgxResource* resource)
	{
		return resource ? static_cast<DXGI_FORMAT>(VulkanPolicy::ColorFormat(static_cast<uint32_t>(resource->format)).dxgi)
		                : DXGI_FORMAT_UNKNOWN;
	}

	// The model this evaluation runs: the ported one unless the menu picks DP4a or the port isn't there. Intel GPUs never
	// get here, so XeFG's own XMX network (17) isn't an option
	Plugin::Path EffectiveVulkanPath(const Device& d)
	{
		return Plugin::ResolveEffectivePath(
			Plugin::options.path.load(std::memory_order_relaxed),
			d.vendor,
			d.vendor == 0x8086u && d.coopInstalled,
			d.tensorInstalled,
			d.coopInstalled,
			d.portInstalled);
	}

	int32_t WantedModel(const Device& d)
	{
		const auto eff = EffectiveVulkanPath(d);
		return (d.portInstalled && eff != Plugin::Path::Fallback) ? MfgKernels::MfgModel : MfgKernels::Dp4aModel;
	}

	// The tensor-core route replaces layers while the ported model runs on the CUDA path (VulkanPolicy::UseTensorCores).
	// The path can change without a new context, CUDA, WMMA, and XMX Port run the same model and only what executes the
	// layers differs, so this is set on every batch. A layer the route can't take stays a SPIR-V program in the same frame
	void DrainRetiredFeaturesLocked(Device& d, uint64_t timeoutNs) noexcept;

	void ApplyTensorGate(const Device& d, int32_t model)
	{
		const auto path = EffectiveVulkanPath(d);
		const bool portModel = model == MfgKernels::MfgModel;
		const bool on = d.tensorInstalled && VulkanPolicy::UseTensorCores(portModel,
			path == Plugin::Path::Cuda, path == Plugin::Path::XmxNative);

		xvk::SetTensorCoresEnabled(d.d3d.Get(), on);

		if (gTensorGate.exchange(on) != on)
			Log::Info("Vulkan tensor cores: {}", on ? "on, the CUDA path runs the int8 layers on the tensor cores" : "off");

		// Native XeFG XMX is D3D12 Intel-only; on Vulkan, XMX (native) only maps to Intel cooperative-matrix hardware.
		const bool intelNativeXmx = (path == Plugin::Path::XmxNative && d.vendor == 0x8086);
		const bool coopOn = !on && d.coopInstalled && VulkanPolicy::UseCoopMatrix(portModel,
			path == Plugin::Path::XmxPort, intelNativeXmx, path == Plugin::Path::Cuda,
			path == Plugin::Path::Wmma);

		xvk::SetCoopMatrixEnabled(d.d3d.Get(), coopOn);

		if (gCoopGate.exchange(coopOn) != coopOn)
		{
			SetModelName(model);
			Log::Info("Vulkan cooperative matrix (WMMA/XMX): {}",
				coopOn ? "on, VK_KHR_cooperative_matrix runs the int8 layers on hardware matrix units" : "off");
		}
	}

	// The network the menu and OptiScaler show for the model that runs
	Plugin::Network NetworkOf(int32_t model)
	{
		if (model != MfgKernels::MfgModel)
			return Plugin::Network::Fallback;

		// The route turns itself off after a driver fault, then the layers are the SPIR-V ones
		if (gTensorGate.load() && xefg_cuda::VulkanActive())
			return Plugin::Network::Cuda;

		if (gCoopGate.load() && xefg_cuda::CoopMatrixActive() &&
			(Plugin::options.path.load() == Plugin::Path::Wmma || gDevice.vendor == 0x1002))
			return Plugin::Network::Wmma;

		return Plugin::Network::XmxPort;
	}

	// The images of one feature: back buffer size and format, depth size (the motion vectors are resampled to it). XeFG's
	// context is destroyed and created again when any of them changes. Inside the recording: XeFG creates its resources and
	// uploads at Init
	bool EnsureFeature(Device& d, Feature& f, const Inputs& in, bool inverted, bool jittered)
	{
		const auto& bb = *in.backbuffer;
		const auto& depth = *in.depth;
		const auto& motion = *in.motion;

		const auto color = VulkanPolicy::ColorFormat(static_cast<uint32_t>(bb.format));

		if (color.dxgi == VulkanPolicy::Dxgi::Unknown)
		{
			f.fault = "backbuffer_format_" + std::to_string(static_cast<int>(bb.format));
			return false;
		}

		const auto format = static_cast<DXGI_FORMAT>(color.dxgi);
		const auto motionFormat = static_cast<DXGI_FORMAT>(VulkanPolicy::MotionFormat(static_cast<uint32_t>(motion.format)));
		const DXGI_FORMAT hudlessFormat = ColorFormat(in.hudless);
		const DXGI_FORMAT uiFormat = ColorFormat(in.ui);
		const int32_t uiAlpha = Plugin::options.uiAlphaMode.load(std::memory_order_relaxed);
		uint32_t wantedUiMode = XEFGD_UI_MODE_AUTO;
		if (uiFormat != DXGI_FORMAT_UNKNOWN)
		{
			if (uiAlpha == 1)
				wantedUiMode = XEFGD_UI_MODE_UI_TEXTURE;
			else if (uiAlpha == 2)
				wantedUiMode = XEFGD_UI_MODE_UI_TEXTURE_PREMULTIPLIED;
		}
		const int32_t model = WantedModel(d);

		ApplyTensorGate(d, model);

		const auto path = static_cast<int32_t>(Plugin::options.path.load());
		if (path != f.loggedPath)
		{
			f.loggedPath = path;
			f.createFailed = false;
		}

		const bool sameSettings = f.width == bb.width && f.height == bb.height && f.format == format &&
			f.depthWidth == depth.width && f.depthHeight == depth.height && f.motionFormat == motionFormat &&
			f.hudlessFormat == hudlessFormat && f.uiFormat == uiFormat && f.uiMode == wantedUiMode &&
			f.inverted == inverted && f.jittered == jittered && f.model == model;

		if (sameSettings)
		{
			if (f.context)
				return true;
			if (!f.initFault.empty())
			{
				f.fault = f.initFault;
				return false;
			}
			if (f.createFailed)
				return false;
		}
		f.initFault.clear();

		if (f.context)
		{
			const bool idle = !d.d3d || xvk::WaitRecordedIdle(d.d3d.Get(), 150'000'000ull);
			if (idle)
			{
				DrainRetiredFeaturesLocked(d, 0ull);
				CallXeFG("Destroy", [&] { return XeFG::Destroy(f.context); });
			}
			else
			{
				Log::Warn("Vulkan: context settings change idle wait timed out (150 ms); deferring old XeFG context destroy");
				d.retiredFeatures.push_back({ VK_NULL_HANDLE, f.context });
			}
			f.context = nullptr;
		}

		// Clean resume after a successful resize/settings change following any prior stall or failure (RTX-Encore)
		f.havePrevious = false;
		f.createFailed = false;
		f.consecutiveFailures = 0;
		f.stallCooldownBatches = 0;
		f.uiMode = wantedUiMode;
		f.width = bb.width;
		f.height = bb.height;
		f.format = format;
		f.depthWidth = depth.width;
		f.depthHeight = depth.height;
		f.motionFormat = motionFormat;
		f.hudlessFormat = hudlessFormat;
		f.uiFormat = uiFormat;
		f.inverted = inverted;
		f.jittered = jittered;
		f.model = model;

		ID3D12Device* d3d = d.d3d.Get();
		const VkFormat motionView = motionFormat == DXGI_FORMAT_R32G32_FLOAT ? VK_FORMAT_R32G32_SFLOAT
		                                                                     : VK_FORMAT_R16G16_SFLOAT;

		for (Slot& slot : f.slots)
		{
			slot = {};

			if (FAILED(CreateTexture(d3d, f.width, f.height, format, slot.backbuffer)) ||
				FAILED(CreateTexture(d3d, f.depthWidth, f.depthHeight, DXGI_FORMAT_R32_FLOAT, slot.depth)) ||
				FAILED(CreateTexture(d3d, f.depthWidth, f.depthHeight, motionFormat, slot.motion)) ||
				(hudlessFormat != DXGI_FORMAT_UNKNOWN &&
					FAILED(CreateTexture(d3d, in.hudless->width, in.hudless->height, hudlessFormat, slot.hudless))) ||
				(uiFormat != DXGI_FORMAT_UNKNOWN &&
					FAILED(CreateTexture(d3d, in.ui->width, in.ui->height, uiFormat, slot.ui))))
			{
				f.fault = "input_images";
				f.createFailed = true;
				return false;
			}

			const VkFormat vkColorView = (format == DXGI_FORMAT_R10G10B10A2_UNORM)
				? VK_FORMAT_A2B10G10R10_UNORM_PACK32
				: (format == DXGI_FORMAT_B8G8R8A8_UNORM ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM);
			slot.bbView = xvk::StorageImageView(slot.backbuffer.Get(), vkColorView);
			slot.hudlessView = slot.hudless ? xvk::StorageImageView(slot.hudless.Get(), vkColorView) : VK_NULL_HANDLE;
			slot.depthView = xvk::StorageImageView(slot.depth.Get(), VK_FORMAT_R32_SFLOAT);
			slot.motionView = xvk::StorageImageView(slot.motion.Get(), motionView);

			if (!slot.depthView || !slot.motionView)
			{
				f.fault = "input_views";
				f.createFailed = true;
				return false;
			}
		}

		const VkFormat vkOutColorView = (format == DXGI_FORMAT_R10G10B10A2_UNORM)
			? VK_FORMAT_A2B10G10R10_UNORM_PACK32
			: (format == DXGI_FORMAT_B8G8R8A8_UNORM ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM);
		for (size_t i = 0; i < f.outputs.size(); ++i)
		{
			if (FAILED(CreateTexture(d3d, f.width, f.height, format, f.outputs[i])))
			{
				f.fault = "output_images";
				f.createFailed = true;
				return false;
			}
			f.outputViews[i] = xvk::StorageImageView(f.outputs[i].Get(), vkOutColorView);
		}

		// Reset lazy Tier 3 Tone-Map Matcher resources on feature recreation (re-allocated lazily only if Tier 3 needs them)
		f.hudTmResourcesReady = false;
		f.hudTmHistogram.Reset();
		f.hudTmHistogramBuf = VK_NULL_HANDLE;
		f.hudTmLut.Reset();
		f.hudTmLutBuf = VK_NULL_HANDLE;

		// Allocate 3-tier HUD compute resources (R8_UINT static map + 16-byte GPU counter + 2-slot mapped readback)
		f.hudResourcesReady = false;
		f.staticCountTex.Reset();
		f.staticCountView = VK_NULL_HANDLE;
		f.hudCounterGpu.Reset();
		f.hudCounterGpuBuf = VK_NULL_HANDLE;
		for (int i = 0; i < 2; ++i)
		{
			f.hudCounterReadback[i].Reset();
			f.hudCounterReadbackBuf[i] = VK_NULL_HANDLE;
			f.hudMappedReadback[i] = nullptr;
			f.hudReadbackHasDiff[i] = false;
			f.hudReadbackDiffFromToneMatch[i] = false;
			f.hudReadbackHasStatic[i] = false;
		}
		f.hudRealFrameCount = 0;
		f.hudReadbackSlot = 0;

		if (d.hudPipelinesReady && SUCCEEDED(CreateTexture(d3d, f.width, f.height, DXGI_FORMAT_R8_UINT, f.staticCountTex)))
		{
			f.staticCountView = xvk::StorageImageView(f.staticCountTex.Get(), VK_FORMAT_R8_UINT);
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
			D3D12_HEAP_PROPERTIES defaultHeap{ D3D12_HEAP_TYPE_DEFAULT };
			D3D12_HEAP_PROPERTIES readbackHeap{ D3D12_HEAP_TYPE_READBACK };

			bool ok = (f.staticCountView != VK_NULL_HANDLE) &&
				SUCCEEDED(d3d->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &gpuBufDesc,
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&f.hudCounterGpu)));
			if (ok)
				f.hudCounterGpuBuf = xvk::BufferOf(f.hudCounterGpu.Get());
			ok = ok && (f.hudCounterGpuBuf != VK_NULL_HANDLE);

			for (int i = 0; ok && i < 2; ++i)
			{
				ok = SUCCEEDED(d3d->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
					D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&f.hudCounterReadback[i])));
				if (ok)
				{
					f.hudCounterReadbackBuf[i] = xvk::BufferOf(f.hudCounterReadback[i].Get());
					void* mapped = nullptr;
					const D3D12_RANGE range{ 0, 16 };
					ok = (f.hudCounterReadbackBuf[i] != VK_NULL_HANDLE) &&
						SUCCEEDED(f.hudCounterReadback[i]->Map(0, &range, &mapped)) && mapped != nullptr;
					if (ok)
						f.hudMappedReadback[i] = static_cast<const uint32_t*>(mapped);
				}
			}
			f.hudResourcesReady = ok;
		}

		if (!f.allocator && FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&f.allocator))))
		{
			f.fault = "allocator";
			f.createFailed = true;
			return false;
		}

		// Process wide, they also affect other XeFG contexts. Must happen before the context picks its model
		XeFG::SetMaxInterpolatedFrames(MaxGenerated);
		XeFG::OverrideModel(model);

		const auto missingBefore = xvk::MissingProgramCount();
		xefgd_handle_t context = nullptr;
		xefgd_result_t result = CallXeFG("CreateContext", [&] { return XeFG::CreateContext(d3d, &context); });

		if (result != XEFGD_RESULT_SUCCESS || !context)
		{
			f.fault = "create_context_" + std::to_string(result);
			f.createFailed = true;
			return false;
		}

		XeFG::SetLoggingCallback(context, XEFGD_LOGGING_LEVEL_INFO, &XeFGLog);

		xefgd_init_params_t init{};
		init.outputResolution = { f.width, f.height };
		init.backBufferFormat = static_cast<uint32_t>(format);
		init.initFlags = (inverted ? XEFGD_INIT_FLAG_INVERTED_DEPTH : 0) | (jittered ? XEFGD_INIT_FLAG_JITTERED_MV : 0);
		init.maxInterpolatedFrames = MaxGenerated;
		init.creationNodeMask = init.visibleNodeMask = 1;
		// Automatic: from the inputs given, the HUDless color and the UI texture
		init.uiMode = wantedUiMode;

		result = CallXeFG("Init", [&] { return XeFG::Init(context, init); });

		if (result != XEFGD_RESULT_SUCCESS)
		{
			CallXeFG("Destroy", [&] { return XeFG::Destroy(context); });
			f.fault = "init_" + std::to_string(result);
			f.createFailed = true;
			if (xvk::MissingProgramCount() != missingBefore)
			{
				f.initFault = f.fault + "_missing_spirv";
				f.fault = f.initFault;
				Log::Error("Vulkan: missing shader during XeFG Init; retries stopped for this input configuration. "
					"Install the corrected Vulkan payload and restart the game");
			}
			return false;
		}

		f.context = context;
		SetModelName(model);

		Log::Info("Vulkan XeFG context: {}x{}, back buffer format {}{}, depth {}x{}, motion vector format {}, HUDless {}, "
			"UI texture {}, {}{}, model {} ({})",
			f.width, f.height, static_cast<uint32_t>(format), color.srgb ? " (sRGB)" : "", f.depthWidth, f.depthHeight,
			static_cast<uint32_t>(motionFormat), hudlessFormat != DXGI_FORMAT_UNKNOWN, uiFormat != DXGI_FORMAT_UNKNOWN,
			inverted ? "inverted depth" : "regular depth", jittered ? ", jittered motion vectors" : "", model,
			ModelText(model));

		return true;
	}

	xefgd_resource_t Region(ID3D12Resource* resource, uint32_t width, uint32_t height,
		xefgd_2d_t base = {}, xefgd_2d_t size = {})
	{
		xefgd_resource_t out{};
		out.pResource = resource;
		out.base = base;
		out.size = (size.x > 0 && size.y > 0) ? size : xefgd_2d_t{ width, height };

		return out;
	}

	void BuildViewMatrix(const NGXParameter& params, float out[16]) noexcept
	{
		auto finite = [](float v) noexcept { return std::isfinite(v) ? v : 0.0f; };
		const float px = finite(params.GetOr<float>("DLSSG.CameraPosX", 0.0f));
		const float py = finite(params.GetOr<float>("DLSSG.CameraPosY", 0.0f));
		const float pz = finite(params.GetOr<float>("DLSSG.CameraPosZ", 0.0f));
		const float rx = finite(params.GetOr<float>("DLSSG.CameraRightX", 0.0f));
		const float ry = finite(params.GetOr<float>("DLSSG.CameraRightY", 0.0f));
		const float rz = finite(params.GetOr<float>("DLSSG.CameraRightZ", 0.0f));
		const float ux = finite(params.GetOr<float>("DLSSG.CameraUpX", 0.0f));
		const float uy = finite(params.GetOr<float>("DLSSG.CameraUpY", 0.0f));
		const float uz = finite(params.GetOr<float>("DLSSG.CameraUpZ", 0.0f));
		const float fx = finite(params.GetOr<float>("DLSSG.CameraFwdX", 0.0f));
		const float fy = finite(params.GetOr<float>("DLSSG.CameraFwdY", 0.0f));
		const float fz = finite(params.GetOr<float>("DLSSG.CameraFwdZ", 0.0f));

		// Match D3D12 FrameGenerator::GetViewMatrix so XeFG receives real 3D camera motion for disocclusion/reprojection;
		// fall back to identity only when camera vectors are absent.
		if (fx == 0.0f && fy == 0.0f && fz == 0.0f)
		{
			for (int i = 0; i < 16; ++i)
				out[i] = (i % 5 == 0) ? 1.0f : 0.0f;
			return;
		}

		const float view[16] = {
			rx, ux, fx, 0.0f,
			ry, uy, fy, 0.0f,
			rz, uz, fz, 0.0f,
			-(px * rx + py * ry + pz * rz),
			-(px * ux + py * uy + pz * uz),
			-(px * fx + py * fy + pz * fz),
			1.0f
		};
		std::memcpy(out, view, sizeof(view));
	}

	bool CameraFromViewToClip(const NGXParameter& params, float& cameraNear, float& cameraFar, float& fov) noexcept
	{
		float(*viewToClip)[4] = nullptr;
		params.Get("DLSSG.CameraViewToClip", reinterpret_cast<void**>(&viewToClip));
		if (params.GetOr<uint32_t>("DLSSG.OrthoProjection", 0) || !viewToClip)
			return false;

		float m[4][4] = {};
		if (std::memcmp(viewToClip, m, sizeof(m)) == 0)
			return false;
		m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f;
		if (std::memcmp(viewToClip, m, sizeof(m)) == 0)
			return false;

		const double b = viewToClip[1][1];
		const double c = viewToClip[2][2];
		const double d = viewToClip[3][2];
		const double e = viewToClip[2][3];
		if (!std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d) || !std::isfinite(e) || b == 0.0)
			return false;

		const double sign = (e < 0.0) ? -1.0 : 1.0;
		const double nearDist = (c != 0.0) ? (-d / c) : 0.0;
		const double farDist = ((c - sign) != 0.0) ? (-d / (c - sign)) : 0.0;
		const double computedFov = 2.0 * std::atan(1.0 / std::abs(b));
		if (!std::isfinite(computedFov) || computedFov <= 0.0001 || computedFov >= 3.13)
			return false;

		const float n = static_cast<float>(std::abs(nearDist));
		const float f = static_cast<float>(std::abs(farDist));
		if (!std::isfinite(n) || n <= 0.0f)
			return false;
		cameraNear = n;
		cameraFar = (std::isfinite(f) && f > 0.0f) ? f : 10000.0f;
		fov = static_cast<float>(computedFov);
		return true;
	}

	void ReadClampedSubrect(const NGXParameter& params, const char* baseXKey, const char* baseYKey,
		const char* widthKey, const char* heightKey, uint32_t maxW, uint32_t maxH,
		xefgd_2d_t& outBase, xefgd_2d_t& outSize) noexcept
	{
		uint32_t bx = params.GetOr<uint32_t>(baseXKey, 0u);
		uint32_t by = params.GetOr<uint32_t>(baseYKey, 0u);
		uint32_t w = params.GetOr<uint32_t>(widthKey, 0u);
		uint32_t h = params.GetOr<uint32_t>(heightKey, 0u);
		if (bx >= maxW) bx = 0;
		if (by >= maxH) by = 0;
		if (w == 0 || bx + w > maxW) w = maxW - bx;
		if (h == 0 || by + h > maxH) h = maxH - by;
		outBase = { bx, by };
		outSize = { w, h };
	}

	bool EnsureVkToneMapResources(const Device& d, Feature& f) noexcept
	{
		if (f.hudTmResourcesReady)
			return true;
		if (!d.hudToneMatchReady || !d.d3d || f.width == 0 || f.height == 0 || !IsVkToneMapSupported(f.format))
			return false;

		ID3D12Device* d3d = d.d3d.Get();
		D3D12_HEAP_PROPERTIES defaultHeap{ D3D12_HEAP_TYPE_DEFAULT };
		auto makeBuf = [&](uint64_t bytes, ComPtr<ID3D12Resource>& res, VkBuffer& vkBuf) -> bool {
			D3D12_RESOURCE_DESC bd{};
			bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			bd.Width = bytes;
			bd.Height = 1;
			bd.DepthOrArraySize = 1;
			bd.MipLevels = 1;
			bd.SampleDesc.Count = 1;
			bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
			if (FAILED(d3d->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &bd,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&res))))
				return false;
			vkBuf = xvk::BufferOf(res.Get());
			return vkBuf != VK_NULL_HANDLE;
		};

		constexpr uint32_t kBins = 256u, kCodes = 1024u;
		if (!makeBuf(uint64_t(3u) * kBins * kCodes * 4ull, f.hudTmHistogram, f.hudTmHistogramBuf) ||
			!makeBuf(uint64_t(3u) * kBins * 4ull, f.hudTmLut, f.hudTmLutBuf))
			return false;

		const VkFormat vkColorView = (f.format == DXGI_FORMAT_R10G10B10A2_UNORM)
			? VK_FORMAT_A2B10G10R10_UNORM_PACK32
			: (f.format == DXGI_FORMAT_B8G8R8A8_UNORM ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM);
		for (Slot& s : f.slots)
		{
			if (FAILED(CreateTexture(d3d, f.width, f.height, f.format, s.matchedHudless)))
				return false;
			s.matchedHudlessView = xvk::StorageImageView(s.matchedHudless.Get(), vkColorView);
			if (!s.matchedHudlessView)
				return false;
		}
		f.hudTmResourcesReady = true;
		Log::Info("Vulkan: lazy-allocated Tier 3 Hybrid Tone-Map Matcher resources ({}x{})", f.width, f.height);
		return true;
	}

	bool EnsureVkBilateralResources(const Device& d, Feature& f) noexcept
	{
		if (f.hudBgResourcesReady)
			return true;
		if (!d.hudBgReady || !EnsureVkToneMapResources(d, f) || !d.d3d)
			return false;

		constexpr uint32_t kCell = 128u, kBins = 8u;
		const uint32_t cellsX = std::max(1u, (f.width + kCell - 1u) / kCell);
		const uint32_t cellsY = std::max(1u, (f.height + kCell - 1u) / kCell);
		const uint32_t totalCells = cellsX * cellsY * kBins;
		const uint64_t statsBytes = uint64_t(totalCells) * 22ull * 4ull;
		const uint64_t modelBytes = uint64_t(totalCells) * 12ull * 4ull;

		ID3D12Device* d3d = d.d3d.Get();
		D3D12_HEAP_PROPERTIES defaultHeap{ D3D12_HEAP_TYPE_DEFAULT };
		auto makeBuf = [&](uint64_t bytes, ComPtr<ID3D12Resource>& res, VkBuffer& vkBuf) -> bool {
			D3D12_RESOURCE_DESC bd{};
			bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			bd.Width = bytes;
			bd.Height = 1;
			bd.DepthOrArraySize = 1;
			bd.MipLevels = 1;
			bd.SampleDesc.Count = 1;
			bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
			if (FAILED(d3d->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &bd,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&res))))
				return false;
			vkBuf = xvk::BufferOf(res.Get());
			return vkBuf != VK_NULL_HANDLE;
		};

		if (!makeBuf(statsBytes, f.hudBgStats, f.hudBgStatsBuf) ||
			!makeBuf(modelBytes, f.hudBgModel, f.hudBgModelBuf))
			return false;

		f.hudBgCellsX = cellsX;
		f.hudBgCellsY = cellsY;
		f.hudBgHasModel = false;
		f.hudBgFitCountdown = 0;
		f.hudBgResourcesReady = true;
		Log::Info("Vulkan: lazy-allocated 3D Bilateral-Grid Tone-Map Matcher ({}x{}x{} grid)", cellsX, cellsY, kBins);
		return true;
	}

	void RecordVkToneMapMatch(const Device& d, Feature& f, VkCommandBuffer command, Slot& slot,
		VkImageView rawHudlessView, VkImageView rawBackbufferView) noexcept
	{
		if (!EnsureVkToneMapResources(d, f) || !rawHudlessView || !rawBackbufferView || !slot.matchedHudlessView)
			return;

		const int32_t tmPref = std::clamp(Plugin::options.toneMapMatchMode.load(std::memory_order_relaxed), 0, 2);
		const bool want3D = tmPref == 2;
		if (want3D && EnsureVkBilateralResources(d, f))
		{
			struct VkBilateralPushConstants {
				uint32_t width, height;
				uint32_t cellsX, cellsY;
				uint32_t cell, bins;
				float lambda;
				uint32_t totalStatsUints;
			};
			constexpr uint32_t kCell = 128u, kBins = 8u, kStride = 4u, kFitInterval = 4u;
			const uint32_t totalCells = f.hudBgCellsX * f.hudBgCellsY * kBins;
			const uint32_t totalStatsUints = totalCells * 22u;
			const VkBilateralPushConstants pc{ f.width, f.height, f.hudBgCellsX, f.hudBgCellsY, kCell, kBins, 1e-3f, totalStatsUints };
			const int fmtIdx = (f.format == DXGI_FORMAT_R10G10B10A2_UNORM) ? 1 : 0;

			VkDescriptorImageInfo img0{ d.sampler, rawHudlessView, GameLayout };
			VkDescriptorImageInfo img1{ d.sampler, rawBackbufferView, GameLayout };
			VkDescriptorImageInfo img2{ VK_NULL_HANDLE, slot.matchedHudlessView, VK_IMAGE_LAYOUT_GENERAL };
			VkDescriptorBufferInfo buf3{ f.hudBgStatsBuf, 0, VK_WHOLE_SIZE };
			VkDescriptorBufferInfo buf4{ f.hudBgModelBuf, 0, VK_WHOLE_SIZE };
			VkWriteDescriptorSet writes[5]{};
			writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img0 };
			writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img1 };
			writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img2 };
			writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf3 };
			writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf4 };

			d.vkCmdPushDescriptorSetKHR(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudTmPipelineLayout, 0, 5, writes);
			d.vkCmdPushConstants(command, d.hudTmPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

			const bool runFit = !f.hudBgHasModel || (f.hudBgFitCountdown == 0u);
			if (runFit)
			{
				d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudBgPipelines[0]);
				d.vkCmdDispatch(command, (totalStatsUints + 63u) / 64u, 1u, 1u);
				StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

				d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudBgPipelines[1]);
				const uint32_t sampleW = (f.width + kStride - 1u) / kStride;
				const uint32_t sampleH = (f.height + kStride - 1u) / kStride;
				d.vkCmdDispatch(command, (sampleW + 7u) / 8u, (sampleH + 7u) / 8u, 1u);
				StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

				d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudBgPipelines[2]);
				d.vkCmdDispatch(command, (totalCells + 63u) / 64u, 1u, 1u);
				StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

				f.hudBgHasModel = true;
				f.hudBgFitCountdown = kFitInterval - 1u;
			}
			else
			{
				--f.hudBgFitCountdown;
			}

			d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudBgApplyPipelines[fmtIdx]);
			d.vkCmdDispatch(command, (f.width + 7u) / 8u, (f.height + 7u) / 8u, 1u);
			StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
			f.activeToneMapMode = 2;
			return;
		}
		f.activeToneMapMode = 1;

		struct VkToneMatchPushConstants {
			uint32_t width, height;
			float logMin, binsPerOctave;
			uint32_t bins, codes, minimumCount, unused;
		};
		constexpr uint32_t kBins = 256u, kCodes = 1024u, kStride = 4u;
		const VkToneMatchPushConstants pc{ f.width, f.height, -16.0f, 12.0f, kBins, kCodes, 8u, 0u };
		const int fmtIdx = (f.format == DXGI_FORMAT_R10G10B10A2_UNORM) ? 1 : 0;

		VkDescriptorImageInfo img0{ d.sampler, rawHudlessView, GameLayout };
		VkDescriptorImageInfo img1{ d.sampler, rawBackbufferView, GameLayout };
		VkDescriptorImageInfo img2{ VK_NULL_HANDLE, slot.matchedHudlessView, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorBufferInfo buf3{ f.hudTmHistogramBuf, 0, VK_WHOLE_SIZE };
		VkDescriptorBufferInfo buf4{ f.hudTmLutBuf, 0, VK_WHOLE_SIZE };
		VkWriteDescriptorSet writes[5]{};
		writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img0 };
		writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img1 };
		writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img2 };
		writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf3 };
		writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf4 };

		d.vkCmdPushDescriptorSetKHR(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudTmPipelineLayout, 0, 5, writes);
		d.vkCmdPushConstants(command, d.hudTmPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

		// 1. clear
		d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudTmPipelines[0]);
		d.vkCmdDispatch(command, (3u * kBins * kCodes + 63u) / 64u, 1u, 1u);
		StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

		// 2. accumulate (1/16 stride)
		d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudTmPipelines[1]);
		const uint32_t sampleW = (f.width + kStride - 1u) / kStride;
		const uint32_t sampleH = (f.height + kStride - 1u) / kStride;
		d.vkCmdDispatch(command, (sampleW + 7u) / 8u, (sampleH + 7u) / 8u, 1u);
		StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

		// 3. median
		d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudTmPipelines[2]);
		d.vkCmdDispatch(command, kBins, 3u, 1u);
		StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

		// 4. fill
		d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudTmPipelines[3]);
		d.vkCmdDispatch(command, 3u, 1u, 1u);
		StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

		// 5. apply
		d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.hudTmApplyPipelines[fmtIdx]);
		d.vkCmdDispatch(command, (f.width + 7u) / 8u, (f.height + 7u) / 8u, 1u);
		StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
	}

	void DescribeFrame(const Feature& f, const Slot& slot, xefgd_frame_t& frame)
	{
		frame = {};
		ID3D12Resource* effectiveHudless = (f.hudDecision.runToneMapMatch && slot.matchedHudless)
			? slot.matchedHudless.Get()
			: (slot.hasHudless ? slot.hudless.Get() : nullptr);
		const bool useHudlessScene = f.hudDecision.useHudlessAsSceneSource && effectiveHudless != nullptr;
		frame.backBuffer = Region(useHudlessScene ? effectiveHudless : slot.backbuffer.Get(),
			f.width, f.height, slot.bbBase, slot.bbSize);
		frame.depth = Region(slot.depth.Get(), f.depthWidth, f.depthHeight, slot.depthBase, slot.depthSize);
		frame.motionVectors = Region(slot.motion.Get(), f.depthWidth, f.depthHeight, slot.motionBase, slot.motionSize);

		if (effectiveHudless && f.hudDecision.useXeFgNativeUi)
			frame.hudlessColor = Region(effectiveHudless, f.width, f.height, slot.hudlessBase, slot.hudlessSize);

		if (slot.hasUi)
			frame.uiTexture = Region(slot.ui.Get(), f.width, f.height);

		std::memcpy(frame.viewMatrix, slot.view, sizeof(frame.viewMatrix));
		std::memcpy(frame.projectionMatrix, slot.projection, sizeof(frame.projectionMatrix));

		frame.outputResolution[0] = static_cast<float>(f.width);
		frame.outputResolution[1] = static_cast<float>(f.height);
		frame.jitterOffset[0] = slot.jitterX;
		frame.jitterOffset[1] = slot.jitterY;
		frame.motionVectorScale[0] = slot.motionScaleX;
		frame.motionVectorScale[1] = slot.motionScaleY;
	}

	// XeFG records the outputs `range` names (from output 0 also the pair's shared work) with the batch's call, into the
	// open recording
	bool RecordOutputs(Device& d, Feature& f, VkCommandBuffer command, const VulkanPolicy::OutputRange& range)
	{
		f.exec.firstFrameIndex = range.first;
		f.exec.frameCount = range.count;

		ID3D12Device* d3d = d.d3d.Get();

		if (!f.list)
		{
			if (FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, f.allocator.Get(), nullptr,
					IID_PPV_ARGS(&f.list))))
			{
				f.fault = "command_list";
				return false;
			}
		}
		else if (FAILED(f.allocator->Reset()) || FAILED(f.list->Reset(f.allocator.Get(), nullptr)))
		{
			f.fault = "command_list_reset";
			return false;
		}

		const bool profWanted = Plugin::options.gpuProfiler.load(std::memory_order_relaxed) &&
			d.vkCreateQueryPool && d.vkCmdResetQueryPool && d.vkCmdWriteTimestamp && d.vkGetQueryPoolResults &&
			d.timestampPeriodNs > 0.0f;
		bool profActive = false;
		uint32_t profBase = 0;
		if (profWanted)
		{
			if (!f.profPool)
			{
				VkQueryPoolCreateInfo qci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
				qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
				qci.queryCount = Feature::kProfRing * 4u;
				(void)d.vkCreateQueryPool(d.device, &qci, nullptr, &f.profPool);
			}
			if (f.profPool)
			{
				const uint32_t readSlot = f.profSlot % Feature::kProfRing;
				if (f.profWritten[readSlot])
				{
					uint64_t ts[4]{};
					if (d.vkGetQueryPoolResults(d.device, f.profPool, readSlot * 4u, 4u, sizeof(ts), ts, sizeof(uint64_t),
						VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[3] >= ts[0])
					{
						const float scaleMs = d.timestampPeriodNs * 1e-6f;
						const float tot = static_cast<float>(ts[3] - ts[0]) * scaleMs;
						const float interp = (ts[2] >= ts[1]) ? static_cast<float>(ts[2] - ts[1]) * scaleMs : 0.0f;
						const float hud = (ts[3] >= ts[2]) ? static_cast<float>(ts[3] - ts[2]) * scaleMs : 0.0f;
						if (!f.profHasReading)
						{
							f.profTotalMs = tot;
							f.profInterpMs = interp;
							f.profHudMs = hud;
							f.profHasReading = true;
						}
						else
						{
							f.profTotalMs = f.profTotalMs * 0.85f + tot * 0.15f;
							f.profInterpMs = f.profInterpMs * 0.85f + interp * 0.15f;
							f.profHudMs = f.profHudMs * 0.85f + hud * 0.15f;
						}
					}
				}
				profBase = readSlot * 4u;
				d.vkCmdResetQueryPool(command, f.profPool, profBase, 4u);
				d.vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, f.profPool, profBase + 0u);
				d.vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, f.profPool, profBase + 1u);
				profActive = true;
			}
		}

		const xefgd_result_t result = CallXeFG("Execute", [&] { return XeFG::Execute(f.context, f.list.Get(), f.exec); });
		f.list->Close();
		if (profActive)
			d.vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, f.profPool, profBase + 2u);
		StageBarrier(d, command,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT);

		if (result != XEFGD_RESULT_SUCCESS)
		{
			f.fault = "execute_" + std::to_string(result);
			return false;
		}

		// Inline Tier 2 (HUD-Diff Composite) or Tier 3 (Static HUD Mask) on generated outputs in-place (zero extra submit / PCL).
		if (d.hudPipelinesReady && f.hudResourcesReady &&
			(f.hudDecision.runHudDiffComposite || f.hudDecision.runStaticHudComposite))
		{
			const int fmtIdx = (f.format == DXGI_FORMAT_R10G10B10A2_UNORM) ? 1 : 0;
			const Slot& curSlot = f.slots[f.activeBatchSlot];
			const bool debugMagenta = Plugin::options.hudDebugMagenta.load(std::memory_order_relaxed);
			bool dispatched = false;

			for (uint32_t k = 0; k < range.count; ++k)
			{
				const uint32_t outIdx = range.first + k;
				if (outIdx >= f.outputViews.size() || !f.outputViews[outIdx])
					continue;

				const bool useMatched = f.hudDecision.runToneMapMatch && curSlot.matchedHudlessView != VK_NULL_HANDLE;
				VkImageView compositeHudlessView = useMatched ? curSlot.matchedHudlessView : curSlot.hudlessView;
				const bool canRunDiff = f.hudDecision.runHudDiffComposite && curSlot.bbView && compositeHudlessView;
				const bool canRunStatic = f.hudDecision.runStaticHudComposite && curSlot.bbView && f.staticCountView;
				if (canRunDiff && canRunStatic && d.hudHybridReady)
				{
					// Single fused Tier 3 Hybrid pass: 1/2 VRAM bandwidth, 0 intermediate compute barrier per generated frame.
					const bool updateCounter = (outIdx == 0);
					if (updateCounter)
					{
						d.vkCmdFillBuffer(command, f.hudCounterGpuBuf, 0, 4, 0u);
						StageBarrier(d, command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
					}
					const uint32_t flags = (updateCounter ? 1u : 0u) | (debugMagenta ? 2u : 0u);
					RecordHudHybridDispatch(d, command, d.hudHybridPipelines[fmtIdx],
						curSlot.bbView, compositeHudlessView, f.outputViews[outIdx], f.hudCounterGpuBuf,
						f.staticCountView, f.width, f.height, 0.02f, flags);
					dispatched = true;
					if (updateCounter)
					{
						StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
							VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
						const VkBufferCopy copy{ 0, 0, 4 };
						d.vkCmdCopyBuffer(command, f.hudCounterGpuBuf, f.hudCounterReadbackBuf[f.hudReadbackSlot], 1, &copy);
						f.hudReadbackHasDiff[f.hudReadbackSlot] = true;
						f.hudReadbackDiffFromToneMatch[f.hudReadbackSlot] = useMatched;
					}
				}
				else
				{
					if (canRunDiff)
					{
						const bool updateCounter = (outIdx == 0);
						if (updateCounter)
						{
							d.vkCmdFillBuffer(command, f.hudCounterGpuBuf, 0, 4, 0u);
							StageBarrier(d, command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
						}
						const uint32_t flags = (updateCounter ? 1u : 0u) | (debugMagenta ? 2u : 0u);
						RecordHudDispatch(d, command, d.hudPipelines[1][fmtIdx],
							curSlot.bbView, compositeHudlessView, f.outputViews[outIdx], f.hudCounterGpuBuf,
							f.width, f.height, 0.02f, flags);
						dispatched = true;
						if (updateCounter)
						{
							StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
								VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
							const VkBufferCopy copy{ 0, 0, 4 };
							d.vkCmdCopyBuffer(command, f.hudCounterGpuBuf, f.hudCounterReadbackBuf[f.hudReadbackSlot], 1, &copy);
							f.hudReadbackHasDiff[f.hudReadbackSlot] = true;
							f.hudReadbackDiffFromToneMatch[f.hudReadbackSlot] = useMatched;
						}
					}
					// In Tier 3 Hybrid fallback, Static HUD Mask runs alongside Tone-Map Match to lock fixed UI sharp.
					if (canRunStatic)
					{
						if (dispatched)
						{
							StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
						}
						const uint32_t flags = debugMagenta ? 2u : 0u;
						RecordHudDispatch(d, command, d.hudPipelines[3][fmtIdx],
							curSlot.bbView, f.staticCountView, f.outputViews[outIdx], f.hudCounterGpuBuf,
							f.width, f.height, 0.008f, flags);
						dispatched = true;
					}
				}
			}

			if (dispatched)
			{
				StageBarrier(d, command,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
			}
		}

		if (profActive)
		{
			d.vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, f.profPool, profBase + 3u);
			f.profWritten[f.profSlot % Feature::kProfRing] = true;
			++f.profSlot;
		}

		// Debug split-screen comparison on Vulkan (0% overhead when OFF): left = real backbuffer, right = generated/HUD output.
		if (Plugin::options.hudSplitCompare.load(std::memory_order_relaxed) && f.width > 1u && f.height > 0u)
		{
			const Slot& curSlot = f.slots[f.activeBatchSlot];
			const VkImage bbImg = curSlot.backbuffer ? xvk::ImageOf(curSlot.backbuffer.Get()) : VK_NULL_HANDLE;
			if (bbImg)
			{
				const float pos = std::clamp(Plugin::options.hudSplitPos.load(std::memory_order_relaxed), 0.05f, 0.95f);
				const uint32_t splitX = std::clamp(static_cast<uint32_t>(pos * static_cast<float>(f.width)), 1u, f.width - 1u);
				StageBarrier(d, command,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
				VkImageCopy region{};
				region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
				region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
				region.extent = { splitX, f.height, 1 };
				for (uint32_t k = 0; k < range.count; ++k)
				{
					const uint32_t outIdx = range.first + k;
					if (outIdx >= f.outputs.size() || !f.outputs[outIdx])
						continue;
					const VkImage outImg = xvk::ImageOf(f.outputs[outIdx].Get());
					if (outImg)
						d.vkCmdCopyImage(command, bbImg, VK_IMAGE_LAYOUT_GENERAL, outImg, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
				}
				StageBarrier(d, command,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
			}
		}

		f.generatedThrough = range.through;
		return true;
	}

	// The batch's first evaluation: the inputs into the current slot, XeFG's call for the batch, then the outputs `range`
	// names. Inside the recording
	bool GenerateBatch(Device& d, Feature& f, VkCommandBuffer command, const NGXParameter& parameters, uint32_t count,
		const VulkanPolicy::OutputRange& range)
	{
		Inputs in;
		in.backbuffer = ImageParameter(parameters, "DLSSG.Backbuffer");
		in.depth = ImageParameter(parameters, "DLSSG.Depth");
		in.motion = ImageParameter(parameters, "DLSSG.MVecs");

		const NgxResource* rawGameHudless = ImageParameter(parameters, "DLSSG.HUDLess");
		if (!rawGameHudless && in.backbuffer &&
			in.backbuffer->format != VK_FORMAT_R16G16B16A16_SFLOAT &&
			in.backbuffer->format != VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
			in.backbuffer->format != VK_FORMAT_A2R10G10B10_UNORM_PACK32)
		{
			if (const NgxResource* optiHudless = ImageParameter(parameters, "Opti.Hudless");
				optiHudless && optiHudless->image && optiHudless->image != in.backbuffer->image)
			{
				rawGameHudless = optiHudless;
				static bool loggedVkOptiHudless = false;
				if (!std::exchange(loggedVkOptiHudless, true))
					Log::Info("Vulkan UnityUpscalerBridge: using Opti.Hudless as HUD-less source (SDR)");
			}
		}
		const NgxResource* rawGameUi = ImageParameter(parameters, "DLSSG.UI");
		const auto vkHudMode = static_cast<HudTier::Mode>(
			std::clamp(Plugin::options.hudTierMode.load(std::memory_order_relaxed), 0, 4));
		if (Plugin::options.hudlessUi)
		{
			in.hudless = rawGameHudless;
			in.ui = rawGameUi;
		}


		if (!in.backbuffer || !in.depth || !in.motion)
		{
			f.fault = "inputs_missing";
			return false;
		}

		if (!in.depth->imageView || !in.motion->imageView ||
			!VulkanPolicy::SampledAspect(in.depth->range.aspectMask) ||
			in.motion->range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
			in.backbuffer->range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT)
		{
			f.fault = "input_view_or_aspect";
			return false;
		}

		// XeFG's UI inputs must cover the output extent and use a supported color format (dlssg-0.4.2 C86/C139).
		// If HUD-less is mismatched in size or format, fall back cleanly to UI Auto with a one-shot warning.
		const DXGI_FORMAT bbFormat = ColorFormat(in.backbuffer);
		auto usableUi = [&](const NgxResource* image, bool requireBbFormatMatch) {
			if (!image)
				return false;
			const DXGI_FORMAT fmt = ColorFormat(image);
			return image->width == in.backbuffer->width && image->height == in.backbuffer->height &&
				image->range.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT &&
				fmt != DXGI_FORMAT_UNKNOWN &&
				(!requireBbFormatMatch || fmt == bbFormat);
		};
		static const bool isNoMansSky = []() noexcept {
			wchar_t exePath[MAX_PATH]{};
			const DWORD len = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
			if (!len || len >= MAX_PATH)
				return false;
			const wchar_t* leaf = exePath;
			for (const wchar_t* p = exePath; *p; ++p)
				if (*p == L'\\' || *p == L'/')
					leaf = p + 1;
			return _wcsicmp(leaf, L"NMS.exe") == 0;
		}();
		if (in.hudless && isNoMansSky)
		{
			if (!f.hudlessMismatchLogged)
			{
				f.hudlessMismatchLogged = true;
				Log::Info("Vulkan: disabling HUD-less UI on No Man's Sky (NMS.exe) to prevent doubled outlines "
					"during camera motion (using UI Auto)");
			}
			in.hudless = nullptr;
		}
		bool vkHudlessFormatMismatch = false;
		const NgxResource* toneMatchHudless = nullptr;
		if (in.hudless)
		{
			const bool sameAsBackbuffer = (in.hudless->image == in.backbuffer->image);
			const uint32_t bbSubX = parameters.GetOr<uint32_t>("DLSSG.BackbufferSubrectBaseX", 0u);
			const uint32_t bbSubY = parameters.GetOr<uint32_t>("DLSSG.BackbufferSubrectBaseY", 0u);
			const uint32_t bbSubW = parameters.GetOr<uint32_t>("DLSSG.BackbufferSubrectWidth", in.backbuffer->width);
			const uint32_t bbSubH = parameters.GetOr<uint32_t>("DLSSG.BackbufferSubrectHeight", in.backbuffer->height);
			const uint32_t hlSubX = parameters.GetOr<uint32_t>("DLSSG.HUDLessSubrectBaseX", 0u);
			const uint32_t hlSubY = parameters.GetOr<uint32_t>("DLSSG.HUDLessSubrectBaseY", 0u);
			const uint32_t hlSubW = parameters.GetOr<uint32_t>("DLSSG.HUDLessSubrectWidth", in.hudless->width);
			const uint32_t hlSubH = parameters.GetOr<uint32_t>("DLSSG.HUDLessSubrectHeight", in.hudless->height);
			const bool subrectMismatch = (hlSubX != bbSubX || hlSubY != bbSubY || hlSubW != bbSubW || hlSubH != bbSubH);
			if (sameAsBackbuffer || subrectMismatch || !usableUi(in.hudless, false))
			{
				if (!f.hudlessMismatchLogged)
				{
					f.hudlessMismatchLogged = true;
					Log::Warn("Vulkan: HUD-less input ({}x{}, VkFormat {}, subrect {},{} {}x{}) rejected against backbuffer "
						"({}x{}, VkFormat {}, subrect {},{} {}x{}); falling back to UI Auto",
						in.hudless->width, in.hudless->height, static_cast<int>(in.hudless->format),
						hlSubX, hlSubY, hlSubW, hlSubH,
						in.backbuffer->width, in.backbuffer->height, static_cast<int>(in.backbuffer->format),
						bbSubX, bbSubY, bbSubW, bbSubH);
				}
				in.hudless = nullptr;
			}
			else if (!usableUi(in.hudless, true))
			{
				// Same geometry/subrect, different color format (e.g. RGBA16F HUDLess + R10G10B10A2 HDR10 Backbuffer):
				// keep raw pointer for Tier 3 Hybrid Tone-Map Matcher if supported, otherwise reject cleanly.
				if (d.hudToneMatchReady && IsVkToneMapSupported(bbFormat) && in.hudless->imageView && in.backbuffer->imageView)
				{
					vkHudlessFormatMismatch = true;
					toneMatchHudless = in.hudless;
					in.hudless = nullptr; // do not copy mismatched format into slot.hudless
				}
				else
				{
					if (!f.hudlessMismatchLogged)
					{
						f.hudlessMismatchLogged = true;
						Log::Warn("Vulkan: HUD-less format mismatch (VkFormat {} vs backbuffer VkFormat {}); falling back to UI Auto",
							static_cast<int>(in.hudless->format), static_cast<int>(in.backbuffer->format));
					}
					in.hudless = nullptr;
				}
			}
			else
			{
				toneMatchHudless = in.hudless;
			}
		}
		if (!usableUi(in.ui, false)) in.ui = nullptr;

		// 3-Tier Adaptive HUD Protection: poll non-blocking 2-frame-old readback (0 GPU stall / 0 PCL overhead)
		if (f.hudResourcesReady && f.hudRealFrameCount >= 2u && f.width > 0 && f.height > 0)
		{
			const uint32_t rslot = f.hudReadbackSlot;
			const uint64_t sampledTotal = static_cast<uint64_t>((f.width + 3u) / 4u) * static_cast<uint64_t>((f.height + 3u) / 4u);
			if (sampledTotal > 0 && f.hudMappedReadback[rslot])
			{
				if (f.hudReadbackHasDiff[rslot])
				{
					const float frac = std::clamp(static_cast<float>(f.hudMappedReadback[rslot][0]) / static_cast<float>(sampledTotal), 0.0f, 1.0f);
					f.hudTierState.FeedHudDiffFraction(frac, f.hudReadbackDiffFromToneMatch[rslot]);
					f.hudReadbackHasDiff[rslot] = false;
					f.hudReadbackDiffFromToneMatch[rslot] = false;
				}
				if (f.hudReadbackHasStatic[rslot])
				{
					const float frac = std::clamp(static_cast<float>(f.hudMappedReadback[rslot][1]) / static_cast<float>(sampledTotal), 0.0f, 1.0f);
					f.hudTierState.FeedStaticHudFraction(frac);
					f.hudReadbackHasStatic[rslot] = false;
				}
			}
		}

		HudTier::Inputs vkHudInputs{};
		vkHudInputs.mode = vkHudMode;
		vkHudInputs.hudlessUiEnabled = Plugin::options.hudlessUi.load(std::memory_order_relaxed);
		vkHudInputs.hasGameHudless = rawGameHudless != nullptr;
		vkHudInputs.hudlessStructurallyValid = in.hudless != nullptr;
		vkHudInputs.hudlessFormatMismatch = vkHudlessFormatMismatch;
		vkHudInputs.hasSeparateUiPlane = in.ui != nullptr;
		vkHudInputs.gpuComputeAvailable = d.hudPipelinesReady;
		vkHudInputs.toneMapMatchAvailable = d.hudToneMatchReady && IsVkToneMapSupported(bbFormat) &&
			toneMatchHudless != nullptr && toneMatchHudless->imageView != VK_NULL_HANDLE && in.backbuffer->imageView != VK_NULL_HANDLE;
		f.hudDecision = f.hudTierState.Evaluate(vkHudInputs);

		if (!f.hasLoggedHudTier || f.hudDecision.activeTier != f.loggedHudTier)
		{
			f.hasLoggedHudTier = true;
			f.loggedHudTier = f.hudDecision.activeTier;
			Log::Info("Vulkan HUD protection tier: {} (reason: {}, diff={:.2f}%, static={:.2f}%)",
				HudTier::ActiveTierShortName(f.hudDecision.activeTier), f.hudDecision.reason,
				f.hudTierState.LastDiffFraction() * 100.0f, f.hudTierState.LastStaticFraction() * 100.0f);
		}

		if (!f.hudDecision.useXeFgNativeUi && !f.hudDecision.useHudlessAsSceneSource && !f.hudDecision.runHudDiffGuard && !f.hudDecision.runToneMapMatch)
			in.hudless = nullptr;
		if (!f.hudDecision.useXeFgNativeUi)
			in.ui = nullptr;

		const bool inverted = XeFgEnabler::ApplyTriStateOverride(
			Plugin::options.depthInvertedOverride.load(std::memory_order_relaxed),
			parameters.GetOr<uint32_t>("DLSSG.DepthInverted", 0) != 0);
		const bool jittered = XeFgEnabler::ApplyTriStateOverride(
			Plugin::options.jitteredMvOverride.load(std::memory_order_relaxed),
			parameters.GetOr<uint32_t>("DLSSG.MvecJittered", 0) != 0);

		if (!EnsureFeature(d, f, in, inverted, jittered))
			return false;

		// Takes effect from XeFG's next Execute on; only call into libxess_fg.dll when the toggle changes
		const bool debugLines = Plugin::options.debugLines.load(std::memory_order_relaxed);
		static std::atomic<int> sLastDebugLines{ -1 };
		if (sLastDebugLines.exchange(debugLines ? 1 : 0, std::memory_order_relaxed) != (debugLines ? 1 : 0))
			XeFG::SetDebugLines(debugLines);

		const bool ignoreGameReset = Plugin::options.ignoreGameReset.load(std::memory_order_relaxed);
		bool reset = !ignoreGameReset && (parameters.GetOr<uint32_t>("DLSSG.Reset", 0) != 0);

		// Long-pause / Alt+Tab / window-minimize history reset (>500 ms between real frames), matching DX12.
		const auto nowReal = std::chrono::steady_clock::now();
		if (f.hasLastRealFrameTime &&
			std::chrono::duration_cast<std::chrono::milliseconds>(nowReal - f.lastRealFrameTime).count() > 500)
		{
			reset = true;
		}
		f.lastRealFrameTime = nowReal;
		f.hasLastRealFrameTime = true;

		// DRS / Subrect Change Detector (dlssg-0.4.2 M3): force history reset if any active subrect changes
		// between consecutive batches even when the game omits DLSSG.Reset.
		const std::array<uint32_t, 12> curSubrects = {
			parameters.GetOr<uint32_t>("DLSSG.BackbufferSubrectBaseX", 0u),
			parameters.GetOr<uint32_t>("DLSSG.BackbufferSubrectBaseY", 0u),
			parameters.GetOr<uint32_t>("DLSSG.BackbufferSubrectWidth", in.backbuffer->width),
			parameters.GetOr<uint32_t>("DLSSG.BackbufferSubrectHeight", in.backbuffer->height),
			parameters.GetOr<uint32_t>("DLSSG.DepthSubrectBaseX", 0u),
			parameters.GetOr<uint32_t>("DLSSG.DepthSubrectBaseY", 0u),
			parameters.GetOr<uint32_t>("DLSSG.DepthSubrectWidth", in.depth->width),
			parameters.GetOr<uint32_t>("DLSSG.DepthSubrectHeight", in.depth->height),
			parameters.GetOr<uint32_t>("DLSSG.MVecsSubrectBaseX", 0u),
			parameters.GetOr<uint32_t>("DLSSG.MVecsSubrectBaseY", 0u),
			parameters.GetOr<uint32_t>("DLSSG.MVecsSubrectWidth", in.motion->width),
			parameters.GetOr<uint32_t>("DLSSG.MVecsSubrectHeight", in.motion->height),
		};
		if (f.hasPrevSubrects && curSubrects != f.prevSubrects)
			reset = true;
		f.prevSubrects = curSubrects;
		f.hasPrevSubrects = true;

		// Camera Cut & Teleport Detector + GhostBuster anti-ghosting (internal safety resets still apply when
		// ignoreGameReset is on; adaptive camera-step scale prevents meter-threshold false positives in UE cm-scale games)
		constexpr uint32_t kSyntheticResetQuietFrames = 4u;
		if (reset)
			f.syntheticResetQuietFrames = kSyntheticResetQuietFrames;
		else if (f.syntheticResetQuietFrames > 0u)
			--f.syntheticResetQuietFrames;

		const bool camCutOn = Plugin::options.cameraCutDetection.load(std::memory_order_relaxed);
		const bool ghostBusterOn = Plugin::options.ghostBuster.load(std::memory_order_relaxed);
		if (camCutOn || ghostBusterOn)
		{
			auto finiteOr = [](float v, float fallback) noexcept { return std::isfinite(v) ? v : fallback; };
			const float px = finiteOr(parameters.GetOr<float>("DLSSG.CameraPosX", 0.0f), 0.0f);
			const float py = finiteOr(parameters.GetOr<float>("DLSSG.CameraPosY", 0.0f), 0.0f);
			const float pz = finiteOr(parameters.GetOr<float>("DLSSG.CameraPosZ", 0.0f), 0.0f);
			const float fx = finiteOr(parameters.GetOr<float>("DLSSG.CameraFwdX", 0.0f), 0.0f);
			const float fy = finiteOr(parameters.GetOr<float>("DLSSG.CameraFwdY", 0.0f), 0.0f);
			const float fz = finiteOr(parameters.GetOr<float>("DLSSG.CameraFwdZ", 0.0f), 0.0f);
			const float fwdLenSq = fx * fx + fy * fy + fz * fz;
			const bool hasCamera = fwdLenSq > 1e-6f;
			float curFov = finiteOr(parameters.GetOr<float>("DLSSG.CameraFOV", 0.0f), 0.0f);
			if (curFov > 10.0f)
				curFov *= 3.14159265358979323846f / 180.0f;

			float distSq = 0.0f;
			float fwdDot = 1.0f;
			if (hasCamera && f.hasPrevCamera)
			{
				const float dx = px - f.prevCameraPos[0];
				const float dy = py - f.prevCameraPos[1];
				const float dz = pz - f.prevCameraPos[2];
				distSq = dx * dx + dy * dy + dz * dz;

				const float prevLenSq = f.prevCameraFwd[0] * f.prevCameraFwd[0] +
					f.prevCameraFwd[1] * f.prevCameraFwd[1] +
					f.prevCameraFwd[2] * f.prevCameraFwd[2];
				fwdDot = (prevLenSq > 1e-6f)
					? ((fx * f.prevCameraFwd[0] + fy * f.prevCameraFwd[1] + fz * f.prevCameraFwd[2]) /
					   std::sqrt(fwdLenSq * prevLenSq))
					: 1.0f;

				const float cutThresholdSq = f.cameraMotionScale.CutThresholdSq();
				const float hardTeleportSq = f.cameraMotionScale.HardTeleportThresholdSq();
				const bool quietActive = f.syntheticResetQuietFrames > 0u;
				if (camCutOn && !reset && (distSq > hardTeleportSq || (!quietActive && (distSq > cutThresholdSq || (!ghostBusterOn && fwdDot < -0.5f)))))
				{
					reset = true;
					f.syntheticResetQuietFrames = kSyntheticResetQuietFrames;
				}
			}

			if (ghostBusterOn && !reset && f.syntheticResetQuietFrames == 0u)
			{
				const bool camMvIncluded = parameters.GetOr<uint32_t>("DLSSG.CameraMotionIncluded", 1) != 0;
				if (XeFgEnabler::ShouldGhostBusterReset(true, hasCamera, f.hasPrevCamera, fwdDot,
					curFov, f.prevCameraFov, f.hasPrevFov, camMvIncluded, distSq,
					f.cameraMotionScale.GhostBusterMoveThresholdSq()))
				{
					reset = true;
					f.syntheticResetQuietFrames = kSyntheticResetQuietFrames;
				}
			}

			if (hasCamera && f.hasPrevCamera)
				f.cameraMotionScale.ObserveCandidate(std::sqrt(distSq), reset);

			if (hasCamera)
			{
				f.prevCameraPos[0] = px;
				f.prevCameraPos[1] = py;
				f.prevCameraPos[2] = pz;
				f.prevCameraFwd[0] = fx;
				f.prevCameraFwd[1] = fy;
				f.prevCameraFwd[2] = fz;
				f.hasPrevCamera = true;
			}
			if (curFov > 0.01f)
			{
				f.prevCameraFov = curFov;
				f.hasPrevFov = true;
			}
		}

		// UI / HUD-less transition cooldown (DLSS Enabler XeFG): track raw game HUD-less/UI presence
		// so internal HudTierPolicy Tier 1 <-> Tier 3 switches never trigger a 10-batch FG dropout.
		const bool curUiState = (rawGameHudless != nullptr) || (rawGameUi != nullptr);
		if (f.hasPrevUiState && curUiState != f.prevUiState &&
			Plugin::options.uiTransitionCooldown.load(std::memory_order_relaxed))
		{
			f.uiTransitionCooldown = 10u;
			Log::Info("Vulkan: HUD-less/UI state changed ({} -> {}), skipping interpolation for 10 batches",
				f.prevUiState, curUiState);
		}
		f.prevUiState = curUiState;
		f.hasPrevUiState = true;
		if (f.uiTransitionCooldown > 0u)
		{
			--f.uiTransitionCooldown;
			f.havePrevious = false;
			f.fault = "ui_transition";
			return false;
		}

		Slot& slot = f.slots[f.current];

		// The frame's inputs: wait for the game's prior writes before transfer/compute reads
		StageBarrier(d, command,
			VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_MEMORY_WRITE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
		CopyImage(d, command, in.backbuffer->image, GameLayout, xvk::ImageOf(slot.backbuffer.Get()),
			VK_IMAGE_LAYOUT_GENERAL, f.width, f.height, in.backbuffer);

		slot.hasHudless = in.hudless && slot.hudless;

		if (slot.hasHudless)
			CopyImage(d, command, in.hudless->image, GameLayout, xvk::ImageOf(slot.hudless.Get()),
				VK_IMAGE_LAYOUT_GENERAL, f.width, f.height, in.hudless);

		slot.hasUi = in.ui && slot.ui;

		if (slot.hasUi)
			CopyImage(d, command, in.ui->image, GameLayout, xvk::ImageOf(slot.ui.Get()), VK_IMAGE_LAYOUT_GENERAL,
				f.width, f.height, in.ui);

		const auto& depth = *in.depth;
		const auto& motion = *in.motion;

		RecordSampledCopy(d, command, d.pipelines[0], depth.imageView, slot.depthView, f.depthWidth, f.depthHeight,
			depth.width, depth.height);
		RecordSampledCopy(d, command, d.pipelines[VulkanPolicy::MotionCopyShader(static_cast<uint32_t>(f.motionFormat))],
			motion.imageView, slot.motionView, f.depthWidth, f.depthHeight, motion.width, motion.height);
		StageBarrier(d, command,
			VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

		// Record inline Tier 3 Hybrid Tone-Map Match, Tier 1 Pixel-Diff Guard, and/or Tier 3 Static-HUD update (0 extra submits / 0 PCL)
		f.activeBatchSlot = f.current;
		if (d.hudPipelinesReady && f.hudResourcesReady)
		{
			++f.hudRealFrameCount;
			f.hudReadbackSlot = static_cast<uint32_t>(f.hudRealFrameCount & 1u);
			const int fmtIdx = (f.format == DXGI_FORMAT_R10G10B10A2_UNORM) ? 1 : 0;

			if (f.hudDecision.runToneMapMatch && toneMatchHudless && toneMatchHudless->imageView && in.backbuffer->imageView)
			{
				RecordVkToneMapMatch(d, f, command, slot, toneMatchHudless->imageView, in.backbuffer->imageView);
			}

			const bool doDiffGuard = f.hudDecision.runHudDiffGuard && !f.hudDecision.runHudDiffComposite &&
				slot.hasHudless && slot.bbView && slot.hudlessView;
			const bool doStaticUpdate = f.hudDecision.runStaticHudUpdate && slot.bbView && f.staticCountView;
			if (doDiffGuard || doStaticUpdate)
			{
				const VkDeviceSize fillOffset = doDiffGuard ? 0 : 4;
				const VkDeviceSize fillSize = (doDiffGuard && doStaticUpdate) ? 8 : 4;
				d.vkCmdFillBuffer(command, f.hudCounterGpuBuf, fillOffset, fillSize, 0u);
				StageBarrier(d, command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

				if (doDiffGuard)
				{
					RecordHudDispatch(d, command, d.hudPipelines[0][fmtIdx],
						slot.bbView, slot.hudlessView, slot.bbView, f.hudCounterGpuBuf,
						f.width, f.height, 0.02f, 1u, 4u);
					f.hudReadbackHasDiff[f.hudReadbackSlot] = true;
					f.hudReadbackDiffFromToneMatch[f.hudReadbackSlot] = false;
				}
				if (doStaticUpdate)
				{
					const Slot& prevSlot = f.slots[f.current ^ 1];
					const bool staticReset = !f.havePrevious || reset || !prevSlot.bbView;
					RecordHudDispatch(d, command, d.hudPipelines[2][fmtIdx],
						slot.bbView, staticReset ? slot.bbView : prevSlot.bbView, f.staticCountView, f.hudCounterGpuBuf,
						f.width, f.height, 0.008f, staticReset ? 1u : 0u);
					f.hudReadbackHasStatic[f.hudReadbackSlot] = true;
				}

				const VkPipelineStageFlags dstStages = doStaticUpdate
					? static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
					: static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_TRANSFER_BIT);
				const VkAccessFlags dstAccess = doStaticUpdate
					? static_cast<VkAccessFlags>(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT)
					: static_cast<VkAccessFlags>(VK_ACCESS_TRANSFER_READ_BIT);
				StageBarrier(d, command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
					dstStages, dstAccess);
				const VkBufferCopy copy{ fillOffset, fillOffset, fillSize };
				d.vkCmdCopyBuffer(command, f.hudCounterGpuBuf, f.hudCounterReadbackBuf[f.hudReadbackSlot], 1, &copy);
			}
		}

		// The frame's constants. DLSSG.MvecScale is already a pixel scale of the motion vectors' extent (not
		// Streamline's normalized one), carried to the depth size they are resampled to. XeFG's left handed projection is
		// built from DLSS-G's near, far and field of view; DLSSG.CameraViewToClip is OptiScaler's right handed one, with
		// which XeFG reconstructs depth the wrong way round
		auto sanitizeFinite = [](float v, float fallback) noexcept { return std::isfinite(v) ? v : fallback; };
		const float rawMvecX = sanitizeFinite(parameters.GetOr<float>("DLSSG.MvecScaleX", 1.0f), 1.0f);
		const float rawMvecY = sanitizeFinite(parameters.GetOr<float>("DLSSG.MvecScaleY", 1.0f), 1.0f);
		const float mvecX = (rawMvecX != 0.0f) ? rawMvecX : 1.0f;
		const float mvecY = (rawMvecY != 0.0f) ? rawMvecY : 1.0f;
		const auto [scaleX, scaleY] =
			VulkanPolicy::MotionScale(mvecX, mvecY, motion.width, motion.height, f.depthWidth, f.depthHeight);

		slot.motionScaleX = sanitizeFinite(scaleX, 1.0f);
		slot.motionScaleY = sanitizeFinite(scaleY, 1.0f);
		slot.jitterX = sanitizeFinite(parameters.GetOr<float>("DLSSG.JitterOffsetX", 0.0f), 0.0f);
		slot.jitterY = sanitizeFinite(parameters.GetOr<float>("DLSSG.JitterOffsetY", 0.0f), 0.0f);
		slot.reset = reset;

		BuildViewMatrix(parameters, slot.view);
		ReadClampedSubrect(parameters, "DLSSG.BackbufferSubrectBaseX", "DLSSG.BackbufferSubrectBaseY",
			"DLSSG.BackbufferSubrectWidth", "DLSSG.BackbufferSubrectHeight", f.width, f.height, slot.bbBase, slot.bbSize);
		ReadClampedSubrect(parameters, "DLSSG.DepthSubrectBaseX", "DLSSG.DepthSubrectBaseY",
			"DLSSG.DepthSubrectWidth", "DLSSG.DepthSubrectHeight", f.depthWidth, f.depthHeight, slot.depthBase, slot.depthSize);
		// Motion vectors are resampled into f.depthWidth x f.depthHeight; scale motion subrect proportionally if present.
		ReadClampedSubrect(parameters, "DLSSG.MVecsSubrectBaseX", "DLSSG.MVecsSubrectBaseY",
			"DLSSG.MVecsSubrectWidth", "DLSSG.MVecsSubrectHeight", motion.width, motion.height, slot.motionBase, slot.motionSize);
		if (motion.width != f.depthWidth || motion.height != f.depthHeight)
		{
			slot.motionBase = { static_cast<uint32_t>((uint64_t(slot.motionBase.x) * f.depthWidth) / std::max(motion.width, 1u)),
				static_cast<uint32_t>((uint64_t(slot.motionBase.y) * f.depthHeight) / std::max(motion.height, 1u)) };
			slot.motionSize = { static_cast<uint32_t>((uint64_t(slot.motionSize.x) * f.depthWidth) / std::max(motion.width, 1u)),
				static_cast<uint32_t>((uint64_t(slot.motionSize.y) * f.depthHeight) / std::max(motion.height, 1u)) };
			if (slot.motionBase.x >= f.depthWidth) slot.motionBase.x = 0;
			if (slot.motionBase.y >= f.depthHeight) slot.motionBase.y = 0;
			if (slot.motionSize.x == 0 || slot.motionBase.x + slot.motionSize.x > f.depthWidth) slot.motionSize.x = f.depthWidth - slot.motionBase.x;
			if (slot.motionSize.y == 0 || slot.motionBase.y + slot.motionSize.y > f.depthHeight) slot.motionSize.y = f.depthHeight - slot.motionBase.y;
		}
		if (slot.hasHudless)
		{
			ReadClampedSubrect(parameters, "DLSSG.HUDLessSubrectBaseX", "DLSSG.HUDLessSubrectBaseY",
				"DLSSG.HUDLessSubrectWidth", "DLSSG.HUDLessSubrectHeight", f.width, f.height, slot.hudlessBase, slot.hudlessSize);
		}

		const float defaultAspect = static_cast<float>(f.width) / static_cast<float>(f.height);
		const float rawAspect = sanitizeFinite(parameters.GetOr<float>("DLSSG.CameraAspectRatio", defaultAspect), defaultAspect);
		const float aspect = (rawAspect > 0.0f) ? rawAspect : defaultAspect;
		float cameraNear = sanitizeFinite(parameters.GetOr<float>("DLSSG.CameraNear", 0.0f), 0.0f);
		float cameraFar = sanitizeFinite(parameters.GetOr<float>("DLSSG.CameraFar", 0.0f), 0.0f);
		float fov = sanitizeFinite(parameters.GetOr<float>("DLSSG.CameraFOV", 0.0f), 0.0f);
		if (cameraNear <= 0.0f || cameraFar <= 0.0f || fov <= 0.0001f)
			CameraFromViewToClip(parameters, cameraNear, cameraFar, fov);

		VulkanPolicy::Projection(aspect, cameraNear, cameraFar, fov, inverted, slot.projection);

		if (!f.cameraLogged)
		{
			f.cameraLogged = true;
			Log::Info("Vulkan camera: aspect {}, near {}, far {}, field of view {}, motion vectors {}x{} scale {} x {} "
				"(DLSS-G gave {} x {}), DLSSG.OutputReal {}",
				aspect, cameraNear, cameraFar, fov, motion.width, motion.height, scaleX, scaleY, mvecX, mvecY,
				ImageParameter(parameters, "DLSSG.OutputReal") != nullptr);
		}

		// A reset, or the first frame, has no previous frame: the current one stands in for it
		const Slot& previous = f.havePrevious && !reset ? f.slots[f.current ^ 1] : slot;

		// The batch's call, kept for its later evaluations (the slots it names stay until the next batch)
		f.exec = {};
		DescribeFrame(f, previous, f.exec.prev);
		DescribeFrame(f, slot, f.exec.next);
		f.exec.prev.resetHistory = 0;
		f.exec.next.resetHistory = (reset || !f.havePrevious) ? 1 : 0;

		f.execOutputs = {};
		f.execBases = {};

		for (uint32_t i = 0; i < count; ++i)
		{
			f.execOutputs[i] = f.outputs[i].Get();
			f.execBases[i] = slot.bbBase;
		}

		f.exec.ppOutputs = f.execOutputs.data();
		f.exec.pOutputBases = f.execBases.data();
		f.exec.numInterpolatedFrames = static_cast<uint8_t>(count);
		f.exec.uiCompositionEnabled = (f.hudDecision.useXeFgNativeUi && (slot.hasHudless || slot.hasUi)) ? 1u : 0u;

		if (!RecordOutputs(d, f, command, range))
		{
			f.havePrevious = false;
			return false;
		}
		f.current ^= 1;
		f.havePrevious = true;
		return true;
	}

	// ------------------------------------------------------------------------------------- the provider table

	NGXResult Init(void* instance, void* physical, void* device, void*, void*)
	{
		try
		{
			std::scoped_lock lock(gDevice.mutex);
			Device& d = gDevice;
			const auto vkDevice = reinterpret_cast<VkDevice>(device);

			if (!vkDevice)
				return NGX_FAIL_INVALID_PARAMETER;

			// One device: the game has one
			if (d.initialized && d.device != vkDevice)
			{
				Log::Warn("Vulkan: NGX was given a second device {}, the first one stays", device);
				return NGX_FAIL_PLATFORM_ERROR;
			}

			d.initialized = true;
			d.device = vkDevice;
			d.instance = reinterpret_cast<VkInstance>(instance);
			d.physical = reinterpret_cast<VkPhysicalDevice>(physical);

			if (d.reason == "not_initialized")
				d.reason = "pending";

			// Whether the hooks prepared this device is known now: NVSDK_NGX_VULKAN_Init and the menu say so before a
			// feature is asked for
			if (!d.ready && d.reason == "pending")
			{
				VulkanHost::DeviceRecord record;

				if (!VulkanHost::Find(vkDevice, record))
					d.reason = NotPreparedReason(d.physical);
			}

			Publish(d);

			if (d.reason != "pending")
				Log::Warn("Vulkan: device {} given to NGX, XeMFG can't run on it: {}", device,
					VulkanPolicy::Describe(d.reason));
			else
				Log::Info("Vulkan: device {} given to NGX", device);

			return NGX_SUCCESS;
		}
		catch (...)
		{
			return NGX_FAIL;
		}
	}

	NGXResult CreateFeature(void* device, void*, NGXParameter*, void** feature)
	{
		if (!feature)
			return NGX_FAIL_INVALID_PARAMETER;

		*feature = nullptr;

		try
		{
			std::scoped_lock lock(gDevice.mutex);

			if (device && gDevice.device && reinterpret_cast<VkDevice>(device) != gDevice.device)
				return NGX_FAIL_PLATFORM_ERROR;

			if (!EnsureDevice(gDevice))
			{
				static std::string logged;

				if (logged != gDevice.reason)
				{
					logged = gDevice.reason;
					Log::Warn("Vulkan: no feature, {}", VulkanPolicy::Describe(gDevice.reason));
				}

				return NGX_FAIL_FEATURE_NOT_SUPPORTED;
			}

			// XeFG's context comes with the first evaluation, once the inputs are known
			*feature = new Feature();
			return NGX_SUCCESS;
		}
		catch (...)
		{
			return NGX_FAIL;
		}
	}

	NGXResult Evaluate(void* commandBuffer, void* handle, NGXParameter* parameters)
	{
		if (!commandBuffer || !handle || !parameters)
			return NGX_FAIL_INVALID_PARAMETER;

		auto& f = *static_cast<Feature*>(handle);
		const auto command = reinterpret_cast<VkCommandBuffer>(commandBuffer);
		bool recording = false;

		try
		{
			std::scoped_lock lock(gDevice.mutex);
			Device& d = gDevice;

			if (!d.ready)
				return NGX_FAIL_NOT_INITIALIZED;

			if (!d.retiredFeatures.empty())
				DrainRetiredFeaturesLocked(d, 0ull);

			const uint32_t count = VulkanPolicy::ClampCount(parameters->GetOr<uint32_t>("DLSSG.MultiFrameCount", 1u));
			const uint32_t index = VulkanPolicy::ClampIndex(parameters->GetOr<uint32_t>("DLSSG.MultiFrameIndex", 1u), count);

			const NgxResource* output = ImageParameter(*parameters, "DLSSG.OutputInterpolated");

			if (!output || output->range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT)
				return NGX_FAIL_INVALID_PARAMETER;

			// Vulkan dispatchable objects from another device must never reach xvk.
			if (*reinterpret_cast<const void* const*>(command) !=
				*reinterpret_cast<const void* const*>(d.device))
				return NGX_FAIL_PLATFORM_ERROR;

			const NgxResource* real = ImageParameter(*parameters, "DLSSG.OutputReal");
			const NgxResource* backbuffer = ImageParameter(*parameters, "DLSSG.Backbuffer");

			if (!backbuffer || backbuffer->range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
				!VulkanPolicy::CopyCompatible(backbuffer->format, output->format) ||
				output->width != backbuffer->width || output->height != backbuffer->height ||
				(real && (real->range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
					!VulkanPolicy::CopyCompatible(backbuffer->format, real->format) ||
					SameSubresource(*real, *output))))
				return NGX_FAIL_INVALID_PARAMETER;

			if (!xvk::BeginRecording(d.d3d.Get(), command))
				return NGX_FAIL_PLATFORM_ERROR;

			recording = true;
			f.evaluations++;

			// DLSSG.EnableInterp 0 or DLSSG.OutputDisableInterpolation 1 pauses generation for this frame
			// (menu, cutscene, loading screen): copy the real frame cleanly without invoking XeFG (dlssg-0.4.2 M1).
			const bool disableInterp = parameters->GetOr<uint32_t>("DLSSG.OutputDisableInterpolation", 0) != 0;
			const bool ignoreGameResetEval = Plugin::options.ignoreGameReset.load(std::memory_order_relaxed);
			const bool resetRequested = !ignoreGameResetEval && (parameters->GetOr<uint32_t>("DLSSG.Reset", 0) != 0);
			const bool interpolate = parameters->GetOr<uint32_t>("DLSSG.EnableInterp", 1) != 0 && !disableInterp;
			bool didExecute = false;

			if (!interpolate)
			{
				f.batchReady = false;
				f.havePrevious = false;
				f.generatedThrough = 0;
				f.lastCopiedRealImage = VK_NULL_HANDLE;
				f.consecutiveFailures = 0;
				f.stallCooldownBatches = 0;
				f.uiTransitionCooldown = 0;
				f.paused++;
				VulkanHost::NotifyPaused();
			}
			else if (VulkanPolicy::NewBatch(index, f.lastIndex, count, f.requestedBatchCount, f.batches))
			{
				if (resetRequested)
				{
					f.consecutiveFailures = 0;
					f.stallCooldownBatches = 0;
				}

				const bool vramGuarded = PollVkVramGuard(d, f);
				const bool vramClamped = vramGuarded && count > 1u;
				const uint32_t effectiveCount = vramClamped ? 1u : count;

				f.requestedBatchCount = count;
				f.batchCount = effectiveCount;
				f.fault.clear();
				f.generatedThrough = 0;
				f.lastCopiedRealImage = VK_NULL_HANDLE;

				// Stall circuit breaker (dlssg-0.4.2): after 3 consecutive Execute failures, bypass XeFG for 60 batches
				// and copy the real frame cleanly so the game never freezes or flickers.
				if (f.stallCooldownBatches > 0)
				{
					--f.stallCooldownBatches;
					f.batchReady = false;
					f.havePrevious = false;
					f.fault = "stall_cooldown";
					VulkanHost::NotifyPaused();
				}
				else
				{
					f.batchReady = GenerateBatch(d, f, command, *parameters, effectiveCount,
						VulkanPolicy::ExecuteRange(true, index, effectiveCount, false));
				}
				didExecute = f.batchReady;
				f.batches++;

				const bool profOn = Plugin::options.gpuProfiler.load(std::memory_order_relaxed) && f.profHasReading;
				const int32_t tmActive = (f.hudDecision.runToneMapMatch && f.hudTmResourcesReady) ? f.activeToneMapMode : 0;
				Plugin::Report({ f.width, f.height, effectiveCount, f.hudDecision.activeTier != HudTier::ActiveTier::Interpolated,
					static_cast<int32_t>(f.hudDecision.activeTier), NetworkOf(f.model),
					f.context == nullptr, f.format == DXGI_FORMAT_R16G16B16A16_FLOAT,
					count, vramClamped, f.vramUsageMiB, f.vramBudgetMiB,
					tmActive, profOn, f.profTotalMs, f.profInterpMs, f.profHudMs });

				if (!f.batchReady)
				{
					f.havePrevious = false;
					VulkanHost::NotifyPaused();
					if (f.fault != "stall_cooldown" && f.fault != "ui_transition")
					{
						if (++f.failures <= 16)
							Log::Error("Vulkan: batch failed ({}), index {}, count {}", f.fault, index, count);
						if (++f.consecutiveFailures >= 3u)
						{
							f.stallCooldownBatches = 60u;
							f.consecutiveFailures = 0;
							Log::Warn("Vulkan: XeFG stalled after 3 consecutive batch failures ({}); pausing FG for 60 batches", f.fault);
						}
					}
				}
				else
				{
					f.consecutiveFailures = 0;
					// Streamline still submits `count` presents per real frame even when VRAM Guard clamps
					// extra slots to real-frame copies, so pass `count` to keep PresentPacer aligned.
					VulkanHost::NotifyBatchStart(count);
					if (f.batches == 1 || f.batches % 6000 == 0)
					{
						Log::Info("Vulkan: {} batches of {} generated frames, {} evaluations, {} failures, {} paused", f.batches,
							count, f.evaluations, f.failures, f.paused);
					}
				}
			}
			else if (f.batchReady && index > f.generatedThrough && index <= f.batchCount)
			{
				const auto range = VulkanPolicy::ExecuteRange(false, index, f.batchCount, false);

				if (range.record && f.context)
				{
					if (!RecordOutputs(d, f, command, range))
					{
						f.batchReady = false;
						f.havePrevious = false;
						f.generatedThrough = 0;
						VulkanHost::NotifyPaused();
						if (++f.failures <= 16)
							Log::Error("Vulkan: output {} of {} failed ({})", index, f.batchCount, f.fault);
					}
					else
					{
						didExecute = true;
					}
				}
			}

			f.lastIndex = index;

			const bool copied = f.batchReady && index <= f.generatedThrough && index <= f.batchCount &&
				f.outputs[index - 1];
			const bool copyReal = real && !SameSubresource(*real, *backbuffer) &&
				(real->image != f.lastCopiedRealImage);
			const bool copyFallback = !copied && backbuffer && !SameSubresource(*output, *backbuffer) &&
				VulkanPolicy::CopyCompatible(output->format, backbuffer->format);
			const bool anyCopy = copyReal || copied || copyFallback;

			// RecordOutputs already emits a COMPUTE -> TRANSFER barrier; only emit a pre-copy barrier when
			// no execute ran in this evaluation (paused, fallback, or pre-generated cache hit).
			if (anyCopy && !didExecute)
			{
				StageBarrier(d, command,
					VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_ACCESS_MEMORY_WRITE_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
			}

			// Preserve the real frame before writing an interpolated output that may alias the backbuffer.
			// real and output are verified disjoint above, so both copies share one exit barrier.
			if (copyReal)
			{
				if (copied)
					CopyImage(d, command, xvk::ImageOf(f.slots[f.current ^ 1].backbuffer.Get()),
						VK_IMAGE_LAYOUT_GENERAL, real->image, GameLayout,
						std::min(f.width, real->width), std::min(f.height, real->height), nullptr, real);
				else
					CopyImage(d, command, backbuffer->image, GameLayout, real->image, GameLayout,
						std::min(backbuffer->width, real->width), std::min(backbuffer->height, real->height), backbuffer, real);
				f.lastCopiedRealImage = real->image;
				f.realCopies++;
			}

			if (copied)
			{
				CopyImage(d, command, xvk::ImageOf(f.outputs[index - 1].Get()), VK_IMAGE_LAYOUT_GENERAL, output->image,
					GameLayout, std::min(output->width, f.width), std::min(output->height, f.height), nullptr, output);
			}
			else if (copyFallback)
			{
				// Paused, or the batch failed: the real frame takes the output's place, never its stale content
				CopyImage(d, command, backbuffer->image, GameLayout, output->image, GameLayout,
					std::min(backbuffer->width, output->width), std::min(backbuffer->height, output->height), backbuffer, output);
			}

			if (anyCopy)
			{
				StageBarrier(d, command,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
			}

			xvk::EndRecording(d.d3d.Get());
			recording = false;

			return copied || !interpolate ? NGX_SUCCESS : NGX_FAIL;
		}
		catch (...)
		{
			std::scoped_lock lock(gDevice.mutex);
			f.batchReady = false;
			f.havePrevious = false;
			f.generatedThrough = 0;
			if (recording)
				xvk::EndRecording(gDevice.d3d.Get());

			return NGX_FAIL;
		}
	}

	void DrainRetiredFeaturesLocked(Device& d, uint64_t timeoutNs) noexcept
	{
		if (d.retiredFeatures.empty())
			return;
		if (d.d3d && !xvk::WaitRecordedIdle(d.d3d.Get(), timeoutNs))
			return;
		for (auto& rf : d.retiredFeatures)
		{
			if (rf.profPool && d.device && d.vkDestroyQueryPool)
				d.vkDestroyQueryPool(d.device, rf.profPool, nullptr);
			if (rf.context)
				CallXeFG("Destroy", [&] { return XeFG::Destroy(rf.context); });
		}
		d.retiredFeatures.clear();
	}

	NGXResult ReleaseFeature(void* handle)
	{
		if (!handle)
			return NGX_FAIL_INVALID_PARAMETER;

		auto* f = static_cast<Feature*>(handle);

		try
		{
			std::scoped_lock lock(gDevice.mutex);

			// Drain in-flight recorded command buffers before destroying XeFG's context and internal xvk resources,
			// preventing deferred-resource buildup and XeFG "Invalid internal state" (init_-1000) after rapid
			// multiplier toggles on Vulkan. If GPU work has not finished within 150 ms, defer destruction
			// rather than freeing an in-use query pool or XeFG context.
			const bool idle = !gDevice.d3d || xvk::WaitRecordedIdle(gDevice.d3d.Get(), 150'000'000ull);
			if (idle)
			{
				DrainRetiredFeaturesLocked(gDevice, 0ull);
				if (f->profPool && gDevice.device && gDevice.vkDestroyQueryPool)
				{
					gDevice.vkDestroyQueryPool(gDevice.device, f->profPool, nullptr);
					f->profPool = VK_NULL_HANDLE;
				}

				if (f->context)
					CallXeFG("Destroy", [&] { return XeFG::Destroy(f->context); });

				f->context = nullptr;
				delete f;

				if (gDevice.d3d)
					xvk::WaitRecordedIdle(gDevice.d3d.Get(), 50'000'000ull);
			}
			else
			{
				Log::Warn("Vulkan: ReleaseFeature idle wait timed out (150 ms); deferring query pool and XeFG context destroy");
				if (f->profPool || f->context)
					gDevice.retiredFeatures.push_back({ f->profPool, f->context });
				f->profPool = VK_NULL_HANDLE;
				f->context = nullptr;
				delete f;
			}

			return NGX_SUCCESS;
		}
		catch (...)
		{
			return NGX_FAIL;
		}
	}

	NGXResult Shutdown()
	{
		try
		{
			std::scoped_lock lock(gDevice.mutex);

			DrainRetiredFeaturesLocked(gDevice, 2'000'000'000ull);
			if (gDevice.d3d)
				xvk::WaitRecordedIdle(gDevice.d3d.Get(), 2'000'000'000ull);

			return NGX_SUCCESS;
		}
		catch (...)
		{
			return NGX_FAIL;
		}
	}

	// Why XeMFG can't run on Vulkan here, a VulkanPolicy::Describe code, empty when it can
	std::string Blocker()
	{
		if (!VulkanPrograms::Installed().Usable())
			return "no_vulkan_models";

		if (XeFG::State() == XeFG::LoadState::Failed)
			return "no_xefg";

		{
			std::scoped_lock lock(gStatusMutex);

			if (gDeviceReason != "not_initialized" && gDeviceReason != "pending")
				return gDeviceReason;

			// No device yet: the hooks are what can still keep it from running
			if (gDeviceReason == "not_initialized")
			{
				const auto host = VulkanHost::Snapshot();

				if (!host.enabled)
					return "hooks_off";

				if (host.hooks == VulkanRuntime::Hooks::Failed)
					return "hooks_failed";
			}
		}

		return {};
	}

	uint32_t Supported()
	{
		try
		{
			return Blocker().empty() ? MaxGenerated : 0;
		}
		catch (...)
		{
			return 0;
		}
	}

	int Reason(char* buffer, uint32_t size)
	{
		if (!buffer || !size)
			return 0;

		buffer[0] = '\0';

		try
		{
			const auto code = Blocker();

			if (code.empty())
				return 0;

			const auto text = VulkanPolicy::Describe(code);
			const size_t length = std::min<size_t>(text.size(), size - 1);
			std::memcpy(buffer, text.data(), length);
			buffer[length] = '\0';

			return 1;
		}
		catch (...)
		{
			return 0;
		}
	}
} // namespace

bool VulkanRuntime::Built()
{
	return true;
}

const VulkanProvider::Api* VulkanRuntime::Table()
{
	static const VulkanProvider::Api table = { sizeof(VulkanProvider::Api), VulkanProvider::ApiVersion, &Init,
		&CreateFeature, &Evaluate, &ReleaseFeature, &Shutdown, &Supported, &Reason };

	return &table;
}

void VulkanRuntime::Start(const char* caller)
{
	static std::once_flag once;
	std::call_once(once,
		[]
		{
			if (EnvironmentOff(L"XEFG_VULKAN_HOOKS"))
			{
				VulkanHost::SetEnabled(false);
				Log::Info("XEFG_VULKAN_HOOKS is off, the Vulkan hooks aren't installed");
			}
		});

	// Off, nothing is hooked at all
	if (!VulkanHost::Snapshot().enabled)
		return;

	VulkanHost::Start(caller);
}

void VulkanRuntime::SetHooksEnabled(bool enabled)
{
	VulkanHost::SetEnabled(enabled);
}

VulkanRuntime::Status VulkanRuntime::GetStatus()
{
	return VulkanHost::Snapshot();
}

std::string VulkanRuntime::ModelName()
{
	std::scoped_lock lock(gStatusMutex);
	return gModelName;
}

VulkanRuntime::TensorStatus VulkanRuntime::GetTensorStatus()
{
	TensorStatus status;
	const auto route = xefg_cuda::VulkanSnapshot();

	switch (route.state)
	{
	case xefg_cuda::State::Active:
		status.state = TensorState::Active;
		break;
	case xefg_cuda::State::Unavailable:
		status.state = TensorState::Unavailable;
		break;
	case xefg_cuda::State::Disabled:
		status.state = TensorState::Disabled;
		break;
	default:
		break;
	}

	status.reason = route.reason;
	status.major = route.major;
	status.minor = route.minor;
	status.launches = route.launches;
	status.fallbacks = route.fallbacks;
	status.replacing = status.state == TensorState::Active && gTensorGate.load();

	// The route reports nothing for a GPU it was never an option for, the device knows why
	if (status.state == TensorState::NotAttempted)
	{
		std::scoped_lock lock(gStatusMutex);
		status.reason = gTensorReason;
	}

	return status;
}

VulkanRuntime::CoopMatrixStatus VulkanRuntime::GetCoopMatrixStatus()
{
	CoopMatrixStatus status;
	const auto route = xefg_cuda::CoopMatrixSnapshot();

	switch (route.state)
	{
	case xefg_cuda::State::Active:
		status.state = TensorState::Active;
		break;
	case xefg_cuda::State::Unavailable:
		status.state = TensorState::Unavailable;
		break;
	case xefg_cuda::State::Disabled:
		status.state = TensorState::Disabled;
		break;
	default:
		break;
	}

	status.reason = route.reason;
	status.tileM = route.major;
	status.tileN = (route.minor >> 8) & 0xFF;
	status.tileK = route.minor & 0xFF;
	status.launches = route.launches;
	status.fallbacks = route.fallbacks;
	status.replacing = status.state == TensorState::Active && gCoopGate.load();

	if (status.state == TensorState::NotAttempted)
	{
		std::scoped_lock lock(gStatusMutex);
		status.reason = gCoopReason;
	}

	return status;
}
