// The Vulkan runtime built into this dll, see VulkanRuntime.h: XeFG on a game's VkDevice.
//
// This is MFG-Unlocker's xess/vulkan/vk_provider.cpp (MIT, Copyright (c) 2026 Michael Robles, see NOTICE.md) ported to
// this repository: XeFG is driven through XeFG.h, the ported model comes from MfgKernels, the model follows the menu's
// network path, and the programs are the ones in xefg_mfg_programs\vulkan (VulkanPrograms.h). The tensor-core route of an
// NVIDIA GPU is in (VulkanTensor.h), selected by the CUDA network path. Left out: its GPU timing and its registration as an
// OptiScaler provider.
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
#include "../VulkanPrograms.h"
#include "../XeFG.h"
#include "VulkanHost.h"
#include "VulkanInputCopy.h"
#include "VulkanPolicy.h"
#include "VulkanTensor.h"
#include "xvk_extra.h"

#include <wrl/client.h>

#include <algorithm>
#include <array>
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
	X(vkDestroyShaderModule) X(vkCreateComputePipelines) X(vkDestroyPipeline)

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
		PFN_vkGetInstanceProcAddr gipa = nullptr;
		uint32_t family = 0;
		uint32_t vendor = 0;
		ComPtr<ID3D12Device> d3d; // the layer's recorded device on `device`
		bool portInstalled = false;
		// The tensor-core route (VulkanTensor.h) is installed on the layer's device. When it isn't, tensorReason is the
		// VulkanPolicy::DescribeTensor code
		bool tensorInstalled = false;
		std::string tensorReason = "not_attempted";

#define VULKAN_RUNTIME_FIELD(name) PFN_##name name = nullptr;
		VULKAN_RUNTIME_FNS(VULKAN_RUNTIME_FIELD)
#undef VULKAN_RUNTIME_FIELD

		VkSampler sampler = VK_NULL_HANDLE;
		VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipeline pipelines[VulkanInputCopy::Count] = {}; // 0 r32f (depth), 1 rg16f, 2 rg32f (motion vectors)
	};

	// The game owns the VkDevice and destroys it before the CRT is torn down. The layer's device is kept for the life of
	// the process: its destructor must not call into a dead VkDevice
	Device& gDevice = *new Device;

	// What the menu and supported() read, without waiting for an evaluation that holds gDevice.mutex
	std::mutex gStatusMutex;
	std::string gDeviceReason = "not_initialized";
	std::string gModelName;
	std::string gTensorReason = "not_attempted";

	// The route is replacing layers: installed, and the CUDA path selected with the ported model (ApplyTensorGate)
	std::atomic_bool gTensorGate = false;

	// One real frame's inputs, kept for the next batch as its previous frame
	struct Slot
	{
		ComPtr<ID3D12Resource> backbuffer, hudless, ui, depth, motion;
		VkImageView depthView = VK_NULL_HANDLE;
		VkImageView motionView = VK_NULL_HANDLE;
		bool hasHudless = false;
		bool hasUi = false;
		bool reset = false;
		float projection[16]{};
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
		bool inverted = false;
		bool jittered = false;
		int32_t model = 0;

		std::array<Slot, 2> slots;
		uint32_t current = 0;
		bool havePrevious = false;
		std::array<ComPtr<ID3D12Resource>, MaxGenerated> outputs;

		uint32_t lastIndex = 0; // the previous evaluation's DLSSG.MultiFrameIndex
		uint32_t batchCount = 0;
		bool batchReady = false;
		bool cameraLogged = false;
		uint64_t batches = 0, failures = 0, evaluations = 0, realCopies = 0, paused = 0;
		std::string fault;

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
		return model == MfgKernels::MfgModel ? "port" : "DP4a";
	}

	// With gDevice.mutex held
	void Publish(const Device& d)
	{
		std::scoped_lock lock(gStatusMutex);
		gDeviceReason = d.reason;
		gTensorReason = d.tensorReason;
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

	void FullBarrier(const Device& d, VkCommandBuffer command)
	{
		VkMemoryBarrier memory{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
			VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };

		d.vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
			&memory, 0, nullptr, 0, nullptr);
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

		if (FAILED(xvk::CreateDevice(desc, IID_PPV_ARGS(&d.d3d))) || !d.d3d)
		{
			d.reason = "xvk_device";
			return false;
		}

		// Publishing the ported model patches libxess_fg.dll process wide, so it runs before the first context. It is
		// installed even when the menu selects the DP4a model, so switching the path needs no new process
		d.portInstalled = MfgKernels::Install(XeFG::Module());

		InstallTensorRoute(d, record);

		if (!CreatePipelines(d))
		{
			d.reason = "copy_pipelines";
			return false;
		}

		d.ready = true;
		d.reason.clear();

		Log::Info("Vulkan XeMFG ready on queue family {}, vendor {:#06x}: {} programs, ported model {}, tensor cores {}, up to {}x",
			d.family, d.vendor, VulkanPrograms::Installed().programs.size(), d.portInstalled ? "installed" : "not available",
			d.tensorInstalled ? "ready" : "not available", MaxGenerated + 1);

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
	int32_t WantedModel(const Device& d)
	{
		const bool dp4a = Plugin::options.path.load() == Plugin::Path::Fallback;

		return d.portInstalled && !dp4a ? MfgKernels::MfgModel : MfgKernels::Dp4aModel;
	}

	// The tensor-core route replaces layers while the ported model runs on the CUDA path (VulkanPolicy::UseTensorCores).
	// The path can change without a new context, CUDA and XMX Port run the same model and only what executes the layers
	// differs, so this is set on every batch. A layer the route can't take stays a SPIR-V program in the same frame
	void ApplyTensorGate(const Device& d, int32_t model)
	{
		const auto path = Plugin::options.path.load();
		const bool on = d.tensorInstalled && VulkanPolicy::UseTensorCores(model == MfgKernels::MfgModel,
			path == Plugin::Path::Cuda, path == Plugin::Path::XmxNative);

		xvk::SetTensorCoresEnabled(d.d3d.Get(), on);

		if (gTensorGate.exchange(on) != on)
			Log::Info("Vulkan tensor cores: {}", on ? "on, the CUDA path runs the int8 layers on the tensor cores" : "off");
	}

	// The network the menu and OptiScaler show for the model that runs
	Plugin::Network NetworkOf(int32_t model)
	{
		if (model != MfgKernels::MfgModel)
			return Plugin::Network::Fallback;

		// The route turns itself off after a driver fault, then the layers are the SPIR-V ones
		return gTensorGate.load() && xefg_cuda::VulkanActive() ? Plugin::Network::Cuda : Plugin::Network::XmxPort;
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
		const int32_t model = WantedModel(d);

		ApplyTensorGate(d, model);

		const bool same = f.context && f.width == bb.width && f.height == bb.height && f.format == format &&
			f.depthWidth == depth.width && f.depthHeight == depth.height && f.motionFormat == motionFormat &&
			f.hudlessFormat == hudlessFormat && f.uiFormat == uiFormat && f.inverted == inverted &&
			f.jittered == jittered && f.model == model;

		if (same)
			return true;

		if (f.context)
		{
			CallXeFG("Destroy", [&] { return XeFG::Destroy(f.context); });
			f.context = nullptr;
		}

		f.havePrevious = false;
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
				return false;
			}

			slot.depthView = xvk::StorageImageView(slot.depth.Get(), VK_FORMAT_R32_SFLOAT);
			slot.motionView = xvk::StorageImageView(slot.motion.Get(), motionView);

			if (!slot.depthView || !slot.motionView)
			{
				f.fault = "input_views";
				return false;
			}
		}

		for (auto& output : f.outputs)
		{
			if (FAILED(CreateTexture(d3d, f.width, f.height, format, output)))
			{
				f.fault = "output_images";
				return false;
			}
		}

		if (!f.allocator && FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&f.allocator))))
		{
			f.fault = "allocator";
			return false;
		}

		// Process wide, they also affect other XeFG contexts. Must happen before the context picks its model
		XeFG::SetMaxInterpolatedFrames(MaxGenerated);
		XeFG::OverrideModel(model);

		xefgd_handle_t context = nullptr;
		xefgd_result_t result = CallXeFG("CreateContext", [&] { return XeFG::CreateContext(d3d, &context); });

		if (result != XEFGD_RESULT_SUCCESS || !context)
		{
			f.fault = "create_context_" + std::to_string(result);
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
		init.uiMode = XEFGD_UI_MODE_AUTO;

		result = CallXeFG("Init", [&] { return XeFG::Init(context, init); });

		if (result != XEFGD_RESULT_SUCCESS)
		{
			CallXeFG("Destroy", [&] { return XeFG::Destroy(context); });
			f.fault = "init_" + std::to_string(result);
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

	xefgd_resource_t Region(ID3D12Resource* resource, uint32_t width, uint32_t height)
	{
		xefgd_resource_t out{};
		out.pResource = resource;
		out.size = { width, height };

		return out;
	}

	void DescribeFrame(const Feature& f, const Slot& slot, xefgd_frame_t& frame)
	{
		frame = {};
		frame.backBuffer = Region(slot.backbuffer.Get(), f.width, f.height);
		frame.depth = Region(slot.depth.Get(), f.depthWidth, f.depthHeight);
		frame.motionVectors = Region(slot.motion.Get(), f.depthWidth, f.depthHeight);

		if (slot.hasHudless)
			frame.hudlessColor = Region(slot.hudless.Get(), f.width, f.height);

		if (slot.hasUi)
			frame.uiTexture = Region(slot.ui.Get(), f.width, f.height);

		// Identity: no camera matrices are given, which also keeps XeFG's scene change detection on
		for (int i = 0; i < 16; ++i)
			frame.viewMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;

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

		const xefgd_result_t result = CallXeFG("Execute", [&] { return XeFG::Execute(f.context, f.list.Get(), f.exec); });
		f.list->Close();
		FullBarrier(d, command);

		if (result != XEFGD_RESULT_SUCCESS)
		{
			f.fault = "execute_" + std::to_string(result);
			return false;
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

		// The menu's HUDless UI switch is for both: off, the UI is interpolated with the scene
		if (Plugin::options.hudlessUi)
		{
			in.hudless = ImageParameter(parameters, "DLSSG.HUDLess");
			in.ui = ImageParameter(parameters, "DLSSG.UI");
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

		// XeFG's UI inputs cover the output extent. A differently sized optional
		// image cannot be described as the full backbuffer: ignore it this batch.
		auto usableUi = [&](const NgxResource* image) {
			return image && image->width == in.backbuffer->width && image->height == in.backbuffer->height &&
				image->range.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT;
		};
		if (!usableUi(in.hudless)) in.hudless = nullptr;
		if (!usableUi(in.ui)) in.ui = nullptr;

		const bool inverted = parameters.GetOr<uint32_t>("DLSSG.DepthInverted", 0) != 0;
		const bool jittered = parameters.GetOr<uint32_t>("DLSSG.MvecJittered", 0) != 0;

		if (!EnsureFeature(d, f, in, inverted, jittered))
			return false;

		// Takes effect from XeFG's next Execute on
		XeFG::SetDebugLines(Plugin::options.debugLines);

		const bool reset = parameters.GetOr<uint32_t>("DLSSG.Reset", 0) != 0;
		Slot& slot = f.slots[f.current];

		// The frame's inputs
		FullBarrier(d, command);
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
		FullBarrier(d, command);

		// The frame's constants. DLSSG.MvecScale is already a pixel scale of the motion vectors' extent (not
		// Streamline's normalized one), carried to the depth size they are resampled to. XeFG's left handed projection is
		// built from DLSS-G's near, far and field of view; DLSSG.CameraViewToClip is OptiScaler's right handed one, with
		// which XeFG reconstructs depth the wrong way round
		const float mvecX = parameters.GetOr<float>("DLSSG.MvecScaleX", 1.0f);
		const float mvecY = parameters.GetOr<float>("DLSSG.MvecScaleY", 1.0f);
		const auto [scaleX, scaleY] =
			VulkanPolicy::MotionScale(mvecX, mvecY, motion.width, motion.height, f.depthWidth, f.depthHeight);

		slot.motionScaleX = scaleX;
		slot.motionScaleY = scaleY;
		slot.jitterX = parameters.GetOr<float>("DLSSG.JitterOffsetX", 0.0f);
		slot.jitterY = parameters.GetOr<float>("DLSSG.JitterOffsetY", 0.0f);
		slot.reset = reset;

		const float aspect = parameters.GetOr<float>("DLSSG.CameraAspectRatio",
			static_cast<float>(f.width) / static_cast<float>(f.height));
		const float cameraNear = parameters.GetOr<float>("DLSSG.CameraNear", 0.0f);
		const float cameraFar = parameters.GetOr<float>("DLSSG.CameraFar", 0.0f);
		const float fov = parameters.GetOr<float>("DLSSG.CameraFOV", 0.0f);

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
			f.execOutputs[i] = f.outputs[i].Get();

		f.exec.ppOutputs = f.execOutputs.data();
		f.exec.pOutputBases = f.execBases.data();
		f.exec.numInterpolatedFrames = static_cast<uint8_t>(count);
		f.exec.uiCompositionEnabled = (slot.hasHudless || slot.hasUi) ? 1u : 0u;

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

			// DLSSG.EnableInterp 0 pauses generation for this frame (a menu, a loading screen): no XeFG work and the next
			// batch starts without a previous frame
			const bool interpolate = parameters->GetOr<uint32_t>("DLSSG.EnableInterp", 1) != 0;

			if (!interpolate)
			{
				f.batchReady = false;
				f.havePrevious = false;
				f.generatedThrough = 0;
				f.paused++;
			}
			else if (VulkanPolicy::NewBatch(index, f.lastIndex, count, f.batchCount, f.batches))
			{
				f.batchCount = count;
				f.fault.clear();
				f.generatedThrough = 0;
				f.batchReady = GenerateBatch(d, f, command, *parameters, count,
					VulkanPolicy::ExecuteRange(true, index, count, false));
				f.batches++;

				Plugin::Report({ f.width, f.height, count, f.exec.uiCompositionEnabled != 0, NetworkOf(f.model),
					f.context == nullptr, f.format == DXGI_FORMAT_R16G16B16A16_FLOAT });

				if (!f.batchReady)
				{
					f.havePrevious = false;
					if (++f.failures <= 16)
						Log::Error("Vulkan: batch failed ({}), index {}, count {}", f.fault, index, count);
				}
				else if (f.batches == 1 || f.batches % 6000 == 0)
				{
					Log::Info("Vulkan: {} batches of {} generated frames, {} evaluations, {} failures, {} paused", f.batches,
						count, f.evaluations, f.failures, f.paused);
				}
			}
			else if (f.batchReady && index > f.generatedThrough)
			{
				const auto range = VulkanPolicy::ExecuteRange(false, index, f.batchCount, false);

				if (range.record && f.context && !RecordOutputs(d, f, command, range))
				{
					f.batchReady = false;
					f.havePrevious = false;
					f.generatedThrough = 0;
					if (++f.failures <= 16)
						Log::Error("Vulkan: output {} of {} failed ({})", index, f.batchCount, f.fault);
				}
			}

			f.lastIndex = index;

			const bool copied = f.batchReady && index <= f.generatedThrough && index <= f.batchCount &&
				f.outputs[index - 1];

			// Preserve the real frame before writing an interpolated output that may
			// alias the backbuffer. Later evaluations use this batch's saved input,
			// because that game backbuffer may already hold an interpolated frame.
			if (real && !SameSubresource(*real, *backbuffer))
			{
				FullBarrier(d, command);
				if (copied)
					CopyImage(d, command, xvk::ImageOf(f.slots[f.current ^ 1].backbuffer.Get()),
						VK_IMAGE_LAYOUT_GENERAL, real->image, GameLayout,
						std::min(f.width, real->width), std::min(f.height, real->height), nullptr, real);
				else
					CopyImage(d, command, backbuffer->image, GameLayout, real->image, GameLayout,
						std::min(backbuffer->width, real->width), std::min(backbuffer->height, real->height), backbuffer, real);
				FullBarrier(d, command);
				f.realCopies++;
			}

			if (copied)
			{
				FullBarrier(d, command);
				CopyImage(d, command, xvk::ImageOf(f.outputs[index - 1].Get()), VK_IMAGE_LAYOUT_GENERAL, output->image,
					GameLayout, std::min(output->width, f.width), std::min(output->height, f.height), nullptr, output);
				FullBarrier(d, command);
			}
			else if (backbuffer && !SameSubresource(*output, *backbuffer) && VulkanPolicy::CopyCompatible(output->format, backbuffer->format))
			{
				// Paused, or the batch failed: the real frame takes the output's place, never its stale content
				FullBarrier(d, command);
				CopyImage(d, command, backbuffer->image, GameLayout, output->image, GameLayout,
					std::min(backbuffer->width, output->width), std::min(backbuffer->height, output->height), backbuffer, output);
				FullBarrier(d, command);
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

	NGXResult ReleaseFeature(void* handle)
	{
		if (!handle)
			return NGX_FAIL_INVALID_PARAMETER;

		auto* f = static_cast<Feature*>(handle);

		try
		{
			std::scoped_lock lock(gDevice.mutex);

			// The layer keeps the released images and XeFG's objects until the GPU has run the recordings that used them
			if (f->context)
				CallXeFG("Destroy", [&] { return XeFG::Destroy(f->context); });

			f->context = nullptr;
			delete f;

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
