#include "Plugin.h"
// The nvngx_dlssg exports for Vulkan, the same 12 mfgunlocker_xemfg.dll has. XeFG has no Vulkan API, so none of this
// drives XeFG itself: the calls go to the XeMFG provider (see VulkanProvider.h), which runs XeFG on the game's VkDevice
// with the Vulkan models. The provider is the runtime built into this dll (VulkanRuntime.h, models read from
// xefg_mfg_programs\vulkan), or in a build without the Vulkan headers the one of a MFGUnlocker.asi loaded in the game.
// Without either every export reports that frame generation isn't available and OptiScaler goes on without it. The D3D12
// exports in Exports.cpp don't depend on this file in any way.
//
// The Vulkan handles are passed as void*, a VkInstance, VkPhysicalDevice, VkDevice and VkCommandBuffer are pointers
// on 64-bit and no Vulkan header is needed.
#include "Log.h"
#include "NGX.h"
#include "VulkanProvider.h"
#include "VulkanRuntime.h"

#include <atomic>
#include <cstddef>
#include <cstring>

#define NGX_EXPORT extern "C" __declspec(dllexport)

// Bit 0x10 of NVSDK_NGX_FeatureRequirement::FeatureSupported, NVSDK_NGX_FeatureSupportResult_NotImplemented
constexpr uint32_t NGX_FEATURE_NOT_IMPLEMENTED = 0x10;

static_assert(sizeof(NGXFeatureRequirement) == 0x108, "NVSDK_NGX_FeatureRequirement");

// The handle given out is ours, the caller hands it back untouched. impl is the provider's own feature
struct VulkanFeatureHandle : NGXHandle
{
	void* impl;
};

namespace
{
	using VulkanProvider::State;

	void LogOnce(std::atomic_bool& logged, const char* message)
	{
		if (!logged.exchange(true))
			Log::Warn("{}", message);
	}

	NGXResult Init(void* instance, void* physicalDevice, void* device, void* getInstanceProcAddr,
		void* getDeviceProcAddr)
	{
		[[maybe_unused]] static const bool logged = []
			{
				Log::Info("dlssg-to-xefg, XeFG replacing DLSS Frame Generation on Vulkan, {}. Free software under GPLv3, if "
					"you paid for it you've been scammed",
					VulkanRuntime::Built() ? "with MFG-Unlocker's D3D12-on-Vulkan layer built in"
					                       : "through MFG-Unlocker's XeMFG provider");
				return true;
			}();

		static std::atomic_bool noProvider;
		static std::atomic_bool blocked;

		// The hooks that give the game's device what XeFG's programs need are meant to be in before the game creates it,
		// this dll's start-up put them there when it could. A device created before is too late, the log says so
		VulkanRuntime::Start("NVSDK_NGX_VULKAN_Init");

		const auto* api = VulkanProvider::Get();
		if (!api)
		{
			LogOnce(noProvider, "Vulkan: no XeMFG provider in the process, MFGUnlocker.asi has to be loaded for Vulkan "
				"games (this build has no Vulkan runtime of its own)");
			return NGX_FAIL_FEATURE_NOT_SUPPORTED;
		}

		if (!VulkanRuntime::Built())
		{
			// The same check mfgunlocker_xemfg.dll makes when it loads: when MFG-Unlocker says it can't run, say so now
			// and let OptiScaler use something else. Asked again here, an earlier answer may be from before the asi was
			// ready
			VulkanProvider::DeviceChanged();
			if (VulkanProvider::Query().state != State::Ready)
			{
				LogOnce(blocked, "Vulkan: the XeMFG provider can't run here, see the line above");
				return NGX_FAIL_FEATURE_NOT_SUPPORTED;
			}
		}

		const NGXResult result = api->init(instance, physicalDevice, device, getInstanceProcAddr, getDeviceProcAddr);
		if (result != NGX_SUCCESS)
		{
			Log::Error("Vulkan: the XeMFG provider refused the device, 0x{:X}", result);
			return result;
		}

		if (VulkanRuntime::Built())
		{
			// A game gave NGX a Vulkan device, what the runtime can't do on it is news from here on. It knows whether the
			// hooks prepared this device only now it has it, and says why when they didn't. Refusing here lets OptiScaler
			// go on without frame generation instead of failing at the first feature
			VulkanProvider::initialized = true;
			VulkanProvider::DeviceChanged();
			if (VulkanProvider::Query().state != State::Ready)
			{
				LogOnce(blocked, "Vulkan: XeMFG can't run on this device, see the line above");
				return NGX_FAIL_FEATURE_NOT_SUPPORTED;
			}
		}

		VulkanProvider::initialized = true;

		return NGX_SUCCESS;
	}

	NGXResult Create(void* device, void* cmdBuf, uint32_t feature, NGXParameter* params, NGXHandle** handle)
	{
		if (!handle || !params)
			return NGX_FAIL_INVALID_PARAMETER;

		*handle = nullptr;

		const auto* api = VulkanProvider::Get();
		if (!api || feature != NGX_FEATURE_FRAME_GENERATION)
			return NGX_FAIL_FEATURE_NOT_SUPPORTED;

		void* impl = nullptr;
		const NGXResult result = api->create(device, cmdBuf, params, &impl);

		if (result != NGX_SUCCESS)
		{
			Log::Error("Vulkan: the XeMFG provider couldn't create a feature, 0x{:X}", result);
			return result;
		}

		if (!impl)
			return NGX_FAIL;

		static std::atomic_uint32_t nextId = 1;

		// The XeFG context is created on the first evaluation, like on D3D12
		*handle = new VulkanFeatureHandle{ { NGX_HANDLE_VULKAN | (nextId++ & ~NGX_HANDLE_VULKAN) }, impl };
		VulkanProvider::liveFeatures++;

		Plugin::activeApi.store(Plugin::ActiveApi::Vulkan, std::memory_order_release);
		Log::WarnForeignFgMods(true, VulkanProvider::Query().module != "built into this dll");
		Log::Info("Vulkan: feature created");

		return NGX_SUCCESS;
	}

	NGXResult GetCurrentSettingsCallback(NGXHandle* handle, NGXParameter* params)
	{
		if (!handle || !params)
			return NGX_FAIL_INVALID_PARAMETER;

		params->Set("DLSSG.MustCallEval", 1);
		params->Set("DLSSG.BurstCaptureRunning", 0);

		return NGX_SUCCESS;
	}

	NGXResult EstimateVRAMCallback(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
		uint32_t, size_t* estimatedSize)
	{
		// The same guess mfgunlocker_xemfg.dll gives, the Vulkan route keeps XeFG's resources and its own on top
		if (estimatedSize)
			*estimatedSize = 640 * 1024 * 1024;

		return NGX_SUCCESS;
	}
} // namespace

// Init and Init_Ext2 share their argument order up to the debug pointers, Init_Ext has no vkGetInstanceProcAddr and
// vkGetDeviceProcAddr: the provider then finds them itself
NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_Init(unsigned long long, const wchar_t*, void* instance, void* physicalDevice,
	void* device, void* getInstanceProcAddr, void* getDeviceProcAddr, const void*, uint32_t)
{
	return Init(instance, physicalDevice, device, getInstanceProcAddr, getDeviceProcAddr);
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_Init_Ext(unsigned long long, const wchar_t*, void* instance,
	void* physicalDevice, void* device, uint32_t, const void*)
{
	return Init(instance, physicalDevice, device, nullptr, nullptr);
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_Init_Ext2(unsigned long long, const wchar_t*, void* instance,
	void* physicalDevice, void* device, void* getInstanceProcAddr, void* getDeviceProcAddr, uint32_t, const void*)
{
	return Init(instance, physicalDevice, device, getInstanceProcAddr, getDeviceProcAddr);
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_Shutdown()
{
	if (const auto* api = VulkanProvider::Get())
		api->shutdown();

	VulkanProvider::initialized = false;

	return NGX_SUCCESS;
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_Shutdown1(void*) { return NVSDK_NGX_VULKAN_Shutdown(); }

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_GetScratchBufferSize(uint32_t, const NGXParameter*, size_t* size)
{
	if (!size)
		return NGX_FAIL_INVALID_PARAMETER;

	*size = 0;
	return NGX_SUCCESS;
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_GetFeatureRequirements(void*, void*, const void*,
	NGXFeatureRequirement* requirement)
{
	if (!requirement)
		return NGX_FAIL_INVALID_PARAMETER;

	std::memset(requirement, 0, sizeof(*requirement));

	if (VulkanProvider::Query().state != State::Ready)
		requirement->FeatureSupported = NGX_FEATURE_NOT_IMPLEMENTED;

	return NGX_SUCCESS;
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_CreateFeature(void* cmdBuf, uint32_t feature, NGXParameter* params,
	NGXHandle** handle)
{
	return Create(nullptr, cmdBuf, feature, params, handle);
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_CreateFeature1(void* device, void* cmdBuf, uint32_t feature,
	NGXParameter* params, NGXHandle** handle)
{
	return Create(device, cmdBuf, feature, params, handle);
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_ReleaseFeature(NGXHandle* handle)
{
	if (!handle)
		return NGX_FAIL_INVALID_PARAMETER;

	// A D3D12 feature isn't ours to read as a Vulkan one
	if (!IsVulkanHandle(handle))
		return NGX_FAIL_FEATURE_NOT_FOUND;

	auto feature = static_cast<VulkanFeatureHandle*>(handle);

	NGXResult result = NGX_SUCCESS;
	if (const auto* api = VulkanProvider::Get())
		result = api->release(feature->impl);

	delete feature;
	VulkanProvider::liveFeatures--;

	return result;
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_EvaluateFeature(void* cmdBuf, const NGXHandle* handle, NGXParameter* params,
	void*)
{
	if (!cmdBuf || !handle || !params)
		return NGX_FAIL_INVALID_PARAMETER;

	if (!IsVulkanHandle(handle))
		return NGX_FAIL_FEATURE_NOT_FOUND;

	const auto* api = VulkanProvider::Get();
	if (!api)
		return NGX_FAIL_FEATURE_NOT_SUPPORTED;

	const NGXResult result = api->evaluate(cmdBuf, static_cast<const VulkanFeatureHandle*>(handle)->impl, params);

	// The provider names its model once XeFG has a context, after the first frames. Looked for a bounded number of
	// evaluations in case the asi never publishes one
	static std::atomic_uint32_t modelLookups;
	static std::atomic_bool modelLogged;

	if (result == NGX_SUCCESS && !modelLogged.load(std::memory_order_relaxed) && modelLookups++ < 600)
	{
		if (const auto model = VulkanProvider::ModelName(); !model.empty() && !modelLogged.exchange(true))
			Log::Info("Vulkan: XeFG model {}", model);
	}

	return result;
}

NGX_EXPORT NGXResult NVSDK_NGX_VULKAN_PopulateParameters_Impl(NGXParameter* params)
{
	if (!params)
		return NGX_FAIL_INVALID_PARAMETER;

	const auto info = VulkanProvider::Query();
	const uint32_t frames = info.state == State::Ready ? info.maxFrames : 0;

	params->Set("FrameGeneration.Available", frames != 0 ? 1 : 0);
	params->Set("FrameGeneration.NeedsUpdatedDriver", 0);
	params->Set("FrameGeneration.FeatureInitResult", frames != 0 ? (int)NGX_SUCCESS : (int)NGX_FAIL_FEATURE_NOT_SUPPORTED);
	params->Set("DLSSG.MultiFrameCountMax", (int)frames);

	// Only offered while it can run, so a caller that checks for them sees DLSS-G as absent
	if (frames != 0)
	{
		params->Set("DLSSG.GetCurrentSettingsCallback", reinterpret_cast<void*>(&GetCurrentSettingsCallback));
		params->Set("DLSSG.EstimateVRAMCallback", reinterpret_cast<void*>(&EstimateVRAMCallback));
	}

	return NGX_SUCCESS;
}
