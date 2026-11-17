// The Vulkan half of the mod.
//
// XeFG only has a D3D12 API. MFG-Unlocker runs libxess_fg and libxell on a game's VkDevice anyway, through a small
// D3D12-on-Vulkan layer (xvk) that runs every XeMFG compute program as SPIR-V: the Vulkan models. This dll carries that
// layer (src/vulkan, see VulkanRuntime.h) and reads the models from xefg_mfg_programs\vulkan next to it
// (VulkanPrograms.h), so the NVSDK_NGX_VULKAN_* exports need no other file.
//
// The exports reach the layer through a provider table, the shape of the one MFGUnlocker.asi publishes as
// MfgXeMFGProviderGetApi and mfgunlocker_xemfg.dll calls. Get() is
//   - the table of the runtime built into this dll (VulkanRuntime::Table()), in a build that has the Vulkan headers,
//   - else the table of a MFGUnlocker.asi loaded in the process, found the same way mfgunlocker_xemfg.dll finds it.
//
// The asi's table layout isn't published by MFG-Unlocker, it was read off the asi (1.7.1-beta2fix.1, the same in
// 1.7.1-beta.3, 1.7.1-rc.2 and 1.7.1) and off mfgunlocker_xemfg.dll's call sites. Only the Vulkan half of it is used here,
// the D3D12 half (+0x40 and up) is this mod's own code.
#pragma once

#include "NGX.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace VulkanProvider
{
	// What MfgXeMFGProviderGetApi(1) returns, a static table of the asi
	struct Api
	{
		uint32_t size;    // 0x68 in 1.7.1
		uint32_t version; // 1
		// instance, physical device, device and the loader's vkGetInstanceProcAddr/vkGetDeviceProcAddr (null when the
		// caller has none). A second init with another device is refused (NGX_FAIL_PLATFORM_ERROR)
		NGXResult (*init)(void* instance, void* physicalDevice, void* device, void* getInstanceProcAddr,
			void* getDeviceProcAddr);
		// device is null for the NGX entry point that doesn't take one. The context itself is made on the first evaluation
		NGXResult (*create)(void* device, void* cmdBuf, NGXParameter* params, void** impl);
		// Reads DLSSG.MultiFrameCount, DLSSG.MultiFrameIndex, DLSSG.OutputInterpolated and DLSSG.OutputReal, the same
		// parameters the D3D12 exports take
		NGXResult (*evaluate)(void* cmdBuf, void* impl, NGXParameter* params);
		NGXResult (*release)(void* impl);
		NGXResult (*shutdown)();
		// 0 when XeMFG can't run on Vulkan here, else DLSSG.MultiFrameCountMax: the most frames generated per real frame
		// (5 in 1.7.1, 6x)
		uint32_t (*supported)();
		// 0 when it can run (the buffer gets an empty string), else 1 and the buffer says why it can't. Missing in
		// tables shorter than 0x40 bytes
		int (*reason)(char* buffer, uint32_t size);
	};

	constexpr uint32_t ApiVersion = 1;
	// Everything up to and including supported(), reason() is optional
	constexpr uint32_t MinApiSize = offsetof(Api, reason);
	constexpr uint32_t ReasonApiSize = MinApiSize + sizeof(void*);

	static_assert(sizeof(void*) == 8, "The table is the asi's 64-bit layout");
	static_assert(offsetof(Api, init) == 0x08 && offsetof(Api, create) == 0x10 && offsetof(Api, evaluate) == 0x18 &&
		offsetof(Api, release) == 0x20 && offsetof(Api, shutdown) == 0x28 && offsetof(Api, supported) == 0x30 &&
		offsetof(Api, reason) == 0x38);

	// Can this dll call into the table
	inline bool Usable(const Api* api)
	{
		return api && api->version == ApiVersion && api->size >= MinApiSize && api->init && api->create &&
			api->evaluate && api->release && api->shutdown && api->supported;
	}

	// The provider's reason() exists
	inline bool HasReason(const Api* api)
	{
		return api->size >= ReasonApiSize && api->reason;
	}

	enum class State : int32_t
	{
		NotLoaded, // No runtime in this dll and no module of the process exports the provider
		Blocked,   // Found, but it says XeMFG can't run here, Info::reason says why
		Ready,
	};

	struct Info
	{
		State state = State::NotLoaded;
		uint32_t maxFrames = 0; // Most generated frames per real frame (DLSSG.MultiFrameCountMax), 0 unless Ready
		std::string reason;
		std::string module; // Path of the module the provider is in, "built into this dll" for the runtime
	};

	// The provider table: the runtime built into this dll when there is one, else null while no asi is loaded. The
	// process is searched for the asi on the first call and then now and then until it's found, it may be loaded after
	// this dll
	const Api* Get();

	// What the provider says about this machine, refreshed after Init and now and then, cheap enough to call from
	// the menu every frame. The built-in runtime answers from memory and is asked every time
	Info Query();

	// The model the provider runs, "port" or "DP4a" ("2X" for the asi), empty until XeFG has been created. The asi's
	// own XeFGModelName
	std::string ModelName();

	// Called by NVSDK_NGX_VULKAN_Init*, what the provider says is asked again
	void DeviceChanged();

	// Vulkan features created and not released yet, for the menu
	inline std::atomic<uint32_t> liveFeatures = 0;
	// A game gave NVSDK_NGX_VULKAN_Init a device
	inline std::atomic_bool initialized = false;
} // namespace VulkanProvider
