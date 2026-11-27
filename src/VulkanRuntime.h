// The Vulkan runtime built into this dll, what the NVSDK_NGX_VULKAN_* exports run on.
//
// XeFG only has a D3D12 API. This dll gives it one on a game's VkDevice the way MFG-Unlocker (MFGUnlocker.asi) does: a
// small D3D12-on-Vulkan layer (src/vulkan/xvk_*.cpp, MFG-Unlocker's own code) that runs XeFG's programs as the SPIR-V
// models of xefg_mfg_programs\vulkan (src/VulkanPrograms.h), and hooks that give the game's Vulkan device what those
// programs need when it is created (src/vulkan/VulkanHost.cpp). MFGUnlocker.asi isn't needed.
//
// It needs the Vulkan headers to build. A build without them has no runtime (Built() is false) and the exports fall
// back to the provider table of an MFGUnlocker.asi loaded in the game, if there is one (src/VulkanProvider.h).
//
// Nothing here includes Vulkan or Windows, the menu and the exports use it as it is.
#pragma once

#include "VulkanProvider.h"

#include <cstdint>
#include <string>

namespace VulkanRuntime
{
	// This build carries the runtime
	bool Built();

	// The provider table of the runtime, null when it isn't built. The same shape as MFG-Unlocker's, so the exports
	// call either the same way
	const VulkanProvider::Api* Table();

	// Hooks the Vulkan loader, now when vulkan-1.dll is loaded and else as soon as it is. The game's VkInstance and
	// VkDevice must be created after this for the runtime to use them, so it is called when this dll starts running
	// (InitializeASI, OptiPluginInit) and again by NVSDK_NGX_VULKAN_Init, which is too late for a game that already has
	// its device but says so in the log. Safe to call more than once, not from DllMain
	void Start(const char* caller);

	// The hooks only pass the game's calls on while they are off. On by default; XEFG_VULKAN_HOOKS=0 in the
	// environment or VulkanHooks=0 in OptiScaler.ini turn them off
	void SetHooksEnabled(bool enabled);

	// Marks the tracked Vulkan device as abandoned when its VkInstance is destroyed, so teardown skips
	// calling back into a dead ICD (0% runtime overhead).
	void AbandonInstance(const void* instance) noexcept;

	enum class Hooks
	{
		Off,       // not started, or turned off
		Waiting,   // vulkan-1.dll isn't loaded yet, they go in when it is
		Installed,
		Failed,    // detail says why
	};

	struct Status
	{
		Hooks hooks = Hooks::Off;
		bool enabled = true;
		std::string detail;      // why they are off or failed
		uint32_t instances = 0;  // vkCreateInstance calls the hooks saw
		uint32_t devices = 0;    // vkCreateDevice calls the hooks saw
		uint32_t prepared = 0;   // of those, the devices that got what XeFG needs
		std::string refusal;     // reason code of the last device they turned down, empty when none
	};

	Status GetStatus();

	// The model the runtime runs: "port" (the XMX Port, XeFG's model 16) or "DP4a" (XeFG's stock model 18), empty until
	// the runtime has created XeFG
	std::string ModelName();

	// What became of the tensor-core route (src/vulkan/VulkanTensor.h): on an NVIDIA RTX GPU the CUDA network path runs the
	// ported model's int8 layers as CUDA kernels in the game's command buffers, instead of as SPIR-V programs
	enum class TensorState
	{
		NotAttempted, // no Vulkan device was prepared yet, or its GPU isn't an NVIDIA one (reason says which)
		Active,       // installed: it takes layers while the CUDA path is selected
		Unavailable,  // reason says why, the SPIR-V programs run the layers
		Disabled,     // XEFG_TENSOR_CORES is off in the environment
	};

	struct TensorStatus
	{
		TensorState state = TensorState::NotAttempted;
		std::string reason;       // a VulkanPolicy::DescribeTensor code, empty when it's active
		int major = 0;            // the GPU's CUDA compute capability, 0 when it isn't known
		int minor = 0;
		bool replacing = false;   // it takes layers now: active, the CUDA path selected and the ported model running
		uint64_t launches = 0;    // layers it ran on the tensor cores
		uint64_t fallbacks = 0;   // layers it left to their SPIR-V programs: a shape or a view it can't take
	};

	TensorStatus GetTensorStatus();

	// What became of the VK_KHR_cooperative_matrix route (src/vulkan/VulkanCoopMat.h): on AMD RDNA3/RDNA4 (WMMA) and
	// Intel Arc/Battlemage (XMX) GPUs, the XMX Port and XMX (native) paths run the ported model's int8 layers as
	// SPV_KHR_cooperative_matrix shaders in the game's command buffers, instead of DP4a SPIR-V programs
	struct CoopMatrixStatus
	{
		TensorState state = TensorState::NotAttempted;
		std::string reason;       // a VulkanPolicy::DescribeCoopMatrix code, empty when it's active
		int tileM = 0;            // the GPU's selected SINT8xSINT8->SINT32 cooperative matrix tile M dimension
		int tileN = 0;            // the GPU's selected SINT8xSINT8->SINT32 cooperative matrix tile N dimension
		int tileK = 0;            // the GPU's selected SINT8xSINT8->SINT32 cooperative matrix tile K dimension
		bool replacing = false;   // it takes layers now: active, the XMX Port / XMX (native) path selected and ported model running
		uint64_t launches = 0;    // layers it ran on the cooperative matrix units
		uint64_t fallbacks = 0;   // layers it left to their DP4a SPIR-V programs
	};

	CoopMatrixStatus GetCoopMatrixStatus();
} // namespace VulkanRuntime
