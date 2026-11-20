// The part of the Vulkan runtime that has to run before the game creates its VkDevice. XeFG's programs need features
// and an extension the game didn't ask for (Vulkan 1.3, timeline semaphores, synchronization2, maintenance4, shader
// integer dot product, extended storage image formats, VK_KHR_push_descriptor), and those can only be given to a device
// when it is created. So this hooks vkCreateInstance and vkCreateDevice in vulkan-1.dll with MinHook, raises the
// game's own request and remembers which devices it did that for. VulkanRuntime.cpp runs XeFG on those.
//
// It follows MFG-Unlocker's vk_host.cpp (MIT, Copyright (c) 2026 Michael Robles, see NOTICE.md) for what a device must
// have and what refuses it, without its swap chain and latency parts: only the features of a device that the game keeps
// rendering to are changed, never its queues, extensions it didn't ask for apart from push descriptors, on an NVIDIA
// GPU the two NVX extensions of the tensor-core route (VulkanTensor.h), and on GPUs supporting SINT8 cooperative
// matrices VK_KHR_cooperative_matrix (VulkanCoopMat.h), or its swap chains.
#pragma once

#include "../VulkanRuntime.h"
#include "xvk.h"

#include <cstdint>
#include <string>

namespace VulkanHost
{
	// A device this dll prepared: what the runtime needs to put XeFG on it
	struct DeviceRecord
	{
		VkPhysicalDevice physical = VK_NULL_HANDLE;
		VkInstance instance = VK_NULL_HANDLE;
		PFN_vkGetInstanceProcAddr gipa = nullptr; // the loader's own
		uint32_t family = 0;                      // the graphics queue family of the game's device
		uint32_t vendor = 0;                      // PCI vendor ID
		// NVIDIA's VK_NVX_binary_import and VK_NVX_image_view_handle, which the tensor-core route (VulkanTensor.h) launches
		// its CUDA kernels with. The specs are the versions the driver offers (0: it doesn't), tensorExtensions says both
		// are enabled on the device: the hooks add them when the route is wanted, a game that enabled them itself has them too
		bool tensorExtensions = false;
		uint32_t binaryImportSpec = 0;
		uint32_t imageViewHandleSpec = 0;
		// VK_KHR_cooperative_matrix (AMD RDNA3/RDNA4 WMMA and Intel Arc/Battlemage XMX)
		bool coopMatrixExtensions = false;
		uint32_t coopMatrixSpec = 0;
		// Hardware flip metering / low-latency pacing (VK_NV_present_metering or VK_NV_low_latency2)
		bool hwPacing = false;
	};

	// Installs the hooks, or arranges for them to be installed when vulkan-1.dll is loaded. Later calls do nothing. Not
	// for DllMain: MinHook suspends the process' other threads
	void Start(const char* caller);

	// The hooks edit what the game asks for while this is on, they only pass its calls on while it's off. On by default
	void SetEnabled(bool enabled);

	// A device the hooks prepared, false for any other: the game created it before the hooks, or the hooks turned it
	// down (RefusalFor says why)
	bool Find(VkDevice device, DeviceRecord& out);

	// The reason code (VulkanPolicy::Describe) the hooks turned this physical device down for, empty when they didn't
	std::string RefusalFor(VkPhysicalDevice physical);

	// Temporarily bypasses VulkanHost instance/device hooks on the current thread (used by D3D12VulkanWmma
	// so intermediate wrappers like OptiScaler/overlays cannot mask the internal bridge caller).
	void SetInternalBridgeBypass(bool bypass) noexcept;

	// Frame-pacing cadence signals from VulkanRuntime::Evaluate (recorded-mode MFG)
	void NotifyBatchStart(uint32_t generatedCount) noexcept;
	void NotifyPaused() noexcept;

	// For the log and the menu
	VulkanRuntime::Status Snapshot();
} // namespace VulkanHost
