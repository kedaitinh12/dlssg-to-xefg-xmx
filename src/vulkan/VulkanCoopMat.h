// The VK_KHR_cooperative_matrix route of the Vulkan runtime: on AMD RDNA3/RDNA4 (WMMA) and Intel Arc/Battlemage (XMX)
// GPUs, the 17 int8 layers of the ported model can run on hardware matrix units via SPV_KHR_cooperative_matrix inside
// the game's own VkCommandBuffer. Layers without an external cooperative-matrix module in
// xefg_mfg_programs\vulkan\coopmat keep their SPV_KHR_integer_dot_product program seamlessly in the same frame.
#pragma once

#include "xvk.h"

#include <string>

namespace VulkanCoopMat
{
	// False when XEFG_COOP_MATRIX is 0, off or false in the environment.
	bool Wanted();

	// True when XEFG_ALLOW_INTEL_VULKAN or XEFG_COOP_MATRIX allows Intel Arc GPUs with VK_KHR_cooperative_matrix on Vulkan.
	bool AllowIntelAdapter();

	// Runs the one-time Wave32 16x16x16 matC register layout micro-probe on an AMD Vulkan device.
	// Uses a transient VkCommandPool/VkCommandBuffer/VkFence so caller command buffers remain untouched.
	xvk::MatCLayout ProbeMatCLayout(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
		uint32_t queueFamily, VkQueue queue, PFN_vkGetInstanceProcAddr gipa,
		PFN_vkGetDeviceProcAddr gdpa, const char* logPrefix) noexcept;

	// Polls (or bounded-waits on teardown) any matC probe resources that timed out while still in flight
	// on the GPU, destroying them only after vkGetFenceStatus reports completion (or device lost).
	// Returns true if no pending probe remains for `device` (safe to destroy `device`).
	bool ReapPendingProbes(VkDevice device, bool waitBounded = false, bool deviceLost = false) noexcept;

	// Installs the cooperative-matrix route on the layer's device (`device` is what xvk::CreateDevice gave).
	bool Install(ID3D12Device* device, std::string& reason, xvk::MatCLayout matCLayout = xvk::MatCLayout::Unknown);
} // namespace VulkanCoopMat
