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

	// Installs the cooperative-matrix route on the layer's device (`device` is what xvk::CreateDevice gave).
	bool Install(ID3D12Device* device, std::string& reason);
} // namespace VulkanCoopMat
