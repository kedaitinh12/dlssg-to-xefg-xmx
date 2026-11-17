// The tensor-core route of the Vulkan runtime: on an NVIDIA RTX GPU the int8 layers of the ported model run as the CUDA
// kernels of xefg_mfg_programs\xefg_cuda_layers.fatbin, launched through VK_NVX_binary_import in the command buffer the
// game gave NGX, instead of as SPIR-V programs. The launching itself is MFG-Unlocker's xvk_tensor.cpp (vendored with
// the rest of xvk); this is what MFG-Unlocker's vk_host.cpp does around it, written against this repository.
//
// A layer the route can't take (a shape it doesn't know, an unsupported view) keeps its SPIR-V program in the same frame,
// so nothing here can leave a frame without its layers.
#pragma once

#include "xvk.h"

#include <string>

namespace VulkanTensor
{
	// False when XEFG_TENSOR_CORES is 0, off or false in the environment, the switch the D3D12 route has too. The hooks ask
	// before they give an NVIDIA device VK_NVX_binary_import and VK_NVX_image_view_handle, which a device needs to have the
	// route at all
	bool Wanted();

	// Installs the route on the layer's device (`device` is what xvk::CreateDevice gave): reads the fatbin beside the
	// programs, takes XEFG_TENSOR_CORES_MASK (layers to launch, hexadecimal, default all) and
	// XEFG_TENSOR_CORES_VULKAN_BARRIERS (on: a full barrier after each launch, for diagnosis) from the environment and
	// reports what became of it to xefg_cuda::VulkanSnapshot and the log. True when the route is active, else `reason` is
	// a VulkanPolicy::DescribeTensor code. That is no error: the SPIR-V programs run the layers. Never throws
	bool Install(ID3D12Device* device, std::string& reason);
} // namespace VulkanTensor
