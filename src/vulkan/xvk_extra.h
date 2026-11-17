// What the runtime (VulkanRuntime.cpp) needs of the D3D12-on-Vulkan layer beyond xvk.h. It is one function here, so that
// VulkanRuntime.cpp doesn't include xvk_internal.h and with it every class of the layer.
#pragma once

#include "xvk.h"

namespace xvk
{
	// A 2D storage image view of an xvk texture, null for any other resource or when the view can't be made. The layer
	// caches it on the texture and destroys it with it
	VkImageView StorageImageView(ID3D12Resource* resource, VkFormat format) noexcept;
} // namespace xvk
