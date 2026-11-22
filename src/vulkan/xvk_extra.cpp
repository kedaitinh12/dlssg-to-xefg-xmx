#include "xvk_extra.h"
#include "xvk_internal.h"

namespace xvk
{
	VkImageView StorageImageView(ID3D12Resource* resource, VkFormat format) noexcept
	{
		if (!resource)
			return VK_NULL_HANDLE;

		try
		{
			return Own<Resource>(resource)->View(format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
				VK_IMAGE_USAGE_STORAGE_BIT);
		}
		catch (...)
		{
			return VK_NULL_HANDLE;
		}
	}

	VkBuffer BufferOf(ID3D12Resource* resource) noexcept
	{
		if (!resource)
			return VK_NULL_HANDLE;

		try
		{
			return Own<Resource>(resource)->buffer();
		}
		catch (...)
		{
			return VK_NULL_HANDLE;
		}
	}
} // namespace xvk
