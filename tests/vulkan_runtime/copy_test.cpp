// Executes the runtime's actual copy helper with a captured vkCmdCopyImage.
// Other runtime entry points are discarded by --gc-sections; no GPU is used.
#include "../../src/vulkan/VulkanRuntime.cpp"
#include <cassert>
#include <cstdio>

static VkImageCopy captured{};
static VkImage capturedSource{}, capturedTarget{};
static void VKAPI_CALL Capture(VkCommandBuffer, VkImage source, VkImageLayout, VkImage target,
    VkImageLayout, uint32_t count, const VkImageCopy* regions)
{
    assert(count == 1);
    captured = *regions;
    capturedSource = source;
    capturedTarget = target;
}
int main()
{
    gDevice.vkCmdCopyImage = Capture;
    NgxResource source{}, target{};
    source.image = reinterpret_cast<VkImage>(1);
    target.image = reinterpret_cast<VkImage>(2);
    source.range = {VK_IMAGE_ASPECT_COLOR_BIT, 3, 1, 2, 1};
    target.range = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 4, 1};
    CopyImage(gDevice, nullptr, source.image, GameLayout, target.image, GameLayout, 640, 360, &source, &target);
    assert(capturedSource == source.image && capturedTarget == target.image);
    assert(captured.srcSubresource.mipLevel == 3 && captured.srcSubresource.baseArrayLayer == 2);
    assert(captured.dstSubresource.mipLevel == 1 && captured.dstSubresource.baseArrayLayer == 4);
    assert(captured.extent.width == 640 && captured.extent.height == 360 && captured.extent.depth == 1);
    assert(captured.srcSubresource.layerCount == 1 && captured.dstSubresource.layerCount == 1);
    CopyImage(gDevice, nullptr, source.image, GameLayout, target.image, GameLayout, 128, 64, nullptr, &target);
    assert(captured.srcSubresource.mipLevel == 0 && captured.srcSubresource.baseArrayLayer == 0);
    assert(captured.dstSubresource.mipLevel == 1 && captured.dstSubresource.baseArrayLayer == 4);
    CopyImage(gDevice, nullptr, source.image, GameLayout, target.image, GameLayout, 128, 64, &source);
    assert(captured.srcSubresource.mipLevel == 3 && captured.dstSubresource.mipLevel == 0);
    assert(!SameSubresource(source, target));
    target.image = source.image;
    assert(!SameSubresource(source, target));
    target.range = source.range;
    assert(SameSubresource(source, target));
    std::puts("runtime copies: source/destination mip/layer, extent, alias detection: PASS");
}
