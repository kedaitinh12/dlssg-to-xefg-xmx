// Owned copy of the prefix containing XeFG feature requests. The untouched tail
// remains borrowed for vkCreateDevice. Never write through a game's const pNext.
#pragma once
#include <vulkan/vulkan.h>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

namespace VulkanHost
{
class FeatureChain
{
    std::vector<std::unique_ptr<std::max_align_t[]>> copies;
    const void* head = nullptr;

    static bool NeedsEdit(const VkBaseInStructure* node)
    {
        switch (node->sType)
        {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:
            return reinterpret_cast<const VkPhysicalDeviceFeatures2*>(node)->features.shaderStorageImageExtendedFormats != VK_TRUE;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
            return reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(node)->timelineSemaphore != VK_TRUE;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
            const auto* f = reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(node);
            return f->synchronization2 != VK_TRUE || f->maintenance4 != VK_TRUE || f->shaderIntegerDotProduct != VK_TRUE;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES:
            return reinterpret_cast<const VkPhysicalDeviceTimelineSemaphoreFeatures*>(node)->timelineSemaphore != VK_TRUE;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES:
            return reinterpret_cast<const VkPhysicalDeviceSynchronization2Features*>(node)->synchronization2 != VK_TRUE;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES:
            return reinterpret_cast<const VkPhysicalDeviceMaintenance4Features*>(node)->maintenance4 != VK_TRUE;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES:
            return reinterpret_cast<const VkPhysicalDeviceShaderIntegerDotProductFeatures*>(node)->shaderIntegerDotProduct != VK_TRUE;
        default: return false;
        }
    }
    static size_t Size(VkStructureType type)
    {
#define X(tag, name) case tag: return sizeof(name)
        switch (type)
        {
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, VkPhysicalDeviceFeatures2);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, VkPhysicalDeviceVulkan13Features);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, VkPhysicalDeviceTimelineSemaphoreFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, VkPhysicalDeviceSynchronization2Features);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, VkPhysicalDeviceMaintenance4Features);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, VkPhysicalDeviceShaderIntegerDotProductFeatures);
#ifdef VK_KHR_cooperative_matrix
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, VkPhysicalDeviceCooperativeMatrixFeaturesKHR);
#endif
#ifdef VK_VERSION_1_1
        X(VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO, VkDeviceGroupDeviceCreateInfo);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, VkPhysicalDevice16BitStorageFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, VkPhysicalDeviceMultiviewFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES, VkPhysicalDeviceVariablePointersFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES, VkPhysicalDeviceSamplerYcbcrConversionFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, VkPhysicalDeviceShaderDrawParametersFeatures);
#endif
#ifdef VK_VERSION_1_2
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, VkPhysicalDeviceVulkan11Features);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, VkPhysicalDeviceDescriptorIndexingFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, VkPhysicalDeviceBufferDeviceAddressFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, VkPhysicalDeviceScalarBlockLayoutFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, VkPhysicalDeviceShaderFloat16Int8Features);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, VkPhysicalDeviceVulkanMemoryModelFeatures);
#endif
        default: return 0;
        }
#undef X
    }
public:
    bool Copy(const void* chain)
    {
        copies.clear();
        head = chain;
        const VkBaseInStructure* last = nullptr;
        unsigned links = 0;
        for (auto* n = static_cast<const VkBaseInStructure*>(chain); n; n = n->pNext)
        {
            if (++links > 64) return false;
            if (NeedsEdit(n)) last = n;
        }
        if (!last) return true;
        VkBaseOutStructure* previous = nullptr;
        for (auto* n = static_cast<const VkBaseInStructure*>(chain);; n = n->pNext)
        {
            const size_t size = Size(n->sType);
            if (!size) return false; // no way to copy an unknown Vulkan structure safely
            auto& storage = copies.emplace_back(std::make_unique<std::max_align_t[]>(
                (size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)));
            std::memcpy(storage.get(), n, size);
            auto* copy = reinterpret_cast<VkBaseOutStructure*>(storage.get());
            if (previous) previous->pNext = copy; else head = copy;
            previous = copy;
            if (n == last) return true;
        }
    }
    const void* Head() const { return head; }
};
}
