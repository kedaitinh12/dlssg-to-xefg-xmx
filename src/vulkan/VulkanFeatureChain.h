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
#ifdef VK_VERSION_1_3
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, VkPhysicalDeviceDynamicRenderingFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES, VkPhysicalDeviceInlineUniformBlockFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES, VkPhysicalDevicePipelineCreationCacheControlFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES, VkPhysicalDevicePrivateDataFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES, VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, VkPhysicalDeviceSubgroupSizeControlFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES, VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures);
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT, VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT);
#endif
#ifdef VK_EXT_mesh_shader
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, VkPhysicalDeviceMeshShaderFeaturesEXT);
#endif
#ifdef VK_EXT_graphics_pipeline_library
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT);
#endif
#ifdef VK_EXT_shader_module_identifier
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_FEATURES_EXT, VkPhysicalDeviceShaderModuleIdentifierFeaturesEXT);
#endif
#ifdef VK_EXT_custom_border_color
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT, VkPhysicalDeviceCustomBorderColorFeaturesEXT);
#endif
#ifdef VK_EXT_extended_dynamic_state
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, VkPhysicalDeviceExtendedDynamicStateFeaturesEXT);
#endif
#ifdef VK_EXT_extended_dynamic_state2
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT, VkPhysicalDeviceExtendedDynamicState2FeaturesEXT);
#endif
#ifdef VK_EXT_extended_dynamic_state3
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT, VkPhysicalDeviceExtendedDynamicState3FeaturesEXT);
#endif
#ifdef VK_EXT_depth_clip_control
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, VkPhysicalDeviceDepthClipControlFeaturesEXT);
#endif
#ifdef VK_EXT_robustness2
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, VkPhysicalDeviceRobustness2FeaturesEXT);
#endif
#ifdef VK_EXT_index_type_uint8
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, VkPhysicalDeviceIndexTypeUint8FeaturesEXT);
#endif
#ifdef VK_EXT_conditional_rendering
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT, VkPhysicalDeviceConditionalRenderingFeaturesEXT);
#endif
#ifdef VK_EXT_provoking_vertex
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT, VkPhysicalDeviceProvokingVertexFeaturesEXT);
#endif
#ifdef VK_EXT_vertex_attribute_divisor
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT, VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT);
#endif
#ifdef VK_EXT_image_2d_view_of_3d
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT, VkPhysicalDeviceImage2DViewOf3DFeaturesEXT);
#endif
#ifdef VK_EXT_device_fault
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT, VkPhysicalDeviceFaultFeaturesEXT);
#endif
#ifdef VK_EXT_mutable_descriptor_type
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT, VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT);
#endif
#ifdef VK_AMD_anti_lag
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ANTI_LAG_FEATURES_AMD, VkPhysicalDeviceAntiLagFeaturesAMD);
#endif
#ifdef VK_EXT_color_write_enable
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT, VkPhysicalDeviceColorWriteEnableFeaturesEXT);
#endif
#ifdef VK_EXT_line_rasterization
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT, VkPhysicalDeviceLineRasterizationFeaturesEXT);
#endif
#ifdef VK_EXT_primitive_topology_list_restart
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT, VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT);
#endif
#ifdef VK_EXT_transform_feedback
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT, VkPhysicalDeviceTransformFeedbackFeaturesEXT);
#endif
#ifdef VK_EXT_vertex_input_dynamic_state
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT, VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT);
#endif
#ifdef VK_EXT_4444_formats
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT, VkPhysicalDevice4444FormatsFeaturesEXT);
#endif
#ifdef VK_EXT_border_color_swizzle
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES_EXT, VkPhysicalDeviceBorderColorSwizzleFeaturesEXT);
#endif
#ifdef VK_EXT_non_seamless_cube_map
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT, VkPhysicalDeviceNonSeamlessCubeMapFeaturesEXT);
#endif
#ifdef VK_EXT_primitives_generated_query
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT, VkPhysicalDevicePrimitivesGeneratedQueryFeaturesEXT);
#endif
#ifdef VK_KHR_workgroup_memory_explicit_layout
        X(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR, VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR);
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
