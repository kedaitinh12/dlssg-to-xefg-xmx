// The part of the Vulkan API the Vulkan runtime's sources use, with the real names, member order and numeric values, so
// that tests/vulkan_runtime/check_syntax.sh can syntax check them on Linux without the Vulkan headers. It proves the
// sources agree with themselves and with this list, it cannot prove the list is complete or right: the build on
// Windows (the Vulkan-Headers the CI clones) is the real check.
#pragma once
#include <cstddef>
#include <cstdint>

#define VKAPI_ATTR
#define VKAPI_CALL
#define VKAPI_PTR

#define VK_MAKE_API_VERSION(variant, major, minor, patch) \
	((((uint32_t)(variant)) << 29U) | (((uint32_t)(major)) << 22U) | (((uint32_t)(minor)) << 12U) | ((uint32_t)(patch)))
#define VK_API_VERSION_1_0 VK_MAKE_API_VERSION(0, 1, 0, 0)
#define VK_API_VERSION_1_1 VK_MAKE_API_VERSION(0, 1, 1, 0)
#define VK_API_VERSION_1_2 VK_MAKE_API_VERSION(0, 1, 2, 0)
#define VK_API_VERSION_1_3 VK_MAKE_API_VERSION(0, 1, 3, 0)
#define VK_API_VERSION_VARIANT(version) ((uint32_t)(version) >> 29U)
#define VK_API_VERSION_MAJOR(version) (((uint32_t)(version) >> 22U) & 0x7FU)
#define VK_API_VERSION_MINOR(version) (((uint32_t)(version) >> 12U) & 0x3FFU)
#define VK_API_VERSION_PATCH(version) ((uint32_t)(version) & 0xFFFU)

#define VK_DEFINE_HANDLE(object) typedef struct object##_T* object;
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef struct object##_T* object;
#define VK_WHOLE_SIZE (~0ULL)
#define VK_NULL_HANDLE nullptr

#define VK_TRUE 1
#define VK_FALSE 0
#define VK_UUID_SIZE 16
#define VK_LUID_SIZE 8
#define VK_MAX_EXTENSION_NAME_SIZE 256
#define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256

#define VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME "VK_KHR_push_descriptor"
#define VK_NVX_BINARY_IMPORT_SPEC_VERSION 2
#define VK_NVX_BINARY_IMPORT_EXTENSION_NAME "VK_NVX_binary_import"
#define VK_NVX_IMAGE_VIEW_HANDLE_SPEC_VERSION 3
#define VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME "VK_NVX_image_view_handle"
#define VK_KHR_COOPERATIVE_MATRIX_SPEC_VERSION 2
#define VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME "VK_KHR_cooperative_matrix"

typedef uint32_t VkFlags;
typedef uint64_t VkFlags64;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef VkFlags VkInstanceCreateFlags;
typedef VkFlags VkDeviceCreateFlags;
typedef VkFlags VkDeviceQueueCreateFlags;
typedef VkFlags VkQueueFlags;
typedef VkFlags VkShaderStageFlags;
typedef VkFlags VkPipelineStageFlags;
typedef VkFlags VkAccessFlags;
typedef VkFlags VkDependencyFlags;
typedef VkFlags VkImageAspectFlags;
typedef VkFlags VkSubgroupFeatureFlags;
typedef VkFlags VkDescriptorSetLayoutCreateFlags;
typedef VkFlags VkPipelineLayoutCreateFlags;
typedef VkFlags VkSamplerCreateFlags;
typedef VkFlags VkShaderModuleCreateFlags;
typedef VkFlags VkPipelineShaderStageCreateFlags;
typedef VkFlags VkPipelineCreateFlags;
typedef VkFlags VkPipelineCacheCreateFlags;
typedef VkFlags64 VkPipelineStageFlags2;
typedef VkFlags64 VkAccessFlags2;

VK_DEFINE_HANDLE(VkInstance)
VK_DEFINE_HANDLE(VkPhysicalDevice)
VK_DEFINE_HANDLE(VkDevice)
VK_DEFINE_HANDLE(VkQueue)
VK_DEFINE_HANDLE(VkCommandBuffer)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkImage)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkImageView)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkBuffer)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkSampler)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkShaderModule)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkPipeline)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkPipelineCache)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkPipelineLayout)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkDescriptorSet)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkDescriptorSetLayout)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkSurfaceKHR)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkCuModuleNVX)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkCuFunctionNVX)

typedef enum VkResult
{
	VK_SUCCESS = 0,
	VK_NOT_READY = 1,
	VK_TIMEOUT = 2,
	VK_EVENT_SET = 3,
	VK_EVENT_RESET = 4,
	VK_INCOMPLETE = 5,
	VK_ERROR_OUT_OF_HOST_MEMORY = -1,
	VK_ERROR_OUT_OF_DEVICE_MEMORY = -2,
	VK_ERROR_INITIALIZATION_FAILED = -3,
	VK_ERROR_DEVICE_LOST = -4,
	VK_ERROR_MEMORY_MAP_FAILED = -5,
	VK_ERROR_LAYER_NOT_PRESENT = -6,
	VK_ERROR_EXTENSION_NOT_PRESENT = -7,
	VK_ERROR_FEATURE_NOT_PRESENT = -8,
	VK_ERROR_INCOMPATIBLE_DRIVER = -9,
	VK_ERROR_TOO_MANY_OBJECTS = -10,
	VK_ERROR_FORMAT_NOT_SUPPORTED = -11,
	VK_ERROR_FRAGMENTED_POOL = -12,
	VK_ERROR_UNKNOWN = -13,
} VkResult;

typedef enum VkStructureType
{
	VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
	VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
	VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
	VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
	VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
	VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO = 17,
	VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
	VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
	VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
	VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO = 31,
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
	VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
	VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46,
	VK_STRUCTURE_TYPE_CU_MODULE_CREATE_INFO_NVX = 1000029000,
	VK_STRUCTURE_TYPE_CU_FUNCTION_CREATE_INFO_NVX = 1000029001,
	VK_STRUCTURE_TYPE_CU_LAUNCH_INFO_NVX = 1000029002,
	VK_STRUCTURE_TYPE_CU_MODULE_TEXTURING_MODE_CREATE_INFO_NVX = 1000029004,
	VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX = 1000030000,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES = 1000071004,
	VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 = 1000314000,
	VK_STRUCTURE_TYPE_DEPENDENCY_INFO = 1000314003,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES = 50,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES = 51,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES = 53,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES = 1000280000,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES = 1000207000,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES = 1000211000,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES = 1000314007,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES = 1000413000,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR = 1000506000,
	VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR = 1000506001,
	VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR = 1000506002,
} VkStructureType;

typedef enum VkComponentTypeKHR
{
	VK_COMPONENT_TYPE_FLOAT16_KHR = 0,
	VK_COMPONENT_TYPE_FLOAT32_KHR = 1,
	VK_COMPONENT_TYPE_FLOAT64_KHR = 2,
	VK_COMPONENT_TYPE_SINT8_KHR = 3,
	VK_COMPONENT_TYPE_SINT16_KHR = 4,
	VK_COMPONENT_TYPE_SINT32_KHR = 5,
	VK_COMPONENT_TYPE_SINT64_KHR = 6,
	VK_COMPONENT_TYPE_UINT8_KHR = 7,
	VK_COMPONENT_TYPE_UINT16_KHR = 8,
	VK_COMPONENT_TYPE_UINT32_KHR = 9,
	VK_COMPONENT_TYPE_UINT64_KHR = 10,
} VkComponentTypeKHR;

typedef enum VkScopeKHR
{
	VK_SCOPE_DEVICE_KHR = 1,
	VK_SCOPE_WORKGROUP_KHR = 2,
	VK_SCOPE_SUBGROUP_KHR = 3,
	VK_SCOPE_QUEUE_FAMILY_KHR = 5,
} VkScopeKHR;

typedef enum VkFormat
{
	VK_FORMAT_UNDEFINED = 0,
	VK_FORMAT_R8_UINT = 13,
	VK_FORMAT_R8G8B8A8_UNORM = 37,
	VK_FORMAT_R8G8B8A8_SRGB = 43,
	VK_FORMAT_B8G8R8A8_UNORM = 44,
	VK_FORMAT_B8G8R8A8_SRGB = 50,
	VK_FORMAT_A2B10G10R10_UNORM_PACK32 = 64,
	VK_FORMAT_R16G16_SFLOAT = 83,
	VK_FORMAT_R16G16B16A16_SFLOAT = 97,
	VK_FORMAT_R32_SFLOAT = 100,
	VK_FORMAT_R32G32_SFLOAT = 103,
	VK_FORMAT_R32G32B32A32_SFLOAT = 109,
	VK_FORMAT_B10G11R11_UFLOAT_PACK32 = 122,
} VkFormat;

typedef enum VkImageLayout
{
	VK_IMAGE_LAYOUT_UNDEFINED = 0,
	VK_IMAGE_LAYOUT_GENERAL = 1,
} VkImageLayout;

typedef enum VkImageAspectFlagBits
{
	VK_IMAGE_ASPECT_COLOR_BIT = 0x00000001,
} VkImageAspectFlagBits;

typedef enum VkImageUsageFlagBits
{
	VK_IMAGE_USAGE_STORAGE_BIT = 0x00000008,
} VkImageUsageFlagBits;

typedef enum VkImageViewType
{
	VK_IMAGE_VIEW_TYPE_1D = 0,
	VK_IMAGE_VIEW_TYPE_2D = 1,
} VkImageViewType;

typedef enum VkPipelineBindPoint
{
	VK_PIPELINE_BIND_POINT_GRAPHICS = 0,
	VK_PIPELINE_BIND_POINT_COMPUTE = 1,
} VkPipelineBindPoint;

typedef enum VkShaderStageFlagBits
{
	VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020,
} VkShaderStageFlagBits;

typedef enum VkDescriptorType
{
	VK_DESCRIPTOR_TYPE_SAMPLER = 0,
	VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1,
	VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE = 2,
	VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3,
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
} VkDescriptorType;

typedef enum VkDescriptorSetLayoutCreateFlagBits
{
	VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR = 0x00000001,
} VkDescriptorSetLayoutCreateFlagBits;

typedef enum VkFilter
{
	VK_FILTER_NEAREST = 0,
	VK_FILTER_LINEAR = 1,
} VkFilter;

typedef enum VkSamplerMipmapMode
{
	VK_SAMPLER_MIPMAP_MODE_NEAREST = 0,
	VK_SAMPLER_MIPMAP_MODE_LINEAR = 1,
} VkSamplerMipmapMode;

typedef enum VkSamplerAddressMode
{
	VK_SAMPLER_ADDRESS_MODE_REPEAT = 0,
	VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 2,
} VkSamplerAddressMode;

typedef enum VkCompareOp
{
	VK_COMPARE_OP_NEVER = 0,
} VkCompareOp;

typedef enum VkBorderColor
{
	VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK = 0,
} VkBorderColor;

typedef enum VkAccessFlagBits
{
	VK_ACCESS_SHADER_READ_BIT = 0x00000020,
	VK_ACCESS_SHADER_WRITE_BIT = 0x00000040,
	VK_ACCESS_TRANSFER_READ_BIT = 0x00000800,
	VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000,
	VK_ACCESS_MEMORY_READ_BIT = 0x00008000,
	VK_ACCESS_MEMORY_WRITE_BIT = 0x00010000,
} VkAccessFlagBits;

typedef enum VkPipelineStageFlagBits
{
	VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800,
	VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000,
	VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT = 0x00008000,
	VK_PIPELINE_STAGE_ALL_COMMANDS_BIT = 0x00010000,
} VkPipelineStageFlagBits;

typedef enum VkQueueFlagBits
{
	VK_QUEUE_GRAPHICS_BIT = 0x00000001,
	VK_QUEUE_COMPUTE_BIT = 0x00000002,
} VkQueueFlagBits;

typedef enum VkSubgroupFeatureFlagBits
{
	VK_SUBGROUP_FEATURE_BASIC_BIT = 0x00000001,
	VK_SUBGROUP_FEATURE_VOTE_BIT = 0x00000002,
	VK_SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x00000004,
} VkSubgroupFeatureFlagBits;

typedef enum VkPhysicalDeviceType
{
	VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
} VkPhysicalDeviceType;

typedef enum VkPointClippingBehavior
{
	VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES = 0,
} VkPointClippingBehavior;

// xvk.h
typedef enum VkFullScreenExclusiveEXT
{
	VK_FULL_SCREEN_EXCLUSIVE_DEFAULT_EXT = 0,
} VkFullScreenExclusiveEXT;

typedef enum VkPresentModeKHR
{
	VK_PRESENT_MODE_IMMEDIATE_KHR = 0,
	VK_PRESENT_MODE_MAILBOX_KHR = 1,
	VK_PRESENT_MODE_FIFO_KHR = 2,
	VK_PRESENT_MODE_FIFO_RELAXED_KHR = 3,
} VkPresentModeKHR;

typedef enum VkLatencyMarkerNV
{
	VK_LATENCY_MARKER_SIMULATION_START_NV = 0,
} VkLatencyMarkerNV;

typedef enum VkAntiLagStageAMD
{
	VK_ANTI_LAG_STAGE_INPUT_AMD = 0,
} VkAntiLagStageAMD;

typedef struct VkAllocationCallbacks
{
	void* pUserData;
	void* pfnAllocation;
	void* pfnReallocation;
	void* pfnFree;
	void* pfnInternalAllocation;
	void* pfnInternalFree;
} VkAllocationCallbacks;

typedef struct VkBaseInStructure
{
	VkStructureType sType;
	const struct VkBaseInStructure* pNext;
} VkBaseInStructure;

typedef struct VkBaseOutStructure
{
	VkStructureType sType;
	struct VkBaseOutStructure* pNext;
} VkBaseOutStructure;

typedef struct VkOffset3D
{
	int32_t x;
	int32_t y;
	int32_t z;
} VkOffset3D;

typedef struct VkExtent3D
{
	uint32_t width;
	uint32_t height;
	uint32_t depth;
} VkExtent3D;

typedef struct VkApplicationInfo
{
	VkStructureType sType;
	const void* pNext;
	const char* pApplicationName;
	uint32_t applicationVersion;
	const char* pEngineName;
	uint32_t engineVersion;
	uint32_t apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkInstanceCreateFlags flags;
	const VkApplicationInfo* pApplicationInfo;
	uint32_t enabledLayerCount;
	const char* const* ppEnabledLayerNames;
	uint32_t enabledExtensionCount;
	const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkDeviceQueueCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkDeviceQueueCreateFlags flags;
	uint32_t queueFamilyIndex;
	uint32_t queueCount;
	const float* pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkPhysicalDeviceFeatures
{
	VkBool32 robustBufferAccess;
	VkBool32 fullDrawIndexUint32;
	VkBool32 imageCubeArray;
	VkBool32 independentBlend;
	VkBool32 geometryShader;
	VkBool32 tessellationShader;
	VkBool32 sampleRateShading;
	VkBool32 dualSrcBlend;
	VkBool32 logicOp;
	VkBool32 multiDrawIndirect;
	VkBool32 drawIndirectFirstInstance;
	VkBool32 depthClamp;
	VkBool32 depthBiasClamp;
	VkBool32 fillModeNonSolid;
	VkBool32 depthBounds;
	VkBool32 wideLines;
	VkBool32 largePoints;
	VkBool32 alphaToOne;
	VkBool32 multiViewport;
	VkBool32 samplerAnisotropy;
	VkBool32 textureCompressionETC2;
	VkBool32 textureCompressionASTC_LDR;
	VkBool32 textureCompressionBC;
	VkBool32 occlusionQueryPrecise;
	VkBool32 pipelineStatisticsQuery;
	VkBool32 vertexPipelineStoresAndAtomics;
	VkBool32 fragmentStoresAndAtomics;
	VkBool32 shaderTessellationAndGeometryPointSize;
	VkBool32 shaderImageGatherExtended;
	VkBool32 shaderStorageImageExtendedFormats;
	VkBool32 shaderStorageImageMultisample;
	VkBool32 shaderStorageImageReadWithoutFormat;
	VkBool32 shaderStorageImageWriteWithoutFormat;
	VkBool32 shaderUniformBufferArrayDynamicIndexing;
	VkBool32 shaderSampledImageArrayDynamicIndexing;
	VkBool32 shaderStorageBufferArrayDynamicIndexing;
	VkBool32 shaderStorageImageArrayDynamicIndexing;
	VkBool32 shaderClipDistance;
	VkBool32 shaderCullDistance;
	VkBool32 shaderFloat64;
	VkBool32 shaderInt64;
	VkBool32 shaderInt16;
	VkBool32 shaderResourceResidency;
	VkBool32 shaderResourceMinLod;
	VkBool32 sparseBinding;
	VkBool32 sparseResidencyBuffer;
	VkBool32 sparseResidencyImage2D;
	VkBool32 sparseResidencyImage3D;
	VkBool32 sparseResidency2Samples;
	VkBool32 sparseResidency4Samples;
	VkBool32 sparseResidency8Samples;
	VkBool32 sparseResidency16Samples;
	VkBool32 sparseResidencyAliased;
	VkBool32 variableMultisampleRate;
	VkBool32 inheritedQueries;
} VkPhysicalDeviceFeatures;

typedef struct VkDeviceCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkDeviceCreateFlags flags;
	uint32_t queueCreateInfoCount;
	const VkDeviceQueueCreateInfo* pQueueCreateInfos;
	uint32_t enabledLayerCount;
	const char* const* ppEnabledLayerNames;
	uint32_t enabledExtensionCount;
	const char* const* ppEnabledExtensionNames;
	const VkPhysicalDeviceFeatures* pEnabledFeatures;
} VkDeviceCreateInfo;

typedef struct VkPhysicalDeviceFeatures2
{
	VkStructureType sType;
	void* pNext;
	VkPhysicalDeviceFeatures features;
} VkPhysicalDeviceFeatures2;

typedef struct VkPhysicalDeviceLimits
{
	uint32_t maxImageDimension1D; // the rest of the real structure is not used
	float timestampPeriod;
} VkPhysicalDeviceLimits;

typedef struct VkPhysicalDeviceSparseProperties
{
	VkBool32 residencyStandard2DBlockShape;
	VkBool32 residencyStandard2DMultisampleBlockShape;
	VkBool32 residencyStandard3DBlockShape;
	VkBool32 residencyAlignedMipSize;
	VkBool32 residencyNonResidentStrict;
} VkPhysicalDeviceSparseProperties;

typedef struct VkPhysicalDeviceProperties
{
	uint32_t apiVersion;
	uint32_t driverVersion;
	uint32_t vendorID;
	uint32_t deviceID;
	VkPhysicalDeviceType deviceType;
	char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
	uint8_t pipelineCacheUUID[VK_UUID_SIZE];
	VkPhysicalDeviceLimits limits;
	VkPhysicalDeviceSparseProperties sparseProperties;
} VkPhysicalDeviceProperties;

typedef struct VkPhysicalDeviceProperties2
{
	VkStructureType sType;
	void* pNext;
	VkPhysicalDeviceProperties properties;
} VkPhysicalDeviceProperties2;

typedef struct VkPhysicalDeviceIDProperties
{
	VkStructureType sType;
	void* pNext;
	uint8_t deviceUUID[VK_UUID_SIZE];
	uint8_t driverUUID[VK_UUID_SIZE];
	uint8_t deviceLUID[VK_LUID_SIZE];
	uint32_t deviceNodeMask;
	VkBool32 deviceLUIDValid;
} VkPhysicalDeviceIDProperties;

typedef struct VkPhysicalDeviceVulkan11Properties
{
	VkStructureType sType;
	void* pNext;
	uint8_t deviceUUID[VK_UUID_SIZE];
	uint8_t driverUUID[VK_UUID_SIZE];
	uint8_t deviceLUID[VK_LUID_SIZE];
	uint32_t deviceNodeMask;
	VkBool32 deviceLUIDValid;
	uint32_t subgroupSize;
	VkShaderStageFlags subgroupSupportedStages;
	VkSubgroupFeatureFlags subgroupSupportedOperations;
	VkBool32 subgroupQuadOperationsInAllStages;
	VkPointClippingBehavior pointClippingBehavior;
	uint32_t maxMultiviewViewCount;
	uint32_t maxMultiviewInstanceIndex;
	VkBool32 protectedNoFault;
	uint32_t maxPerSetDescriptors;
	VkDeviceSize maxMemoryAllocationSize;
} VkPhysicalDeviceVulkan11Properties;

typedef struct VkPhysicalDeviceVulkan12Features
{
	VkStructureType sType;
	void* pNext;
	VkBool32 samplerMirrorClampToEdge;
	VkBool32 drawIndirectCount;
	VkBool32 storageBuffer8BitAccess;
	VkBool32 uniformAndStorageBuffer8BitAccess;
	VkBool32 storagePushConstant8;
	VkBool32 shaderBufferInt64Atomics;
	VkBool32 shaderSharedInt64Atomics;
	VkBool32 shaderFloat16;
	VkBool32 shaderInt8;
	VkBool32 descriptorIndexing;
	VkBool32 shaderInputAttachmentArrayDynamicIndexing;
	VkBool32 shaderUniformTexelBufferArrayDynamicIndexing;
	VkBool32 shaderStorageTexelBufferArrayDynamicIndexing;
	VkBool32 shaderUniformBufferArrayNonUniformIndexing;
	VkBool32 shaderSampledImageArrayNonUniformIndexing;
	VkBool32 shaderStorageBufferArrayNonUniformIndexing;
	VkBool32 shaderStorageImageArrayNonUniformIndexing;
	VkBool32 shaderInputAttachmentArrayNonUniformIndexing;
	VkBool32 shaderUniformTexelBufferArrayNonUniformIndexing;
	VkBool32 shaderStorageTexelBufferArrayNonUniformIndexing;
	VkBool32 descriptorBindingUniformBufferUpdateAfterBind;
	VkBool32 descriptorBindingSampledImageUpdateAfterBind;
	VkBool32 descriptorBindingStorageImageUpdateAfterBind;
	VkBool32 descriptorBindingStorageBufferUpdateAfterBind;
	VkBool32 descriptorBindingUniformTexelBufferUpdateAfterBind;
	VkBool32 descriptorBindingStorageTexelBufferUpdateAfterBind;
	VkBool32 descriptorBindingUpdateUnusedWhilePending;
	VkBool32 descriptorBindingPartiallyBound;
	VkBool32 descriptorBindingVariableDescriptorCount;
	VkBool32 runtimeDescriptorArray;
	VkBool32 samplerFilterMinmax;
	VkBool32 scalarBlockLayout;
	VkBool32 imagelessFramebuffer;
	VkBool32 uniformBufferStandardLayout;
	VkBool32 shaderSubgroupExtendedTypes;
	VkBool32 separateDepthStencilLayouts;
	VkBool32 hostQueryReset;
	VkBool32 timelineSemaphore;
	VkBool32 bufferDeviceAddress;
	VkBool32 bufferDeviceAddressCaptureReplay;
	VkBool32 bufferDeviceAddressMultiDevice;
	VkBool32 vulkanMemoryModel;
	VkBool32 vulkanMemoryModelDeviceScope;
	VkBool32 vulkanMemoryModelAvailabilityVisibilityChains;
	VkBool32 shaderOutputViewportIndex;
	VkBool32 shaderOutputLayer;
	VkBool32 subgroupBroadcastDynamicId;
} VkPhysicalDeviceVulkan12Features;

typedef struct VkPhysicalDeviceVulkan13Features
{
	VkStructureType sType;
	void* pNext;
	VkBool32 robustImageAccess;
	VkBool32 inlineUniformBlock;
	VkBool32 descriptorBindingInlineUniformBlockUpdateAfterBind;
	VkBool32 pipelineCreationCacheControl;
	VkBool32 privateData;
	VkBool32 shaderDemoteToHelperInvocation;
	VkBool32 shaderTerminateInvocation;
	VkBool32 subgroupSizeControl;
	VkBool32 computeFullSubgroups;
	VkBool32 synchronization2;
	VkBool32 textureCompressionASTC_HDR;
	VkBool32 shaderZeroInitializeWorkgroupMemory;
	VkBool32 dynamicRendering;
	VkBool32 shaderIntegerDotProduct;
	VkBool32 maintenance4;
} VkPhysicalDeviceVulkan13Features;

typedef struct VkPhysicalDeviceTimelineSemaphoreFeatures
{
	VkStructureType sType;
	void* pNext;
	VkBool32 timelineSemaphore;
} VkPhysicalDeviceTimelineSemaphoreFeatures;

typedef struct VkPhysicalDeviceSynchronization2Features
{
	VkStructureType sType;
	void* pNext;
	VkBool32 synchronization2;
} VkPhysicalDeviceSynchronization2Features;

typedef struct VkPhysicalDeviceMaintenance4Features
{
	VkStructureType sType;
	void* pNext;
	VkBool32 maintenance4;
} VkPhysicalDeviceMaintenance4Features;

typedef struct VkPhysicalDeviceShaderIntegerDotProductFeatures
{
	VkStructureType sType;
	void* pNext;
	VkBool32 shaderIntegerDotProduct;
} VkPhysicalDeviceShaderIntegerDotProductFeatures;

typedef struct VkPhysicalDeviceVulkanMemoryModelFeatures
{
	VkStructureType sType;
	void* pNext;
	VkBool32 vulkanMemoryModel;
	VkBool32 vulkanMemoryModelDeviceScope;
	VkBool32 vulkanMemoryModelAvailabilityVisibilityChains;
} VkPhysicalDeviceVulkanMemoryModelFeatures;

typedef struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR
{
	VkStructureType sType;
	void* pNext;
	VkBool32 cooperativeMatrix;
	VkBool32 cooperativeMatrixRobustBufferAccess;
} VkPhysicalDeviceCooperativeMatrixFeaturesKHR;

typedef struct VkPhysicalDeviceCooperativeMatrixPropertiesKHR
{
	VkStructureType sType;
	void* pNext;
	VkShaderStageFlags cooperativeMatrixSupportedStages;
} VkPhysicalDeviceCooperativeMatrixPropertiesKHR;

typedef struct VkCooperativeMatrixPropertiesKHR
{
	VkStructureType sType;
	void* pNext;
	uint32_t MSize;
	uint32_t NSize;
	uint32_t KSize;
	VkComponentTypeKHR AType;
	VkComponentTypeKHR BType;
	VkComponentTypeKHR CType;
	VkComponentTypeKHR ResultType;
	VkBool32 saturatingAccumulation;
	VkScopeKHR scope;
} VkCooperativeMatrixPropertiesKHR;

typedef struct VkExtensionProperties
{
	char extensionName[VK_MAX_EXTENSION_NAME_SIZE];
	uint32_t specVersion;
} VkExtensionProperties;

typedef struct VkQueueFamilyProperties
{
	VkQueueFlags queueFlags;
	uint32_t queueCount;
	uint32_t timestampValidBits;
	VkExtent3D minImageTransferGranularity;
} VkQueueFamilyProperties;

typedef struct VkImageSubresourceLayers
{
	VkImageAspectFlags aspectMask;
	uint32_t mipLevel;
	uint32_t baseArrayLayer;
	uint32_t layerCount;
} VkImageSubresourceLayers;

typedef struct VkImageSubresourceRange
{
	VkImageAspectFlags aspectMask;
	uint32_t baseMipLevel;
	uint32_t levelCount;
	uint32_t baseArrayLayer;
	uint32_t layerCount;
} VkImageSubresourceRange;

typedef struct VkBufferCopy
{
	VkDeviceSize srcOffset;
	VkDeviceSize dstOffset;
	VkDeviceSize size;
} VkBufferCopy;

typedef struct VkImageCopy
{
	VkImageSubresourceLayers srcSubresource;
	VkOffset3D srcOffset;
	VkImageSubresourceLayers dstSubresource;
	VkOffset3D dstOffset;
	VkExtent3D extent;
} VkImageCopy;

typedef struct VkMemoryBarrier
{
	VkStructureType sType;
	const void* pNext;
	VkAccessFlags srcAccessMask;
	VkAccessFlags dstAccessMask;
} VkMemoryBarrier;

typedef struct VkBufferMemoryBarrier VkBufferMemoryBarrier;
typedef struct VkImageMemoryBarrier VkImageMemoryBarrier;
typedef struct VkBufferMemoryBarrier2 VkBufferMemoryBarrier2;
typedef struct VkImageMemoryBarrier2 VkImageMemoryBarrier2;

typedef struct VkMemoryBarrier2
{
	VkStructureType sType;
	const void* pNext;
	VkPipelineStageFlags2 srcStageMask;
	VkAccessFlags2 srcAccessMask;
	VkPipelineStageFlags2 dstStageMask;
	VkAccessFlags2 dstAccessMask;
} VkMemoryBarrier2;

typedef struct VkDependencyInfo
{
	VkStructureType sType;
	const void* pNext;
	VkDependencyFlags dependencyFlags;
	uint32_t memoryBarrierCount;
	const VkMemoryBarrier2* pMemoryBarriers;
	uint32_t bufferMemoryBarrierCount;
	const VkBufferMemoryBarrier2* pBufferMemoryBarriers;
	uint32_t imageMemoryBarrierCount;
	const VkImageMemoryBarrier2* pImageMemoryBarriers;
} VkDependencyInfo;

// VK_NVX_binary_import and VK_NVX_image_view_handle (src/vulkan/xvk_tensor.h)
typedef struct VkCuModuleCreateInfoNVX
{
	VkStructureType sType;
	const void* pNext;
	size_t dataSize;
	const void* pData;
} VkCuModuleCreateInfoNVX;

typedef struct VkCuModuleTexturingModeCreateInfoNVX
{
	VkStructureType sType;
	const void* pNext;
	VkBool32 use64bitTexturing;
} VkCuModuleTexturingModeCreateInfoNVX;

typedef struct VkCuFunctionCreateInfoNVX
{
	VkStructureType sType;
	const void* pNext;
	VkCuModuleNVX module;
	const char* pName;
} VkCuFunctionCreateInfoNVX;

typedef struct VkCuLaunchInfoNVX
{
	VkStructureType sType;
	const void* pNext;
	VkCuFunctionNVX function;
	uint32_t gridDimX;
	uint32_t gridDimY;
	uint32_t gridDimZ;
	uint32_t blockDimX;
	uint32_t blockDimY;
	uint32_t blockDimZ;
	uint32_t sharedMemBytes;
	size_t paramCount;
	const void* const* pParams;
	size_t extraCount;
	const void* const* pExtras;
} VkCuLaunchInfoNVX;

typedef struct VkImageViewHandleInfoNVX
{
	VkStructureType sType;
	const void* pNext;
	VkImageView imageView;
	VkDescriptorType descriptorType;
	VkSampler sampler;
} VkImageViewHandleInfoNVX;

typedef struct VkDescriptorImageInfo
{
	VkSampler sampler;
	VkImageView imageView;
	VkImageLayout imageLayout;
} VkDescriptorImageInfo;

typedef struct VkDescriptorBufferInfo
{
	VkBuffer buffer;
	VkDeviceSize offset;
	VkDeviceSize range;
} VkDescriptorBufferInfo;
typedef struct VkBufferView_T* VkBufferView;

typedef struct VkWriteDescriptorSet
{
	VkStructureType sType;
	const void* pNext;
	VkDescriptorSet dstSet;
	uint32_t dstBinding;
	uint32_t dstArrayElement;
	uint32_t descriptorCount;
	VkDescriptorType descriptorType;
	const VkDescriptorImageInfo* pImageInfo;
	const VkDescriptorBufferInfo* pBufferInfo;
	const VkBufferView* pTexelBufferView;
} VkWriteDescriptorSet;

typedef struct VkSamplerCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkSamplerCreateFlags flags;
	VkFilter magFilter;
	VkFilter minFilter;
	VkSamplerMipmapMode mipmapMode;
	VkSamplerAddressMode addressModeU;
	VkSamplerAddressMode addressModeV;
	VkSamplerAddressMode addressModeW;
	float mipLodBias;
	VkBool32 anisotropyEnable;
	float maxAnisotropy;
	VkBool32 compareEnable;
	VkCompareOp compareOp;
	float minLod;
	float maxLod;
	VkBorderColor borderColor;
	VkBool32 unnormalizedCoordinates;
} VkSamplerCreateInfo;

typedef struct VkDescriptorSetLayoutBinding
{
	uint32_t binding;
	VkDescriptorType descriptorType;
	uint32_t descriptorCount;
	VkShaderStageFlags stageFlags;
	const VkSampler* pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct VkDescriptorSetLayoutCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkDescriptorSetLayoutCreateFlags flags;
	uint32_t bindingCount;
	const VkDescriptorSetLayoutBinding* pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct VkPushConstantRange
{
	VkShaderStageFlags stageFlags;
	uint32_t offset;
	uint32_t size;
} VkPushConstantRange;

typedef struct VkPipelineLayoutCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkPipelineLayoutCreateFlags flags;
	uint32_t setLayoutCount;
	const VkDescriptorSetLayout* pSetLayouts;
	uint32_t pushConstantRangeCount;
	const VkPushConstantRange* pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct VkShaderModuleCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkShaderModuleCreateFlags flags;
	size_t codeSize;
	const uint32_t* pCode;
} VkShaderModuleCreateInfo;

typedef struct VkSpecializationMapEntry
{
	uint32_t constantID;
	uint32_t offset;
	size_t size;
} VkSpecializationMapEntry;

typedef struct VkSpecializationInfo
{
	uint32_t mapEntryCount;
	const VkSpecializationMapEntry* pMapEntries;
	size_t dataSize;
	const void* pData;
} VkSpecializationInfo;

typedef struct VkPipelineShaderStageCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkPipelineShaderStageCreateFlags flags;
	VkShaderStageFlagBits stage;
	VkShaderModule module;
	const char* pName;
	const VkSpecializationInfo* pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct VkPipelineCacheCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkPipelineCacheCreateFlags flags;
	size_t initialDataSize;
	const void* pInitialData;
} VkPipelineCacheCreateInfo;

typedef struct VkComputePipelineCreateInfo
{
	VkStructureType sType;
	const void* pNext;
	VkPipelineCreateFlags flags;
	VkPipelineShaderStageCreateInfo stage;
	VkPipelineLayout layout;
	VkPipeline basePipelineHandle;
	int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

typedef void(VKAPI_PTR* PFN_vkVoidFunction)(void);

typedef PFN_vkVoidFunction(VKAPI_PTR* PFN_vkGetInstanceProcAddr)(VkInstance instance, const char* pName);
typedef PFN_vkVoidFunction(VKAPI_PTR* PFN_vkGetDeviceProcAddr)(VkDevice device, const char* pName);
typedef VkResult(VKAPI_PTR* PFN_vkCreateInstance)(
	const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance);
typedef VkResult(VKAPI_PTR* PFN_vkCreateDevice)(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDevice* pDevice);
typedef VkResult(VKAPI_PTR* PFN_vkEnumerateInstanceVersion)(uint32_t* pApiVersion);
typedef VkResult(VKAPI_PTR* PFN_vkEnumeratePhysicalDevices)(
	VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices);
typedef VkResult(VKAPI_PTR* PFN_vkEnumerateDeviceExtensionProperties)(VkPhysicalDevice physicalDevice,
	const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties);
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceFeatures2)(
	VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures);
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceProperties2)(
	VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* pProperties);
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceQueueFamilyProperties)(
	VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties);

typedef void(VKAPI_PTR* PFN_vkCmdCopyBuffer)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer,
	VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions);
typedef void(VKAPI_PTR* PFN_vkCmdFillBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
	VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data);
typedef struct VkQueryPool_T* VkQueryPool;
typedef uint32_t VkQueryType;
typedef uint32_t VkQueryResultFlags;
typedef uint32_t VkQueryPoolCreateFlags;
constexpr VkQueryType VK_QUERY_TYPE_TIMESTAMP = 2;
constexpr VkQueryResultFlags VK_QUERY_RESULT_64_BIT = 0x00000001;
constexpr VkStructureType VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = static_cast<VkStructureType>(11);
constexpr VkPipelineStageFlagBits VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = static_cast<VkPipelineStageFlagBits>(0x00002000);
typedef struct VkQueryPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkQueryPoolCreateFlags flags;
    VkQueryType queryType;
    uint32_t queryCount;
    uint32_t pipelineStatistics;
} VkQueryPoolCreateInfo;
typedef VkResult(VKAPI_PTR* PFN_vkCreateQueryPool)(VkDevice, const VkQueryPoolCreateInfo*, const VkAllocationCallbacks*, VkQueryPool*);
typedef void(VKAPI_PTR* PFN_vkDestroyQueryPool)(VkDevice, VkQueryPool, const VkAllocationCallbacks*);
typedef void(VKAPI_PTR* PFN_vkCmdResetQueryPool)(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t);
typedef void(VKAPI_PTR* PFN_vkCmdWriteTimestamp)(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t);
typedef VkResult(VKAPI_PTR* PFN_vkGetQueryPoolResults)(VkDevice, VkQueryPool, uint32_t, uint32_t, size_t, void*, VkDeviceSize, VkQueryResultFlags);

typedef void(VKAPI_PTR* PFN_vkCmdCopyImage)(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout,
	VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageCopy* pRegions);
typedef void(VKAPI_PTR* PFN_vkCmdPipelineBarrier)(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
	VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount,
	const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount,
	const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount,
	const VkImageMemoryBarrier* pImageMemoryBarriers);
typedef void(VKAPI_PTR* PFN_vkCmdBindPipeline)(
	VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline);
typedef void(VKAPI_PTR* PFN_vkCmdBindDescriptorSets)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
	VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets,
	uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets);
typedef void(VKAPI_PTR* PFN_vkCmdPushDescriptorSetKHR)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
	VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites);
typedef void(VKAPI_PTR* PFN_vkCmdPushConstants)(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
	VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues);
typedef void(VKAPI_PTR* PFN_vkCmdDispatch)(
	VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
typedef VkResult(VKAPI_PTR* PFN_vkCreateSampler)(VkDevice device, const VkSamplerCreateInfo* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkSampler* pSampler);
typedef void(VKAPI_PTR* PFN_vkDestroySampler)(
	VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator);
typedef VkResult(VKAPI_PTR* PFN_vkCreateDescriptorSetLayout)(VkDevice device,
	const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
	VkDescriptorSetLayout* pSetLayout);
typedef void(VKAPI_PTR* PFN_vkDestroyDescriptorSetLayout)(
	VkDevice device, VkDescriptorSetLayout descriptorSetLayout, const VkAllocationCallbacks* pAllocator);
typedef VkResult(VKAPI_PTR* PFN_vkCreatePipelineLayout)(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout);
typedef void(VKAPI_PTR* PFN_vkDestroyPipelineLayout)(
	VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks* pAllocator);
typedef VkResult(VKAPI_PTR* PFN_vkCreateShaderModule)(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule);
typedef void(VKAPI_PTR* PFN_vkDestroyShaderModule)(
	VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator);
typedef VkResult(VKAPI_PTR* PFN_vkCreatePipelineCache)(VkDevice device, const VkPipelineCacheCreateInfo* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkPipelineCache* pPipelineCache);
typedef void(VKAPI_PTR* PFN_vkDestroyPipelineCache)(
	VkDevice device, VkPipelineCache pipelineCache, const VkAllocationCallbacks* pAllocator);
typedef VkResult(VKAPI_PTR* PFN_vkGetPipelineCacheData)(
	VkDevice device, VkPipelineCache pipelineCache, size_t* pDataSize, void* pData);
typedef VkResult(VKAPI_PTR* PFN_vkCreateComputePipelines)(VkDevice device, VkPipelineCache pipelineCache,
	uint32_t createInfoCount, const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator,
	VkPipeline* pPipelines);
typedef void(VKAPI_PTR* PFN_vkDestroyPipeline)(
	VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator);

typedef VkResult(VKAPI_PTR* PFN_vkCreateCuModuleNVX)(VkDevice device, const VkCuModuleCreateInfoNVX* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkCuModuleNVX* pModule);
typedef VkResult(VKAPI_PTR* PFN_vkCreateCuFunctionNVX)(VkDevice device, const VkCuFunctionCreateInfoNVX* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkCuFunctionNVX* pFunction);
typedef void(VKAPI_PTR* PFN_vkDestroyCuModuleNVX)(
	VkDevice device, VkCuModuleNVX module, const VkAllocationCallbacks* pAllocator);
typedef void(VKAPI_PTR* PFN_vkDestroyCuFunctionNVX)(
	VkDevice device, VkCuFunctionNVX function, const VkAllocationCallbacks* pAllocator);
typedef void(VKAPI_PTR* PFN_vkCmdCuLaunchKernelNVX)(VkCommandBuffer commandBuffer, const VkCuLaunchInfoNVX* pLaunchInfo);
typedef uint32_t(VKAPI_PTR* PFN_vkGetImageViewHandleNVX)(VkDevice device, const VkImageViewHandleInfoNVX* pInfo);
typedef uint64_t(VKAPI_PTR* PFN_vkGetImageViewHandle64NVX)(VkDevice device, const VkImageViewHandleInfoNVX* pInfo);
typedef VkResult(VKAPI_PTR* PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)(VkPhysicalDevice physicalDevice,
	uint32_t* pPropertyCount, VkCooperativeMatrixPropertiesKHR* pProperties);
typedef void(VKAPI_PTR* PFN_vkCmdPipelineBarrier2)(VkCommandBuffer commandBuffer, const VkDependencyInfo* pDependencyInfo);
typedef struct VkQueue_T* VkQueue;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;
typedef struct VkSwapchainKHR_T* VkSwapchainKHR;
typedef struct VkSurfaceCapabilitiesKHR
{
	uint32_t minImageCount;
	uint32_t maxImageCount;
} VkSurfaceCapabilitiesKHR;
typedef VkResult(VKAPI_PTR* PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)(VkPhysicalDevice physicalDevice,
	VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR* pSurfaceCapabilities);
typedef struct VkExtent2D { uint32_t width; uint32_t height; } VkExtent2D;
typedef struct VkSwapchainCreateInfoKHR
{
	VkStructureType sType;
	const void* pNext;
	VkSurfaceKHR surface;
	uint32_t minImageCount;
	VkExtent2D imageExtent;
	VkPresentModeKHR presentMode;
} VkSwapchainCreateInfoKHR;
typedef struct VkPresentInfoKHR
{
	VkStructureType sType;
	const void* pNext;
	uint32_t waitSemaphoreCount;
	const void* pWaitSemaphores;
	uint32_t swapchainCount;
	const VkSwapchainKHR* pSwapchains;
	const uint32_t* pImageIndices;
	VkResult* pResults;
} VkPresentInfoKHR;
typedef VkResult(VKAPI_PTR* PFN_vkCreateSwapchainKHR)(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain);
typedef void(VKAPI_PTR* PFN_vkDestroySwapchainKHR)(VkDevice device, VkSwapchainKHR swapchain,
	const VkAllocationCallbacks* pAllocator);
typedef VkResult(VKAPI_PTR* PFN_vkQueuePresentKHR)(VkQueue queue, const VkPresentInfoKHR* pPresentInfo);
