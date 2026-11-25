// SPDX-License-Identifier: MIT
//
// Standalone Vulkan cooperative-matrix diagnostic tool (no game / D3D12 / OptiScaler required).
// Queries all SINT8->SINT32 subgroup shapes, runs the 16x16x16 Wave32 matC register-layout probe,
// and dumps VK_KHR_pipeline_executable_properties ISA stats (VGPR/SGPR/LDS/Scratch) for all 16
// XeFG cooperative-matrix kernels (standard sAcc vs radv_reg, local_x=128 vs 256).
//
// Build (MSVC x64):
//   cl /nologo /O2 /EHsc /std:c++20 /DVK_NO_PROTOTYPES /I<vulkan_include> tools/CoopMatProbe.cpp /Fe:xefg_coopmat_probe.exe

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
struct KernelSpec
{
	const char* name;
	const char* family;
	uint32_t tileH;
	bool hasSkip;
	uint32_t pushBytes;
};

constexpr KernelSpec kKernels[] = {
	{"L_aa016078",     "Pool",    16, false, 16},
	{"L_198d6bf8",     "Pool",     8, false, 16},
	{"L_dcd8becf",     "Pool",    16, false, 16},
	{"L_efbe3788",     "Pool",     8, false, 16},
	{"L_50f0c205",     "Pool",     4, false, 16},
	{"L_a3c3caa1",     "Pool",     8, false, 16},
	{"L_9481d452",     "Pool",    16, false, 16},
	{"L_06e97933",     "Pool",    16, false, 16},
	{"L_3ed03daf",     "Decoder", 16, true,  16},
	{"L_748b0baf",     "Decoder",  8, true,  16},
	{"L_cb391582",     "Decoder", 16, true,  16},
	{"L_fc1b8a04",     "Flow",     8, false, 48},
	{"L_3459485",      "Flow",     8, false, 48},
	{"L_3452733",      "Flow",     8, false, 48},
	{"L_flow16_acc17", "Flow",     8, false, 48},
	{"L_3453044",      "Flow",     8, false, 48},
};

std::vector<uint32_t> LoadSpv(const std::filesystem::path& path)
{
	std::error_code ec;
	const auto bytes = std::filesystem::file_size(path, ec);
	if (ec || bytes < 20 || bytes > 4 * 1024 * 1024 || (bytes % 4) != 0)
		return {};
	std::vector<uint32_t> words(static_cast<size_t>(bytes / 4));
	std::ifstream in(path, std::ios::binary);
	if (!in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(bytes)))
		return {};
	if (words[0] != 0x07230203u)
		return {};
	return words;
}

bool HasExt(const std::vector<VkExtensionProperties>& exts, const char* name)
{
	for (const auto& e : exts)
		if (std::strcmp(e.extensionName, name) == 0)
			return true;
	return false;
}

uint32_t FindMemType(const VkPhysicalDeviceMemoryProperties& props, uint32_t typeBits, VkMemoryPropertyFlags want)
{
	for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
		if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & want) == want)
			return i;
	return UINT32_MAX;
}

std::filesystem::path FindCoopMatDir(const wchar_t* argPath)
{
	std::error_code ec;
	if (argPath && *argPath)
	{
		std::filesystem::path p(argPath);
		if (std::filesystem::is_directory(p, ec))
			return p;
	}
	wchar_t exeBuf[MAX_PATH]{};
	const DWORD len = GetModuleFileNameW(nullptr, exeBuf, MAX_PATH);
	const std::filesystem::path exeDir = (len > 0 && len < MAX_PATH)
		? std::filesystem::path(exeBuf).parent_path()
		: std::filesystem::current_path(ec);

	const std::filesystem::path candidates[] = {
		exeDir / L"xefg_mfg_programs" / L"vulkan" / L"coopmat",
		exeDir / L"coopmat",
		exeDir / L"third_party" / L"xefg_mfg_programs" / L"vulkan" / L"coopmat",
		std::filesystem::current_path(ec) / L"xefg_mfg_programs" / L"vulkan" / L"coopmat",
		std::filesystem::current_path(ec) / L"coopmat",
	};
	for (const auto& c : candidates)
		if (std::filesystem::is_directory(c, ec))
			return c;
	return {};
}
} // namespace

int wmain(int argc, wchar_t** argv)
{
	std::printf("=== XeFG Vulkan Cooperative-Matrix Offline Probe ===\n");
	const std::filesystem::path coopDir = FindCoopMatDir(argc > 1 ? argv[1] : nullptr);
	if (coopDir.empty())
		std::printf("[warn] coopmat directory not found (pass path as arg1); will still query GPU shapes.\n");
	else
		std::printf("CoopMat directory: %s\n", coopDir.string().c_str());

	HMODULE vkDll = LoadLibraryW(L"vulkan-1.dll");
	if (!vkDll)
	{
		std::printf("[error] Failed to load vulkan-1.dll (err=%lu)\n", GetLastError());
		return 1;
	}
	const auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vkDll, "vkGetInstanceProcAddr"));
	if (!gipa)
	{
		std::printf("[error] vkGetInstanceProcAddr not found\n");
		return 1;
	}
	const auto createInstance = reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
	if (!createInstance)
	{
		std::printf("[error] vkCreateInstance not found\n");
		return 1;
	}

	VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	appInfo.pApplicationName = "XeFgCoopMatProbe";
	appInfo.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instInfo.pApplicationInfo = &appInfo;

	VkInstance instance = VK_NULL_HANDLE;
	if (createInstance(&instInfo, nullptr, &instance) != VK_SUCCESS || !instance)
	{
		std::printf("[error] vkCreateInstance (Vulkan 1.3) failed\n");
		return 1;
	}

	const auto destroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(gipa(instance, "vkDestroyInstance"));
	const auto enumPhys = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(instance, "vkEnumeratePhysicalDevices"));
	const auto getProps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gipa(instance, "vkGetPhysicalDeviceProperties2"));
	const auto getFeats2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(gipa(instance, "vkGetPhysicalDeviceFeatures2"));
	const auto enumDevExts = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(gipa(instance, "vkEnumerateDeviceExtensionProperties"));
	const auto getQueueFams = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
	const auto getMemProps = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(gipa(instance, "vkGetPhysicalDeviceMemoryProperties"));
	const auto getCoopProps = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
		gipa(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
	const auto createDevice = reinterpret_cast<PFN_vkCreateDevice>(gipa(instance, "vkCreateDevice"));
	const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));

	uint32_t physCount = 0;
	enumPhys(instance, &physCount, nullptr);
	std::vector<VkPhysicalDevice> physDevs(physCount);
	if (physCount)
		enumPhys(instance, &physCount, physDevs.data());
	std::printf("Physical devices found: %u\n", physCount);

	for (uint32_t d = 0; d < physCount; ++d)
	{
		VkPhysicalDevice phys = physDevs[d];
		VkPhysicalDeviceSubgroupSizeControlProperties subCtrl{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
		VkPhysicalDeviceDriverProperties drvProps{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &subCtrl};
		VkPhysicalDeviceSubgroupProperties subProps{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &drvProps};
		VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subProps};
		getProps2(phys, &props2);

		std::printf("\n--- GPU #%u: %s (vendor=0x%04x, device=0x%04x, api=%u.%u.%u) ---\n",
			d, props2.properties.deviceName, props2.properties.vendorID, props2.properties.deviceID,
			VK_API_VERSION_MAJOR(props2.properties.apiVersion),
			VK_API_VERSION_MINOR(props2.properties.apiVersion),
			VK_API_VERSION_PATCH(props2.properties.apiVersion));
		std::printf("Driver: id=%d, name=\"%s\", info=\"%s\"\n",
			static_cast<int>(drvProps.driverID), drvProps.driverName, drvProps.driverInfo);
		std::printf("Subgroup: default=%u, range=%u..%u, computeCtrl=%d\n",
			subProps.subgroupSize, subCtrl.minSubgroupSize, subCtrl.maxSubgroupSize,
			(subCtrl.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ? 1 : 0);

		uint32_t extCount = 0;
		enumDevExts(phys, nullptr, &extCount, nullptr);
		std::vector<VkExtensionProperties> exts(extCount);
		if (extCount)
			enumDevExts(phys, nullptr, &extCount, exts.data());

		const bool hasCoop = HasExt(exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
		const bool hasPush = HasExt(exts, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
		const bool hasPipeExec = HasExt(exts, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
		std::printf("Extensions: coop_matrix=%d, push_descriptor=%d, pipeline_executable_properties=%d\n",
			hasCoop ? 1 : 0, hasPush ? 1 : 0, hasPipeExec ? 1 : 0);
		if (!hasCoop || !hasPush || !getCoopProps)
			continue;

		uint32_t shapeCount = 0;
		getCoopProps(phys, &shapeCount, nullptr);
		std::vector<VkCooperativeMatrixPropertiesKHR> shapes(
			shapeCount, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
		if (shapeCount)
			getCoopProps(phys, &shapeCount, shapes.data());

		bool has16x16x16 = false;
		std::string s8Shapes;
		for (const auto& s : shapes)
		{
			if (s.scope == VK_SCOPE_SUBGROUP_KHR &&
				s.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
				s.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
				s.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
				s.ResultType == VK_COMPONENT_TYPE_SINT32_KHR)
			{
				if (!s8Shapes.empty()) s8Shapes += ", ";
				s8Shapes += std::to_string(s.MSize) + "x" + std::to_string(s.NSize) + "x" + std::to_string(s.KSize);
				if (s.MSize == 16u && s.NSize == 16u && s.KSize == 16u)
					has16x16x16 = true;
			}
		}
		std::printf("Subgroup SINT8->SINT32 shapes: %s\n", s8Shapes.empty() ? "(none)" : s8Shapes.c_str());
		if (!has16x16x16)
			continue;

		VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipeExecFeats{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
		VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeats{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
			hasPipeExec ? &pipeExecFeats : nullptr};
		VkPhysicalDeviceVulkan13Features vk13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &coopFeats};
		VkPhysicalDeviceVulkan12Features vk12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &vk13};
		VkPhysicalDeviceVulkan11Features vk11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &vk12};
		VkPhysicalDeviceFeatures2 feats2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vk11};
		getFeats2(phys, &feats2);
		if (!coopFeats.cooperativeMatrix)
			continue;

		uint32_t famCount = 0;
		getQueueFams(phys, &famCount, nullptr);
		std::vector<VkQueueFamilyProperties> fams(famCount);
		getQueueFams(phys, &famCount, fams.data());
		uint32_t qFam = UINT32_MAX;
		for (uint32_t i = 0; i < famCount; ++i)
			if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qFam = i; break; }
		if (qFam == UINT32_MAX)
			continue;

		const bool usePipeExec = hasPipeExec && (pipeExecFeats.pipelineExecutableInfo == VK_TRUE);
		VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR enPipeExec{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
		enPipeExec.pipelineExecutableInfo = VK_TRUE;
		VkPhysicalDeviceCooperativeMatrixFeaturesKHR enCoop{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
			usePipeExec ? &enPipeExec : nullptr};
		enCoop.cooperativeMatrix = VK_TRUE;
		VkPhysicalDeviceVulkan13Features en13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &enCoop};
		en13.synchronization2 = vk13.synchronization2;
		en13.maintenance4 = vk13.maintenance4;
		en13.shaderIntegerDotProduct = vk13.shaderIntegerDotProduct;
		en13.subgroupSizeControl = vk13.subgroupSizeControl;
		en13.computeFullSubgroups = vk13.computeFullSubgroups;
		VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &en13};
		en12.shaderInt8 = vk12.shaderInt8;
		en12.storageBuffer8BitAccess = vk12.storageBuffer8BitAccess;
		en12.uniformAndStorageBuffer8BitAccess = vk12.uniformAndStorageBuffer8BitAccess;
		en12.shaderFloat16 = vk12.shaderFloat16;
		en12.vulkanMemoryModel = vk12.vulkanMemoryModel;
		en12.vulkanMemoryModelDeviceScope = vk12.vulkanMemoryModelDeviceScope;
		en12.scalarBlockLayout = vk12.scalarBlockLayout;
		VkPhysicalDeviceVulkan11Features en11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &en12};
		en11.storageBuffer16BitAccess = vk11.storageBuffer16BitAccess;
		VkPhysicalDeviceFeatures2 enFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &en11};
		enFeats.features.shaderStorageImageExtendedFormats = feats2.features.shaderStorageImageExtendedFormats;
		enFeats.features.shaderInt16 = feats2.features.shaderInt16;

		std::vector<const char*> devExts = {
			VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
			VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
		};
		if (usePipeExec)
			devExts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);

		const float prio = 1.0f;
		VkDeviceQueueCreateInfo qInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		qInfo.queueFamilyIndex = qFam;
		qInfo.queueCount = 1;
		qInfo.pQueuePriorities = &prio;
		VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enFeats};
		devInfo.queueCreateInfoCount = 1;
		devInfo.pQueueCreateInfos = &qInfo;
		devInfo.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
		devInfo.ppEnabledExtensionNames = devExts.data();

		VkDevice dev = VK_NULL_HANDLE;
		if (createDevice(phys, &devInfo, nullptr, &dev) != VK_SUCCESS || !dev)
		{
			std::printf("[warn] vkCreateDevice failed on GPU #%u\n", d);
			continue;
		}

#define DEV_FN(var, name) const auto var = reinterpret_cast<PFN_##name>(gdpa(dev, #name))
		DEV_FN(destroyDevice, vkDestroyDevice);
		DEV_FN(getQueue, vkGetDeviceQueue);
		DEV_FN(createShaderModule, vkCreateShaderModule);
		DEV_FN(destroyShaderModule, vkDestroyShaderModule);
		DEV_FN(createDsl, vkCreateDescriptorSetLayout);
		DEV_FN(destroyDsl, vkDestroyDescriptorSetLayout);
		DEV_FN(createPipeLayout, vkCreatePipelineLayout);
		DEV_FN(destroyPipeLayout, vkDestroyPipelineLayout);
		DEV_FN(createComputePipelines, vkCreateComputePipelines);
		DEV_FN(destroyPipeline, vkDestroyPipeline);
		DEV_FN(createCmdPool, vkCreateCommandPool);
		DEV_FN(destroyCmdPool, vkDestroyCommandPool);
		DEV_FN(allocCmdBufs, vkAllocateCommandBuffers);
		DEV_FN(beginCmd, vkBeginCommandBuffer);
		DEV_FN(endCmd, vkEndCommandBuffer);
		DEV_FN(cmdBindPipeline, vkCmdBindPipeline);
		DEV_FN(cmdPushDesc, vkCmdPushDescriptorSetKHR);
		DEV_FN(cmdDispatch, vkCmdDispatch);
		DEV_FN(queueSubmit2, vkQueueSubmit2);
		DEV_FN(createFence, vkCreateFence);
		DEV_FN(destroyFence, vkDestroyFence);
		DEV_FN(waitForFences, vkWaitForFences);
		DEV_FN(createBuffer, vkCreateBuffer);
		DEV_FN(destroyBuffer, vkDestroyBuffer);
		DEV_FN(getBufMemReq, vkGetBufferMemoryRequirements);
		DEV_FN(allocMem, vkAllocateMemory);
		DEV_FN(freeMem, vkFreeMemory);
		DEV_FN(bindBufMem, vkBindBufferMemory);
		DEV_FN(mapMem, vkMapMemory);
		DEV_FN(unmapMem, vkUnmapMemory);
		DEV_FN(getExecProps, vkGetPipelineExecutablePropertiesKHR);
		DEV_FN(getExecStats, vkGetPipelineExecutableStatisticsKHR);
#undef DEV_FN

		VkPhysicalDeviceMemoryProperties memProps{};
		getMemProps(phys, &memProps);
		const bool reqSubgroup = (vk13.subgroupSizeControl == VK_TRUE) &&
			(subCtrl.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
			subCtrl.minSubgroupSize <= 32u && subCtrl.maxSubgroupSize >= 32u;

		// 1. Run matC register layout probe if matc_probe_16x16x16.spv is present
		if (!coopDir.empty())
		{
			const auto probeSpv = LoadSpv(coopDir / L"matc_probe_16x16x16.spv");
			if (probeSpv.empty())
			{
				std::printf("matC probe: skipped (matc_probe_16x16x16.spv not found in coopmat folder)\n");
			}
			else
			{
				VkQueue queue = VK_NULL_HANDLE;
				getQueue(dev, qFam, 0, &queue);
				VkShaderModule sm = VK_NULL_HANDLE;
				VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
				VkPipelineLayout pl = VK_NULL_HANDLE;
				VkPipeline pipe = VK_NULL_HANDLE;
				VkCommandPool pool = VK_NULL_HANDLE;
				VkCommandBuffer cmd = VK_NULL_HANDLE;
				VkBuffer buf = VK_NULL_HANDLE;
				VkDeviceMemory mem = VK_NULL_HANDLE;
				VkFence fence = VK_NULL_HANDLE;

				VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
				smi.codeSize = probeSpv.size() * sizeof(uint32_t);
				smi.pCode = probeSpv.data();
				VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
				VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
				dsli.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
				dsli.bindingCount = 1;
				dsli.pBindings = &b0;
				VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
				pli.setLayoutCount = 1;
				VkCommandPoolCreateInfo pooli{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
				pooli.queueFamilyIndex = qFam;
				constexpr VkDeviceSize kBufSize = 257u * sizeof(int32_t);
				VkBufferCreateInfo bufi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
				bufi.size = kBufSize;
				bufi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
				bufi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

				bool ok = (createShaderModule(dev, &smi, nullptr, &sm) == VK_SUCCESS) &&
					(createDsl(dev, &dsli, nullptr, &dsl) == VK_SUCCESS);
				pli.pSetLayouts = &dsl;
				ok = ok && (createPipeLayout(dev, &pli, nullptr, &pl) == VK_SUCCESS);

				VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subReq{
					VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, nullptr, 32u};
				VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
				cpi.stage = {
					VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					reqSubgroup ? &subReq : nullptr,
					reqSubgroup ? VkPipelineShaderStageCreateFlags(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : 0u,
					VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr
				};
				cpi.layout = pl;
				ok = ok && (createComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe) == VK_SUCCESS);
				ok = ok && (createCmdPool(dev, &pooli, nullptr, &pool) == VK_SUCCESS);
				VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
				cbai.commandPool = pool;
				cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				cbai.commandBufferCount = 1;
				ok = ok && (allocCmdBufs(dev, &cbai, &cmd) == VK_SUCCESS);
				ok = ok && (createBuffer(dev, &bufi, nullptr, &buf) == VK_SUCCESS);
				if (ok)
				{
					VkMemoryRequirements req{};
					getBufMemReq(dev, buf, &req);
					const uint32_t mt = FindMemType(memProps, req.memoryTypeBits,
						VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
					VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, mt};
					ok = (mt != UINT32_MAX) && (allocMem(dev, &mai, nullptr, &mem) == VK_SUCCESS) &&
						(bindBufMem(dev, buf, mem, 0) == VK_SUCCESS);
				}
				VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
				ok = ok && (createFence(dev, &fi, nullptr, &fence) == VK_SUCCESS);
				if (ok)
				{
					void* ptr = nullptr;
					if (mapMem(dev, mem, 0, kBufSize, 0, &ptr) == VK_SUCCESS && ptr)
					{
						std::memset(ptr, 0xFF, static_cast<size_t>(kBufSize));
						unmapMem(dev, mem);
					}
					VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
					bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
					beginCmd(cmd, &bi);
					cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
					VkDescriptorBufferInfo dbi{buf, 0, kBufSize};
					VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
					w.dstBinding = 0; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &dbi;
					cmdPushDesc(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &w);
					cmdDispatch(cmd, 1, 1, 1);
					endCmd(cmd);

					VkCommandBufferSubmitInfo cbs{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
					cbs.commandBuffer = cmd;
					VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
					sub.commandBufferInfoCount = 1;
					sub.pCommandBufferInfos = &cbs;
					if (queueSubmit2(queue, 1, &sub, fence) == VK_SUCCESS &&
						waitForFences(dev, 1, &fence, VK_TRUE, 2'000'000'000ull) == VK_SUCCESS &&
						mapMem(dev, mem, 0, kBufSize, 0, &ptr) == VK_SUCCESS && ptr)
					{
						int32_t vals[257]{};
						std::memcpy(vals, ptr, sizeof(vals));
						unmapMem(dev, mem);
						uint32_t matched = 0;
						int32_t fl = -1, fiIdx = -1, fgot = 0, fexp = 0;
						for (uint32_t lane = 0; lane < 32u; ++lane)
						{
							const uint32_t g = lane >> 4u, p = lane & 15u;
							for (uint32_t i = 0; i < 8u; ++i)
							{
								const int32_t exp = static_cast<int32_t>((g + i * 2u) * 16u + p);
								if (vals[lane * 8u + i] == exp) ++matched;
								else if (fl < 0) { fl = int32_t(lane); fiIdx = int32_t(i); fgot = vals[lane * 8u + i]; fexp = exp; }
							}
						}
						if (vals[256] == 8 && matched == 256u)
							std::printf("matC register layout probe: radv_match (256/256 cells exact, len=8)\n");
						else
							std::printf("matC register layout probe: mismatch (matched %u/256, len=%d, first diff lane=%d i=%d got=%d exp=%d, lane0=[%d,%d,%d,%d], lane16=[%d,%d,%d,%d])\n",
								matched, vals[256], fl, fiIdx, fgot, fexp,
								vals[0], vals[1], vals[2], vals[3],
								vals[16 * 8 + 0], vals[16 * 8 + 1], vals[16 * 8 + 2], vals[16 * 8 + 3]);
					}
					else
					{
						std::printf("matC register layout probe: execution failed\n");
					}
				}
				if (fence) destroyFence(dev, fence, nullptr);
				if (buf) destroyBuffer(dev, buf, nullptr);
				if (mem) freeMem(dev, mem, nullptr);
				if (pool) destroyCmdPool(dev, pool, nullptr);
				if (pipe) destroyPipeline(dev, pipe, nullptr);
				if (pl) destroyPipeLayout(dev, pl, nullptr);
				if (dsl) destroyDsl(dev, dsl, nullptr);
				if (sm) destroyShaderModule(dev, sm, nullptr);
			}
		}

		// 2. Compile all 16 kernels (std sAcc + radv_reg, local_x=128 & 256) and dump ISA stats
		if (!coopDir.empty() && usePipeExec && getExecProps && getExecStats)
		{
			std::printf("\nISA statistics (VK_KHR_pipeline_executable_properties):\n");
			auto dumpKernel = [&](const KernelSpec& ks, const std::filesystem::path& spvPath, const char* variant, uint32_t localX) {
				const auto words = LoadSpv(spvPath);
				if (words.empty())
					return;
				std::vector<VkDescriptorSetLayoutBinding> bindings = {
					{0,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
					{1,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
					{2,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
					{3,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
					{64, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
				};
				if (ks.hasSkip)
					bindings.push_back({4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});

				VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
				dsli.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
				dsli.bindingCount = static_cast<uint32_t>(bindings.size());
				dsli.pBindings = bindings.data();
				VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
				if (createDsl(dev, &dsli, nullptr, &dsl) != VK_SUCCESS)
					return;

				VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, ks.pushBytes};
				VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
				pli.setLayoutCount = 1;
				pli.pSetLayouts = &dsl;
				pli.pushConstantRangeCount = 1;
				pli.pPushConstantRanges = &pcr;
				VkPipelineLayout pl = VK_NULL_HANDLE;
				if (createPipeLayout(dev, &pli, nullptr, &pl) != VK_SUCCESS) { destroyDsl(dev, dsl, nullptr); return; }

				VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
				smi.codeSize = words.size() * sizeof(uint32_t);
				smi.pCode = words.data();
				VkShaderModule sm = VK_NULL_HANDLE;
				if (createShaderModule(dev, &smi, nullptr, &sm) != VK_SUCCESS) { destroyPipeLayout(dev, pl, nullptr); destroyDsl(dev, dsl, nullptr); return; }

				const uint32_t specData[5] = {16u, 16u, 16u, 32u, localX};
				const VkSpecializationMapEntry entries[5] = {
					{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}
				};
				VkSpecializationInfo spec{5, entries, sizeof(specData), specData};
				VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subReq{
					VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, nullptr, 32u};
				VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
				cpi.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
				cpi.stage = {
					VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					reqSubgroup ? &subReq : nullptr,
					reqSubgroup ? VkPipelineShaderStageCreateFlags(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : 0u,
					VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", &spec
				};
				cpi.layout = pl;
				VkPipeline pipe = VK_NULL_HANDLE;
				if (createComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe) == VK_SUCCESS && pipe)
				{
					VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, nullptr, pipe};
					uint32_t ec = 0;
					getExecProps(dev, &pi, &ec, nullptr);
					std::vector<VkPipelineExecutablePropertiesKHR> ep(ec, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
					if (ec && getExecProps(dev, &pi, &ec, ep.data()) == VK_SUCCESS)
					{
						VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, nullptr, pipe, 0};
						uint32_t sc = 0;
						getExecStats(dev, &ei, &sc, nullptr);
						std::vector<VkPipelineExecutableStatisticKHR> st(sc, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
						if (sc && getExecStats(dev, &ei, &sc, st.data()) == VK_SUCCESS)
						{
							std::printf("  %-14s %-8s local=%3u wave=%2u | ", ks.name, variant, localX, ep[0].subgroupSize);
							for (uint32_t i = 0; i < sc; ++i)
							{
								if (i) std::printf("; ");
								std::printf("%s=", st[i].name);
								switch (st[i].format)
								{
								case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:  std::printf("%u", st[i].value.b32 ? 1u : 0u); break;
								case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:   std::printf("%lld", static_cast<long long>(st[i].value.i64)); break;
								case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:  std::printf("%llu", static_cast<unsigned long long>(st[i].value.u64)); break;
								case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: std::printf("%.2f", st[i].value.f64); break;
								default: std::printf("?"); break;
								}
							}
							std::printf("\n");
						}
					}
					destroyPipeline(dev, pipe, nullptr);
				}
				destroyShaderModule(dev, sm, nullptr);
				destroyPipeLayout(dev, pl, nullptr);
				destroyDsl(dev, dsl, nullptr);
			};

			for (const auto& ks : kKernels)
			{
				const std::string fname = std::string(ks.name) + ".spv";
				dumpKernel(ks, coopDir / fname, "sAcc", 128u);
				dumpKernel(ks, coopDir / L"radv_reg" / fname, "radv_reg", 128u);
				if (ks.tileH >= 16u && std::strcmp(ks.family, "Flow") != 0)
				{
					dumpKernel(ks, coopDir / fname, "sAcc", 256u);
					dumpKernel(ks, coopDir / L"radv_reg" / fname, "radv_reg", 256u);
				}
			}
		}

		destroyDevice(dev, nullptr);
	}

	destroyInstance(instance, nullptr);
	FreeLibrary(vkDll);
	std::printf("\nDone.\n");
	return 0;
}
