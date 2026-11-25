// See VulkanCoopMat.h.
#include "VulkanCoopMat.h"

#include "../Log.h"
#include "../MfgKernels.h"
#include "../MfgPort/Cuda/cuda_layers.h"
#include "../MfgPort/Cuda/cuda_route.h"
#include "../VulkanPrograms.h"
#include "VulkanPolicy.h"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

namespace
{
	constexpr uint32_t AllLayers = (1u << std::size(xefg_cuda::kLayers)) - 1;

	std::wstring Environment(const wchar_t* name)
	{
		wchar_t value[64]{};
		const DWORD length = GetEnvironmentVariableW(name, value, static_cast<DWORD>(std::size(value)));
		return length && length < std::size(value) ? std::wstring(value, length) : std::wstring();
	}

	bool Refused(std::string& reason, const char* code)
	{
		reason = code;
		xefg_cuda::ReportCoopMatrix(xefg_cuda::State::Unavailable, 0, 0, 0, 0, code);
		return false;
	}

	struct PendingProbe
	{
		VkDevice dev = VK_NULL_HANDLE;
		VkShaderModule shader = VK_NULL_HANDLE;
		VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
		VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;
		VkBuffer outBuf = VK_NULL_HANDLE;
		VkDeviceMemory outMem = VK_NULL_HANDLE;
		VkCommandPool cmdPool = VK_NULL_HANDLE;
		VkFence fence = VK_NULL_HANDLE;
		PFN_vkGetFenceStatus getFenceStatus = nullptr;
		PFN_vkWaitForFences waitForFences = nullptr;
		PFN_vkDestroyFence dFence = nullptr;
		PFN_vkDestroyCommandPool dPool = nullptr;
		PFN_vkDestroyPipeline dPipe = nullptr;
		PFN_vkDestroyPipelineLayout dPipeLayout = nullptr;
		PFN_vkDestroyDescriptorSetLayout dDsl = nullptr;
		PFN_vkDestroyShaderModule dShader = nullptr;
		PFN_vkDestroyBuffer dBuf = nullptr;
		PFN_vkFreeMemory dMem = nullptr;

		void DestroyNow() noexcept
		{
			if (fence && dFence) dFence(dev, fence, nullptr);
			if (cmdPool && dPool) dPool(dev, cmdPool, nullptr);
			if (pipeline && dPipe) dPipe(dev, pipeline, nullptr);
			if (pipeLayout && dPipeLayout) dPipeLayout(dev, pipeLayout, nullptr);
			if (dsl && dDsl) dDsl(dev, dsl, nullptr);
			if (shader && dShader) dShader(dev, shader, nullptr);
			if (outBuf && dBuf) dBuf(dev, outBuf, nullptr);
			if (outMem && dMem) dMem(dev, outMem, nullptr);
			fence = VK_NULL_HANDLE;
			cmdPool = VK_NULL_HANDLE;
			pipeline = VK_NULL_HANDLE;
			pipeLayout = VK_NULL_HANDLE;
			dsl = VK_NULL_HANDLE;
			shader = VK_NULL_HANDLE;
			outBuf = VK_NULL_HANDLE;
			outMem = VK_NULL_HANDLE;
		}
	};

	std::mutex gPendingProbesMutex;
	std::vector<PendingProbe> gPendingProbes;
	std::atomic<bool> gHasPendingProbes{ false };
} // namespace

bool VulkanCoopMat::Wanted()
{
	return !VulkanPolicy::IsOffWord(Environment(L"XEFG_COOP_MATRIX"));
}

bool VulkanCoopMat::AllowIntelAdapter()
{
	return Wanted() && !VulkanPolicy::IsOffWord(Environment(L"XEFG_INTEL_VULKAN"));
}

bool VulkanCoopMat::Install(ID3D12Device* device, std::string& reason, xvk::MatCLayout matCLayout)
{
	reason.clear();

	try
	{
		if (!Wanted())
		{
			reason = "environment_off";
			xefg_cuda::ReportCoopMatrix(xefg_cuda::State::Disabled, 0, 0, 0, 0, reason.c_str());
			Log::Info("Vulkan cooperative matrix: off ({})", VulkanPolicy::DescribeCoopMatrix(reason));
			return false;
		}

		if (!device)
			return Refused(reason, "device_inputs");

		const auto dir = VulkanPrograms::DirectoryFor(MfgKernels::ProgramsDirectory()) / L"coopmat";
		xefg_cuda::SetCoopMatrixLayerMask(VulkanPolicy::ParseLayerMask(Environment(L"XEFG_COOP_MATRIX_MASK"), AllLayers));

		xvk::CoopMatrixOptions options;
		options.fullBarriers = VulkanPolicy::IsOnWord(Environment(L"XEFG_COOP_MATRIX_VULKAN_BARRIERS"));
		options.matCLayout = matCLayout;

		if (!xvk::InstallCoopMatrix(device, dir.wstring().c_str(), options, reason))
		{
			if (reason.empty())
				reason = "exception";

			xefg_cuda::ReportCoopMatrix(xefg_cuda::State::Unavailable, 0, 0, 0, 0, reason.c_str());
			Log::Info("Vulkan cooperative matrix: unavailable, {}", VulkanPolicy::DescribeCoopMatrix(reason));
			return false;
		}

		const auto status = xefg_cuda::CoopMatrixSnapshot();
		const uint32_t tileM = static_cast<uint32_t>(status.major);
		const uint32_t tileN = static_cast<uint32_t>((status.minor >> 8) & 0xFF);
		const uint32_t tileK = static_cast<uint32_t>(status.minor & 0xFF);
		Log::Info("Vulkan cooperative matrix: ready ({}x{}x{} s8->s32), {} layer modules in {}{}",
			tileM, tileN, tileK, status.layers, VulkanPrograms::PathText(dir),
			options.fullBarriers ? ", full barriers" : "");

		return true;
	}
	catch (...)
	{
		return Refused(reason, "exception");
	}
}

xvk::MatCLayout VulkanCoopMat::ProbeMatCLayout(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
	uint32_t queueFamily, VkQueue queue, PFN_vkGetInstanceProcAddr gipa,
	PFN_vkGetDeviceProcAddr gdpa, const char* logPrefix) noexcept
{
	const char* prefix = (logPrefix && *logPrefix) ? logPrefix : "Vulkan coopmat";
	try
	{
		if (!instance || !physical || !device || !queue || !gipa || !gdpa)
			return xvk::MatCLayout::Unknown;

		const auto probePath = VulkanPrograms::DirectoryFor(MfgKernels::ProgramsDirectory()) / L"coopmat" / L"matc_probe_16x16x16.spv";
		std::error_code ec;
		const auto bytes = std::filesystem::file_size(probePath, ec);
		if (ec || bytes < 20 || bytes > 262144 || (bytes % 4) != 0)
			return xvk::MatCLayout::Unknown;

		std::vector<uint32_t> spv(static_cast<size_t>(bytes / 4));
		{
			std::ifstream in(probePath, std::ios::binary);
			if (!in.read(reinterpret_cast<char*>(spv.data()), static_cast<std::streamsize>(bytes)))
				return xvk::MatCLayout::Unknown;
		}
		if (spv[0] != 0x07230203u)
			return xvk::MatCLayout::Unknown;

		const auto getPhysMemProps = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
			gipa(instance, "vkGetPhysicalDeviceMemoryProperties"));
		auto load = [&](const char* name) { return gdpa(device, name); };
		const auto createShaderModule = reinterpret_cast<PFN_vkCreateShaderModule>(load("vkCreateShaderModule"));
		const auto destroyShaderModule = reinterpret_cast<PFN_vkDestroyShaderModule>(load("vkDestroyShaderModule"));
		const auto createDsl = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(load("vkCreateDescriptorSetLayout"));
		const auto destroyDsl = reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(load("vkDestroyDescriptorSetLayout"));
		const auto createPipeLayout = reinterpret_cast<PFN_vkCreatePipelineLayout>(load("vkCreatePipelineLayout"));
		const auto destroyPipeLayout = reinterpret_cast<PFN_vkDestroyPipelineLayout>(load("vkDestroyPipelineLayout"));
		const auto createComputePipelines = reinterpret_cast<PFN_vkCreateComputePipelines>(load("vkCreateComputePipelines"));
		const auto destroyPipeline = reinterpret_cast<PFN_vkDestroyPipeline>(load("vkDestroyPipeline"));
		const auto createBuffer = reinterpret_cast<PFN_vkCreateBuffer>(load("vkCreateBuffer"));
		const auto destroyBuffer = reinterpret_cast<PFN_vkDestroyBuffer>(load("vkDestroyBuffer"));
		const auto getBufMemReq = reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(load("vkGetBufferMemoryRequirements"));
		const auto allocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(load("vkAllocateMemory"));
		const auto freeMemory = reinterpret_cast<PFN_vkFreeMemory>(load("vkFreeMemory"));
		const auto bindBufMem = reinterpret_cast<PFN_vkBindBufferMemory>(load("vkBindBufferMemory"));
		const auto mapMem = reinterpret_cast<PFN_vkMapMemory>(load("vkMapMemory"));
		const auto unmapMem = reinterpret_cast<PFN_vkUnmapMemory>(load("vkUnmapMemory"));
		const auto createCmdPool = reinterpret_cast<PFN_vkCreateCommandPool>(load("vkCreateCommandPool"));
		const auto destroyCmdPool = reinterpret_cast<PFN_vkDestroyCommandPool>(load("vkDestroyCommandPool"));
		const auto allocCmdBufs = reinterpret_cast<PFN_vkAllocateCommandBuffers>(load("vkAllocateCommandBuffers"));
		const auto beginCmdBuf = reinterpret_cast<PFN_vkBeginCommandBuffer>(load("vkBeginCommandBuffer"));
		const auto endCmdBuf = reinterpret_cast<PFN_vkEndCommandBuffer>(load("vkEndCommandBuffer"));
		const auto cmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(load("vkCmdBindPipeline"));
		const auto cmdPushDesc = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(load("vkCmdPushDescriptorSetKHR"));
		const auto cmdBarrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(load("vkCmdPipelineBarrier"));
		const auto cmdDispatch = reinterpret_cast<PFN_vkCmdDispatch>(load("vkCmdDispatch"));
		const auto createFence = reinterpret_cast<PFN_vkCreateFence>(load("vkCreateFence"));
		const auto destroyFence = reinterpret_cast<PFN_vkDestroyFence>(load("vkDestroyFence"));
		const auto waitForFences = reinterpret_cast<PFN_vkWaitForFences>(load("vkWaitForFences"));
		const auto getFenceStatus = reinterpret_cast<PFN_vkGetFenceStatus>(load("vkGetFenceStatus"));
		const auto queueSubmit2 = reinterpret_cast<PFN_vkQueueSubmit2>(load("vkQueueSubmit2"));

		if (!getPhysMemProps || !createShaderModule || !destroyShaderModule || !createDsl || !destroyDsl ||
			!createPipeLayout || !destroyPipeLayout || !createComputePipelines || !destroyPipeline ||
			!createBuffer || !destroyBuffer || !getBufMemReq || !allocateMemory || !freeMemory ||
			!bindBufMem || !mapMem || !unmapMem || !createCmdPool || !destroyCmdPool || !allocCmdBufs ||
			!beginCmdBuf || !endCmdBuf || !cmdBindPipeline || !cmdPushDesc || !cmdBarrier || !cmdDispatch ||
			!createFence || !destroyFence || !waitForFences || !getFenceStatus || !queueSubmit2)
			return xvk::MatCLayout::Unknown;

		ReapPendingProbes(device, false);

		VkShaderModule shader = VK_NULL_HANDLE;
		VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
		VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;
		VkBuffer outBuf = VK_NULL_HANDLE;
		VkDeviceMemory outMem = VK_NULL_HANDLE;
		VkCommandPool cmdPool = VK_NULL_HANDLE;
		VkFence fence = VK_NULL_HANDLE;

		struct Cleanup
		{
			VkDevice dev;
			VkShaderModule& shader;
			VkDescriptorSetLayout& dsl;
			VkPipelineLayout& pipeLayout;
			VkPipeline& pipeline;
			VkBuffer& outBuf;
			VkDeviceMemory& outMem;
			VkCommandPool& cmdPool;
			VkFence& fence;
			PFN_vkDestroyFence dFence;
			PFN_vkDestroyCommandPool dPool;
			PFN_vkDestroyPipeline dPipe;
			PFN_vkDestroyPipelineLayout dPipeLayout;
			PFN_vkDestroyDescriptorSetLayout dDsl;
			PFN_vkDestroyShaderModule dShader;
			PFN_vkDestroyBuffer dBuf;
			PFN_vkFreeMemory dMem;
			bool deferred = false;
			~Cleanup()
			{
				if (deferred)
					return;
				if (fence) dFence(dev, fence, nullptr);
				if (cmdPool) dPool(dev, cmdPool, nullptr);
				if (pipeline) dPipe(dev, pipeline, nullptr);
				if (pipeLayout) dPipeLayout(dev, pipeLayout, nullptr);
				if (dsl) dDsl(dev, dsl, nullptr);
				if (shader) dShader(dev, shader, nullptr);
				if (outBuf) dBuf(dev, outBuf, nullptr);
				if (outMem) dMem(dev, outMem, nullptr);
			}
		} cleanup{ device, shader, dsl, pipeLayout, pipeline, outBuf, outMem, cmdPool, fence,
			destroyFence, destroyCmdPool, destroyPipeline, destroyPipeLayout, destroyDsl,
			destroyShaderModule, destroyBuffer, freeMemory, false };

		VkShaderModuleCreateInfo smInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
		smInfo.codeSize = spv.size() * sizeof(uint32_t);
		smInfo.pCode = spv.data();
		if (createShaderModule(device, &smInfo, nullptr, &shader) != VK_SUCCESS || !shader)
			return xvk::MatCLayout::Unknown;

		VkDescriptorSetLayoutBinding binding{};
		binding.binding = 0;
		binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		binding.descriptorCount = 1;
		binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo dslInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		dslInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
		dslInfo.bindingCount = 1;
		dslInfo.pBindings = &binding;
		if (createDsl(device, &dslInfo, nullptr, &dsl) != VK_SUCCESS || !dsl)
			return xvk::MatCLayout::Unknown;

		VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		plInfo.setLayoutCount = 1;
		plInfo.pSetLayouts = &dsl;
		if (createPipeLayout(device, &plInfo, nullptr, &pipeLayout) != VK_SUCCESS || !pipeLayout)
			return xvk::MatCLayout::Unknown;

		VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSub{
			VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, nullptr, 32u };
		VkComputePipelineCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		cpInfo.stage = {
			VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			&reqSub,
			VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
			VK_SHADER_STAGE_COMPUTE_BIT,
			shader,
			"main",
			nullptr
		};
		cpInfo.layout = pipeLayout;
		if (createComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pipeline) != VK_SUCCESS || !pipeline)
		{
			return xvk::MatCLayout::Unknown;
		}

		constexpr VkDeviceSize kBufSize = 257u * sizeof(int32_t);
		VkBufferCreateInfo bufInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bufInfo.size = kBufSize;
		bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (createBuffer(device, &bufInfo, nullptr, &outBuf) != VK_SUCCESS || !outBuf)
			return xvk::MatCLayout::Unknown;

		VkPhysicalDeviceMemoryProperties memProps{};
		getPhysMemProps(physical, &memProps);
		VkMemoryRequirements memReq{};
		getBufMemReq(device, outBuf, &memReq);
		uint32_t memType = UINT32_MAX;
		const VkMemoryPropertyFlags wantFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
		{
			if ((memReq.memoryTypeBits & (1u << i)) != 0 &&
				(memProps.memoryTypes[i].propertyFlags & wantFlags) == wantFlags)
			{
				memType = i;
				break;
			}
		}
		if (memType == UINT32_MAX)
			return xvk::MatCLayout::Unknown;

		VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
		allocInfo.allocationSize = memReq.size;
		allocInfo.memoryTypeIndex = memType;
		if (allocateMemory(device, &allocInfo, nullptr, &outMem) != VK_SUCCESS || !outMem)
			return xvk::MatCLayout::Unknown;
		if (bindBufMem(device, outBuf, outMem, 0) != VK_SUCCESS)
			return xvk::MatCLayout::Unknown;

		void* mapped = nullptr;
		if (mapMem(device, outMem, 0, kBufSize, 0, &mapped) != VK_SUCCESS || !mapped)
			return xvk::MatCLayout::Unknown;
		std::memset(mapped, 0xFF, static_cast<size_t>(kBufSize));
		unmapMem(device, outMem);

		VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
		poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		poolInfo.queueFamilyIndex = queueFamily;
		if (createCmdPool(device, &poolInfo, nullptr, &cmdPool) != VK_SUCCESS || !cmdPool)
			return xvk::MatCLayout::Unknown;

		VkCommandBuffer cmd = VK_NULL_HANDLE;
		VkCommandBufferAllocateInfo cbInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
		cbInfo.commandPool = cmdPool;
		cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbInfo.commandBufferCount = 1;
		if (allocCmdBufs(device, &cbInfo, &cmd) != VK_SUCCESS || !cmd)
			return xvk::MatCLayout::Unknown;

		VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		if (createFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS || !fence)
			return xvk::MatCLayout::Unknown;

		VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		if (beginCmdBuf(cmd, &beginInfo) != VK_SUCCESS)
			return xvk::MatCLayout::Unknown;

		cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		VkDescriptorBufferInfo descBuf{ outBuf, 0, kBufSize };
		VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
		write.dstBinding = 0;
		write.descriptorCount = 1;
		write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write.pBufferInfo = &descBuf;
		cmdPushDesc(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &write);
		cmdDispatch(cmd, 1, 1, 1);

		// Make the storage-buffer writes visible to the CPU before reading the mapped result.
		VkMemoryBarrier readback{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		readback.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		cmdBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
			1, &readback, 0, nullptr, 0, nullptr);

		if (endCmdBuf(cmd) != VK_SUCCESS)
			return xvk::MatCLayout::Unknown;

		VkCommandBufferSubmitInfo cmdSub{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
		cmdSub.commandBuffer = cmd;
		VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
		submitInfo.commandBufferInfoCount = 1;
		submitInfo.pCommandBufferInfos = &cmdSub;
		if (queueSubmit2(queue, 1, &submitInfo, fence) != VK_SUCCESS)
			return xvk::MatCLayout::Unknown;
		const VkResult waitRes = waitForFences(device, 1, &fence, VK_TRUE, 1'000'000'000ull);
		if (waitRes != VK_SUCCESS)
		{
			if (waitRes != VK_ERROR_DEVICE_LOST)
			{
				// GPU execution may still be in flight after VK_TIMEOUT: park all probe handles until
				// vkGetFenceStatus confirms completion rather than destroying active GPU resources.
				PendingProbe pending{};
				pending.dev = device;
				pending.shader = shader;
				pending.dsl = dsl;
				pending.pipeLayout = pipeLayout;
				pending.pipeline = pipeline;
				pending.outBuf = outBuf;
				pending.outMem = outMem;
				pending.cmdPool = cmdPool;
				pending.fence = fence;
				pending.getFenceStatus = getFenceStatus;
				pending.waitForFences = waitForFences;
				pending.dFence = destroyFence;
				pending.dPool = destroyCmdPool;
				pending.dPipe = destroyPipeline;
				pending.dPipeLayout = destroyPipeLayout;
				pending.dDsl = destroyDsl;
				pending.dShader = destroyShaderModule;
				pending.dBuf = destroyBuffer;
				pending.dMem = freeMemory;
				cleanup.deferred = true;
				{
					std::lock_guard lock(gPendingProbesMutex);
					gPendingProbes.push_back(pending);
					gHasPendingProbes.store(true, std::memory_order_release);
				}
				Log::Warn("{}: Wave32 16x16x16 matC probe wait returned {} (still in flight); deferring probe cleanup until fence completes",
					prefix, static_cast<int>(waitRes));
			}
			return xvk::MatCLayout::Unknown;
		}

		mapped = nullptr;
		if (mapMem(device, outMem, 0, kBufSize, 0, &mapped) != VK_SUCCESS || !mapped)
			return xvk::MatCLayout::Unknown;
		int32_t vals[257]{};
		std::memcpy(vals, mapped, sizeof(vals));
		unmapMem(device, outMem);

		const int32_t matLen = vals[256];
		if (matLen > 0 && matLen != 8)
		{
			Log::Warn("{}: Wave32 16x16x16 matC probe reported len={} (emulated/non-Wave32 WMMA); skipping coopmat to avoid VGPR spill", prefix, matLen);
			return xvk::MatCLayout::Emulated;
		}

		uint32_t matchedGfx11 = 0, matchedGfx12 = 0;
		int32_t firstLane = -1, firstI = -1, firstGot = 0, firstExp = 0;
		for (uint32_t lane = 0; lane < 32u; ++lane)
		{
			const uint32_t g = lane >> 4u;
			const uint32_t p = lane & 15u;
			for (uint32_t i = 0; i < 8u; ++i)
			{
				const int32_t exp11 = static_cast<int32_t>((g + i * 2u) * 16u + p);
				const int32_t exp12 = static_cast<int32_t>((g * 8u + i) * 16u + p);
				const int32_t got = vals[lane * 8u + i];
				if (got == exp11)
					++matchedGfx11;
				if (got == exp12)
					++matchedGfx12;
				if (got != exp11 && got != exp12 && firstLane < 0)
				{
					firstLane = static_cast<int32_t>(lane);
					firstI = static_cast<int32_t>(i);
					firstGot = got;
					firstExp = exp11;
				}
			}
		}

		if (matLen == 8 && matchedGfx11 == 256u)
		{
			Log::Info("{}: Wave32 16x16x16 matC register layout probe: gfx11_reg (256/256 cells exact, len=8)", prefix);
			return xvk::MatCLayout::Gfx11Reg;
		}
		if (matLen == 8 && matchedGfx12 == 256u)
		{
			Log::Info("{}: Wave32 16x16x16 matC register layout probe: gfx12_reg (256/256 cells exact, len=8)", prefix);
			return xvk::MatCLayout::Gfx12Reg;
		}
		Log::Info("{}: Wave32 16x16x16 matC register layout probe: unknown (len={}, gfx11={}/256, gfx12={}/256, first mismatch lane={} i={} got={} exp11={})",
			prefix, matLen, matchedGfx11, matchedGfx12, firstLane, firstI, firstGot, firstExp);
		return xvk::MatCLayout::Unknown;
	}
	catch (...)
	{
		return xvk::MatCLayout::Unknown;
	}
}

bool VulkanCoopMat::ReapPendingProbes(VkDevice device, bool waitBounded, bool deviceLost) noexcept
{
	if (!gHasPendingProbes.load(std::memory_order_acquire))
		return true;

	std::lock_guard lock(gPendingProbesMutex);
	bool deviceClear = true;
	for (auto it = gPendingProbes.begin(); it != gPendingProbes.end(); )
	{
		if (device && it->dev != device)
		{
			++it;
			continue;
		}
		VkResult st = VK_ERROR_DEVICE_LOST;
		if (!deviceLost || it->dev != device)
			st = it->getFenceStatus ? it->getFenceStatus(it->dev, it->fence) : VK_NOT_READY;
		if (st == VK_NOT_READY && waitBounded && it->waitForFences)
		{
			const VkResult w = it->waitForFences(it->dev, 1, &it->fence, VK_TRUE, 500'000'000ull);
			if (w == VK_SUCCESS || w == VK_ERROR_DEVICE_LOST)
				st = w;
			else if (it->getFenceStatus)
				st = it->getFenceStatus(it->dev, it->fence);
		}
		if (st == VK_SUCCESS || st == VK_ERROR_DEVICE_LOST)
		{
			it->DestroyNow();
			it = gPendingProbes.erase(it);
		}
		else
		{
			if (!device || it->dev == device)
				deviceClear = false;
			++it;
		}
	}
	gHasPendingProbes.store(!gPendingProbes.empty(), std::memory_order_release);
	return deviceClear;
}
