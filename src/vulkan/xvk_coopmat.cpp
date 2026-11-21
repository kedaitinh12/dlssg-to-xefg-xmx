#include "xvk_coopmat.h"
#include "xvk_internal.h"
#include "../MfgPort/Cuda/cuda_layers.h"
#include "../MfgPort/Sha256.h"
#include <algorithm>
#include <cstring>
#include <fstream>

namespace xvk
{
namespace
{
constexpr size_t kMaxModuleBytes = 4 * 1024 * 1024;

bool IsSpirvWords(const std::vector<uint32_t>& words) noexcept
{
    return words.size() >= 5 && words[0] == 0x07230203u;
}

// Checks whether a SPIR-V module declares OpExtension "SPV_KHR_cooperative_matrix" or OpCapability CooperativeMatrixKHR (6022)
bool UsesCoopMatrix(const std::vector<uint32_t>& words) noexcept
{
    if (!IsSpirvWords(words))
        return false;

    for (size_t i = 5; i < words.size();)
    {
        const uint32_t word = words[i];
        const uint32_t count = word >> 16;
        const uint32_t op = word & 0xFFFFu;
        if (!count || i + count > words.size())
            break;

        if (op == 17 && count >= 2 && words[i + 1] == 6022u) // OpCapability CooperativeMatrixKHR
            return true;

        if (op == 10 && count >= 2) // OpExtension
        {
            const char* ext = reinterpret_cast<const char*>(&words[i + 1]);
            const size_t maxBytes = (count - 1) * sizeof(uint32_t);
            if (std::strncmp(ext, "SPV_KHR_cooperative_matrix", maxBytes) == 0)
                return true;
        }

        i += count;
    }

    return false;
}

std::vector<uint32_t> LoadSpirvFile(const std::filesystem::path& path) noexcept
{
    try
    {
        std::error_code ec;
        const auto bytes = std::filesystem::file_size(path, ec);
        if (ec || bytes < 20 || bytes > kMaxModuleBytes || (bytes % 4) != 0)
            return {};

        std::ifstream file(path, std::ios::binary);
        if (!file)
            return {};

        std::vector<uint32_t> words(static_cast<size_t>(bytes / 4));
        if (!file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(bytes)))
            return {};

        if (!UsesCoopMatrix(words))
            return {};

        return words;
    }
    catch (...)
    {
        return {};
    }
}
} // namespace

bool CoopMatrixRoute::SelectTile(const VkCooperativeMatrixPropertiesKHR* properties, uint32_t count,
                                 uint32_t& m, uint32_t& n, uint32_t& k) noexcept
{
    m = n = k = 0;
    if (!properties || !count)
        return false;

    // xefg_coopmat_layers.comp uses fixed 16x16x16 subgroup-scope SINT8 x SINT8 -> SINT32 fragments
    // (AMD RDNA 4 / RDNA 3 WMMA v_wmma_i32_16x16x16_iu8 and any GPU advertising 16x16x16 SINT8->SINT32).
    // Only admit devices that actually support 16x16x16 so unsupported GPUs cleanly stay on DP4a.
    for (uint32_t i = 0; i < count; ++i)
    {
        const auto& p = properties[i];
        if (p.scope == VK_SCOPE_SUBGROUP_KHR &&
            p.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
            p.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
            p.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
            p.ResultType == VK_COMPONENT_TYPE_SINT32_KHR &&
            p.MSize == 16u && p.NSize == 16u && p.KSize == 16u)
        {
            m = 16u;
            n = 16u;
            k = 16u;
            return true;
        }
    }

    return false;
}

CoopMatrixRoute::~CoopMatrixRoute()
{
    if (!device_ || !destroyPipeline_)
        return;

    for (auto& pipeline : pipelines_)
    {
        if (pipeline)
        {
            destroyPipeline_(device_, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
    }
}

bool CoopMatrixRoute::Install(Device& owner, const DeviceDesc& desc, const std::filesystem::path& moduleDir,
                              const CoopMatrixOptions& options, std::string& reason) noexcept
{
    try
    {
        if (device_)
        {
            reason = "already_attempted";
            return false;
        }
        if (!desc.coopMatrixExtensions)
        {
            reason = "no_extensions";
            return false;
        }
        if (!desc.device || !desc.physicalDevice || !desc.getInstanceProcAddr)
        {
            reason = "device_inputs";
            return false;
        }

        const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            desc.getInstanceProcAddr(desc.instance, "vkGetDeviceProcAddr"));
        const auto props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            desc.getInstanceProcAddr(desc.instance, "vkGetPhysicalDeviceProperties2"));
        const auto coopPropsFn = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            desc.getInstanceProcAddr(desc.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

        if (!gdpa || !props2 || !coopPropsFn)
        {
            reason = "instance_functions";
            return false;
        }

        uint32_t propCount = 0;
        if (coopPropsFn(desc.physicalDevice, &propCount, nullptr) != VK_SUCCESS || !propCount || propCount > 256)
        {
            reason = "no_s8_matrix_shape";
            return false;
        }

        std::vector<VkCooperativeMatrixPropertiesKHR> props(
            propCount, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        if (coopPropsFn(desc.physicalDevice, &propCount, props.data()) != VK_SUCCESS)
        {
            reason = "no_s8_matrix_shape";
            return false;
        }

        uint32_t m = 0, n = 0, k = 0;
        if (!SelectTile(props.data(), propCount, m, n, k))
        {
            reason = "no_s8_matrix_shape";
            return false;
        }

        VkPhysicalDeviceSubgroupSizeControlProperties subgroupCtrl{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroup{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &subgroupCtrl};
        VkPhysicalDeviceProperties2 devProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
        props2(desc.physicalDevice, &devProps);

        const bool supportsWave32 =
            (subgroupCtrl.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
            subgroupCtrl.minSubgroupSize <= 32u && subgroupCtrl.maxSubgroupSize >= 32u;
        if (!supportsWave32 && subgroup.subgroupSize != 32u)
        {
            reason = "no_s8_matrix_shape";
            return false;
        }
        subgroupSize_ = 32u;
        requireSubgroupSize32_ = supportsWave32;

        device_ = desc.device;
        options_ = options;
        tileM_ = m;
        tileN_ = n;
        tileK_ = k;

#define LOAD_FN(member, name) member = reinterpret_cast<PFN_##name>(gdpa(device_, #name))
        LOAD_FN(createShaderModule_, vkCreateShaderModule);
        LOAD_FN(destroyShaderModule_, vkDestroyShaderModule);
        LOAD_FN(createComputePipelines_, vkCreateComputePipelines);
        LOAD_FN(destroyPipeline_, vkDestroyPipeline);
        LOAD_FN(bindPipeline_, vkCmdBindPipeline);
        LOAD_FN(pushDescriptorSet_, vkCmdPushDescriptorSetKHR);
        LOAD_FN(pushConstants_, vkCmdPushConstants);
        LOAD_FN(dispatch_, vkCmdDispatch);
        LOAD_FN(barrier_, vkCmdPipelineBarrier2);
#undef LOAD_FN

        if (!createShaderModule_ || !destroyShaderModule_ || !createComputePipelines_ || !destroyPipeline_ ||
            !bindPipeline_ || !pushDescriptorSet_ || !pushConstants_ || !dispatch_ || !barrier_)
        {
            reason = "device_functions";
            return false;
        }

        uint32_t loadedCount = 0;
        for (size_t i = 0; i < std::size(xefg_cuda::kLayers); ++i)
        {
            const auto& layer = xefg_cuda::kLayers[i];
            auto words = LoadSpirvFile(moduleDir / (std::string(layer.info.program) + ".spv"));
            if (words.empty())
                words = LoadSpirvFile(moduleDir / (std::string(layer.info.kernel) + ".spv"));

            if (!words.empty())
            {
                spirvModules_[i] = std::move(words);
                ++loadedCount;
            }
        }

        installedLayers_ = loadedCount;
        active_.store(true, std::memory_order_release);
        xefg_cuda::ReportCoopMatrix(xefg_cuda::State::Active, tileM_, tileN_, tileK_, installedLayers_, "");
        xefg_cuda::Report("coopmat_route_vk",
            "\"state\":\"active\",\"vendor\":" + std::to_string(devProps.properties.vendorID) +
            ",\"tile\":\"" + std::to_string(tileM_) + "x" + std::to_string(tileN_) + "x" + std::to_string(tileK_) +
            "\",\"subgroup\":" + std::to_string(subgroupSize_) +
            ",\"modules\":" + std::to_string(installedLayers_) +
            ",\"full_barriers\":" + (options_.fullBarriers ? "true" : "false"));
        (void)owner;
        reason.clear();
        return true;
    }
    catch (...)
    {
        reason = "exception";
        return false;
    }
}

bool CoopMatrixRoute::EnsureLayerPipeline(Device& device, uint8_t layerIdx, PipelineState& basePso) noexcept
{
    if (!layerIdx || layerIdx > spirvModules_.size())
        return false;

    const size_t idx = layerIdx - 1;
    if (pipelineReady_[idx].load(std::memory_order_acquire))
        return true;
    if (pipelineFailed_[idx].load(std::memory_order_relaxed))
        return false;

    std::lock_guard lock(pipelineMutex_);
    if (pipelines_[idx])
        return true;

    if (spirvModules_[idx].empty() || !basePso.layout())
    {
        pipelineFailed_[idx].store(true, std::memory_order_relaxed);
        return false;
    }

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = spirvModules_[idx].size() * sizeof(uint32_t);
    moduleInfo.pCode = spirvModules_[idx].data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (createShaderModule_(device_, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS || !shaderModule)
    {
        pipelineFailed_[idx].store(true, std::memory_order_relaxed);
        return false;
    }

    // Specialization constants 0..3 pass hardware cooperative-matrix tile (M, N, K) and subgroup size to the SPIR-V module.
    const uint32_t specData[4] = {tileM_, tileN_, tileK_, subgroupSize_};
    const VkSpecializationMapEntry specEntries[4] = {
        {0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
        {1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
        {2, 2 * sizeof(uint32_t), sizeof(uint32_t)},
        {3, 3 * sizeof(uint32_t), sizeof(uint32_t)},
    };
    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = 4;
    specInfo.pMapEntries = specEntries;
    specInfo.dataSize = sizeof(specData);
    specInfo.pData = specData;

    // Enforce Wave32 (subgroup size 32) on AMD RDNA 3 / RDNA 4 so the 128-thread workgroup always maps to
    // 4 uniform 32-lane warps (warp = tid >> 5) for coopMatLoad / coopMatMulAdd.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSubgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        nullptr,
        32u
    };
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        requireSubgroupSize32_ ? &reqSubgroup : nullptr,
        (requireSubgroupSize32_ || subgroupSize_ == 16u || subgroupSize_ == 32u)
            ? VkPipelineShaderStageCreateFlags(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : 0u,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shaderModule,
        "main",
        &specInfo
    };
    pipelineInfo.layout = basePso.layout();

    VkPipeline created = VK_NULL_HANDLE;
    const VkResult res = createComputePipelines_(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &created);
    destroyShaderModule_(device_, shaderModule, nullptr);

    if (res != VK_SUCCESS || !created)
    {
        pipelineFailed_[idx].store(true, std::memory_order_relaxed);
        return false;
    }

    (void)device;
    pipelines_[idx] = created;
    pipelineReady_[idx].store(true, std::memory_order_release);
    return true;
}

bool CoopMatrixRoute::TryLaunch(Device& device, VkCommandBuffer buffer, uint8_t id, PipelineState& basePso,
                                const std::array<const Descriptor*, 64>& tables,
                                const std::array<std::array<uint32_t, 64>, 16>& constants,
                                const std::array<uint32_t, 16>& constantMasks,
                                UINT x, UINT y, UINT z) noexcept
{
    if (!active() || !xefg_cuda::CoopMatrixLayerEnabled(id))
        return false;

    const auto& layer = xefg_cuda::kLayers[id - 1];
    auto fallback = [&](const char* reason) {
        ++fallbacks_;
        xefg_cuda::CountCoopMatrix(0, 1);
        try
        {
            std::lock_guard lock(reasonsMutex_);
            if (reasons_.insert(reason).second)
            {
                xefg_cuda::Report("coopmat_vk_fallback",
                    std::string("\"reason\":\"") + reason + "\",\"program\":\"" + layer.info.program + "\"");
            }
        }
        catch (...)
        {
        }
        return false;
    };

    try
    {
        if (!xefg_cuda::DispatchShapeOk(layer, x, y, z))
            return fallback("dispatch_shape");

        const Descriptor* t0 = tables[0];
        const Descriptor* t1 = tables[1];
        if (!t0 || !t1 || (constantMasks[2] & layer.constants) != layer.constants)
            return fallback("bindings_incomplete");

        if (!EnsureLayerPipeline(device, id, basePso))
            return fallback("no_coopmat_module");

        const auto& ranges = basePso.ranges();
        // Avoid zero-initializing 6 KB of stack memory on every WMMA Dispatch; only [0..count) are written and passed.
        std::array<VkWriteDescriptorSet, 64> writes;
        std::array<VkDescriptorImageInfo, 64> images;
        std::array<VkDescriptorBufferInfo, 64> buffers;
        if (ranges.size() > writes.size())
            return fallback("bindings_overflow");

        for (size_t i = 0; i < ranges.size(); ++i)
        {
            const BoundRange& r = ranges[i];
            const Descriptor* table = tables[r.parameter];
            if (!table)
                return fallback("bindings_incomplete");

            const Descriptor& d = table[r.offset];
            VkWriteDescriptorSet& w = writes[i];
            w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, r.binding, 0, 1, r.type};
            switch (r.type)
            {
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                if (d.kind != DescriptorKind::SrvImage) return fallback("view_type");
                images[i] = d.image;
                w.pImageInfo = &images[i];
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                if (d.kind != DescriptorKind::UavImage) return fallback("view_type");
                images[i] = d.image;
                w.pImageInfo = &images[i];
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                if (d.kind != DescriptorKind::SrvBuffer && d.kind != DescriptorKind::UavBuffer) return fallback("view_type");
                buffers[i] = d.buffer;
                w.pBufferInfo = &buffers[i];
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                if (d.kind != DescriptorKind::Cbv) return fallback("view_type");
                buffers[i] = d.buffer;
                w.pBufferInfo = &buffers[i];
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                if (d.kind != DescriptorKind::Sampler) return fallback("view_type");
                images[i] = d.image;
                w.pImageInfo = &images[i];
                break;
            default:
                return fallback("view_type");
            }
        }

        size_t count = ranges.size();
        const auto& samplers = basePso.staticSamplers();
        if (count + samplers.size() > writes.size())
            return fallback("bindings_overflow");

        for (size_t s = 0; s < samplers.size(); ++s, ++count)
        {
            images[count] = {samplers[s], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
            writes[count] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE,
                             basePso.staticBindings()[s], 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &images[count]};
        }

        if (!seen_[id - 1].exchange(true))
        {
            xefg_cuda::Report("coopmat_layer_vk",
                std::string("\"program\":\"") + layer.info.program + "\",\"kernel\":\"" + layer.info.kernel + "\"");
        }

        bindPipeline_(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[id - 1]);
        if (count)
            pushDescriptorSet_(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, basePso.layout(), 0, uint32_t(count), writes.data());

        if (basePso.pushConstantBytes() && basePso.constantsParameter() >= 0)
        {
            pushConstants_(buffer, basePso.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, basePso.pushConstantBytes(),
                           constants[basePso.constantsParameter()].data());
        }

        const uint32_t gx = (x * layer.intelW + layer.tileW - 1) / layer.tileW;
        const uint32_t gy = (y * layer.intelH + layer.tileH - 1) / layer.tileH;
        dispatch_(buffer, gx, gy, layer.gridZ);

        if (options_.fullBarriers)
        {
            VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            memory.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &memory;
            barrier_(buffer, &dependency);
        }

        ++launches_;
        xefg_cuda::CountCoopMatrix(1, 0);
        return true;
    }
    catch (...)
    {
        return fallback("exception");
    }
}
} // namespace xvk
