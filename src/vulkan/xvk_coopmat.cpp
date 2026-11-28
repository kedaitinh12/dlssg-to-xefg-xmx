#include "xvk_coopmat.h"
#include "xvk_internal.h"
#include "VulkanPolicy.h"
#include "../WineCompat.h"
#include "../MfgPort/Cuda/cuda_layers.h"
#include "../MfgPort/Sha256.h"
#include <algorithm>
#include <charconv>
#include <cstdio>
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
                                 uint32_t& m, uint32_t& n, uint32_t& k, bool prefer16x16x32) noexcept
{
    m = n = k = 0;
    if (!properties || !count)
        return false;

    // Prefer 16x16x32 (RDNA 4 Double-K 128-bit load) when explicitly enabled & available,
    // then 16x16x16 (AMD RDNA 4 / RDNA 3 WMMA Wave32), then 8x8x32 (Intel Arc / Battlemage XMX SIMD16).
    // Only admit devices advertising an exact supported SINT8 x SINT8 -> SINT32 subgroup shape.
    bool has16x16x32 = false;
    bool has16x16x16 = false;
    bool has8x8x32 = false;
    for (uint32_t i = 0; i < count; ++i)
    {
        const auto& p = properties[i];
        if (p.scope != VK_SCOPE_SUBGROUP_KHR || p.saturatingAccumulation ||
            p.AType != VK_COMPONENT_TYPE_SINT8_KHR ||
            p.BType != VK_COMPONENT_TYPE_SINT8_KHR ||
            p.CType != VK_COMPONENT_TYPE_SINT32_KHR ||
            p.ResultType != VK_COMPONENT_TYPE_SINT32_KHR)
            continue;
        if (p.MSize == 16u && p.NSize == 16u && p.KSize == 32u)
            has16x16x32 = true;
        else if (p.MSize == 16u && p.NSize == 16u && p.KSize == 16u)
            has16x16x16 = true;
        else if (p.MSize == 8u && p.NSize == 8u && p.KSize == 32u)
            has8x8x32 = true;
    }
    if (prefer16x16x32 && has16x16x32)
    {
        m = 16u;
        n = 16u;
        k = 32u;
        return true;
    }
    if (has16x16x16)
    {
        m = 16u;
        n = 16u;
        k = 16u;
        return true;
    }
    if (has8x8x32)
    {
        m = 8u;
        n = 8u;
        k = 32u;
        return true;
    }

    return false;
}

CoopMatrixRoute::~CoopMatrixRoute()
{
    SavePipelineCache();
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
    for (auto& pipeline : pipelinesStride5_)
    {
        if (pipeline)
        {
            destroyPipeline_(device_, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
    }
    if (pipelineCache_ && destroyPipelineCache_)
    {
        destroyPipelineCache_(device_, pipelineCache_, nullptr);
        pipelineCache_ = VK_NULL_HANDLE;
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

        VkPhysicalDeviceSubgroupSizeControlProperties subgroupCtrl{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceDriverProperties driverProps{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &subgroupCtrl};
        VkPhysicalDeviceSubgroupProperties subgroup{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &driverProps};
        VkPhysicalDeviceProperties2 devProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
        props2(desc.physicalDevice, &devProps);

        bool allowK32 = true;
        {
            wchar_t envBuf[16]{};
            const DWORD envN = GetEnvironmentVariableW(L"XEFG_COOP_MATRIX_K32", envBuf, static_cast<DWORD>(std::size(envBuf)));
            if (envN > 0 && envN < std::size(envBuf) &&
                (envBuf[0] == L'0' || envBuf[0] == L'f' || envBuf[0] == L'F' || envBuf[0] == L'n' || envBuf[0] == L'N'))
            {
                allowK32 = false;
            }
        }
        std::error_code k32DirEc;
        const std::filesystem::path k32Dir = moduleDir / L"k32_16x16x32";
        const auto& gpu = devProps.properties;
        const bool preferK32 = allowK32 && VulkanPolicy::IsAmdRdna4Device(gpu.vendorID, gpu.deviceID, gpu.deviceName) &&
            std::filesystem::is_regular_file(k32Dir / L"coopmat_layers.tsv", k32DirEc);

        uint32_t m = 0, n = 0, k = 0;
        if (!SelectTile(props.data(), propCount, m, n, k, preferK32))
        {
            reason = "no_s8_matrix_shape";
            return false;
        }

        const uint32_t wantedSubgroup = (m == 8u && n == 8u && k == 32u) ? 16u : 32u;
        const bool supportsWantedSubgroup =
            (subgroupCtrl.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
            subgroupCtrl.minSubgroupSize <= wantedSubgroup && subgroupCtrl.maxSubgroupSize >= wantedSubgroup;
        if (!supportsWantedSubgroup && subgroup.subgroupSize != wantedSubgroup)
        {
            reason = "no_s8_matrix_shape";
            return false;
        }
        if (options.matCLayout == MatCLayout::Emulated)
        {
            reason = "no_s8_matrix_shape";
            return false;
        }
        device_ = desc.device;
        options_ = options;
        tileM_ = m;
        tileN_ = n;
        tileK_ = k;
        subgroupSize_ = wantedSubgroup;
        requireSubgroupSize_ = supportsWantedSubgroup;

        gpuArch_ = GpuArch::Generic;
        if (devProps.properties.vendorID == 0x1002u && tileM_ == 16u && tileN_ == 16u && (tileK_ == 16u || tileK_ == 32u) && subgroupSize_ == 32u)
        {
            const auto amdArch = VulkanPolicy::ClassifyAmdArch(
                devProps.properties.vendorID, devProps.properties.deviceID,
                std::string_view(devProps.properties.deviceName), /*hasWmma16x16x16Wave32=*/true);
            if (amdArch == VulkanPolicy::AmdArch::Rdna4)
                gpuArch_ = GpuArch::Rdna4;
            else if (amdArch == VulkanPolicy::AmdArch::Rdna3)
                gpuArch_ = GpuArch::Rdna3;

            // RX 7700 XT probe on Adrenalin 26.6.2 (LLPC) showed radv_reg + local=256 inflates VGPRs
            // on L_9481d452 (57->78), L_06e97933 (66->96), and L_3ed03daf (57->78), whereas 26.11.2
            // compiles them to 33-35 VGPRs. Keep those 3 layers on local=128 for AMD LLPC < 26.10.
            rdna3LegacyLlpc_ = false;
            if (gpuArch_ == GpuArch::Rdna3 && driverProps.driverID == VK_DRIVER_ID_AMD_PROPRIETARY)
            {
                unsigned drvMajor = 0, drvMinor = 0;
                const std::string_view version(driverProps.driverInfo);
                const char* end = version.data() + version.size();
                const auto major = std::from_chars(version.data(), end, drvMajor);
                if (major.ec == std::errc{} && major.ptr != end && *major.ptr == '.' &&
                    std::from_chars(major.ptr + 1, end, drvMinor).ec == std::errc{} &&
                    (drvMajor < 26u || (drvMajor == 26u && drvMinor < 10u)))
                {
                    rdna3LegacyLlpc_ = true;
                }
            }
        }

#define LOAD_FN(member, name) member = reinterpret_cast<PFN_##name>(gdpa(device_, #name))
        LOAD_FN(createShaderModule_, vkCreateShaderModule);
        LOAD_FN(destroyShaderModule_, vkDestroyShaderModule);
        LOAD_FN(createComputePipelines_, vkCreateComputePipelines);
        LOAD_FN(destroyPipeline_, vkDestroyPipeline);
        LOAD_FN(createPipelineCache_, vkCreatePipelineCache);
        LOAD_FN(destroyPipelineCache_, vkDestroyPipelineCache);
        LOAD_FN(getPipelineCacheData_, vkGetPipelineCacheData);
        LOAD_FN(bindPipeline_, vkCmdBindPipeline);
        LOAD_FN(pushDescriptorSet_, vkCmdPushDescriptorSetKHR);
        LOAD_FN(pushConstants_, vkCmdPushConstants);
        LOAD_FN(dispatch_, vkCmdDispatch);
        LOAD_FN(barrier_, vkCmdPipelineBarrier2);
        if (desc.pipelineExecutableProperties)
        {
            LOAD_FN(getPipelineExecProps_, vkGetPipelineExecutablePropertiesKHR);
            LOAD_FN(getPipelineExecStats_, vkGetPipelineExecutableStatisticsKHR);
            pipelineExecStats_ = (getPipelineExecProps_ != nullptr && getPipelineExecStats_ != nullptr);
        }
#undef LOAD_FN

        if (!createShaderModule_ || !destroyShaderModule_ || !createComputePipelines_ || !destroyPipeline_ ||
            !bindPipeline_ || !pushDescriptorSet_ || !pushConstants_ || !dispatch_ || !barrier_)
        {
            reason = "device_functions";
            return false;
        }

        // Enable register-only (zero-sAcc) WMMA shaders when the Wave32 16x16x16 matC register layout
        // is verified (gfx11 -> radv_reg, gfx12 -> gfx12_reg). Can be disabled via XEFG_COOP_MATRIX_RADV_REG=0.
        bool allowRegOnly = true;
        {
            wchar_t envBuf[16]{};
            const DWORD n = GetEnvironmentVariableW(L"XEFG_COOP_MATRIX_RADV_REG", envBuf, static_cast<DWORD>(std::size(envBuf)));
            if (n > 0 && n < std::size(envBuf) && (envBuf[0] == L'0' || envBuf[0] == L'f' || envBuf[0] == L'F' || envBuf[0] == L'n' || envBuf[0] == L'N'))
                allowRegOnly = false;
        }
        MatCLayout activeLayout = MatCLayout::Unknown;
        if (allowRegOnly && tileM_ == 16u && tileN_ == 16u && tileK_ == 16u && subgroupSize_ == 32u)
        {
            if (options.matCLayout == MatCLayout::Gfx11Reg || options.matCLayout == MatCLayout::Gfx12Reg)
            {
                activeLayout = options.matCLayout;
            }
            // Do not guess register-only layout from GPU arch when MatCLayout::Unknown (e.g. native Vulkan
            // without a verified matC probe); keep standard sAcc shaders unless the hardware probe confirmed
            // Gfx11Reg or Gfx12Reg (which D3D12VulkanWmma always runs on both Windows and Wine/Proton RADV).
        }
        const std::filesystem::path regDir = (activeLayout == MatCLayout::Gfx12Reg)
            ? (moduleDir / L"gfx12_reg")
            : ((activeLayout == MatCLayout::Gfx11Reg) ? (moduleDir / L"radv_reg") : std::filesystem::path{});
        std::error_code dirEc;
        const bool useRegDir = !regDir.empty() && std::filesystem::is_directory(regDir, dirEc);
        const bool useRadvReg = useRegDir;

        // Per-kernel reg-only gate backed by RDNA 3 & RDNA 4 ISA probe data (Windows LLPC + Linux RADV ACO):
        // - RDNA 3 (gfx11): reg_only cuts VGPRs and 5-10 KB LDS across all 16 kernels.
        // - RDNA 4 (gfx12): reg_only yields major VGPR/LDS/occupancy gains on heavy PoolConv (aa016078, 198d6bf8,
        //   dcd8becf, efbe3788, a3c3caa1) and all 5 Flow layers, while sAcc stays faster (fewer VMEM clauses /
        //   lower VGPRs) on the 6 lighter Pool/Decoder layers (50f0c205, 9481d452, 06e97933, 3ed03daf, 748b0baf, cb391582).
        const auto layerWantsReg = [&](size_t idx, const xefg_cuda::Layer& l) -> bool {
            if (!useRegDir)
                return false;
            if (activeLayout == MatCLayout::Gfx11Reg)
                return true;
            if (activeLayout == MatCLayout::Gfx12Reg)
            {
                if (l.family == xefg_cuda::Family::Flow)
                    return true;
                return idx == 0u || idx == 1u || idx == 2u || idx == 3u || idx == 5u;
            }
            return false;
        };

        const bool isXmx = (tileM_ == 8u && tileN_ == 8u && tileK_ == 32u);
        const bool isK32 = (tileM_ == 16u && tileN_ == 16u && tileK_ == 32u);
        const std::filesystem::path xmxDir = moduleDir / L"xmx_8x8x32";
        const std::filesystem::path stride5Dir = moduleDir / L"stride5";
        const std::filesystem::path stride5RegDir = (activeLayout == MatCLayout::Gfx12Reg)
            ? (moduleDir / L"stride5_gfx12_reg")
            : ((activeLayout == MatCLayout::Gfx11Reg) ? (moduleDir / L"stride5_radv_reg") : std::filesystem::path{});
        std::error_code s5Ec;
        // Stride 4 (16B aligned) is locked for RDNA 3/4; skip probing/loading stride5* SPIR-V sets.
        const bool hasStride5Base = false;
        (void)stride5Dir; (void)s5Ec;
        const bool hasStride5Reg = hasStride5Base && !stride5RegDir.empty() && std::filesystem::is_directory(stride5RegDir, s5Ec);
        uint32_t loadedCount = 0;
        uint32_t loadedStride5Count = 0;
        for (size_t i = 0; i < std::size(xefg_cuda::kLayers); ++i)
        {
            const auto& layer = xefg_cuda::kLayers[i];
            const std::filesystem::path primaryDir = isXmx ? xmxDir : (isK32 ? k32Dir : (layerWantsReg(i, layer) ? regDir : moduleDir));
            auto words = LoadSpirvFile(primaryDir / (std::string(layer.info.program) + ".spv"));
            if (words.empty())
                words = LoadSpirvFile(primaryDir / (std::string(layer.info.kernel) + ".spv"));
            if (words.empty() && primaryDir != moduleDir && !isXmx && !isK32)
            {
                words = LoadSpirvFile(moduleDir / (std::string(layer.info.program) + ".spv"));
                if (words.empty())
                    words = LoadSpirvFile(moduleDir / (std::string(layer.info.kernel) + ".spv"));
            }

            if (!words.empty())
            {
                spirvModules_[i] = std::move(words);
                ++loadedCount;
            }

            if (hasStride5Base)
            {
                const std::filesystem::path s5Primary = (layerWantsReg(i, layer) && hasStride5Reg) ? stride5RegDir : stride5Dir;
                auto s5Words = LoadSpirvFile(s5Primary / (std::string(layer.info.program) + ".spv"));
                if (s5Words.empty())
                    s5Words = LoadSpirvFile(s5Primary / (std::string(layer.info.kernel) + ".spv"));
                if (s5Words.empty() && s5Primary != stride5Dir)
                {
                    s5Words = LoadSpirvFile(stride5Dir / (std::string(layer.info.program) + ".spv"));
                    if (s5Words.empty())
                        s5Words = LoadSpirvFile(stride5Dir / (std::string(layer.info.kernel) + ".spv"));
                }
                if (!s5Words.empty())
                {
                    spirvModulesStride5_[i] = std::move(s5Words);
                    ++loadedStride5Count;
                }
            }
        }
        hasStride5_ = (loadedStride5Count == loadedCount && loadedCount > 0);
        if (loadedCount == 0)
        {
            reason = "no_coopmat_modules";
            return false;
        }

        // Load persistent VkPipelineCache from %LOCALAPPDATA%\dlssg-to-xefg to eliminate multi-second
        // shader compilation hitches across sessions.
        if (createPipelineCache_ && destroyPipelineCache_ && getPipelineCacheData_)
        {
            wchar_t localAppData[MAX_PATH]{};
            const DWORD envLen = GetEnvironmentVariableW(L"LOCALAPPDATA", localAppData, MAX_PATH);
            std::vector<uint8_t> initialData;
            if (envLen > 0 && envLen < MAX_PATH)
            {
                const std::filesystem::path cacheDir = std::filesystem::path(localAppData) / L"dlssg-to-xefg";
                std::error_code ec;
                std::filesystem::create_directories(cacheDir, ec);
                char fname[96]{};
                std::snprintf(fname, sizeof(fname), "vk_coopmat_cache_v2_%04x_%04x_%ux%ux%u%s.bin",
                    devProps.properties.vendorID, devProps.properties.deviceID, tileM_, tileN_, tileK_,
                    useRadvReg ? "_radvreg" : "");
                cachePath_ = cacheDir / fname;
                const auto sz = std::filesystem::file_size(cachePath_, ec);
                if (!ec && sz > 0 && sz <= 32 * 1024 * 1024)
                {
                    std::ifstream in(cachePath_, std::ios::binary);
                    initialData.resize(static_cast<size_t>(sz));
                    if (!in.read(reinterpret_cast<char*>(initialData.data()), static_cast<std::streamsize>(sz)))
                        initialData.clear();
                }
            }
            VkPipelineCacheCreateInfo cacheInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            cacheInfo.initialDataSize = initialData.size();
            cacheInfo.pInitialData = initialData.empty() ? nullptr : initialData.data();
            if (createPipelineCache_(device_, &cacheInfo, nullptr, &pipelineCache_) != VK_SUCCESS)
                pipelineCache_ = VK_NULL_HANDLE;
        }

        installedLayers_ = loadedCount;
        active_.store(true, std::memory_order_release);
        xefg_cuda::ReportCoopMatrix(xefg_cuda::State::Active, tileM_, tileN_, tileK_, installedLayers_, "");
        const char* archTag = (gpuArch_ == GpuArch::Rdna4) ? "rdna4_gfx12"
                            : (gpuArch_ == GpuArch::Rdna3) ? "rdna3_gfx11"
                                                           : "generic";
        std::string s8Shapes;
        for (uint32_t i = 0; i < propCount; ++i)
        {
            const auto& p = props[i];
            if (p.scope == VK_SCOPE_SUBGROUP_KHR &&
                p.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
                p.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                p.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
                p.ResultType == VK_COMPONENT_TYPE_SINT32_KHR)
            {
                if (!s8Shapes.empty())
                    s8Shapes += ",";
                s8Shapes += std::to_string(p.MSize) + "x" + std::to_string(p.NSize) + "x" + std::to_string(p.KSize);
            }
        }
        xefg_cuda::Report("coopmat_route_vk",
            "\"state\":\"active\",\"vendor\":" + std::to_string(devProps.properties.vendorID) +
            ",\"arch\":\"" + std::string(archTag) +
            "\",\"tile\":\"" + std::to_string(tileM_) + "x" + std::to_string(tileN_) + "x" + std::to_string(tileK_) +
            "\",\"s8_shapes\":\"" + s8Shapes +
            "\",\"subgroup\":" + std::to_string(subgroupSize_) +
            ",\"modules\":" + std::to_string(installedLayers_) +
            ",\"isa_stats\":" + (pipelineExecStats_ ? "true" : "false") +
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

uint32_t CoopMatrixRoute::SelectWorkgroupSize(size_t layerIdx) const noexcept
{
    if (layerIdx >= std::size(xefg_cuda::kLayers) || subgroupSize_ != 32u)
        return 128u;
    const auto& layer = xefg_cuda::kLayers[layerIdx];
    if (layer.family == xefg_cuda::Family::Flow || layer.tileH < 16u)
        return 128u;

    if (gpuArch_ == GpuArch::Rdna3 && rdna3LegacyLlpc_ &&
        (layer.kernel == 6u || layer.kernel == 7u || layer.kernel == 8u))
    {
        // AMD proprietary LLPC < 26.10 on RDNA 3: keep L_9481d452 / L_06e97933 / L_3ed03daf at local=128
        // (57-66 VGPRs, full occupancy) while keeping local=256 on L_aa016078 / L_dcd8becf / L_cb391582.
        return 128u;
    }
    if (gpuArch_ == GpuArch::Rdna3 || gpuArch_ == GpuArch::Rdna4)
    {
        // RDNA 3 (gfx11) & RDNA 4 (gfx12): 8 Wave32s (256 threads) halve per-wave accumulators on 16-row tiles,
        // cutting VGPR pressure and improving inverse throughput by 24%-50% across all 6 tileH=16 layers.
        return 256u;
    }
    return 128u;
}

void CoopMatrixRoute::LogPipelineStats(size_t layerIdx, uint32_t localSizeX, VkPipeline pipeline, uint32_t strideVal) const noexcept
{
    if (!pipelineExecStats_ || !device_ || !pipeline || layerIdx >= std::size(xefg_cuda::kLayers))
        return;
    try
    {
        VkPipelineInfoKHR pipeInfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, nullptr, pipeline};
        uint32_t execCount = 0;
        if (getPipelineExecProps_(device_, &pipeInfo, &execCount, nullptr) != VK_SUCCESS || !execCount)
            return;
        if (execCount > 8u)
            execCount = 8u;
        std::vector<VkPipelineExecutablePropertiesKHR> execs(
            execCount, VkPipelineExecutablePropertiesKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
        if (getPipelineExecProps_(device_, &pipeInfo, &execCount, execs.data()) != VK_SUCCESS || !execCount)
            return;

        VkPipelineExecutableInfoKHR execInfo{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, nullptr, pipeline, 0u};
        uint32_t statCount = 0;
        if (getPipelineExecStats_(device_, &execInfo, &statCount, nullptr) != VK_SUCCESS || !statCount)
            return;
        if (statCount > 64u)
            statCount = 64u;
        std::vector<VkPipelineExecutableStatisticKHR> stats(
            statCount, VkPipelineExecutableStatisticKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        if (getPipelineExecStats_(device_, &execInfo, &statCount, stats.data()) != VK_SUCCESS || !statCount)
            return;

        auto sanitize = [](const char* s) -> std::string {
            std::string out;
            for (; s && *s; ++s)
            {
                if (*s == '"' || *s == '\\')
                    out.push_back('_');
                else if (*s >= 32 && *s < 127)
                    out.push_back(*s);
            }
            return out;
        };
        std::string fields;
        for (uint32_t i = 0; i < statCount; ++i)
        {
            const auto& st = stats[i];
            if (!fields.empty())
                fields += "; ";
            fields += sanitize(st.name) + "=";
            switch (st.format)
            {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                fields += st.value.b32 ? "1" : "0";
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                fields += std::to_string(st.value.i64);
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                fields += std::to_string(st.value.u64);
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
            {
                char buf[32]{};
                std::snprintf(buf, sizeof(buf), "%.2f", st.value.f64);
                fields += buf;
                break;
            }
            default:
                fields += "?";
                break;
            }
        }
        const auto& layer = xefg_cuda::kLayers[layerIdx];
        xefg_cuda::Report("coopmat_isa_stats",
            "\"program\":\"" + std::string(layer.info.program) +
            "\",\"kernel\":\"" + std::string(layer.info.kernel) +
            "\",\"stride\":" + std::to_string(strideVal) +
            ",\"local_x\":" + std::to_string(localSizeX) +
            ",\"wave\":" + std::to_string(execs[0].subgroupSize) +
            ",\"stats\":\"" + fields + "\"");
    }
    catch (...)
    {
    }
}

bool CoopMatrixRoute::EnsureLayerPipeline(Device& device, uint8_t layerIdx, PipelineState& basePso, bool useStride5) noexcept
{
    if (!layerIdx || layerIdx > spirvModules_.size())
        return false;

    const bool s5 = useStride5 && hasStride5_;
    auto& mods = s5 ? spirvModulesStride5_ : spirvModules_;
    auto& pipes = s5 ? pipelinesStride5_ : pipelines_;
    auto& readyArr = s5 ? pipelineReadyStride5_ : pipelineReady_;
    auto& failedArr = s5 ? pipelineFailedStride5_ : pipelineFailed_;

    const size_t idx = layerIdx - 1;
    if (readyArr[idx].load(std::memory_order_acquire))
        return true;
    if (failedArr[idx].load(std::memory_order_relaxed))
        return false;

    std::lock_guard lock(pipelineMutex_);
    if (pipes[idx])
        return true;

    if (mods[idx].empty() || !basePso.layout())
    {
        failedArr[idx].store(true, std::memory_order_relaxed);
        return false;
    }

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = mods[idx].size() * sizeof(uint32_t);
    moduleInfo.pCode = mods[idx].data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (createShaderModule_(device_, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS || !shaderModule)
    {
        failedArr[idx].store(true, std::memory_order_relaxed);
        return false;
    }

    // Specialization constants 0..4 pass cooperative-matrix tile (M, N, K), subgroup size, and arch-tuned workgroup size.
    const uint32_t localSizeX = SelectWorkgroupSize(idx);
    const uint32_t specData[5] = {tileM_, tileN_, tileK_, subgroupSize_, localSizeX};
    const VkSpecializationMapEntry specEntries[5] = {
        {0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
        {1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
        {2, 2 * sizeof(uint32_t), sizeof(uint32_t)},
        {3, 3 * sizeof(uint32_t), sizeof(uint32_t)},
        {4, 4 * sizeof(uint32_t), sizeof(uint32_t)},
    };
    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = 5;
    specInfo.pMapEntries = specEntries;
    specInfo.dataSize = sizeof(specData);
    specInfo.pData = specData;

    // Enforce Wave32 (16x16x16 AMD WMMA) or SIMD16 (8x8x32 Intel Arc/Battlemage XMX) so the 128-thread
    // workgroup always maps cleanly to uniform subgroups for coopMatLoad / coopMatMulAdd.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSubgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        nullptr,
        subgroupSize_
    };
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    if (pipelineExecStats_)
        pipelineInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    pipelineInfo.stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        requireSubgroupSize_ ? &reqSubgroup : nullptr,
        (requireSubgroupSize_ || subgroupSize_ == 16u || subgroupSize_ == 32u)
            ? VkPipelineShaderStageCreateFlags(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) : 0u,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shaderModule,
        "main",
        &specInfo
    };
    pipelineInfo.layout = basePso.layout();

    VkPipeline created = VK_NULL_HANDLE;
    const VkResult res = createComputePipelines_(device_, pipelineCache_, 1, &pipelineInfo, nullptr, &created);
    destroyShaderModule_(device_, shaderModule, nullptr);

    if (res != VK_SUCCESS || !created)
    {
        failedArr[idx].store(true, std::memory_order_relaxed);
        return false;
    }

    (void)device;
    const uint32_t strideVal = (tileK_ == 32u) ? 12u : (s5 ? 5u : 4u);
    LogPipelineStats(idx, localSizeX, created, strideVal);
    pipes[idx] = created;
    readyArr[idx].store(true, std::memory_order_release);
    cacheDirty_.store(true, std::memory_order_relaxed);
    return true;
}

void CoopMatrixRoute::PrecompileLayer(Device& device, uint8_t layerIdx, PipelineState& basePso) noexcept
{
    if (!active() || !layerIdx || !xefg_cuda::CoopMatrixLayerEnabled(layerIdx))
        return;
    const bool useStride5 = hasStride5_ && (VulkanPolicy::GetWmmaLdsStrideMode() == 1);
    (void)EnsureLayerPipeline(device, layerIdx, basePso, useStride5);
}

void CoopMatrixRoute::SavePipelineCache() noexcept
{
    if (!device_ || !pipelineCache_ || !getPipelineCacheData_ || cachePath_.empty())
        return;
    if (!cacheDirty_.exchange(false, std::memory_order_acq_rel))
        return;
    try
    {
        size_t bytes = 0;
        if (getPipelineCacheData_(device_, pipelineCache_, &bytes, nullptr) != VK_SUCCESS || !bytes || bytes > 32 * 1024 * 1024)
            return;
        std::vector<uint8_t> blob(bytes);
        if (getPipelineCacheData_(device_, pipelineCache_, &bytes, blob.data()) != VK_SUCCESS || !bytes)
            return;
        std::filesystem::path tmpPath = cachePath_;
        tmpPath += L".tmp";
        {
            std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
            if (!out.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(bytes)))
                return;
        }
        std::error_code ec;
        std::filesystem::rename(tmpPath, cachePath_, ec);
        if (ec)
            std::filesystem::remove(tmpPath, ec);
    }
    catch (...)
    {
    }
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
        batchFallbacks_.fetch_add(1u, std::memory_order_relaxed);
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

        const bool wantStride5 = (VulkanPolicy::GetWmmaLdsStrideMode() == 1);
        if (wantStride5 && !hasStride5_ && !stride5MissingWarned_.exchange(true, std::memory_order_relaxed))
        {
            xefg_cuda::Report("coopmat_stride5_unavailable",
                "\"reason\":\"missing_stride5_spirv\",\"fallback_stride\":4");
        }
        bool useStride5 = hasStride5_ && wantStride5;
        if (!EnsureLayerPipeline(device, id, basePso, useStride5))
        {
            if (useStride5 && EnsureLayerPipeline(device, id, basePso, false))
                useStride5 = false;
            else
                return fallback("no_coopmat_module");
        }

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

        bindPipeline_(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, useStride5 ? pipelinesStride5_[id - 1] : pipelines_[id - 1]);
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

        if (useStride5)
            batchStride5_.fetch_add(1u, std::memory_order_relaxed);
        else
            batchStride4_.fetch_add(1u, std::memory_order_relaxed);
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
