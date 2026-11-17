#pragma once
// VK_KHR_cooperative_matrix (SPV_KHR_cooperative_matrix) acceleration route for the 17 INT8 conv layers of the ported
// XeFG model on Vulkan (AMD RDNA3/RDNA4 WMMA and Intel Arc/Battlemage XMX).
//
// Like TensorRoute (xvk_tensor.h) on NVIDIA RTX GPUs, CoopMatrixRoute replaces recognized INT8 conv layer dispatches
// in-place inside the game's own VkCommandBuffer, reading and writing the exact descriptors XeFG bound for that layer
// (t0, [t4,] u0, t1, t2, t3, root constants b0) with zero queue switches and zero layout copies. Any layer without a
// compiled cooperative-matrix SPIR-V module or with an unsupported shape/binding stays on its SPV_KHR_integer_dot_product
// pipeline in the same frame.
#include "xvk.h"
#include "../MfgPort/Cuda/xefg_cuda_params.h"
#include <array>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace xvk
{
class Device;
class PipelineState;
struct Descriptor;

class CoopMatrixRoute
{
public:
    CoopMatrixRoute() = default;
    ~CoopMatrixRoute();
    CoopMatrixRoute(const CoopMatrixRoute&) = delete;
    CoopMatrixRoute& operator=(const CoopMatrixRoute&) = delete;

    static bool SelectTile(const VkCooperativeMatrixPropertiesKHR* properties, uint32_t count,
                           uint32_t& m, uint32_t& n, uint32_t& k) noexcept;

    bool Install(Device& owner, const DeviceDesc& desc, const std::filesystem::path& moduleDir,
                 const CoopMatrixOptions& options, std::string& reason) noexcept;

    bool TryLaunch(Device& device, VkCommandBuffer buffer, uint8_t layer, PipelineState& basePso,
                   const std::array<const Descriptor*, 64>& tables,
                   const std::array<std::array<uint32_t, 64>, 16>& constants,
                   const std::array<uint32_t, 16>& constantMasks,
                   UINT x, UINT y, UINT z) noexcept;

    void Enable(bool on) noexcept { enabled_.store(on, std::memory_order_relaxed); }
    bool active() const noexcept { return active_.load(std::memory_order_acquire) && enabled_.load(std::memory_order_relaxed); }
    uint32_t tileM() const noexcept { return tileM_; }
    uint32_t tileN() const noexcept { return tileN_; }
    uint32_t tileK() const noexcept { return tileK_; }
    uint32_t installedLayers() const noexcept { return installedLayers_; }

private:
    bool EnsureLayerPipeline(Device& device, uint8_t layerIdx, PipelineState& basePso) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    PFN_vkCreateShaderModule createShaderModule_{};
    PFN_vkDestroyShaderModule destroyShaderModule_{};
    PFN_vkCreateComputePipelines createComputePipelines_{};
    PFN_vkDestroyPipeline destroyPipeline_{};
    PFN_vkCmdBindPipeline bindPipeline_{};
    PFN_vkCmdPushDescriptorSetKHR pushDescriptorSet_{};
    PFN_vkCmdPushConstants pushConstants_{};
    PFN_vkCmdDispatch dispatch_{};
    PFN_vkCmdPipelineBarrier2 barrier_{};

    CoopMatrixOptions options_{};
    uint32_t tileM_{}, tileN_{}, tileK_{};
    uint32_t subgroupSize_{};
    uint32_t installedLayers_{};
    std::array<std::vector<uint32_t>, 17> spirvModules_{};
    std::array<VkPipeline, 17> pipelines_{};
    std::array<std::atomic<bool>, 17> pipelineFailed_{};
    std::array<std::atomic<bool>, 17> seen_{};
    std::atomic<bool> active_{};
    std::atomic<bool> enabled_{true};
    std::atomic<uint64_t> launches_{}, fallbacks_{};
    std::mutex pipelineMutex_;
    std::mutex reasonsMutex_;
    std::set<std::string> reasons_;
};
} // namespace xvk
