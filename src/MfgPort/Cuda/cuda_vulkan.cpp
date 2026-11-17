// The Vulkan tensor-core and cooperative-matrix routes' layer masks and statuses, declared in cuda_route.h.
//
// MFG-Unlocker 1.7.1 (MIT, Copyright (c) 2026 Michael Robles, see NOTICE.md) keeps them in xess/cuda/cuda_route.cpp,
// beside the state of its D3D12 route. The D3D12 route of this repository is an older port of that file with its own
// state, so the Vulkan routes' states are kept here and don't touch the D3D12 state.
#include "cuda_route.h"
#include "cuda_layers.h"

#include <atomic>
#include <iterator>
#include <mutex>

namespace xefg_cuda
{
namespace
{
constexpr uint32_t kAllLayers = (1u << std::size(kLayers)) - 1;

std::atomic<uint32_t> gLayerMask{kAllLayers};

// What ReportVulkan last said, for the menu and the log. A launch is on the recording path, so the counts are atomics
std::mutex gMutex;
Status gStatus;
std::atomic<bool> gActive{};
std::atomic<uint64_t> gLaunches{}, gFallbacks{};

// VK_KHR_cooperative_matrix status (AMD WMMA / Intel XMX on Vulkan)
std::atomic<uint32_t> gCoopLayerMask{kAllLayers};
std::mutex gCoopMutex;
Status gCoopStatus;
std::atomic<bool> gCoopActive{};
std::atomic<uint64_t> gCoopLaunches{}, gCoopFallbacks{};
}

void SetLayerMask(uint32_t mask) noexcept
{
    gLayerMask.store(mask & kAllLayers, std::memory_order_relaxed);
}

bool LayerEnabled(uint8_t layerId) noexcept
{
    return layerId && layerId <= std::size(kLayers) && (gLayerMask.load(std::memory_order_relaxed) & (1u << (layerId - 1)));
}

void ReportVulkan(State state, int major, int minor, uint32_t layers, const char* reason) noexcept
{
    try
    {
        std::lock_guard lock(gMutex);
        gStatus.state = state;
        gStatus.major = major;
        gStatus.minor = minor;
        gStatus.layers = layers;
        gStatus.reason = reason ? reason : "";
    }
    catch (...)
    {
        // The status is only shown, a failure to keep it must not reach the route
    }

    gActive.store(state == State::Active, std::memory_order_release);
}

void CountVulkan(uint64_t launches, uint64_t fallbacks) noexcept
{
    if (launches)
        gLaunches.fetch_add(launches, std::memory_order_relaxed);

    if (fallbacks)
        gFallbacks.fetch_add(fallbacks, std::memory_order_relaxed);
}

Status VulkanSnapshot() noexcept
{
    Status status;

    try
    {
        std::lock_guard lock(gMutex);
        status = gStatus;
    }
    catch (...)
    {
    }

    status.launches = gLaunches.load(std::memory_order_relaxed);
    status.fallbacks = gFallbacks.load(std::memory_order_relaxed);

    return status;
}

bool VulkanActive() noexcept
{
    return gActive.load(std::memory_order_acquire);
}

bool VulkanLaunched() noexcept
{
    return gActive.load(std::memory_order_acquire) && gLaunches.load(std::memory_order_relaxed) != 0;
}

void SetCoopMatrixLayerMask(uint32_t mask) noexcept
{
    gCoopLayerMask.store(mask & kAllLayers, std::memory_order_relaxed);
}

bool CoopMatrixLayerEnabled(uint8_t layerId) noexcept
{
    return layerId && layerId <= std::size(kLayers) && (gCoopLayerMask.load(std::memory_order_relaxed) & (1u << (layerId - 1)));
}

void ReportCoopMatrix(State state, uint32_t tileM, uint32_t tileN, uint32_t tileK, uint32_t layers, const char* reason) noexcept
{
    try
    {
        std::lock_guard lock(gCoopMutex);
        gCoopStatus.state = state;
        gCoopStatus.major = static_cast<int>(tileM);
        gCoopStatus.minor = static_cast<int>((tileN << 8) | (tileK & 0xFFu));
        gCoopStatus.layers = layers;
        gCoopStatus.reason = reason ? reason : "";
    }
    catch (...)
    {
    }

    gCoopActive.store(state == State::Active, std::memory_order_release);
}

void CountCoopMatrix(uint64_t launches, uint64_t fallbacks) noexcept
{
    if (launches)
        gCoopLaunches.fetch_add(launches, std::memory_order_relaxed);

    if (fallbacks)
        gCoopFallbacks.fetch_add(fallbacks, std::memory_order_relaxed);
}

Status CoopMatrixSnapshot() noexcept
{
    Status status;

    try
    {
        std::lock_guard lock(gCoopMutex);
        status = gCoopStatus;
    }
    catch (...)
    {
    }

    status.launches = gCoopLaunches.load(std::memory_order_relaxed);
    status.fallbacks = gCoopFallbacks.load(std::memory_order_relaxed);

    return status;
}

bool CoopMatrixActive() noexcept
{
    return gCoopActive.load(std::memory_order_acquire);
}

bool CoopMatrixLaunched() noexcept
{
    return gCoopActive.load(std::memory_order_acquire) && gCoopLaunches.load(std::memory_order_relaxed) != 0;
}
}
