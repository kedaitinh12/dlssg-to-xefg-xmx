#pragma once
// AMD RDNA 3 / RDNA 4 WMMA capability probe for XeFG's ported XMX MFG model (Model 16).
// Hardware 16x16x16 INT8 WMMA runs via D3D12VulkanWmma (Vulkan VK_KHR_cooperative_matrix);
// D3D12 fallback frames and XMX Port run stock D3D12 DXIL directly (the software cs_6_6
// shared-tile DP4a overrides are disabled because they lack hardware matrix acceleration
// and regress LDS/VGPR occupancy vs. stock DXIL).
#include <d3d12.h>
#include <cstdint>
#include <filesystem>
#include <string>

namespace xefg_wmma
{
enum class State : uint8_t { NotAttempted, Disabled, Unavailable, Active };
enum class Arch : uint8_t { Unknown, Rdna3, Rdna4 };

struct Status
{
    State state{State::NotAttempted};
    Arch arch{Arch::Unknown};
    uint32_t vendorId{};
    uint32_t deviceId{};
    uint32_t dxgiVendorId{};
    uint32_t dxgiDeviceId{};
    bool dxgiSpoofed{false};
    uint32_t waveMmaTier{};
    uint32_t layers{};
    uint32_t overrideLayers{};
    uint64_t launches{};
    uint64_t fallbacks{};
    uint64_t binds{};
    std::string gpuName;
    std::string probeSource;
    std::string reason;
};

using LogFunction = void (*)(const char* stage, const std::string& fields);
void SetLog(LogFunction log) noexcept;

// Probes whether the underlying D3D12 adapter is an AMD RDNA 3 / RDNA 4 GPU capable of hardware WMMA
// (bypassing OptiScaler/Streamline device wrappers and DXGI spoofing). Does not install D3D12
// command-list hooks or software cs_6_6 DP4a overrides.
bool Install(ID3D12Device* device, HMODULE providerModule, const std::filesystem::path& wmmaDir, std::string& reason) noexcept;
void Disable(const char* reason) noexcept;
void SetEnabled(bool enabled) noexcept;
bool Enabled() noexcept;
bool Supported() noexcept;
bool IsRdna4() noexcept;
void Adopt(ID3D12GraphicsCommandList* list) noexcept;
void ForgetLayers() noexcept;
Status Snapshot() noexcept;
} // namespace xefg_wmma
