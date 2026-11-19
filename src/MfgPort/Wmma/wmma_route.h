#pragma once
// AMD WMMA INT8 acceleration route for XeFG's ported XMX MFG model (Model 16), built on top of the
// XMX DP4a INT8 graph (one optical flow field per generated frame batch) for D3D12 and Vulkan.
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

// Installs the D3D12 WMMA INT8 route on an AMD RDNA 3 / RDNA 4 adapter. Loads optional precompiled
// WaveMatrix/WMMA DXIL overrides from wmmaDir (<program>.dxbc) and tracks the 17 INT8 layers of Model 16.
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
