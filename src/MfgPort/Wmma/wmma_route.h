#pragma once
// AMD model routes: scoped D3D12 dot4 on RDNA 2 and Vulkan cooperative matrices on RDNA 3/4.
#include <d3d12.h>
#ifdef _WIN32
#include "../../XeFG.h"
#endif
#include <cstdint>
#include <filesystem>
#include <string>

namespace xefg_wmma
{
enum class State : uint8_t { NotAttempted, Disabled, Unavailable, Active };
enum class Arch : uint8_t { Unknown, Rdna2, Rdna3, Rdna4 };

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

bool Install(ID3D12Device* device, HMODULE providerModule, const std::filesystem::path& wmmaDir, std::string& reason) noexcept;
void Disable(const char* reason) noexcept;
void SetEnabled(bool enabled) noexcept;
bool Enabled() noexcept;
bool Supported() noexcept;
bool IsRdna4() noexcept;
bool IsRdna2() noexcept;
bool UsesD3D12Dot4Override() noexcept;
bool Dot4SplitActive() noexcept;
#if defined(_WIN32)
using CreateComputePsoFn = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID, void**);
void OnCreateComputePipelineState(ID3D12Device* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, REFIID iid, void* stockPso, CreateComputePsoFn origCreate) noexcept;
xefgd_result_t ExecuteDot4(xefgd_handle_t context, ID3D12GraphicsCommandList* list, const xefgd_exec_params_t& params,
                           float interpolationTime, bool& usedDot4);
#endif
Status Snapshot() noexcept;
} // namespace xefg_wmma
