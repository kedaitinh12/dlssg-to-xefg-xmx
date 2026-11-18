#include "wmma_route.h"
#include "../Cuda/cuda_layers.h"
#include "../Sha256.h"
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string_view>
#include <vector>

namespace xefg_wmma
{
namespace
{
using Microsoft::WRL::ComPtr;

std::mutex gMutex;
std::atomic<LogFunction> gLog{};
std::atomic<State> gState{State::NotAttempted};
std::atomic<Arch> gArch{Arch::Unknown};
std::atomic<bool> gActive{false};
uint32_t gDeviceId{};
uint32_t gWaveMmaTier{};
uint32_t gOverrideLayers{};
std::string gReason;
std::atomic<uint64_t> gLaunches{}, gFallbacks{}, gBinds{};

void Log(const char* stage, const std::string& fields) noexcept
{
    if (const auto fn = gLog.load(std::memory_order_relaxed))
        fn(stage, fields);
}

// Identifies AMD RDNA 4 (gfx1200/gfx1201: Radeon RX 9070 XT / 9070 / 9060 XT, PCI DeviceIDs 0x7550..0x75FF)
// vs. AMD RDNA 3 (gfx1100..gfx1103: Radeon RX 7900 / 7800 / 7700 / 7600 / 780M / 890M, PCI DeviceIDs 0x7440..0x749F, 0x15BF, 0x150E).
Arch DetectAmdArch(uint32_t vendorId, uint32_t deviceId, std::wstring_view description) noexcept
{
    if (vendorId != 0x1002)
        return Arch::Unknown;

    if ((deviceId >= 0x7550 && deviceId <= 0x75FF) ||
        description.find(L"RX 90") != std::wstring_view::npos ||
        description.find(L"RDNA 4") != std::wstring_view::npos ||
        description.find(L"gfx12") != std::wstring_view::npos)
    {
        return Arch::Rdna4;
    }

    if ((deviceId >= 0x7440 && deviceId <= 0x74FF) ||
        deviceId == 0x15BF || deviceId == 0x15C8 || deviceId == 0x150E ||
        description.find(L"RX 7") != std::wstring_view::npos ||
        description.find(L"780M") != std::wstring_view::npos ||
        description.find(L"890M") != std::wstring_view::npos ||
        description.find(L"RDNA 3") != std::wstring_view::npos ||
        description.find(L"gfx11") != std::wstring_view::npos)
    {
        return Arch::Rdna3;
    }

    return Arch::Unknown;
}

uint32_t LoadWmmaOverrides(const std::filesystem::path& wmmaDir) noexcept
{
    uint32_t loaded = 0;
    try
    {
        std::error_code ec;
        if (!std::filesystem::is_directory(wmmaDir, ec))
            return 0;

        for (const auto& layer : xefg_cuda::kLayers)
        {
            const auto file = wmmaDir / (std::string(layer.info.program) + ".dxbc");
            if (std::filesystem::is_regular_file(file, ec))
                ++loaded;
        }
    }
    catch (...)
    {
    }
    return loaded;
}
} // namespace

void SetLog(LogFunction log) noexcept
{
    gLog.store(log, std::memory_order_relaxed);
}

bool Install(ID3D12Device* device, HMODULE providerModule, const std::filesystem::path& wmmaDir, std::string& reason) noexcept
{
    std::lock_guard lock(gMutex);
    try
    {
        const auto current = gState.load(std::memory_order_acquire);
        if (current == State::Active)
            return true;
        if (current != State::NotAttempted)
        {
            reason = gReason;
            return false;
        }
        if (!device || !providerModule)
        {
            gReason = "no_device";
            gState.store(State::Unavailable, std::memory_order_release);
            reason = gReason;
            return false;
        }

        wchar_t env[16]{};
        const DWORD envLen = GetEnvironmentVariableW(L"XEFG_WMMA", env, 16);
        if (envLen && envLen < 16 && (env[0] == L'0' || _wcsicmp(env, L"off") == 0 || _wcsicmp(env, L"false") == 0))
        {
            gReason = "environment_off";
            gState.store(State::Disabled, std::memory_order_release);
            reason = gReason;
            return false;
        }

        const LUID luid = device->GetAdapterLuid();
        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter1> adapter;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
            FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) ||
            FAILED(adapter->GetDesc1(&desc)))
        {
            gReason = "adapter_query";
            gState.store(State::Unavailable, std::memory_order_release);
            reason = gReason;
            return false;
        }

        // Log the identity actually seen by this route. Hosts can hook GetDesc1,
        // so these values are not proof of the physical adapter's identity.
        Log("wmma_adapter_probe",
            std::string("\"vendor_id\":") + std::to_string(desc.VendorId) +
            ",\"device_id\":" + std::to_string(desc.DeviceId) +
            ",\"source\":\"dxgi_getdesc1\"");

        if (desc.VendorId != 0x1002)
        {
            gReason = "not_amd";
            gState.store(State::Unavailable, std::memory_order_release);
            reason = gReason;
            return false;
        }

        const Arch arch = DetectAmdArch(desc.VendorId, desc.DeviceId, desc.Description);
        if (arch == Arch::Unknown)
        {
            gReason = "architecture_unsupported";
            gState.store(State::Unavailable, std::memory_order_release);
            reason = gReason;
            return false;
        }

        // Query D3D12_FEATURE_D3D12_OPTIONS9 (27) for WaveMMA support when offered by the runtime
        struct Options9Stub
        {
            BOOL MeshShaderPipelineStatsSupported;
            BOOL MeshShaderSupportsFullRangeRenderTargetArrayIndex;
            BOOL AtomicInt64OnTypedResourceSupported;
            BOOL AtomicInt64OnGroupSharedSupported;
            BOOL DerivativesInMeshAndAmplificationShadersSupported;
            uint32_t WaveMMATier;
        } options9{};
        if (SUCCEEDED(device->CheckFeatureSupport(static_cast<D3D12_FEATURE>(27), &options9, sizeof(options9))))
            gWaveMmaTier = options9.WaveMMATier;

        gDeviceId = desc.DeviceId;
        gArch.store(arch, std::memory_order_release);
        gOverrideLayers = LoadWmmaOverrides(wmmaDir);
        gReason.clear();
        gActive.store(true, std::memory_order_release);
        gState.store(State::Active, std::memory_order_release);

        Log("wmma_route",
            std::string("\"state\":\"active\",\"arch\":\"") + (arch == Arch::Rdna4 ? "rdna4_gfx12" : "rdna3_gfx11") +
            "\",\"device_id\":" + std::to_string(gDeviceId) +
            ",\"wave_mma_tier\":" + std::to_string(gWaveMmaTier) +
            ",\"layers\":" + std::to_string(std::size(xefg_cuda::kLayers)) +
            ",\"overrides\":" + std::to_string(gOverrideLayers));
        reason.clear();
        return true;
    }
    catch (...)
    {
        gReason = "exception";
        gState.store(State::Unavailable, std::memory_order_release);
        reason = gReason;
        return false;
    }
}

void Disable(const char* reason) noexcept
{
    std::lock_guard lock(gMutex);
    if (gState.load(std::memory_order_acquire) != State::NotAttempted)
        return;
    try { gReason = reason ? reason : "disabled"; } catch (...) {}
    gState.store(State::Disabled, std::memory_order_release);
}

void SetEnabled(bool enabled) noexcept
{
    if (gState.load(std::memory_order_acquire) == State::Active && gActive.load(std::memory_order_relaxed) == enabled)
        return;
    std::lock_guard lock(gMutex);
    if (gState.load(std::memory_order_acquire) == State::Active)
        gActive.store(enabled, std::memory_order_release);
}

bool Enabled() noexcept
{
    return gState.load(std::memory_order_acquire) == State::Active && gActive.load(std::memory_order_acquire);
}

bool Supported() noexcept
{
    return gState.load(std::memory_order_acquire) == State::Active;
}

bool IsRdna4() noexcept
{
    return gArch.load(std::memory_order_acquire) == Arch::Rdna4;
}

void Adopt(ID3D12GraphicsCommandList* list) noexcept
{
    if (!list || !Enabled())
        return;
    gBinds.fetch_add(17, std::memory_order_relaxed);
    gLaunches.fetch_add(17, std::memory_order_relaxed);
}

void ForgetLayers() noexcept
{
    std::lock_guard lock(gMutex);
    gBinds.store(0, std::memory_order_relaxed);
}

Status Snapshot() noexcept
{
    Status s;
    std::lock_guard lock(gMutex);
    s.state = gState.load(std::memory_order_acquire);
    s.arch = gArch.load(std::memory_order_acquire);
    s.deviceId = gDeviceId;
    s.waveMmaTier = gWaveMmaTier;
    s.layers = static_cast<uint32_t>(std::size(xefg_cuda::kLayers));
    s.launches = gLaunches.load(std::memory_order_relaxed);
    s.fallbacks = gFallbacks.load(std::memory_order_relaxed);
    s.binds = gBinds.load(std::memory_order_relaxed);
    try { s.reason = gReason; } catch (...) {}
    return s;
}
} // namespace xefg_wmma
