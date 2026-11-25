#include "wmma_route.h"
#include "../AdapterIdentity.h"
#include "../Cuda/OverlayNative.h"
#include "../Cuda/cuda_layers.h"
#include <wrl/client.h>
#include <array>
#include <atomic>
#include <cwchar>
#include <iterator>
#include <mutex>
#include <string_view>

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
uint32_t gVendorId{};
uint32_t gDeviceId{};
uint32_t gDxgiVendorId{};
uint32_t gDxgiDeviceId{};
bool gDxgiSpoofed{false};
uint32_t gWaveMmaTier{};
std::string gGpuName;
std::string gProbeSource;
std::string gReason;

void Log(const char* stage, const std::string& fields) noexcept
{
    if (const auto fn = gLog.load(std::memory_order_relaxed))
        fn(stage, fields);
}

void MarkUnavailable(const std::string& why, const std::string& extra = {}) noexcept
{
    try
    {
        gReason = why;
        gState.store(State::Unavailable, std::memory_order_release);
        std::string fields = "\"state\":\"unavailable\",\"reason\":\"" + why + "\"";
        if (!extra.empty())
            fields += "," + extra;
        Log("wmma_route", fields);
    }
    catch (...)
    {
        gState.store(State::Unavailable, std::memory_order_release);
    }
}

// Identifies AMD RDNA 4 (gfx1200/gfx1201: Radeon RX 9070 XT / 9070 / 9060 XT, PCI DeviceIDs 0x7550..0x75FF)
// vs. AMD RDNA 3 (gfx1100..gfx1103: Radeon RX 7900 / 7800 / 7700 / 7600 / 780M / 890M, PCI DeviceIDs 0x7440..0x74FF, 0x15BF, 0x15C8, 0x150E).
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
        deviceId == 0x1506 || deviceId == 0x1586 || deviceId == 0x1681 ||
        description.find(L"RX 7") != std::wstring_view::npos ||
        description.find(L"780M") != std::wstring_view::npos ||
        description.find(L"880M") != std::wstring_view::npos ||
        description.find(L"890M") != std::wstring_view::npos ||
        description.find(L"8060S") != std::wstring_view::npos ||
        description.find(L"8050S") != std::wstring_view::npos ||
        description.find(L"RDNA 3") != std::wstring_view::npos ||
        description.find(L"gfx11") != std::wstring_view::npos)
    {
        return Arch::Rdna3;
    }

    return Arch::Unknown;
}

bool ImageOf(const void* address, std::array<wchar_t, MAX_PATH>& path, const wchar_t*& leaf, bool& system) noexcept
{
    HMODULE owner{};
    std::array<wchar_t, MAX_PATH> directory{};
    if (!address || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            static_cast<LPCWSTR>(address), &owner))
        return false;
    const DWORD length = GetModuleFileNameW(owner, path.data(), DWORD(path.size()));
    const UINT systemLength = GetSystemDirectoryW(directory.data(), UINT(directory.size()));
    if (!length || length >= path.size() || !systemLength || systemLength >= directory.size())
        return false;
    const wchar_t* slash = wcsrchr(path.data(), L'\\');
    if (!slash)
        return false;
    system = size_t(slash - path.data()) == systemLength && !_wcsnicmp(path.data(), directory.data(), systemLength);
    leaf = slash + 1;
    return true;
}

bool RuntimeDevice(ID3D12Device* device, std::string& owner) noexcept
{
    void** table = *reinterpret_cast<void***>(device);
    const std::array<const void*, 2> addresses{table, table ? table[0] : nullptr};
    for (const void* address : addresses)
    {
        std::array<wchar_t, MAX_PATH> path{};
        const wchar_t* leaf{};
        bool system{};
        if (!ImageOf(address, path, leaf, system))
        {
            owner = "not_an_image";
            return false;
        }
        if (!_wcsicmp(leaf, L"D3D12Core.dll") || (system && !_wcsicmp(leaf, L"d3d12.dll")))
            continue;
        owner.clear();
        for (; *leaf && owner.size() < 64; ++leaf)
            owner += *leaf >= 0x20 && *leaf < 0x7F && *leaf != L'"' && *leaf != L'\\' ? char(*leaf) : '?';
        return false;
    }
    return true;
}

bool ResolveNative(ID3D12Device* device, ComPtr<ID3D12Device>& native, std::string& owner, const char*& how) noexcept
{
    how = "runtime";
    if (RuntimeDevice(device, owner))
    {
        native = device;
        return true;
    }
    std::string ignored;
    ComPtr<ID3D12Device> candidate;
    if (single_overlay::native::Unwrap(device, candidate) && candidate && RuntimeDevice(candidate.Get(), ignored))
    {
        how = "unwrap";
        native = std::move(candidate);
        return true;
    }
    ComPtr<ID3D12Fence> fence;
    candidate.Reset();
    if (SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) &&
        SUCCEEDED(fence->GetDevice(IID_PPV_ARGS(&candidate))) && candidate && RuntimeDevice(candidate.Get(), ignored))
    {
        how = "fence";
        native = std::move(candidate);
        return true;
    }
    return false;
}
} // namespace

void SetLog(LogFunction log) noexcept
{
    gLog.store(log, std::memory_order_relaxed);
}

bool Install(ID3D12Device* device, HMODULE providerModule, const std::filesystem::path& /*wmmaDir*/, std::string& reason) noexcept
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
            MarkUnavailable("no_device");
            reason = gReason;
            return false;
        }

        wchar_t env[16]{};
        const DWORD envLen = GetEnvironmentVariableW(L"XEFG_WMMA", env, 16);
        if (envLen && envLen < 16 && (env[0] == L'0' || _wcsicmp(env, L"off") == 0 || _wcsicmp(env, L"false") == 0))
        {
            gReason = "environment_off";
            gState.store(State::Disabled, std::memory_order_release);
            Log("wmma_route", "\"state\":\"disabled\",\"reason\":\"environment_off\"");
            reason = gReason;
            return false;
        }

        std::string owner;
        const char* how{};
        ComPtr<ID3D12Device> native;
        if (!ResolveNative(device, native, owner, how))
        {
            Log("wmma_device", "\"owner\":\"" + owner + "\",\"native\":\"unresolved\"");
            MarkUnavailable("wrapped_device", "\"owner\":\"" + owner + "\"");
            reason = gReason;
            return false;
        }
        device = native.Get();

        const LUID luid = device->GetAdapterLuid();
        const auto probe = adapter_identity::Probe(luid);

        gVendorId = probe.vendorId;
        gDeviceId = probe.deviceId;
        gDxgiVendorId = probe.dxgiVendorId;
        gDxgiDeviceId = probe.dxgiDeviceId;
        gDxgiSpoofed = probe.dxgiSpoofed;
        gGpuName = adapter_identity::WideToAsciiJson(probe.description);
        gProbeSource = probe.source;

        const std::string umdAscii = adapter_identity::WideToAsciiJson(probe.umdLeaf, 64);

        Log("wmma_adapter_probe",
            "\"luid\":\"" + adapter_identity::FormatLuid(luid) +
            "\",\"vendor_id\":" + std::to_string(probe.vendorId) +
            ",\"vendor_hex\":\"" + adapter_identity::Hex16(probe.vendorId) +
            "\",\"vendor_name\":\"" + adapter_identity::VendorName(probe.vendorId) +
            "\",\"device_id\":" + std::to_string(probe.deviceId) +
            ",\"device_hex\":\"" + adapter_identity::Hex16(probe.deviceId) +
            "\",\"gpu\":\"" + gGpuName +
            "\",\"source\":\"" + probe.source +
            "\",\"dxgi_vendor_id\":" + std::to_string(probe.dxgiVendorId) +
            ",\"dxgi_device_id\":" + std::to_string(probe.dxgiDeviceId) +
            ",\"dxgi_spoofed\":" + (probe.dxgiSpoofed ? "true" : "false") +
            ",\"umd\":\"" + umdAscii +
            "\",\"probes\":\"" + probe.probeTrail + "\"");

        if (probe.vendorId == 0)
        {
            MarkUnavailable("adapter_query", "\"probes\":\"" + probe.probeTrail + "\"");
            reason = gReason;
            return false;
        }

        // Query D3D12_FEATURE_D3D12_OPTIONS9 (27) for WaveMMA support before architecture gating so diagnostics include it.
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

        const Arch arch = (probe.vendorId == 0x1002)
            ? DetectAmdArch(probe.vendorId, probe.deviceId, probe.description)
            : Arch::Unknown;

        if (probe.vendorId != 0x1002)
        {
            MarkUnavailable("not_amd",
                "\"vendor_hex\":\"" + adapter_identity::Hex16(probe.vendorId) +
                "\",\"device_hex\":\"" + adapter_identity::Hex16(probe.deviceId) +
                "\",\"gpu\":\"" + gGpuName +
                "\",\"source\":\"" + probe.source +
                "\",\"dxgi_spoofed\":" + (probe.dxgiSpoofed ? "true" : "false"));
            reason = gReason;
            return false;
        }

        if (arch == Arch::Unknown)
        {
            MarkUnavailable("architecture_unsupported",
                "\"vendor_hex\":\"0x1002\",\"device_hex\":\"" + adapter_identity::Hex16(probe.deviceId) +
                "\",\"gpu\":\"" + gGpuName +
                "\",\"wave_mma_tier\":" + std::to_string(gWaveMmaTier) +
                ",\"source\":\"" + probe.source +
                "\",\"umd\":\"" + umdAscii + "\"");
            reason = gReason;
            return false;
        }

        gArch.store(arch, std::memory_order_release);
        gReason.clear();
        gActive.store(true, std::memory_order_release);
        gState.store(State::Active, std::memory_order_release);

        Log("wmma_route",
            std::string("\"state\":\"active\",\"arch\":\"") + (arch == Arch::Rdna4 ? "rdna4_gfx12" : "rdna3_gfx11") +
            "\",\"vendor_hex\":\"0x1002\",\"device_id\":" + std::to_string(gDeviceId) +
            ",\"device_hex\":\"" + adapter_identity::Hex16(gDeviceId) +
            "\",\"gpu\":\"" + gGpuName +
            "\",\"source\":\"" + gProbeSource +
            "\",\"dxgi_spoofed\":" + (gDxgiSpoofed ? "true" : "false") +
            ",\"wave_mma_tier\":" + std::to_string(gWaveMmaTier) +
            ",\"d3d12_shared_tile_overrides\":false" +
            ",\"device\":\"" + how + "\"");
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

void Adopt(ID3D12GraphicsCommandList* /*list*/) noexcept
{
    // Software D3D12 cs_6_6 shared-tile DP4a overrides are intentionally disabled; D3D12 fallback and XMX Port run stock DXIL directly.
}

void ForgetLayers() noexcept
{
}

Status Snapshot() noexcept
{
    Status s;
    std::lock_guard lock(gMutex);
    s.state = gState.load(std::memory_order_acquire);
    s.arch = gArch.load(std::memory_order_acquire);
    s.vendorId = gVendorId;
    s.deviceId = gDeviceId;
    s.dxgiVendorId = gDxgiVendorId;
    s.dxgiDeviceId = gDxgiDeviceId;
    s.dxgiSpoofed = gDxgiSpoofed;
    s.waveMmaTier = gWaveMmaTier;
    s.layers = static_cast<uint32_t>(std::size(xefg_cuda::kLayers));
    s.overrideLayers = 0;
    s.launches = 0;
    s.fallbacks = 0;
    s.binds = 0;
    try
    {
        s.gpuName = gGpuName;
        s.probeSource = gProbeSource;
        s.reason = gReason;
    }
    catch (...)
    {
    }
    return s;
}
} // namespace xefg_wmma
