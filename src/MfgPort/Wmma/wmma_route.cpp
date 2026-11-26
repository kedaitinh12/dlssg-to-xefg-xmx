#include "wmma_route.h"
#include "../AdapterIdentity.h"
#include "../Cuda/OverlayNative.h"
#include "../Cuda/cuda_layers.h"
#include "../Sha256.h"
#include <wrl/client.h>
#include <array>
#include <atomic>
#include <cstring>
#include <cwchar>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string_view>
#include <unordered_map>
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
uint32_t gVendorId{};
uint32_t gDeviceId{};
uint32_t gDxgiVendorId{};
uint32_t gDxgiDeviceId{};
bool gDxgiSpoofed{false};
uint32_t gWaveMmaTier{};
std::string gGpuName;
std::string gProbeSource;
std::string gReason;
std::atomic<bool> gUseD3D12Dot4{false};
std::atomic<bool> gDot4Split{false};
std::atomic<uint32_t> gOverrideLayers{0};
std::atomic<uint64_t> gLaunches{0};
std::atomic<uint64_t> gFallbacks{0};
std::atomic<uint64_t> gBinds{0};

std::array<std::vector<uint8_t>, std::size(xefg_cuda::kLayers)> gOverrideBlobs{};

struct OverrideEntry
{
    ComPtr<ID3D12PipelineState> overridePso;
    uint8_t layerIdx{};
};
std::mutex gPsoMutex;
std::unordered_map<ID3D12PipelineState*, OverrideEntry> gPsoMap;

struct ActiveListLayer
{
    ID3D12PipelineState* stockPso{};
    ID3D12PipelineState* overridePso{};
    uint8_t layerIdx{};
};
using ResetFn = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*, ID3D12PipelineState*);
using DispatchFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
using SetPipelineStateFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12PipelineState*);

struct ListVtableOrig
{
    ResetFn reset{};
    DispatchFn dispatch{};
    SetPipelineStateFn setPso{};
};
std::mutex gListMutex;
std::unordered_map<void**, ListVtableOrig> gListVtables;
std::unordered_map<ID3D12GraphicsCommandList*, ActiveListLayer> gActiveListLayers;

constexpr int kListResetSlot = 10;
constexpr int kListDispatchSlot = 14;
constexpr int kListSetPipelineStateSlot = 25;

bool IsEnvOff(const wchar_t* name) noexcept
{
    wchar_t env[16]{};
    const DWORD len = GetEnvironmentVariableW(name, env, 16);
    return len > 0 && len < 16 &&
        (env[0] == L'0' || _wcsicmp(env, L"off") == 0 || _wcsicmp(env, L"false") == 0 || _wcsicmp(env, L"no") == 0);
}

bool IsEnvOn(const wchar_t* name) noexcept
{
    wchar_t env[16]{};
    const DWORD len = GetEnvironmentVariableW(name, env, 16);
    return len > 0 && len < 16 &&
        (env[0] == L'1' || _wcsicmp(env, L"on") == 0 || _wcsicmp(env, L"true") == 0 || _wcsicmp(env, L"yes") == 0);
}

std::vector<uint8_t> LoadDxbcFile(const std::filesystem::path& path) noexcept
{
    try
    {
        std::error_code ec;
        const auto sz = std::filesystem::file_size(path, ec);
        if (ec || sz < 32 || sz > 4 * 1024 * 1024)
            return {};
        std::ifstream in(path, std::ios::binary);
        if (!in)
            return {};
        std::vector<uint8_t> bytes(static_cast<size_t>(sz));
        if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(sz)))
            return {};
        if (std::memcmp(bytes.data(), "DXBC", 4) != 0)
            return {};
        return bytes;
    }
    catch (...)
    {
        return {};
    }
}

bool MatchesKnownLayerSize(size_t byteSize) noexcept
{
    for (const auto& l : xefg_cuda::kLayers)
    {
        if (l.bytes == byteSize || (l.ownBytes && l.ownBytes == byteSize))
            return true;
    }
    return false;
}

ListVtableOrig LookupListOrig(ID3D12GraphicsCommandList* self) noexcept
{
    if (!self)
        return {};
    void** vtable = *reinterpret_cast<void***>(self);
    std::lock_guard lock(gListMutex);
    const auto it = gListVtables.find(vtable);
    return it != gListVtables.end() ? it->second : ListVtableOrig{};
}

HRESULT STDMETHODCALLTYPE HookListReset(ID3D12GraphicsCommandList* self, ID3D12CommandAllocator* alloc, ID3D12PipelineState* initialState)
{
    const auto orig = LookupListOrig(self);
    {
        std::lock_guard lock(gListMutex);
        gActiveListLayers.erase(self);
    }
    const HRESULT hr = orig.reset ? orig.reset(self, alloc, initialState) : E_FAIL;
    if (SUCCEEDED(hr) && initialState && gUseD3D12Dot4.load(std::memory_order_acquire))
    {
        ID3D12PipelineState* ov = nullptr;
        uint8_t idx = 0;
        {
            std::lock_guard lock(gPsoMutex);
            const auto it = gPsoMap.find(initialState);
            if (it != gPsoMap.end() && it->second.overridePso)
            {
                ov = it->second.overridePso.Get();
                idx = it->second.layerIdx;
            }
        }
        if (ov && idx)
        {
            std::lock_guard lock(gListMutex);
            gActiveListLayers[self] = {initialState, ov, idx};
        }
    }
    return hr;
}

void STDMETHODCALLTYPE HookListSetPipelineState(ID3D12GraphicsCommandList* self, ID3D12PipelineState* pso)
{
    const auto orig = LookupListOrig(self);
    if (gUseD3D12Dot4.load(std::memory_order_acquire))
    {
        ID3D12PipelineState* ov = nullptr;
        uint8_t idx = 0;
        if (pso)
        {
            std::lock_guard lock(gPsoMutex);
            const auto it = gPsoMap.find(pso);
            if (it != gPsoMap.end() && it->second.overridePso)
            {
                ov = it->second.overridePso.Get();
                idx = it->second.layerIdx;
            }
        }
        std::lock_guard lock(gListMutex);
        if (ov && idx)
        {
            gActiveListLayers[self] = {pso, ov, idx};
            gBinds.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            gActiveListLayers.erase(self);
        }
    }
    if (orig.setPso)
        orig.setPso(self, pso);
}

void STDMETHODCALLTYPE HookListDispatch(ID3D12GraphicsCommandList* self, UINT x, UINT y, UINT z)
{
    const auto orig = LookupListOrig(self);
    if (!orig.dispatch)
        return;

    if (gUseD3D12Dot4.load(std::memory_order_acquire) && gActive.load(std::memory_order_acquire))
    {
        ActiveListLayer active{};
        {
            std::lock_guard lock(gListMutex);
            const auto it = gActiveListLayers.find(self);
            if (it != gActiveListLayers.end())
                active = it->second;
        }
        if (active.overridePso && active.layerIdx >= 1 && active.layerIdx <= std::size(xefg_cuda::kLayers))
        {
            const auto& layer = xefg_cuda::kLayers[active.layerIdx - 1];
            if (xefg_cuda::DispatchShapeOk(layer, x, y, z) && orig.setPso)
            {
                const UINT gx = (x * layer.intelW + layer.tileW - 1) / layer.tileW;
                const UINT gy = (y * layer.intelH + layer.tileH - 1) / layer.tileH;
                orig.setPso(self, active.overridePso);
                orig.dispatch(self, gx, gy, layer.gridZ);
                orig.setPso(self, active.stockPso);
                gLaunches.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            gFallbacks.fetch_add(1, std::memory_order_relaxed);
        }
    }
    orig.dispatch(self, x, y, z);
}

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

    if ((deviceId >= 0x73A0 && deviceId <= 0x73FF) ||
        deviceId == 0x163F || deviceId == 0x164D || deviceId == 0x164E ||
        description.find(L"RX 6") != std::wstring_view::npos ||
        description.find(L"680M") != std::wstring_view::npos ||
        description.find(L"660M") != std::wstring_view::npos ||
        description.find(L"610M") != std::wstring_view::npos ||
        description.find(L"Van Gogh") != std::wstring_view::npos ||
        description.find(L"Custom GPU 0405") != std::wstring_view::npos ||
        description.find(L"RDNA 2") != std::wstring_view::npos ||
        description.find(L"gfx103") != std::wstring_view::npos)
    {
        return Arch::Rdna2;
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

        bool useD3D12Dot4 = false;
        bool dot4Split = false;
        uint32_t loadedOverrides = 0;

        if (arch == Arch::Rdna2)
        {
            if (IsEnvOff(L"XEFG_RDNA2_DOT4"))
            {
                gArch.store(arch, std::memory_order_release);
                gReason = "rdna2_dot4_env_off";
                gState.store(State::Disabled, std::memory_order_release);
                Log("wmma_route", "\"state\":\"disabled\",\"arch\":\"rdna2_gfx103\",\"reason\":\"rdna2_dot4_env_off\"");
                reason = gReason;
                return false;
            }
            dot4Split = IsEnvOn(L"XEFG_WMMA_DOT4_SPLIT");
            const std::filesystem::path shaderDir = dot4Split ? (wmmaDir / L"dot4_split") : wmmaDir;
            for (size_t i = 0; i < std::size(xefg_cuda::kLayers); ++i)
            {
                const auto& l = xefg_cuda::kLayers[i];
                auto blob = LoadDxbcFile(shaderDir / (std::string(l.info.program) + ".dxbc"));
                if (blob.empty())
                    blob = LoadDxbcFile(shaderDir / (std::string(l.info.kernel) + ".dxbc"));
                if (!blob.empty())
                {
                    gOverrideBlobs[i] = std::move(blob);
                    ++loadedOverrides;
                }
            }
            if (loadedOverrides == 0)
            {
                gArch.store(arch, std::memory_order_release);
                MarkUnavailable(dot4Split ? "no_rdna2_dot4_split_shaders" : "no_rdna2_dot4_shaders",
                    "\"arch\":\"rdna2_gfx103\",\"dot4_split\":" + std::string(dot4Split ? "true" : "false"));
                reason = gReason;
                return false;
            }
            useD3D12Dot4 = true;
        }

        gArch.store(arch, std::memory_order_release);
        gUseD3D12Dot4.store(useD3D12Dot4, std::memory_order_release);
        gDot4Split.store(dot4Split, std::memory_order_release);
        gOverrideLayers.store(loadedOverrides, std::memory_order_release);
        gReason.clear();
        gActive.store(true, std::memory_order_release);
        gState.store(State::Active, std::memory_order_release);

        const char* archTag = arch == Arch::Rdna4 ? "rdna4_gfx12"
                            : arch == Arch::Rdna3 ? "rdna3_gfx11"
                                                  : "rdna2_gfx103";
        Log("wmma_route",
            std::string("\"state\":\"active\",\"arch\":\"") + archTag +
            "\",\"vendor_hex\":\"0x1002\",\"device_id\":" + std::to_string(gDeviceId) +
            ",\"device_hex\":\"" + adapter_identity::Hex16(gDeviceId) +
            "\",\"gpu\":\"" + gGpuName +
            "\",\"source\":\"" + gProbeSource +
            "\",\"dxgi_spoofed\":" + (gDxgiSpoofed ? "true" : "false") +
            ",\"wave_mma_tier\":" + std::to_string(gWaveMmaTier) +
            ",\"d3d12_shared_tile_overrides\":" + (useD3D12Dot4 ? "true" : "false") +
            ",\"dot4_split\":" + (dot4Split ? "true" : "false") +
            ",\"override_layers\":" + std::to_string(loadedOverrides) +
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

bool IsRdna2() noexcept
{
    return gArch.load(std::memory_order_acquire) == Arch::Rdna2;
}

bool UsesD3D12Dot4Override() noexcept
{
    return gState.load(std::memory_order_acquire) == State::Active && gUseD3D12Dot4.load(std::memory_order_acquire);
}

bool Dot4SplitActive() noexcept
{
    return gDot4Split.load(std::memory_order_acquire);
}

void OnCreateComputePipelineState(ID3D12Device* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc,
                                  REFIID iid, void* stockPsoObj, CreateComputePsoFn origCreate) noexcept
{
    if (!UsesD3D12Dot4Override() || !device || !desc || !stockPsoObj || !origCreate)
        return;
    if (!desc->CS.pShaderBytecode || !MatchesKnownLayerSize(desc->CS.BytecodeLength))
        return;
    try
    {
        sha256_util::Digest digest{};
        if (!sha256_util::Compute(desc->CS.pShaderBytecode, desc->CS.BytecodeLength, digest))
            return;
        const uint8_t layerIdx = xefg_cuda::LayerForSha256(sha256_util::LowerHex(digest));
        if (!layerIdx || layerIdx > std::size(xefg_cuda::kLayers))
            return;
        const auto& blob = gOverrideBlobs[layerIdx - 1];
        if (blob.empty())
            return;

        ComPtr<ID3D12PipelineState> stockPso;
        if (iid == __uuidof(ID3D12PipelineState))
        {
            stockPso = static_cast<ID3D12PipelineState*>(stockPsoObj);
        }
        else
        {
            auto* unk = static_cast<IUnknown*>(stockPsoObj);
            if (FAILED(unk->QueryInterface(IID_PPV_ARGS(&stockPso))) || !stockPso)
                return;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC ovDesc = *desc;
        ovDesc.CS.pShaderBytecode = blob.data();
        ovDesc.CS.BytecodeLength = blob.size();
        ovDesc.CachedPSO = {};
        ComPtr<ID3D12PipelineState> ovPso;
        if (FAILED(origCreate(device, &ovDesc, IID_PPV_ARGS(&ovPso))) || !ovPso)
            return;

        std::lock_guard lock(gPsoMutex);
        gPsoMap[stockPso.Get()] = {std::move(ovPso), layerIdx};
    }
    catch (...)
    {
    }
}

void Adopt(ID3D12GraphicsCommandList* list) noexcept
{
    if (!UsesD3D12Dot4Override() || !list)
        return;
    try
    {
        void** vtable = *reinterpret_cast<void***>(list);
        if (!vtable)
            return;
        std::lock_guard lock(gListMutex);
        if (gListVtables.find(vtable) != gListVtables.end())
            return;

        ListVtableOrig orig{
            reinterpret_cast<ResetFn>(vtable[kListResetSlot]),
            reinterpret_cast<DispatchFn>(vtable[kListDispatchSlot]),
            reinterpret_cast<SetPipelineStateFn>(vtable[kListSetPipelineStateSlot])
        };
        if (!orig.reset || !orig.dispatch || !orig.setPso)
            return;

        DWORD oldProt = 0;
        const size_t patchBytes = (kListSetPipelineStateSlot + 1) * sizeof(void*);
        if (!VirtualProtect(vtable, patchBytes, PAGE_EXECUTE_READWRITE, &oldProt))
            return;
        gListVtables.emplace(vtable, orig);
        vtable[kListResetSlot] = reinterpret_cast<void*>(&HookListReset);
        vtable[kListDispatchSlot] = reinterpret_cast<void*>(&HookListDispatch);
        vtable[kListSetPipelineStateSlot] = reinterpret_cast<void*>(&HookListSetPipelineState);
        DWORD ignored = 0;
        VirtualProtect(vtable, patchBytes, oldProt, &ignored);
    }
    catch (...)
    {
    }
}

void ForgetLayers() noexcept
{
    {
        std::lock_guard lock(gListMutex);
        gActiveListLayers.clear();
    }
    {
        std::lock_guard lock(gPsoMutex);
        gPsoMap.clear();
    }
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
    s.overrideLayers = gOverrideLayers.load(std::memory_order_relaxed);
    s.launches = gLaunches.load(std::memory_order_relaxed);
    s.fallbacks = gFallbacks.load(std::memory_order_relaxed);
    s.binds = gBinds.load(std::memory_order_relaxed);
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
