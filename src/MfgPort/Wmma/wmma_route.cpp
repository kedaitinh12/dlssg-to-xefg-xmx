#include "wmma_route.h"
#include "../AdapterIdentity.h"
#include "../Cuda/AdapterDiscovery.h"
#include "../Cuda/OverlayNative.h"
#include "../Cuda/cuda_layers.h"
#include "../Sha256.h"
#include <dxgi1_4.h>
#include <intrin.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace xefg_wmma
{
namespace
{
using Microsoft::WRL::ComPtr;

constexpr size_t kMaxDxbcBytes = 4 * 1024 * 1024;

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
uint32_t gOverrideLayers{};
std::string gGpuName;
std::string gProbeSource;
std::string gReason;

IUnknown* gDeviceIdentity{};
ID3D12Device* gDevice{};
uintptr_t gProviderBegin{}, gProviderEnd{};
uintptr_t gWrapperBegin{}, gWrapperEnd{};
uintptr_t gRuntimeBegin{}, gRuntimeEnd{};
bool gWrappedDevice{false};

std::atomic<uint64_t> gLaunches{}, gFallbacks{}, gBinds{};
std::atomic<uint32_t> gLayerCount{};
std::atomic<bool> gExecuteListLogged{};
std::array<std::atomic<bool>, std::size(xefg_cuda::kLayers)> gLayerSeen{};
std::array<std::vector<uint8_t>, std::size(xefg_cuda::kLayers)> gShaderBlobs{};

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

std::vector<uint8_t> LoadDxbcFile(const std::filesystem::path& path) noexcept
{
    try
    {
        std::error_code ec;
        const auto bytes = std::filesystem::file_size(path, ec);
        if (ec || bytes < 32 || bytes > kMaxDxbcBytes)
            return {};

        std::ifstream file(path, std::ios::binary);
        if (!file)
            return {};

        std::vector<uint8_t> buf(static_cast<size_t>(bytes));
        if (!file.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(bytes)))
            return {};

        // Verify DXBC container magic ('DXBC' = 0x43425844)
        if (buf[0] != 'D' || buf[1] != 'X' || buf[2] != 'B' || buf[3] != 'C')
            return {};

        return buf;
    }
    catch (...)
    {
        return {};
    }
}

uint32_t LoadWmmaOverrides(const std::filesystem::path& wmmaDir) noexcept
{
    uint32_t loaded = 0;
    try
    {
        std::error_code ec;
        for (auto& blob : gShaderBlobs)
            blob.clear();

        if (!std::filesystem::is_directory(wmmaDir, ec))
            return 0;

        for (size_t i = 0; i < std::size(xefg_cuda::kLayers); ++i)
        {
            const auto& layer = xefg_cuda::kLayers[i];
            auto blob = LoadDxbcFile(wmmaDir / (std::string(layer.info.program) + ".dxbc"));
            if (blob.empty())
                blob = LoadDxbcFile(wmmaDir / (std::string(layer.info.kernel) + ".dxbc"));

            if (!blob.empty())
            {
                gShaderBlobs[i] = std::move(blob);
                ++loaded;
            }
        }
    }
    catch (...)
    {
    }
    return loaded;
}

// ---- recognized pipeline states & companion WMMA replacement PSOs -------------------------------------
// Open addressing on the provider's base ID3D12PipelineState* pointer (inserted under gMutex, read lock-free).
struct HeldEntry
{
    ID3D12PipelineState* base{};
    ID3D12PipelineState* replacement{};
    uint8_t layer{};
};

constexpr size_t kSlots = 512;
std::array<std::atomic<void*>, kSlots> gKeys{};
std::array<std::atomic<uint8_t>, kSlots> gValues{};
std::array<std::atomic<ID3D12PipelineState*>, kSlots> gReplacements{};
std::vector<HeldEntry> gHeld;
std::atomic<bool> gAnyLayer{};

size_t Home(const void* key) noexcept
{
    return size_t(((reinterpret_cast<uintptr_t>(key) >> 4) * 0x9E3779B97F4A7C15ull) >> 55) & (kSlots - 1);
}

uint8_t Lookup(const void* key, ID3D12PipelineState*& replacementOut) noexcept
{
    replacementOut = nullptr;
    if (!key || !gAnyLayer.load(std::memory_order_acquire))
        return 0;

    for (size_t n = 0, i = Home(key); n < kSlots; ++n, i = (i + 1) & (kSlots - 1))
    {
        const void* found = gKeys[i].load(std::memory_order_acquire);
        if (found == key)
        {
            replacementOut = gReplacements[i].load(std::memory_order_relaxed);
            return gValues[i].load(std::memory_order_relaxed);
        }
        if (!found)
            return 0;
    }
    return 0;
}

bool Insert(void* key, uint8_t value, ID3D12PipelineState* replacement) noexcept // gMutex held
{
    if (gHeld.size() >= kSlots / 2)
        return false;

    for (size_t n = 0, i = Home(key); n < kSlots; ++n, i = (i + 1) & (kSlots - 1))
    {
        void* found = gKeys[i].load(std::memory_order_relaxed);
        if (found == key)
        {
            gReplacements[i].store(replacement, std::memory_order_relaxed);
            gValues[i].store(value, std::memory_order_relaxed);
            return true;
        }
        if (!found)
        {
            gReplacements[i].store(replacement, std::memory_order_relaxed);
            gValues[i].store(value, std::memory_order_relaxed);
            gKeys[i].store(key, std::memory_order_release);
            return true;
        }
    }
    return false;
}

void Prune() noexcept // gMutex held
{
    std::vector<HeldEntry> survivors;
    survivors.reserve(gHeld.size());
    for (const HeldEntry& entry : gHeld)
    {
        entry.base->AddRef();
        if (entry.base->Release() > 1)
        {
            survivors.push_back(entry);
        }
        else
        {
            entry.base->Release();
            if (entry.replacement)
                entry.replacement->Release();
        }
    }
    if (survivors.size() == gHeld.size())
        return;

    for (auto& key : gKeys) key.store(nullptr, std::memory_order_release);
    for (auto& val : gValues) val.store(0, std::memory_order_relaxed);
    for (auto& rep : gReplacements) rep.store(nullptr, std::memory_order_relaxed);

    gHeld.swap(survivors);
    gLayerCount.store(uint32_t(gHeld.size()), std::memory_order_relaxed);
    gAnyLayer.store(!gHeld.empty(), std::memory_order_release);
    for (const HeldEntry& entry : gHeld)
        Insert(entry.base, entry.layer, entry.replacement);
}

// ---- D3D12 device & command-list vtable patches ------------------------------------------------------
constexpr int kCreateComputePipelineState = 11, kCreateCommandList = 12, kCreatePipelineState = 47, kCreateCommandList1 = 51;
constexpr int kReset = 10, kDispatch = 14, kSetPipelineState = 25;
constexpr int kVtableSlots = 64;
constexpr std::array<int, 3> kListSlots{kReset, kDispatch, kSetPipelineState};
using Hook = std::pair<int, void*>;

bool HookSlot(void** slot, void*& original, void* hook, bool protect) noexcept
{
    void* expected = *slot;
    while (expected != hook)
    {
        original = expected;
        DWORD old{}, unused{};
        if (protect && !VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old))
            return false;
        void* prior = InterlockedCompareExchangePointer(slot, hook, expected);
        if (protect)
            VirtualProtect(slot, sizeof(void*), old, &unused);
        if (prior == expected)
            return true;
        expected = prior;
    }
    return true;
}

struct Patch
{
    void** vtable{};
    std::array<void*, kVtableSlots> original{};
};
constexpr size_t kMaxPatches = 8;
std::array<Patch, kMaxPatches> gDevicePatches{};
std::atomic<uint32_t> gDevicePatchCount{};

template <class F>
F DeviceOriginal(void* self, int slot) noexcept
{
    void** vtable = *static_cast<void***>(self);
    const uint32_t n = gDevicePatchCount.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n; ++i)
        if (gDevicePatches[i].vtable == vtable)
            return reinterpret_cast<F>(gDevicePatches[i].original[slot]);
    for (uint32_t i = 0; i < n; ++i)
        if (gDevicePatches[i].original[slot])
            return reinterpret_cast<F>(gDevicePatches[i].original[slot]);
    return nullptr;
}

bool ApplyDevice(void** vtable, std::span<const Hook> hooks) noexcept
{
    const uint32_t n = gDevicePatchCount.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n; ++i)
        if (gDevicePatches[i].vtable == vtable)
            return true;
    if (n >= kMaxPatches)
        return false;
    Patch& patch = gDevicePatches[n];
    patch.vtable = vtable;
    for (const auto& [slot, hook] : hooks)
    {
        if (vtable[slot] == hook)
            return false;
        patch.original[slot] = vtable[slot];
    }
    gDevicePatchCount.store(n + 1, std::memory_order_release);
    for (const auto& [slot, hook] : hooks)
        if (!HookSlot(&vtable[slot], patch.original[slot], hook, true))
            return false;
    return true;
}

struct ListTable
{
    std::atomic<void**> table{};
    std::array<void*, kVtableSlots> original{};
    std::atomic<bool> complete{};
    bool shared{};
    bool runtimeClass{};
    ID3D12GraphicsCommandList* list{};
    ID3D12CommandAllocator* allocator{};
};
constexpr size_t kTables = 256;
std::array<ListTable, kTables> gTables{};
size_t gTableKeys{};
std::array<std::array<std::atomic<void*>, kVtableSlots>, 2> gSharedOriginal{};
size_t TypeIndex(D3D12_COMMAND_LIST_TYPE type) noexcept { return type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? 1 : 0; }

size_t TableHome(void** table) noexcept
{
    return size_t(((reinterpret_cast<uintptr_t>(table) >> 3) * 0x9E3779B97F4A7C15ull) >> 56) & (kTables - 1);
}

ListTable* FindTable(void** table) noexcept
{
    if (!table)
        return nullptr;
    for (size_t n = 0, i = TableHome(table); n < kTables; ++n, i = (i + 1) & (kTables - 1))
    {
        void** key = gTables[i].table.load(std::memory_order_acquire);
        if (key == table)
            return &gTables[i];
        if (!key)
            return nullptr;
    }
    return nullptr;
}

const std::array<void*, kVtableSlots>& ListHooks() noexcept;

bool HookedTable(void** table) noexcept
{
    if (!table)
        return false;
    const auto& hooks = ListHooks();
    for (int slot : kListSlots)
        if (table[slot] != hooks[slot])
            return false;
    return true;
}

struct Recording
{
    void* list{};
    void** table{};
    const ListTable* tableRecord{};
    bool tableHooked{};
    ID3D12PipelineState* lastState{};
    ID3D12PipelineState* wmmaPso{};
    uint8_t layer{};
};
thread_local Recording tRecording;

void InitRecording(Recording& r, void* list) noexcept
{
    void** table = *static_cast<void***>(list);
    const ListTable* rec = FindTable(table);
    r = Recording{list, table, rec, rec ? false : HookedTable(table)};
}

Recording& Track(void* list) noexcept
{
    auto& r = tRecording;
    void** table = *static_cast<void***>(list);
    if (r.list != list || r.table != table)
        InitRecording(r, list);
    return r;
}

template <class F>
F ListOriginal(void* self, int slot) noexcept
{
    const auto& r = tRecording;
    if (r.list == self && r.tableRecord)
        return reinterpret_cast<F>(r.tableRecord->original[slot]);
    if (const ListTable* table = FindTable(*static_cast<void***>(self)))
        return reinterpret_cast<F>(table->original[slot]);
    const size_t type = TypeIndex(static_cast<ID3D12GraphicsCommandList*>(self)->GetType());
    void* original = gSharedOriginal[type][slot].load(std::memory_order_acquire);
    if (!original)
        original = gSharedOriginal[1 - type][slot].load(std::memory_order_acquire);
    return reinterpret_cast<F>(original);
}

using ResetFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*, ID3D12PipelineState*);
using DispatchFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
using SetPipelineStateFn = void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12PipelineState*);
using CreateComputeFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID, void**);
using CreateListFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator*,
    ID3D12PipelineState*, REFIID, void**);
using CreateList1Fn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device4*, UINT, D3D12_COMMAND_LIST_TYPE, D3D12_COMMAND_LIST_FLAGS, REFIID, void**);

HRESULT STDMETHODCALLTYPE HookReset(ID3D12GraphicsCommandList* self, ID3D12CommandAllocator* allocator, ID3D12PipelineState* initial)
{
    const HRESULT result = ListOriginal<ResetFn>(self, kReset)(self, allocator, initial);
    if (gActive.load(std::memory_order_relaxed))
    {
        auto& r = tRecording;
        InitRecording(r, self);
        if (SUCCEEDED(result))
        {
            r.lastState = initial;
            r.layer = Lookup(initial, r.wmmaPso);
            if (r.layer)
                gBinds.fetch_add(1, std::memory_order_relaxed);
        }
    }
    else if (tRecording.list == self)
    {
        tRecording.layer = 0;
        tRecording.wmmaPso = nullptr;
        tRecording.lastState = nullptr;
    }
    return result;
}

void STDMETHODCALLTYPE HookSetPipelineState(ID3D12GraphicsCommandList* self, ID3D12PipelineState* state)
{
    if (gActive.load(std::memory_order_relaxed))
    {
        auto& r = Track(self);
        if (r.lastState != state || !state)
        {
            r.lastState = state;
            r.layer = Lookup(state, r.wmmaPso);
        }
        if (r.layer)
            gBinds.fetch_add(1, std::memory_order_relaxed);
    }
    ListOriginal<SetPipelineStateFn>(self, kSetPipelineState)(self, state);
}

enum Reason : uint32_t { kShape = 1, kNoOverride = 2 };
std::atomic<uint32_t> gReasonsLogged{};

void Fallback(Reason reason, const char* text, const xefg_cuda::Layer& layer) noexcept
{
    gFallbacks.fetch_add(1, std::memory_order_relaxed);
    if (!(gReasonsLogged.fetch_or(reason) & reason))
    {
        try
        {
            Log("wmma_dispatch_fallback",
                std::string("\"reason\":\"") + text + "\",\"program\":\"" + layer.info.program + "\"");
        }
        catch (...)
        {
        }
    }
}

void STDMETHODCALLTYPE HookDispatch(ID3D12GraphicsCommandList* self, UINT x, UINT y, UINT z)
{
    if (gActive.load(std::memory_order_acquire))
    {
    auto& r = tRecording;
    if (r.list == self && r.layer && r.table == *reinterpret_cast<void***>(self))
    {
        if ((r.tableRecord && r.tableRecord->complete.load(std::memory_order_acquire)) || (!r.tableRecord && r.tableHooked))
        {
            const auto& layer = xefg_cuda::kLayers[r.layer - 1];
            if (!xefg_cuda::DispatchShapeOk(layer, x, y, z))
            {
                Fallback(kShape, "dispatch_shape", layer);
            }
            else if (!r.wmmaPso)
            {
                Fallback(kNoOverride, "no_wmma_pso", layer);
            }
            else
            {
                const UINT limitW = x * layer.intelW;
                const UINT limitH = y * layer.intelH;
                const UINT gridX  = (limitW + layer.tileW - 1) / layer.tileW;
                const UINT gridY  = (limitH + layer.tileH - 1) / layer.tileH;
                const auto setPso = ListOriginal<SetPipelineStateFn>(self, kSetPipelineState);
                const auto disp   = ListOriginal<DispatchFn>(self, kDispatch);
                setPso(self, r.wmmaPso);
                disp(self, gridX, gridY, layer.gridZ);
                setPso(self, r.lastState);
                gLaunches.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    }
    }
    ListOriginal<DispatchFn>(self, kDispatch)(self, x, y, z);
}

const std::array<void*, kVtableSlots>& ListHooks() noexcept
{
    static const auto hooks = [] {
        std::array<void*, kVtableSlots> h{};
        h[kReset] = reinterpret_cast<void*>(&HookReset);
        h[kDispatch] = reinterpret_cast<void*>(&HookDispatch);
        h[kSetPipelineState] = reinterpret_cast<void*>(&HookSetPipelineState);
        return h;
    }();
    return hooks;
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

bool SystemRuntimeImage(const void* address) noexcept
{
    std::array<wchar_t, MAX_PATH> path{};
    const wchar_t* leaf{};
    bool system{};
    return ImageOf(address, path, leaf, system) && system && (!_wcsicmp(leaf, L"D3D12Core.dll") || !_wcsicmp(leaf, L"d3d12.dll"));
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

bool RuntimeClassTable(void** table) noexcept // gMutex held
{
    const auto inRuntime = [](const void* address) {
        const auto at = reinterpret_cast<uintptr_t>(address);
        return at >= gRuntimeBegin && at < gRuntimeEnd;
    };
    if (!gRuntimeBegin || !inRuntime(table) || !inRuntime(table + kSetPipelineState) || !inRuntime(table[0]))
        return false;
    const auto& hooks = ListHooks();
    for (int slot : kListSlots)
        if (table[slot] != hooks[slot] && !inRuntime(table[slot]))
            return false;
    return true;
}

const char* AdoptList(ID3D12GraphicsCommandList* list, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* temporary) noexcept // gMutex held
{
    void** table = *reinterpret_cast<void***>(list);
    constexpr size_t needed = (kSetPipelineState + 1) * sizeof(void*);
    MEMORY_BASIC_INFORMATION memory{};
    if (!table || reinterpret_cast<uintptr_t>(table) % alignof(void*) ||
        VirtualQuery(table, &memory, sizeof(memory)) != sizeof(memory) ||
        memory.State != MEM_COMMIT || memory.RegionSize < needed ||
        reinterpret_cast<uintptr_t>(table) - reinterpret_cast<uintptr_t>(memory.BaseAddress) > memory.RegionSize - needed)
        return "table_memory";

    const auto at = reinterpret_cast<uintptr_t>(table), object = reinterpret_cast<uintptr_t>(list);
    const bool system = memory.Type == MEM_IMAGE && SystemRuntimeImage(table);
    const bool runtimeClass = !system && memory.Type == MEM_IMAGE && RuntimeClassTable(table);
    const bool wrapperClass = !system && !runtimeClass && memory.Type == MEM_IMAGE &&
        at >= gWrapperBegin && at < gWrapperEnd;
    const bool shared = system || runtimeClass || wrapperClass;
    const bool own = memory.Type == MEM_PRIVATE && memory.Protect == PAGE_READWRITE && at > object && at - object < 0x10000;
    if (!shared && !own)
        return memory.Type == MEM_IMAGE ? "table_runtime_image" : "table_unknown";

    const bool writable = memory.Protect == PAGE_READWRITE || memory.Protect == PAGE_EXECUTE_READWRITE;
    ListTable* record = FindTable(table);
    if (record && record->complete.load(std::memory_order_acquire))
        return shared && record->shared ? nullptr : "table_in_use";
    if (record && record->list)
        return "table_in_use";
    if (!record)
    {
        if (gTableKeys >= kTables / 2)
            return "table_capacity";
        for (size_t n = 0, i = TableHome(table); n < kTables && !record; ++n, i = (i + 1) & (kTables - 1))
            if (!gTables[i].table.load(std::memory_order_relaxed))
                record = &gTables[i];
        if (!record)
            return "table_capacity";
    }

    const auto& hooks = ListHooks();
    auto& sharedOriginal = gSharedOriginal[TypeIndex(type)];
    std::array<void*, kVtableSlots> original{};
    for (int slot : kListSlots)
    {
        void* current = table[slot];
        if (current == hooks[slot])
            current = sharedOriginal[slot].load(std::memory_order_acquire);
        if (!current)
            return "table_original";
        original[slot] = current;
    }
    if (shared)
    {
        for (int slot : kListSlots)
        {
            void* expected{};
            sharedOriginal[slot].compare_exchange_strong(expected, original[slot], std::memory_order_acq_rel);
        }
    }
    record->original = original;
    record->shared = shared;
    record->runtimeClass = runtimeClass;
    if (!record->table.load(std::memory_order_relaxed))
    {
        record->table.store(table, std::memory_order_release);
        ++gTableKeys;
    }
    if (!shared)
    {
        list->AddRef();
        record->list = list;
        if (temporary)
        {
            temporary->AddRef();
            record->allocator = temporary;
        }
    }
    for (int slot : kListSlots)
        if (!HookSlot(&table[slot], record->original[slot], hooks[slot], !writable))
            return "table_exchange";
    record->complete.store(true, std::memory_order_release);
    return nullptr;
}

bool Retire(ListTable& record) noexcept // gMutex held
{
    if (!record.list)
        return true;
    record.complete.store(false, std::memory_order_release);
    void** table = record.table.load(std::memory_order_relaxed);
    const auto& hooks = ListHooks();
    bool restored = true;
    for (int slot : kListSlots)
    {
        InterlockedCompareExchangePointer(&table[slot], record.original[slot], hooks[slot]);
        restored = restored && table[slot] == record.original[slot];
    }
    if (!restored)
        return false;
    record.list->Release();
    record.list = nullptr;
    if (record.allocator)
        record.allocator->Release();
    record.allocator = nullptr;
    return true;
}

void RetireDropped() noexcept // gMutex held
{
    for (auto& record : gTables)
    {
        if (record.list)
        {
            record.list->AddRef();
            if (record.list->Release() == 1)
                Retire(record);
        }
    }
}

bool Routable(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type) noexcept
{
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        return false;
    ComPtr<IUnknown> identity;
    return SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&identity))) && identity.Get() == gDeviceIdentity;
}

void AdoptCreated(ID3D12GraphicsCommandList* list, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* temporary) noexcept
{
    std::lock_guard lock(gMutex);
    RetireDropped();
    AdoptList(list, type, temporary);
}

bool FromProvider(const void* address) noexcept
{
    const auto at = reinterpret_cast<uintptr_t>(address);
    return at >= gProviderBegin && at < gProviderEnd;
}

bool FromWrapper(const void* address) noexcept
{
    const auto at = reinterpret_cast<uintptr_t>(address);
    return at >= gWrapperBegin && at < gWrapperEnd;
}

bool AcceptedCaller(const void* caller) noexcept;

bool ProviderCall(const void* caller) noexcept
{
    if (FromProvider(caller))
        return true;
    if (!FromWrapper(caller))
        return gWrappedDevice && gWrapperBegin == 0 && AcceptedCaller(caller);
    std::array<void*, 16> frames{};
    const USHORT count = RtlCaptureStackBackTrace(0, USHORT(frames.size()), frames.data(), nullptr);
    size_t i = 0;
    while (i < count && !FromWrapper(frames[i])) ++i;
    while (i < count && FromWrapper(frames[i])) ++i;
    return i < count && FromProvider(frames[i]);
}

bool AcceptedCaller(const void* caller) noexcept
{
    if (FromProvider(caller))
        return true;
    std::array<void*, 32> frames{};
    const USHORT count = RtlCaptureStackBackTrace(0, USHORT(frames.size()), frames.data(), nullptr);
    for (USHORT i = 0; i < count; ++i)
        if (FromProvider(frames[i]))
            return true;
    return false;
}

HRESULT STDMETHODCALLTYPE HookCreateCommandList(ID3D12Device* self, UINT mask, D3D12_COMMAND_LIST_TYPE type,
    ID3D12CommandAllocator* allocator, ID3D12PipelineState* initial, REFIID iid, void** out)
{
    void* caller = _ReturnAddress();
    if (out && allocator && ProviderCall(caller) && Routable(self, type))
    {
        ComPtr<ID3D12Device4> device4;
        ComPtr<ID3D12GraphicsCommandList> list;
        CreateList1Fn create1{};
        if (SUCCEEDED(self->QueryInterface(IID_PPV_ARGS(&device4))) &&
            (create1 = DeviceOriginal<CreateList1Fn>(device4.Get(), kCreateCommandList1)) != nullptr &&
            SUCCEEDED(create1(device4.Get(), mask, type, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&list))))
        {
            if (SUCCEEDED(list->Reset(allocator, initial)))
            {
                void* requested{};
                if (SUCCEEDED(list->QueryInterface(iid, &requested)))
                {
                    AdoptCreated(list.Get(), type, nullptr);
                    auto& r = tRecording;
                    InitRecording(r, list.Get());
                    r.lastState = initial;
                    r.layer = Lookup(initial, r.wmmaPso);
                    if (r.layer)
                        gBinds.fetch_add(1, std::memory_order_relaxed);
                    *out = requested;
                    return S_OK;
                }
                list->Close();
            }
        }
    }
    return DeviceOriginal<CreateListFn>(self, kCreateCommandList)(self, mask, type, allocator, initial, iid, out);
}

HRESULT STDMETHODCALLTYPE HookCreateCommandList1(ID3D12Device4* self, UINT mask, D3D12_COMMAND_LIST_TYPE type,
    D3D12_COMMAND_LIST_FLAGS flags, REFIID iid, void** out)
{
    void* caller = _ReturnAddress();
    const HRESULT result = DeviceOriginal<CreateList1Fn>(self, kCreateCommandList1)(self, mask, type, flags, iid, out);
    if (SUCCEEDED(result) && out && *out && ProviderCall(caller) && Routable(self, type))
    {
        ComPtr<ID3D12GraphicsCommandList> list;
        ComPtr<ID3D12CommandAllocator> temporary;
        if (SUCCEEDED(static_cast<IUnknown*>(*out)->QueryInterface(IID_PPV_ARGS(&list))) &&
            SUCCEEDED(self->CreateCommandAllocator(type, IID_PPV_ARGS(&temporary))) &&
            SUCCEEDED(list->Reset(temporary.Get(), nullptr)) && SUCCEEDED(list->Close()))
        {
            AdoptCreated(list.Get(), type, temporary.Get());
        }
    }
    return result;
}

void Recognize(ID3D12Device* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc, REFIID iid, void* created, const void* caller) noexcept
{
    try
    {
        const SIZE_T length = desc.CS.BytecodeLength;
        if (!desc.CS.pShaderBytecode || length < 32 || length > 1000000 ||
            std::none_of(std::begin(xefg_cuda::kLayers), std::end(xefg_cuda::kLayers),
                [length](const xefg_cuda::Layer& layer) { return length == layer.bytes || length == layer.ownBytes; }))
        {
            return;
        }

        sha256_util::Digest digest{};
        if (!sha256_util::Compute(desc.CS.pShaderBytecode, length, digest))
            return;
        const auto hash = sha256_util::LowerHex(digest);
        const uint8_t layerIdx = xefg_cuda::LayerForSha256(hash);
        if (!layerIdx)
            return;

        ComPtr<IUnknown> identity;
        if (FAILED(device->QueryInterface(IID_PPV_ARGS(&identity))) || identity.Get() != gDeviceIdentity)
            return;
        if (!AcceptedCaller(caller))
            return;

        ComPtr<ID3D12PipelineState> baseState;
        if (FAILED(static_cast<IUnknown*>(created)->QueryInterface(IID_PPV_ARGS(&baseState))))
            return;
        if (iid != __uuidof(ID3D12PipelineState) && baseState.Get() != created)
            return;

        const size_t i = layerIdx - 1;
        ComPtr<ID3D12PipelineState> wmmaPso;
        if (!gShaderBlobs[i].empty() && desc.pRootSignature)
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC wmmaDesc = desc;
            wmmaDesc.CS.pShaderBytecode = gShaderBlobs[i].data();
            wmmaDesc.CS.BytecodeLength = gShaderBlobs[i].size();
            wmmaDesc.CachedPSO = {};
            if (const auto createCompute = DeviceOriginal<CreateComputeFn>(device, kCreateComputePipelineState))
            {
                const HRESULT hr = createCompute(device, &wmmaDesc, IID_PPV_ARGS(&wmmaPso));
                if (FAILED(hr))
                {
                    Log("wmma_pso_failed",
                        std::string("\"program\":\"") + xefg_cuda::kLayers[i].info.program +
                        "\",\"kernel\":\"" + xefg_cuda::kLayers[i].info.kernel +
                        "\",\"hr\":\"" + adapter_identity::Hex32(static_cast<uint32_t>(hr)) +
                        "\",\"bytes\":" + std::to_string(gShaderBlobs[i].size()));
                }
            }
        }

        std::lock_guard lock(gMutex);
        if (!Insert(baseState.Get(), layerIdx, wmmaPso.Get()))
        {
            Prune();
            if (!Insert(baseState.Get(), layerIdx, wmmaPso.Get()))
                return;
        }

        ID3D12PipelineState* repRaw = wmmaPso.Detach();
        gHeld.push_back(HeldEntry{baseState.Detach(), repRaw, layerIdx});
        gAnyLayer.store(true, std::memory_order_release);
        gLayerCount.store(uint32_t(gHeld.size()), std::memory_order_relaxed);

        if (!gLayerSeen[i].exchange(true) && gActive.load(std::memory_order_relaxed))
        {
            Log("wmma_layer",
                std::string("\"program\":\"") + xefg_cuda::kLayers[i].info.program +
                "\",\"kernel\":\"" + xefg_cuda::kLayers[i].info.kernel +
                "\",\"override_pso\":" + (repRaw ? "true" : "false"));
        }
    }
    catch (...)
    {
    }
}

HRESULT STDMETHODCALLTYPE HookCreateComputePipelineState(ID3D12Device* self, const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, REFIID iid, void** out)
{
    void* caller = _ReturnAddress();
    const HRESULT result = DeviceOriginal<CreateComputeFn>(self, kCreateComputePipelineState)(self, desc, iid, out);
    if (SUCCEEDED(result) && desc && out && *out)
        Recognize(self, *desc, iid, *out, caller);
    return result;
}

template <class Inner>
constexpr size_t InnerOffset()
{
    return (sizeof(UINT) + alignof(Inner) - 1) / alignof(Inner) * alignof(Inner);
}

template <class Inner>
constexpr size_t SubobjectSize()
{
    constexpr size_t align = (std::max)(alignof(Inner), alignof(void*));
    return (InnerOffset<Inner>() + sizeof(Inner) + align - 1) / align * align;
}

bool StreamComputeDesc(const D3D12_PIPELINE_STATE_STREAM_DESC& desc, D3D12_COMPUTE_PIPELINE_STATE_DESC& outDesc) noexcept
{
    const auto* stream = static_cast<const unsigned char*>(desc.pPipelineStateSubobjectStream);
    if (!stream || reinterpret_cast<uintptr_t>(stream) % alignof(void*))
        return false;

    outDesc = {};
    bool hasCs = false;
    for (size_t at = 0; at + sizeof(UINT) <= desc.SizeInBytes;)
    {
        size_t size{};
        const UINT type = *reinterpret_cast<const UINT*>(stream + at);
        switch (type)
        {
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS:
            if (at + SubobjectSize<D3D12_SHADER_BYTECODE>() > desc.SizeInBytes)
                return false;
            outDesc.CS = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + at + InnerOffset<D3D12_SHADER_BYTECODE>());
            size = SubobjectSize<D3D12_SHADER_BYTECODE>();
            hasCs = true;
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE:
            if (at + SubobjectSize<ID3D12RootSignature*>() > desc.SizeInBytes)
                return false;
            outDesc.pRootSignature = *reinterpret_cast<ID3D12RootSignature* const*>(stream + at + InnerOffset<ID3D12RootSignature*>());
            size = SubobjectSize<ID3D12RootSignature*>();
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK:
            if (at + SubobjectSize<UINT>() > desc.SizeInBytes)
                return false;
            outDesc.NodeMask = *reinterpret_cast<const UINT*>(stream + at + InnerOffset<UINT>());
            size = SubobjectSize<UINT>();
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO:
            size = SubobjectSize<D3D12_CACHED_PIPELINE_STATE>();
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS:
            if (at + SubobjectSize<D3D12_PIPELINE_STATE_FLAGS>() > desc.SizeInBytes)
                return false;
            outDesc.Flags = *reinterpret_cast<const D3D12_PIPELINE_STATE_FLAGS*>(stream + at + InnerOffset<D3D12_PIPELINE_STATE_FLAGS>());
            size = SubobjectSize<D3D12_PIPELINE_STATE_FLAGS>();
            break;
        default:
            return false;
        }
        at += size;
    }
    return hasCs;
}

using CreateStreamFn = HRESULT (STDMETHODCALLTYPE*)(ID3D12Device2*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
HRESULT STDMETHODCALLTYPE HookCreatePipelineState(ID3D12Device2* self, const D3D12_PIPELINE_STATE_STREAM_DESC* desc, REFIID iid, void** out)
{
    void* caller = _ReturnAddress();
    const HRESULT result = DeviceOriginal<CreateStreamFn>(self, kCreatePipelineState)(self, desc, iid, out);
    if (SUCCEEDED(result) && desc && out && *out)
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC compute{};
        if (StreamComputeDesc(*desc, compute))
            Recognize(self, compute, iid, *out, caller);
    }
    return result;
}

template <class Interface>
void AddDeviceVtable(ID3D12Device* device, int level, std::vector<std::pair<void**, int>>& out)
{
    ComPtr<Interface> view;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&view))))
        return;
    void** vtable = *reinterpret_cast<void***>(view.Get());
    for (auto& entry : out)
    {
        if (entry.first == vtable)
        {
            entry.second = (std::max)(entry.second, level);
            return;
        }
    }
    out.emplace_back(vtable, level);
}

bool PatchDevice(ID3D12Device* device, std::string& reason)
{
    std::vector<std::pair<void**, int>> vtables;
    AddDeviceVtable<ID3D12Device>(device, 0, vtables);
    AddDeviceVtable<ID3D12Device1>(device, 1, vtables);
    AddDeviceVtable<ID3D12Device2>(device, 2, vtables);
    AddDeviceVtable<ID3D12Device3>(device, 3, vtables);
    AddDeviceVtable<ID3D12Device4>(device, 4, vtables);
    AddDeviceVtable<ID3D12Device5>(device, 5, vtables);
    AddDeviceVtable<ID3D12Device6>(device, 6, vtables);
    AddDeviceVtable<ID3D12Device7>(device, 7, vtables);
    AddDeviceVtable<ID3D12Device8>(device, 8, vtables);
    AddDeviceVtable<ID3D12Device9>(device, 9, vtables);
    AddDeviceVtable<ID3D12Device10>(device, 10, vtables);
    if (std::none_of(vtables.begin(), vtables.end(), [](const auto& entry) { return entry.second >= 4; }))
    {
        reason = "device4";
        return false;
    }

    const Hook hooks[] = {
        {kCreateComputePipelineState, reinterpret_cast<void*>(&HookCreateComputePipelineState)},
        {kCreateCommandList, reinterpret_cast<void*>(&HookCreateCommandList)},
        {kCreatePipelineState, reinterpret_cast<void*>(&HookCreatePipelineState)},
        {kCreateCommandList1, reinterpret_cast<void*>(&HookCreateCommandList1)}
    };
    for (const auto& [vtable, level] : vtables)
    {
        if (!ApplyDevice(vtable, std::span<const Hook>(hooks, level >= 4 ? 4 : level >= 2 ? 3 : 2)))
        {
            reason = "device_patch";
            return false;
        }
    }
    return true;
}

bool ProviderRange(HMODULE module, uintptr_t& begin, uintptr_t& end)
{
    if (!module)
        return false;
    const auto base = reinterpret_cast<const unsigned char*>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return false;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        return false;
    begin = reinterpret_cast<uintptr_t>(base);
    end = begin + nt->OptionalHeader.SizeOfImage;
    return true;
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
        gWrappedDevice = (native.Get() != device);
        if (gWrappedDevice)
        {
            void** wrapperVtable = *reinterpret_cast<void***>(device);
            HMODULE wrapper{};
            bool pinned = false;
            if (wrapperVtable &&
                GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                    reinterpret_cast<LPCWSTR>(wrapperVtable), &wrapper) &&
                ProviderRange(wrapper, gWrapperBegin, gWrapperEnd))
            {
                pinned = true;
            }
            else if (wrapperVtable && wrapperVtable[0] &&
                GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                    reinterpret_cast<LPCWSTR>(wrapperVtable[0]), &wrapper) &&
                ProviderRange(wrapper, gWrapperBegin, gWrapperEnd))
            {
                pinned = true;
            }
            else
            {
                gWrapperBegin = gWrapperEnd = 0;
            }
            Log("wmma_device",
                "\"owner\":\"" + owner + "\",\"native\":\"" + how +
                "\",\"wrapper_image\":\"" + (pinned ? "pinned" : "dynamic_vtable") + "\"");
        }
        device = native.Get();
        {
            HMODULE runtime{};
            if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                    static_cast<LPCWSTR>(*reinterpret_cast<void**>(device)), &runtime) ||
                !ProviderRange(runtime, gRuntimeBegin, gRuntimeEnd))
            {
                gRuntimeBegin = gRuntimeEnd = 0;
            }
        }

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

        const Arch arch = DetectAmdArch(probe.vendorId, probe.deviceId, probe.description);
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

        if (!ProviderRange(providerModule, gProviderBegin, gProviderEnd))
        {
            MarkUnavailable("provider_image");
            reason = gReason;
            return false;
        }

        ComPtr<IUnknown> identity;
        if (FAILED(device->QueryInterface(IID_PPV_ARGS(&identity))))
        {
            MarkUnavailable("device_identity");
            reason = gReason;
            return false;
        }

        gArch.store(arch, std::memory_order_release);
        std::error_code dirEc;
        const bool wmmaDirExists = std::filesystem::is_directory(wmmaDir, dirEc);
        gOverrideLayers = LoadWmmaOverrides(wmmaDir);
        const size_t expectedLayers = std::size(xefg_cuda::kLayers);
        Log("wmma_overrides",
            "\"dir_exists\":" + std::string(wmmaDirExists ? "true" : "false") +
            ",\"loaded\":" + std::to_string(gOverrideLayers) +
            ",\"expected\":" + std::to_string(expectedLayers) +
            ",\"missing\":" + std::to_string(expectedLayers - gOverrideLayers));

        gDeviceIdentity = identity.Detach();
        gDevice = device;
        gDevice->AddRef();

        std::string patchFailure;
        if (!PatchDevice(device, patchFailure))
        {
            MarkUnavailable(patchFailure,
                "\"device_hex\":\"" + adapter_identity::Hex16(gDeviceId) +
                "\",\"gpu\":\"" + gGpuName + "\"");
            reason = gReason;
            return false;
        }

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
            ",\"layers\":" + std::to_string(expectedLayers) +
            ",\"overrides\":" + std::to_string(gOverrideLayers) +
            ",\"device_vtables\":" + std::to_string(gDevicePatchCount.load()) +
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

void Adopt(ID3D12GraphicsCommandList* list) noexcept
{
    if (!list || gState.load(std::memory_order_acquire) != State::Active)
        return;

    void** fastTable = *reinterpret_cast<void***>(list);
    if (const ListTable* record = FindTable(fastTable); record && record->complete.load(std::memory_order_acquire))
        return;

    try
    {
        std::lock_guard lock(gMutex);
        void** table = *reinterpret_cast<void***>(list);
        if (const ListTable* record = FindTable(table); record && record->complete.load(std::memory_order_acquire))
            return;

        const D3D12_COMMAND_LIST_TYPE type = list->GetType();
        if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
            return;

        RetireDropped();
        if (const char* failure = AdoptList(list, type, nullptr))
        {
            if (!gExecuteListLogged.exchange(true))
            {
                try
                {
                    Log("wmma_list_refused", std::string("\"created\":\"execute\",\"reason\":\"") + failure + "\"");
                }
                catch (...)
                {
                }
            }
            return;
        }
        const ListTable* record = FindTable(table);
        const std::string kind = record && record->runtimeClass ? "runtime_class" : record && record->shared ? "shared" : "own";
        Log("wmma_list", std::string("\"created\":\"execute\",\"type\":\"") +
            (type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? "compute" : "direct") + "\",\"table\":\"" + kind + "\"");
    }
    catch (...)
    {
    }
}

void ForgetLayers() noexcept
{
    std::lock_guard lock(gMutex);
    gAnyLayer.store(false, std::memory_order_release);
    for (auto& key : gKeys) key.store(nullptr, std::memory_order_release);
    for (auto& val : gValues) val.store(0, std::memory_order_relaxed);
    for (auto& rep : gReplacements) rep.store(nullptr, std::memory_order_relaxed);
    for (const HeldEntry& entry : gHeld)
    {
        if (entry.base)
            entry.base->Release();
        if (entry.replacement)
            entry.replacement->Release();
    }
    gHeld.clear();
    gLayerCount.store(0, std::memory_order_relaxed);
    for (auto& table : gTables)
        Retire(table);
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
    const uint32_t recognized = gLayerCount.load(std::memory_order_relaxed);
    s.layers = recognized ? recognized : static_cast<uint32_t>(std::size(xefg_cuda::kLayers));
    s.overrideLayers = gOverrideLayers;
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
