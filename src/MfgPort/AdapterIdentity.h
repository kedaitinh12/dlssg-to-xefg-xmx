#pragma once
// True physical GPU adapter identification by LUID, resilient to DXGI/NVAPI spoofing (e.g. OptiScaler
// SpoofDxgi=true, OptiPatcher.asi, or DLSS upscaler selection on AMD GPUs).
//
// Resolution order (all dynamically loaded, zero extra .lib link dependencies):
//   1. DXCore (dxcore.dll -> DXCoreCreateAdapterFactory -> GetAdapterByLuid -> HardwareID + DriverDescription)
//   2. WDDM Kernel D3DKMT (gdi32.dll -> D3DKMTOpenAdapterFromLuid -> KMTQAITYPE_PHYSICALADAPTERDEVICEIDS,
//      KMTQAITYPE_ADAPTERREGISTRYINFO, KMTQAITYPE_UMDRIVERNAME)
//   3. DXGI (CreateDXGIFactory1 -> EnumAdapterByLuid -> GetDesc1) — always queried to detect spoofing,
//      used as fallback only if DXCore and D3DKMT both fail.
//   4. EnumDisplayDevicesW (user32.dll) PCI VEN_XXXX&DEV_YYYY fallback when both DXCore and D3DKMT fail
//      but driver DLL evidence (e.g. amdxc64.dll) indicates DXGI spoofing.

#include <Windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#if defined(__has_include)
#if __has_include(<dxcore.h>)
#include <dxcore.h>
#define XEFG_HAS_DXCORE_HEADER 1
#endif
#endif

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwctype>
#include <string>
#include <string_view>

namespace adapter_identity
{
struct ProbeResult
{
    uint32_t vendorId{};
    uint32_t deviceId{};
    uint32_t subSysId{};
    uint32_t revision{};
    std::wstring description;
    std::string source;        // "dxcore", "d3dkmt_device_ids", "d3dkmt_umd", "enum_display_devices", "dxgi_getdesc1", "none"
    std::wstring umdLeaf;      // e.g. L"amdxc64.dll"

    bool dxgiQueried{false};
    uint32_t dxgiVendorId{};
    uint32_t dxgiDeviceId{};
    std::wstring dxgiDescription;
    bool dxgiSpoofed{false};

    std::string probeTrail;    // Compact diagnostic summary of every probe layer
};

inline std::string Hex16(uint32_t value)
{
    char buf[16]{};
    std::snprintf(buf, sizeof(buf), "0x%04X", value & 0xFFFFu);
    return buf;
}

inline std::string Hex32(uint32_t value)
{
    char buf[16]{};
    std::snprintf(buf, sizeof(buf), "0x%08X", value);
    return buf;
}

inline std::string FormatLuid(const LUID& luid)
{
    char buf[32]{};
    std::snprintf(buf, sizeof(buf), "0x%lX:0x%08lX",
        static_cast<unsigned long>(luid.HighPart),
        static_cast<unsigned long>(luid.LowPart));
    return buf;
}

inline const char* VendorName(uint32_t vendorId) noexcept
{
    switch (vendorId)
    {
    case 0x1002: return "AMD";
    case 0x10DE: return "NVIDIA";
    case 0x8086: return "Intel";
    case 0x1414: return "Microsoft";
    default: return "Unknown";
    }
}

inline std::string WideToAsciiJson(std::wstring_view text, size_t maxLen = 128)
{
    std::string out;
    out.reserve((text.size() < maxLen ? text.size() : maxLen));
    for (wchar_t ch : text)
    {
        if (ch == L'\0' || out.size() >= maxLen)
            break;
        if (ch >= 0x20 && ch < 0x7F && ch != L'"' && ch != L'\\')
            out.push_back(static_cast<char>(ch));
        else if (ch == L'\\' || ch == L'"')
            out.push_back('_');
    }
    while (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out;
}

inline std::wstring Utf8ToWide(const char* utf8)
{
    if (!utf8 || !utf8[0])
        return {};
    const int needed = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    if (needed <= 1)
        return {};
    std::wstring wide(static_cast<size_t>(needed - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wide.data(), needed);
    return wide;
}

inline std::wstring LeafName(std::wstring_view path)
{
    const size_t slash = path.find_last_of(L"\\/");
    if (slash == std::wstring_view::npos)
        return std::wstring(path);
    return std::wstring(path.substr(slash + 1));
}

inline bool ContainsInsensitive(std::wstring_view haystack, std::wstring_view needle) noexcept
{
    if (needle.empty() || haystack.size() < needle.size())
        return false;
    for (size_t i = 0; i + needle.size() <= haystack.size(); ++i)
    {
        bool match = true;
        for (size_t j = 0; j < needle.size(); ++j)
        {
            if (std::towlower(haystack[i + j]) != std::towlower(needle[j]))
            {
                match = false;
                break;
            }
        }
        if (match)
            return true;
    }
    return false;
}

inline uint32_t VendorFromUmdLeaf(std::wstring_view umdLeaf) noexcept
{
    if (umdLeaf.empty())
        return 0;
    if (ContainsInsensitive(umdLeaf, L"amdxc") ||
        ContainsInsensitive(umdLeaf, L"aticfx") ||
        ContainsInsensitive(umdLeaf, L"atidxx") ||
        ContainsInsensitive(umdLeaf, L"amdh2") ||
        ContainsInsensitive(umdLeaf, L"atiumd"))
        return 0x1002;
    if (ContainsInsensitive(umdLeaf, L"nvwgf2um") ||
        ContainsInsensitive(umdLeaf, L"nvldumdx") ||
        ContainsInsensitive(umdLeaf, L"nvd3dum"))
        return 0x10DE;
    if (ContainsInsensitive(umdLeaf, L"igd12") ||
        ContainsInsensitive(umdLeaf, L"igdumdim") ||
        ContainsInsensitive(umdLeaf, L"igd10") ||
        ContainsInsensitive(umdLeaf, L"igc"))
        return 0x8086;
    return 0;
}

inline bool ParseHex4(std::wstring_view s, uint32_t& out) noexcept
{
    if (s.size() < 4)
        return false;
    uint32_t val = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        wchar_t c = s[i];
        uint32_t nibble = 0;
        if (c >= L'0' && c <= L'9') nibble = static_cast<uint32_t>(c - L'0');
        else if (c >= L'a' && c <= L'f') nibble = static_cast<uint32_t>(10 + c - L'a');
        else if (c >= L'A' && c <= L'F') nibble = static_cast<uint32_t>(10 + c - L'A');
        else return false;
        val = (val << 4) | nibble;
    }
    out = val;
    return true;
}

inline bool ParsePciVenDev(std::wstring_view deviceIdStr, uint32_t& vendorOut, uint32_t& deviceOut) noexcept
{
    vendorOut = 0;
    deviceOut = 0;
    for (size_t i = 0; i + 8 <= deviceIdStr.size(); ++i)
    {
        if (ContainsInsensitive(deviceIdStr.substr(i, 4), L"VEN_"))
        {
            ParseHex4(deviceIdStr.substr(i + 4, 4), vendorOut);
            break;
        }
    }
    for (size_t i = 0; i + 8 <= deviceIdStr.size(); ++i)
    {
        if (ContainsInsensitive(deviceIdStr.substr(i, 4), L"DEV_"))
        {
            ParseHex4(deviceIdStr.substr(i + 4, 4), deviceOut);
            break;
        }
    }
    return vendorOut != 0;
}

namespace detail
{
inline void AppendTrail(std::string& trail, const std::string& item)
{
    if (!trail.empty())
        trail += ",";
    trail += item;
}

inline bool ProbeDxCore(const LUID& luid, uint32_t& vendorId, uint32_t& deviceId,
    uint32_t& subSysId, uint32_t& revision, std::wstring& description, std::string& trail) noexcept
{
#if defined(XEFG_HAS_DXCORE_HEADER)
    using Microsoft::WRL::ComPtr;
    using CreateFactoryFn = HRESULT (WINAPI*)(REFIID, void**);

    HMODULE dxcore = LoadLibraryExW(L"dxcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!dxcore)
    {
        AppendTrail(trail, "dxcore=missing");
        return false;
    }

    auto* createFactory = reinterpret_cast<CreateFactoryFn>(GetProcAddress(dxcore, "DXCoreCreateAdapterFactory"));
    if (!createFactory)
    {
        FreeLibrary(dxcore);
        AppendTrail(trail, "dxcore=no_export");
        return false;
    }

    bool ok = false;
    {
        ComPtr<IDXCoreAdapterFactory> factory;
        ComPtr<IDXCoreAdapter> adapter;
        if (SUCCEEDED(createFactory(IID_PPV_ARGS(&factory))) && factory &&
            SUCCEEDED(factory->GetAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) && adapter)
        {
            DXCoreHardwareID hwId{};
            if (adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareID) &&
                SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::HardwareID, sizeof(hwId), &hwId)) &&
                hwId.vendorID != 0)
            {
                vendorId = hwId.vendorID;
                deviceId = hwId.deviceID;
                subSysId = hwId.subSysID;
                revision = hwId.revision;
                ok = true;
            }

            if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverDescription))
            {
                size_t descSize = 0;
                if (SUCCEEDED(adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &descSize)) &&
                    descSize > 0 && descSize <= 512)
                {
                    std::string utf8(descSize, '\0');
                    if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, utf8.size(), utf8.data())))
                    {
                        utf8.back() = '\0';
                        description = Utf8ToWide(utf8.c_str());
                    }
                }
            }
        }
    }

    FreeLibrary(dxcore);
    if (ok)
        AppendTrail(trail, "dxcore=ok(" + Hex16(vendorId) + ":" + Hex16(deviceId) + ")");
    else
        AppendTrail(trail, "dxcore=fail");
    return ok;
#else
    (void)luid;
    (void)vendorId;
    (void)deviceId;
    (void)subSysId;
    (void)revision;
    (void)description;
    AppendTrail(trail, "dxcore=no_header");
    return false;
#endif
}

struct KmtOpenAdapterFromLuid
{
    LUID AdapterLuid;
    UINT hAdapter;
};

struct KmtCloseAdapter
{
    UINT hAdapter;
};

struct KmtQueryAdapterInfo
{
    UINT hAdapter;
    UINT Type;
    VOID* pPrivateDriverData;
    UINT PrivateDriverDataSize;
};

struct KmtUmdDriverName
{
    UINT Version; // 3 = DX12, 2 = DX11
    WCHAR UmdFileName[MAX_PATH];
};

struct KmtAdapterRegistryInfo
{
    WCHAR AdapterString[MAX_PATH];
    WCHAR BiosString[MAX_PATH];
    WCHAR DacType[MAX_PATH];
    WCHAR ChipType[MAX_PATH];
};

struct KmtDeviceIds
{
    UINT VendorID;
    UINT DeviceID;
    UINT SubVendorID;
    UINT SubSystemID;
    UINT RevisionID;
    UINT BusType;
};

struct KmtQueryDeviceIds
{
    UINT PhysicalAdapterIndex;
    KmtDeviceIds DeviceIds;
};

constexpr UINT kKmtQaiUmdDriverName = 1;
constexpr UINT kKmtQaiAdapterRegistryInfo = 8;
constexpr UINT kKmtQaiPhysicalAdapterDeviceIds = 31;

inline bool ProbeD3dkmt(const LUID& luid, uint32_t& vendorId, uint32_t& deviceId,
    uint32_t& subSysId, uint32_t& revision, std::wstring& description,
    std::wstring& umdLeaf, std::string& trail) noexcept
{
    using OpenFn = LONG (WINAPI*)(KmtOpenAdapterFromLuid*);
    using QueryFn = LONG (WINAPI*)(const KmtQueryAdapterInfo*);
    using CloseFn = LONG (WINAPI*)(const KmtCloseAdapter*);

    HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
    bool loadedHere = false;
    if (!gdi)
    {
        gdi = LoadLibraryExW(L"gdi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        loadedHere = (gdi != nullptr);
    }
    if (!gdi)
    {
        AppendTrail(trail, "d3dkmt=no_gdi32");
        return false;
    }

    auto* openAdapter = reinterpret_cast<OpenFn>(GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid"));
    auto* queryInfo = reinterpret_cast<QueryFn>(GetProcAddress(gdi, "D3DKMTQueryAdapterInfo"));
    auto* closeAdapter = reinterpret_cast<CloseFn>(GetProcAddress(gdi, "D3DKMTCloseAdapter"));
    if (!openAdapter || !queryInfo || !closeAdapter)
    {
        if (loadedHere) FreeLibrary(gdi);
        AppendTrail(trail, "d3dkmt=no_exports");
        return false;
    }

    KmtOpenAdapterFromLuid openParams{};
    openParams.AdapterLuid = luid;
    if (openAdapter(&openParams) < 0 || openParams.hAdapter == 0)
    {
        if (loadedHere) FreeLibrary(gdi);
        AppendTrail(trail, "d3dkmt=open_fail");
        return false;
    }

    bool gotIds = false;
    {
        KmtQueryDeviceIds devIds{};
        devIds.PhysicalAdapterIndex = 0;
        KmtQueryAdapterInfo q{};
        q.hAdapter = openParams.hAdapter;
        q.Type = kKmtQaiPhysicalAdapterDeviceIds;
        q.pPrivateDriverData = &devIds;
        q.PrivateDriverDataSize = sizeof(devIds);
        if (queryInfo(&q) >= 0 && devIds.DeviceIds.VendorID != 0)
        {
            vendorId = devIds.DeviceIds.VendorID;
            deviceId = devIds.DeviceIds.DeviceID;
            subSysId = devIds.DeviceIds.SubSystemID;
            revision = devIds.DeviceIds.RevisionID;
            gotIds = true;
        }
    }

    if (description.empty())
    {
        KmtAdapterRegistryInfo regInfo{};
        KmtQueryAdapterInfo q{};
        q.hAdapter = openParams.hAdapter;
        q.Type = kKmtQaiAdapterRegistryInfo;
        q.pPrivateDriverData = &regInfo;
        q.PrivateDriverDataSize = sizeof(regInfo);
        if (queryInfo(&q) >= 0 && regInfo.AdapterString[0] != L'\0')
        {
            regInfo.AdapterString[MAX_PATH - 1] = L'\0';
            description = regInfo.AdapterString;
        }
    }

    {
        for (UINT version : {3u, 2u})
        {
            KmtUmdDriverName umd{};
            umd.Version = version;
            KmtQueryAdapterInfo q{};
            q.hAdapter = openParams.hAdapter;
            q.Type = kKmtQaiUmdDriverName;
            q.pPrivateDriverData = &umd;
            q.PrivateDriverDataSize = sizeof(umd);
            if (queryInfo(&q) >= 0 && umd.UmdFileName[0] != L'\0')
            {
                umd.UmdFileName[MAX_PATH - 1] = L'\0';
                umdLeaf = LeafName(umd.UmdFileName);
                break;
            }
        }
    }

    KmtCloseAdapter closeParams{openParams.hAdapter};
    closeAdapter(&closeParams);
    if (loadedHere) FreeLibrary(gdi);

    if (gotIds)
        AppendTrail(trail, "d3dkmt=ok(" + Hex16(vendorId) + ":" + Hex16(deviceId) + ")");
    else if (!umdLeaf.empty())
        AppendTrail(trail, "d3dkmt=umd(" + WideToAsciiJson(umdLeaf, 48) + ")");
    else
        AppendTrail(trail, "d3dkmt=no_ids");

    return gotIds;
}

inline bool ProbeEnumDisplayDevices(uint32_t expectedVendor, uint32_t& vendorId,
    uint32_t& deviceId, std::wstring& description, std::string& trail) noexcept
{
    using EnumFn = BOOL (WINAPI*)(LPCWSTR, DWORD, PDISPLAY_DEVICEW, DWORD);
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    bool loadedHere = false;
    if (!user32)
    {
        user32 = LoadLibraryExW(L"user32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        loadedHere = (user32 != nullptr);
    }
    if (!user32)
        return false;

    auto* enumDev = reinterpret_cast<EnumFn>(GetProcAddress(user32, "EnumDisplayDevicesW"));
    if (!enumDev)
    {
        if (loadedHere) FreeLibrary(user32);
        return false;
    }

    bool found = false;
    for (DWORD i = 0; i < 16; ++i)
    {
        DISPLAY_DEVICEW dd{};
        dd.cb = sizeof(dd);
        if (!enumDev(nullptr, i, &dd, 0))
            break;
        if (!(dd.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP))
            continue;
        uint32_t ven = 0, dev = 0;
        if (ParsePciVenDev(dd.DeviceID, ven, dev))
        {
            if (expectedVendor != 0 && ven != expectedVendor)
                continue;
            vendorId = ven;
            deviceId = dev;
            if (description.empty() && dd.DeviceString[0] != L'\0')
                description = dd.DeviceString;
            found = true;
            break;
        }
    }

    if (loadedHere) FreeLibrary(user32);
    if (found)
        AppendTrail(trail, "display_dev=ok(" + Hex16(vendorId) + ":" + Hex16(deviceId) + ")");
    return found;
}
} // namespace detail

// Wine/Proton + DXVK + OptiScaler + ReShade64 can recurse infinitely inside hkCreateDXGIFactory1
// (e.g. Wuthering Waves stack overflow at 11.5 MB). On Wine/Proton, CreateDXGIFactory + QueryInterface
// to IDXGIFactory4 avoids CreateDXGIFactory1 hooks while keeping native Windows behavior unchanged.
inline HRESULT CreateDxgiFactory4Safe(IDXGIFactory4** outFactory) noexcept
{
    if (!outFactory) return E_POINTER;
    *outFactory = nullptr;

    static const bool underWine = []() noexcept {
        const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        return ntdll && GetProcAddress(ntdll, "wine_get_version") != nullptr;
    }();

    if (underWine)
    {
        Microsoft::WRL::ComPtr<IDXGIFactory> baseFactory;
        const HRESULT hr = CreateDXGIFactory(IID_PPV_ARGS(&baseFactory));
        if (FAILED(hr) || !baseFactory)
            return FAILED(hr) ? hr : E_FAIL;
        return baseFactory->QueryInterface(IID_PPV_ARGS(outFactory));
    }

    return CreateDXGIFactory1(IID_PPV_ARGS(outFactory));
}

inline ProbeResult Probe(const LUID& luid) noexcept
{
    ProbeResult result;
    result.source = "none";

    try
    {
        // 1. Always query DXGI GetDesc1 so we can detect & report spoofing (OptiScaler / OptiPatcher)
        {
            using Microsoft::WRL::ComPtr;
            ComPtr<IDXGIFactory4> factory;
            ComPtr<IDXGIAdapter1> adapter;
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(CreateDxgiFactory4Safe(&factory)) &&
                SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) &&
                SUCCEEDED(adapter->GetDesc1(&desc)))
            {
                result.dxgiQueried = true;
                result.dxgiVendorId = desc.VendorId;
                result.dxgiDeviceId = desc.DeviceId;
                result.dxgiDescription = desc.Description;
                detail::AppendTrail(result.probeTrail,
                    "dxgi=ok(" + Hex16(desc.VendorId) + ":" + Hex16(desc.DeviceId) + ")");
            }
            else
            {
                detail::AppendTrail(result.probeTrail, "dxgi=fail");
            }
        }

        // 2. Primary real-GPU probe: DXCore (unhooked by OptiScaler, exact LUID match)
        uint32_t dxcoreVen{}, dxcoreDev{}, dxcoreSub{}, dxcoreRev{};
        std::wstring dxcoreDesc;
        const bool dxcoreOk = detail::ProbeDxCore(
            luid, dxcoreVen, dxcoreDev, dxcoreSub, dxcoreRev, dxcoreDesc, result.probeTrail);

        // 3. Secondary real-GPU probe: WDDM Kernel D3DKMT (exact LUID match + UMD driver DLL name)
        uint32_t kmtVen{}, kmtDev{}, kmtSub{}, kmtRev{};
        std::wstring kmtDesc;
        const bool kmtOk = detail::ProbeD3dkmt(
            luid, kmtVen, kmtDev, kmtSub, kmtRev, kmtDesc, result.umdLeaf, result.probeTrail);

        if (dxcoreOk)
        {
            result.vendorId = dxcoreVen;
            result.deviceId = dxcoreDev;
            result.subSysId = dxcoreSub;
            result.revision = dxcoreRev;
            result.description = !dxcoreDesc.empty() ? dxcoreDesc : kmtDesc;
            result.source = "dxcore";
        }
        else if (kmtOk)
        {
            result.vendorId = kmtVen;
            result.deviceId = kmtDev;
            result.subSysId = kmtSub;
            result.revision = kmtRev;
            result.description = kmtDesc;
            result.source = "d3dkmt_device_ids";
        }
        else
        {
            // Fallback when both DXCore and D3DKMT DeviceIds failed (e.g. older OS / translation layer):
            // Check if D3DKMT UMD driver name (for this exact LUID) or loaded amdxc64.dll reveals vendor.
            const uint32_t umdVendor = VendorFromUmdLeaf(result.umdLeaf);
            const bool amdDllLoaded = GetModuleHandleW(L"amdxc64.dll") != nullptr;
            const uint32_t hintVendor = umdVendor ? umdVendor : (amdDllLoaded && result.dxgiVendorId != 0x1002 ? 0x1002u : 0u);

            uint32_t ddVen{}, ddDev{};
            std::wstring ddDesc = kmtDesc;
            if (hintVendor != 0 && detail::ProbeEnumDisplayDevices(hintVendor, ddVen, ddDev, ddDesc, result.probeTrail))
            {
                result.vendorId = ddVen;
                result.deviceId = ddDev;
                result.description = ddDesc;
                result.source = "enum_display_devices";
            }
            else if (umdVendor != 0)
            {
                result.vendorId = umdVendor;
                result.deviceId = (result.dxgiVendorId == umdVendor) ? result.dxgiDeviceId : 0;
                result.description = !kmtDesc.empty() ? kmtDesc : result.dxgiDescription;
                result.source = "d3dkmt_umd";
            }
            else if (result.dxgiQueried)
            {
                result.vendorId = result.dxgiVendorId;
                result.deviceId = result.dxgiDeviceId;
                result.description = result.dxgiDescription;
                result.source = "dxgi_getdesc1";
            }
        }

        if (result.vendorId != 0 && result.dxgiQueried &&
            (result.vendorId != result.dxgiVendorId ||
             (result.deviceId != 0 && result.dxgiDeviceId != 0 && result.deviceId != result.dxgiDeviceId)))
        {
            result.dxgiSpoofed = true;
        }

        if (result.description.empty() && result.dxgiQueried && !result.dxgiSpoofed)
            result.description = result.dxgiDescription;
    }
    catch (...)
    {
    }

    return result;
}
} // namespace adapter_identity
