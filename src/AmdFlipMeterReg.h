// Automatic AMD RDNA 4+ GPU detection and Gmx_DriverFramePacingMaxGenFrames (6x Flip Meter) registry helper.
// Scans HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\000* so multi-GPU
// systems (e.g. Ryzen iGPU + Radeon RX 9070 XT) target ONLY the exact RDNA 4+ adapter subkey(s).
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cwctype>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace AmdFlipMeterReg
{
	inline constexpr uint32_t kAmdVendorId = 0x1002u;
	inline constexpr uint32_t kTargetMaxGenFrames = 5u; // 5 generated frames per real frame = 6x FG

	inline bool ParseHex4(std::wstring_view s, uint32_t& out) noexcept
	{
		if (s.size() < 4)
			return false;
		uint32_t val = 0;
		for (size_t i = 0; i < 4; ++i)
		{
			const wchar_t c = s[i];
			uint32_t d = 0;
			if (c >= L'0' && c <= L'9') d = static_cast<uint32_t>(c - L'0');
			else if (c >= L'a' && c <= L'f') d = static_cast<uint32_t>(10 + (c - L'a'));
			else if (c >= L'A' && c <= L'F') d = static_cast<uint32_t>(10 + (c - L'A'));
			else return false;
			val = (val << 4) | d;
		}
		out = val;
		return true;
	}

	inline std::wstring ToUpperWide(std::wstring_view s)
	{
		std::wstring out;
		out.reserve(s.size());
		for (wchar_t c : s)
			out.push_back(static_cast<wchar_t>(std::towupper(c)));
		return out;
	}

	// Parses PCI\VEN_1002&DEV_7550... (case-insensitive) into vendorId and deviceId.
	inline bool ParsePciVendorDevice(std::wstring_view matchingId, uint32_t& vendorId, uint32_t& deviceId) noexcept
	{
		const std::wstring up = ToUpperWide(matchingId);
		const size_t vPos = up.find(L"VEN_");
		const size_t dPos = up.find(L"DEV_");
		if (vPos == std::wstring::npos || dPos == std::wstring::npos)
			return false;
		uint32_t v = 0, d = 0;
		if (!ParseHex4(std::wstring_view(up).substr(vPos + 4), v) ||
			!ParseHex4(std::wstring_view(up).substr(dPos + 4), d))
			return false;
		vendorId = v;
		deviceId = d;
		return true;
	}

	// Identifies AMD RDNA 4+ discrete GPUs (Navi 48 / Navi 44: RX 9070 XT / RX 9070 / RX 9060 XT, PCI 0x7550..0x75FF,
	// plus future RDNA 4/5 / gfx12/gfx13) while strictly excluding AMD iGPUs, RDNA 3/2, Intel, and NVIDIA adapters.
	inline bool IsRdna4OrNewerAdapter(uint32_t vendorId, uint32_t deviceId, std::wstring_view driverDesc) noexcept
	{
		if (vendorId != kAmdVendorId)
			return false;

		// Primary hardware check: AMD Navi 4x (RDNA 4: gfx1200/gfx1201) PCI Device IDs 0x7550..0x75FF
		if (deviceId >= 0x7550u && deviceId <= 0x75FFu)
			return true;

		const std::wstring up = ToUpperWide(driverDesc);

		// Explicitly exclude integrated graphics / older RDNA families when deviceId isn't in 0x7550..0x75FF
		if (up.find(L"780M") != std::wstring::npos ||
			up.find(L"880M") != std::wstring::npos ||
			up.find(L"890M") != std::wstring::npos ||
			up.find(L"8060S") != std::wstring::npos ||
			up.find(L"8050S") != std::wstring::npos ||
			up.find(L"RADEON(TM) GRAPHICS") != std::wstring::npos)
		{
			return false;
		}

		// Secondary name match for future RDNA 4/5 / Radeon AI PRO R9xxx / gfx12+ SKUs
		return up.find(L"RX 90") != std::wstring::npos ||
			up.find(L"RADEON AI PRO R9") != std::wstring::npos ||
			up.find(L"RDNA 4") != std::wstring::npos ||
			up.find(L"RDNA 5") != std::wstring::npos ||
			up.find(L"GFX12") != std::wstring::npos ||
			up.find(L"GFX13") != std::wstring::npos;
	}

	struct AdapterEntry
	{
		std::wstring subkey;      // e.g. L"0000" or L"0001"
		std::string subkeyUtf8;   // e.g. "0001"
		std::string driverDesc;   // e.g. "AMD Radeon RX 9070 XT"
		uint32_t vendorId = 0;
		uint32_t deviceId = 0;
		bool valuePresent = false;
		uint32_t currentValue = 1; // Driver default is 1 (2x)
	};

	struct ScanResult
	{
		std::vector<AdapterEntry> rdna4Adapters;
		bool allConfigured = false;
		bool appliedThisSession = false;
		bool rebootRecommended = false;
	};

#ifdef _WIN32
	inline constexpr wchar_t kDisplayClassPath[] =
		L"SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}";
	inline constexpr wchar_t kValueName[] = L"Gmx_DriverFramePacingMaxGenFrames";

	inline std::string WideToUtf8(std::wstring_view w)
	{
		if (w.empty())
			return {};
		const int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
		if (len <= 0)
			return {};
		std::string out(static_cast<size_t>(len), '\0');
		WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), len, nullptr, nullptr);
		return out;
	}

	inline bool ReadRegString(HKEY key, const wchar_t* name, std::wstring& out) noexcept
	{
		DWORD type = 0;
		DWORD bytes = 0;
		if (RegQueryValueExW(key, name, nullptr, &type, nullptr, &bytes) != ERROR_SUCCESS ||
			(type != REG_SZ && type != REG_EXPAND_SZ) || bytes < sizeof(wchar_t))
			return false;

		std::wstring buf(bytes / sizeof(wchar_t), L'\0');
		if (RegQueryValueExW(key, name, nullptr, &type, reinterpret_cast<BYTE*>(buf.data()), &bytes) != ERROR_SUCCESS)
			return false;

		while (!buf.empty() && buf.back() == L'\0')
			buf.pop_back();
		out = std::move(buf);
		return !out.empty();
	}

	inline bool ReadRegDword(HKEY key, const wchar_t* name, uint32_t& out) noexcept
	{
		DWORD type = 0;
		DWORD val = 0;
		DWORD bytes = sizeof(val);
		if (RegQueryValueExW(key, name, nullptr, &type, reinterpret_cast<BYTE*>(&val), &bytes) != ERROR_SUCCESS ||
			type != REG_DWORD || bytes != sizeof(val))
			return false;
		out = val;
		return true;
	}

	inline bool IsFourDigitSubkey(std::wstring_view name) noexcept
	{
		if (name.size() != 4)
			return false;
		for (wchar_t c : name)
			if (c < L'0' || c > L'9')
				return false;
		return true;
	}

	// Scans all display adapters (KEY_READ, works without admin) and returns ONLY AMD RDNA 4+ discrete GPUs.
	inline ScanResult ScanRdna4Adapters() noexcept
	{
		ScanResult res{};
		HKEY classKey = nullptr;
		if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, kDisplayClassPath, 0, KEY_READ, &classKey) != ERROR_SUCCESS)
			return res;

		wchar_t subName[64]{};
		for (DWORD idx = 0;; ++idx)
		{
			DWORD nameLen = static_cast<DWORD>(std::size(subName));
			const LONG er = RegEnumKeyExW(classKey, idx, subName, &nameLen, nullptr, nullptr, nullptr, nullptr);
			if (er == ERROR_NO_MORE_ITEMS)
				break;
			if (er != ERROR_SUCCESS)
				continue;

			const std::wstring_view subView(subName, nameLen);
			if (!IsFourDigitSubkey(subView))
				continue;

			HKEY devKey = nullptr;
			if (RegOpenKeyExW(classKey, subName, 0, KEY_READ, &devKey) != ERROR_SUCCESS)
				continue;

			std::wstring matchId;
			std::wstring desc;
			ReadRegString(devKey, L"MatchingDeviceId", matchId);
			ReadRegString(devKey, L"DriverDesc", desc);

			uint32_t vendorId = 0, deviceId = 0;
			if (ParsePciVendorDevice(matchId, vendorId, deviceId) &&
				IsRdna4OrNewerAdapter(vendorId, deviceId, desc))
			{
				AdapterEntry entry{};
				entry.subkey.assign(subView);
				entry.subkeyUtf8 = WideToUtf8(entry.subkey);
				entry.driverDesc = desc.empty() ? "AMD RDNA 4 GPU" : WideToUtf8(desc);
				entry.vendorId = vendorId;
				entry.deviceId = deviceId;
				uint32_t cur = 1;
				if (ReadRegDword(devKey, kValueName, cur))
				{
					entry.valuePresent = true;
					entry.currentValue = cur;
				}
				res.rdna4Adapters.push_back(std::move(entry));
			}

			RegCloseKey(devKey);
		}

		RegCloseKey(classKey);

		res.allConfigured = !res.rdna4Adapters.empty();
		for (const auto& a : res.rdna4Adapters)
		{
			if (!a.valuePresent || a.currentValue < kTargetMaxGenFrames)
				res.allConfigured = false;
		}
		return res;
	}

	// Attempts a direct HKLM write on every detected RDNA 4+ subkey (succeeds silently if process is elevated;
	// returns false with no UAC popup when running as standard user).
	inline bool TryDirectApply(const ScanResult& scan, bool revert = false) noexcept
	{
		if (scan.rdna4Adapters.empty())
			return false;

		bool allOk = true;
		for (const auto& a : scan.rdna4Adapters)
		{
			const std::wstring fullPath = std::wstring(kDisplayClassPath) + L"\\" + a.subkey;
			HKEY devKey = nullptr;
			if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, fullPath.c_str(), 0, KEY_SET_VALUE, &devKey) != ERROR_SUCCESS)
			{
				allOk = false;
				continue;
			}
			LONG st = ERROR_SUCCESS;
			if (revert)
			{
				st = RegDeleteValueW(devKey, kValueName);
				if (st == ERROR_FILE_NOT_FOUND)
					st = ERROR_SUCCESS;
			}
			else
			{
				const DWORD val = kTargetMaxGenFrames;
				st = RegSetValueExW(devKey, kValueName, 0, REG_DWORD, reinterpret_cast<const BYTE*>(&val), sizeof(val));
			}
			RegCloseKey(devKey);
			if (st != ERROR_SUCCESS)
				allOk = false;
		}
		return allOk;
	}

	// User-initiated elevated apply/revert (invoked ONLY from explicit menu button click, never during fullscreen startup).
	// Targets ONLY the exact detected RDNA 4+ subkey(s) via reg.exe.
	inline bool LaunchElevatedApply(const ScanResult& scan, bool revert = false) noexcept
	{
		if (scan.rdna4Adapters.empty())
			return false;

		if (TryDirectApply(scan, revert))
			return true;

		using ShellExecuteWFn = HINSTANCE(WINAPI*)(HWND, LPCWSTR, LPCWSTR, LPCWSTR, LPCWSTR, INT);
		static ShellExecuteWFn shellExec = []() noexcept -> ShellExecuteWFn {
			HMODULE mod = LoadLibraryW(L"shell32.dll");
			return mod ? reinterpret_cast<ShellExecuteWFn>(GetProcAddress(mod, "ShellExecuteW")) : nullptr;
		}();
		if (!shellExec)
			return false;

		// Build a single elevated cmd.exe command so the user only sees ONE UAC prompt even with multiple RDNA 4 cards.
		std::wstring args = L"/C ";
		for (size_t i = 0; i < scan.rdna4Adapters.size(); ++i)
		{
			if (i > 0)
				args += L" && ";
			const std::wstring key = L"HKLM\\" + std::wstring(kDisplayClassPath) + L"\\" + scan.rdna4Adapters[i].subkey;
			if (revert)
				args += L"reg delete \"" + key + L"\" /v Gmx_DriverFramePacingMaxGenFrames /f";
			else
				args += L"reg add \"" + key + L"\" /v Gmx_DriverFramePacingMaxGenFrames /t REG_DWORD /d 5 /f";
		}

		const HINSTANCE res = shellExec(nullptr, L"runas", L"cmd.exe", args.c_str(), nullptr, SW_HIDE);
		return reinterpret_cast<intptr_t>(res) > 32;
	}
#else
	inline ScanResult ScanRdna4Adapters() noexcept { return {}; }
	inline bool TryDirectApply(const ScanResult&, bool = false) noexcept { return false; }
	inline bool LaunchElevatedApply(const ScanResult&, bool = false) noexcept { return false; }
#endif
} // namespace AmdFlipMeterReg
