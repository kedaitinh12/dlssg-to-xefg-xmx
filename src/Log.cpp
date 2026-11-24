#include "Log.h"
#include "Util.h"

#include <Windows.h>
#include <Psapi.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace
{
	std::atomic_bool gLoggingEnabled{ true };
	LPTOP_LEVEL_EXCEPTION_FILTER gPrevExceptionFilter = nullptr;
	std::atomic_bool gExceptionFilterInstalled{ false };

	struct RtlOsVersionInfoW
	{
		ULONG dwOSVersionInfoSize;
		ULONG dwMajorVersion;
		ULONG dwMinorVersion;
		ULONG dwBuildNumber;
		ULONG dwPlatformId;
		WCHAR szCSDVersion[128];
	};

	std::string FormatWindowsVersion() noexcept
	{
		using RtlGetVersionFn = LONG(NTAPI*)(RtlOsVersionInfoW*);
		if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll"))
		{
			if (auto fn = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion")))
			{
				RtlOsVersionInfoW vi{};
				vi.dwOSVersionInfoSize = sizeof(vi);
				if (fn(&vi) == 0)
					return std::format("Windows {}.{} (build {})", vi.dwMajorVersion, vi.dwMinorVersion, vi.dwBuildNumber);
			}
		}
		return "Windows (version unknown)";
	}

	std::string NarrowModuleName(HMODULE mod) noexcept
	{
		wchar_t buf[MAX_PATH]{};
		const DWORD len = GetModuleFileNameW(mod, buf, MAX_PATH);
		if (len == 0 || len >= MAX_PATH)
			return "<unknown>";
		try
		{
			return std::filesystem::path(buf).filename().string();
		}
		catch (...)
		{
			return "<unknown>";
		}
	}

	const char* ExceptionCodeName(DWORD code) noexcept
	{
		switch (code)
		{
		case EXCEPTION_ACCESS_VIOLATION: return "ACCESS_VIOLATION";
		case EXCEPTION_ILLEGAL_INSTRUCTION: return "ILLEGAL_INSTRUCTION";
		case EXCEPTION_INT_DIVIDE_BY_ZERO: return "INT_DIVIDE_BY_ZERO";
		case EXCEPTION_STACK_OVERFLOW: return "STACK_OVERFLOW";
		case EXCEPTION_PRIV_INSTRUCTION: return "PRIV_INSTRUCTION";
		case EXCEPTION_IN_PAGE_ERROR: return "IN_PAGE_ERROR";
		case 0xE06D7363u: return "CXX_EXCEPTION";
		default: return "EXCEPTION";
		}
	}

	void WriteLineLocked(std::ofstream& file, std::string_view level, std::string_view message) noexcept
	{
		try
		{
			SYSTEMTIME now{};
			GetLocalTime(&now);
			auto line = std::format("[{:02}:{:02}:{:02}.{:03}] [{}] {}\n", now.wHour, now.wMinute, now.wSecond,
				now.wMilliseconds, level, message);

			if (IsDebuggerPresent())
				OutputDebugStringA(line.c_str());

			if (file.is_open())
			{
				file.write(line.data(), static_cast<std::streamsize>(line.size()));
				file.flush();
			}
		}
		catch (...)
		{
		}
	}

	LONG WINAPI CrashLogFilter(EXCEPTION_POINTERS* ep) noexcept
	{
		if (ep && ep->ExceptionRecord && gLoggingEnabled.load(std::memory_order_relaxed))
		{
			const auto* rec = ep->ExceptionRecord;
			HMODULE crashMod = nullptr;
			std::string modName = "<unknown>";
			uintptr_t rva = 0;
			if (GetModuleHandleExW(
				GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(rec->ExceptionAddress), &crashMod) && crashMod)
			{
				modName = NarrowModuleName(crashMod);
				rva = reinterpret_cast<uintptr_t>(rec->ExceptionAddress) - reinterpret_cast<uintptr_t>(crashMod);
			}

			char msg[384]{};
			if (rec->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && rec->NumberParameters >= 2)
			{
				const char* op = rec->ExceptionInformation[0] == 0 ? "read" : (rec->ExceptionInformation[0] == 1 ? "write" : "exec");
				std::snprintf(msg, sizeof(msg),
					"Unhandled exception %s (0x%08lX) at %s+0x%llX (%s addr 0x%llX, tid %lu)",
					ExceptionCodeName(rec->ExceptionCode), rec->ExceptionCode, modName.c_str(),
					static_cast<unsigned long long>(rva), op,
					static_cast<unsigned long long>(rec->ExceptionInformation[1]), GetCurrentThreadId());
			}
			else
			{
				std::snprintf(msg, sizeof(msg),
					"Unhandled exception %s (0x%08lX) at %s+0x%llX (tid %lu)",
					ExceptionCodeName(rec->ExceptionCode), rec->ExceptionCode, modName.c_str(),
					static_cast<unsigned long long>(rva), GetCurrentThreadId());
			}
			Log::Write("error", msg);
		}
		return gPrevExceptionFilter ? gPrevExceptionFilter(ep) : EXCEPTION_CONTINUE_SEARCH;
	}

	struct LogSession
	{
		std::mutex mutex;
		std::ofstream file;

		LogSession()
		{
			const auto logPath = ThisModulePath().replace_extension(L".log");
			wchar_t flag[8]{};
			const DWORD len = GetEnvironmentVariableW(L"XEFG_EARLY_LOG_INITIALIZED", flag, 8);
			const bool appended = (len > 0 && len < 8 && flag[0] == L'1');
			if (appended)
			{
				SetEnvironmentVariableW(L"XEFG_EARLY_LOG_INITIALIZED", nullptr);
				file.open(logPath, std::ios::app);
			}
			else
			{
				file.open(logPath, std::ios::trunc);
			}

			if (gLoggingEnabled.load(std::memory_order_relaxed))
			{
				const std::string exeName = NarrowModuleName(nullptr);
				const std::string modName = ThisModulePath().filename().string();
				WriteLineLocked(file, "info", std::format(
					"Session: build {} {}, pid {}, exe {}, module {}, {}",
					__DATE__, __TIME__, GetCurrentProcessId(), exeName, modName, FormatWindowsVersion()));
			}

			if (!gExceptionFilterInstalled.exchange(true, std::memory_order_acq_rel))
				gPrevExceptionFilter = SetUnhandledExceptionFilter(CrashLogFilter);
		}

		~LogSession()
		{
			if (gLoggingEnabled.load(std::memory_order_relaxed))
			{
				std::scoped_lock lock(mutex);
				WriteLineLocked(file, "info", "Session end (clean exit)");
			}
		}
	};

	LogSession& Session()
	{
		static LogSession session;
		return session;
	}
} // namespace

void Log::SetEnabled(bool enabled) noexcept
{
	gLoggingEnabled.store(enabled, std::memory_order_release);
}

bool Log::IsEnabled() noexcept
{
	return gLoggingEnabled.load(std::memory_order_relaxed);
}

const char* Log::DeviceRemovedName(uint32_t hresult) noexcept
{
	switch (static_cast<HRESULT>(hresult))
	{
	case DXGI_ERROR_DEVICE_HUNG: return "DXGI_ERROR_DEVICE_HUNG";
	case DXGI_ERROR_DEVICE_REMOVED: return "DXGI_ERROR_DEVICE_REMOVED";
	case DXGI_ERROR_DEVICE_RESET: return "DXGI_ERROR_DEVICE_RESET";
	case DXGI_ERROR_DRIVER_INTERNAL_ERROR: return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
	case DXGI_ERROR_INVALID_CALL: return "DXGI_ERROR_INVALID_CALL";
	default: return "HRESULT";
	}
}

void Log::WarnForeignFgMods(bool isVulkan, bool usingExternalVulkanProvider) noexcept
{
	static std::atomic_flag scanned = ATOMIC_FLAG_INIT;
	if (scanned.test_and_set(std::memory_order_acq_rel))
		return;

	try
	{
		std::vector<HMODULE> modules(1024);
		DWORD needed = 0;
		if (!EnumProcessModules(GetCurrentProcess(), modules.data(),
			static_cast<DWORD>(modules.size() * sizeof(HMODULE)), &needed))
			return;

		modules.resize(std::min<size_t>(modules.size(), needed / sizeof(HMODULE)));
		HMODULE self = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			reinterpret_cast<LPCWSTR>(&Log::WarnForeignFgMods), &self);

		auto toLower = [](std::wstring s) {
			for (wchar_t& c : s)
				c = static_cast<wchar_t>(std::towlower(c));
			return s;
		};

		for (HMODULE mod : modules)
		{
			if (!mod || mod == self)
				continue;

			const std::filesystem::path p = ModulePath(mod);
			if (p.empty())
				continue;

			const auto filename = p.filename().u8string();
			const std::string displayName(filename.begin(), filename.end());
			const std::wstring name = toLower(p.filename().wstring());
			const char* kind = nullptr;

			if (name.ends_with(L".addon64") &&
				(name.find(L"mfgunlock") != std::wstring::npos || name.find(L"mfg_unlock") != std::wstring::npos))
				kind = "reshade_fg_addon";
			else if (name.find(L"dlssg-transfusion") != std::wstring::npos ||
				name.find(L"rtx-encore") != std::wstring::npos || name.find(L"rtx_encore") != std::wstring::npos)
				kind = "rtx_encore";
			else if (name.find(L"mfgunlocker") != std::wstring::npos)
			{
				if (!(isVulkan && usingExternalVulkanProvider))
					kind = "mfg_unlocker";
			}
			else if (name.find(L"dlssg_to_fsr3") != std::wstring::npos)
				kind = "dlssg_to_fsr3";
			else if (name.find(L"sm86_backend") != std::wstring::npos || name.find(L"dlssg_sm86") != std::wstring::npos)
				kind = "dlssg_sm86";
			else if (name.find(L"dlssg_to_xefg_intel_is_better") != std::wstring::npos)
				kind = "duplicate_xefg_mod";
			else if (name.find(L"rtsshooks") != std::wstring::npos &&
				(GetModuleHandleW(L"amdxc64.dll") != nullptr || GetModuleHandleW(L"amdvlk64.dll") != nullptr))
			{
				Warn("overlay_warning: {} detected on AMD GPU — RTSS overlay hooks can cause XeFG crashes in some titles (e.g. Crimson Desert); disable RTSS overlay for this game if unstable",
					displayName);
				continue;
			}

			if (kind)
			{
				Warn("foreign_fg_mod detected: {} ({}) — running two frame generation mods together can cause hook conflicts or prevent frame generation",
					displayName, kind);
			}
		}
	}
	catch (...)
	{
	}
}

bool Log::EnableDred() noexcept
{
	static std::atomic_bool sEnabled{ false };
	if (sEnabled.load(std::memory_order_acquire))
		return true;

	try
	{
		HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
		if (!d3d12)
			d3d12 = LoadLibraryW(L"d3d12.dll");
		if (!d3d12)
			return false;

		using GetDebugIfaceFn = HRESULT(WINAPI*)(REFIID, void**);
		auto getDebugIface = reinterpret_cast<GetDebugIfaceFn>(GetProcAddress(d3d12, "D3D12GetDebugInterface"));
		if (!getDebugIface)
			return false;

		Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> dred1;
		if (SUCCEEDED(getDebugIface(IID_PPV_ARGS(&dred1))) && dred1)
		{
			dred1->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			dred1->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			dred1->SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			if (!sEnabled.exchange(true, std::memory_order_acq_rel))
				Info("DRED crash diagnostics enabled (AutoBreadcrumbs + PageFault + Context forced on)");
			return true;
		}

		Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred0;
		if (SUCCEEDED(getDebugIface(IID_PPV_ARGS(&dred0))) && dred0)
		{
			dred0->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			dred0->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			if (!sEnabled.exchange(true, std::memory_order_acq_rel))
				Info("DRED crash diagnostics enabled (AutoBreadcrumbs + PageFault forced on)");
			return true;
		}
	}
	catch (...)
	{
	}
	return false;
}

void Log::DumpDred(ID3D12Device* device) noexcept
{
	static std::atomic_bool sDumped{ false };
	if (!device || SUCCEEDED(device->GetDeviceRemovedReason()) || sDumped.exchange(true, std::memory_order_acq_rel))
		return;

	try
	{
		auto formatObjName = [](const wchar_t* w, const char* a) -> std::string {
			if (a && *a)
				return std::string(a);
			if (w && *w)
			{
				const int needed = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
				if (needed > 1)
				{
					std::string out(static_cast<size_t>(needed - 1), '\0');
					WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), needed, nullptr, nullptr);
					return out;
				}
			}
			return "(unnamed)";
		};

		Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedData1> dred1;
		if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dred1))) && dred1)
		{
			D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 crumbs{};
			if (SUCCEEDED(dred1->GetAutoBreadcrumbsOutput1(&crumbs)))
			{
				int shown = 0;
				for (auto* n = crumbs.pHeadAutoBreadcrumbNode; n != nullptr && shown < 24; n = n->pNext)
				{
					const UINT last = n->pLastBreadcrumbValue ? *n->pLastBreadcrumbValue : 0u;
					if (last >= n->BreadcrumbCount)
						continue;
					++shown;
					std::string ops;
					const UINT start = (last > 3u) ? (last - 3u) : 0u;
					const UINT end = std::min<UINT>(n->BreadcrumbCount, last + 4u);
					for (UINT i = start; i < end; ++i)
					{
						if (i == last) ops += '[';
						ops += std::to_string(static_cast<int>(n->pCommandHistory[i]));
						ops += (i == last) ? "] " : " ";
					}
					Error("DRED unfinished list '{}' on queue '{}': {}/{} ops done; ops {}",
						formatObjName(n->pCommandListDebugNameW, n->pCommandListDebugNameA),
						formatObjName(n->pCommandQueueDebugNameW, n->pCommandQueueDebugNameA),
						last, n->BreadcrumbCount, ops);
				}
				if (shown == 0)
					Info("DRED: every recorded command list had finished");
			}
			else
			{
				Info("DRED: no breadcrumbs (enable 'Force DRED crash diagnostics (DX12)' in OptiScaler menu or set XEFG_FORCE_DRED=1 and restart)");
			}

			D3D12_DRED_PAGE_FAULT_OUTPUT1 fault{};
			if (SUCCEEDED(dred1->GetPageFaultAllocationOutput1(&fault)) && fault.PageFaultVA != 0)
			{
				std::string existing, freed;
				for (auto* a = fault.pHeadExistingAllocationNode; a != nullptr && existing.size() < 240; a = a->pNext)
					existing += formatObjName(a->ObjectNameW, a->ObjectNameA) + "; ";
				for (auto* a = fault.pHeadRecentFreedAllocationNode; a != nullptr && freed.size() < 240; a = a->pNext)
					freed += formatObjName(a->ObjectNameW, a->ObjectNameA) + "; ";
				Error("DRED page fault VA={:#018x}; existing: [{}] recently freed: [{}]",
					static_cast<uint64_t>(fault.PageFaultVA), existing, freed);
			}
			return;
		}

		Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedData> dred0;
		if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dred0))) && dred0)
		{
			D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT crumbs{};
			if (SUCCEEDED(dred0->GetAutoBreadcrumbsOutput(&crumbs)))
			{
				int shown = 0;
				for (auto* n = crumbs.pHeadAutoBreadcrumbNode; n != nullptr && shown < 24; n = n->pNext)
				{
					const UINT last = n->pLastBreadcrumbValue ? *n->pLastBreadcrumbValue : 0u;
					if (last >= n->BreadcrumbCount)
						continue;
					++shown;
					Error("DRED unfinished list '{}' on queue '{}': {}/{} ops done",
						formatObjName(n->pCommandListDebugNameW, n->pCommandListDebugNameA),
						formatObjName(n->pCommandQueueDebugNameW, n->pCommandQueueDebugNameA),
						last, n->BreadcrumbCount);
				}
				if (shown == 0)
					Info("DRED: every recorded command list had finished");
			}
			else
			{
				Info("DRED: no breadcrumbs (enable 'Force DRED crash diagnostics (DX12)' in OptiScaler menu or set XEFG_FORCE_DRED=1 and restart)");
			}
			return;
		}

		Info("DRED: interface not available on this D3D12 runtime");
	}
	catch (...)
	{
	}
}

void Log::Write(std::string_view level, std::string_view message)
{
	if (!gLoggingEnabled.load(std::memory_order_relaxed))
		return;

	auto& s = Session();
	std::scoped_lock lock(s.mutex);
	WriteLineLocked(s.file, level, message);
}
