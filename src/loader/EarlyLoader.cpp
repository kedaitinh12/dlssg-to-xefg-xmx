#include "EarlyLoaderCore.h"
#include "SessionMarkerWin.h"
#include "../optiscaler_plugin.h"

#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace
{
	std::filesystem::path LoaderModulePath()
	{
		HMODULE module = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			reinterpret_cast<LPCWSTR>(&LoaderModulePath), &module);

		wchar_t path[MAX_PATH]{};
		GetModuleFileNameW(module, path, MAX_PATH);
		return path;
	}

	std::mutex gLogMutex;
	std::vector<std::string> gBufferedLines;
	std::filesystem::path gUnifiedLogPath;
	bool gUnifiedLogFlushed = false;

	void WriteLog(std::string_view level, std::string_view message)
	{
		SYSTEMTIME now{};
		GetLocalTime(&now);
		const auto line = std::format("[{:02}:{:02}:{:02}.{:03}] [{}] [EarlyLoader] {}\n",
			now.wHour, now.wMinute, now.wSecond, now.wMilliseconds, level, message);

		std::scoped_lock lock(gLogMutex);
		if (IsDebuggerPresent())
			OutputDebugStringA(line.c_str());

		if (gUnifiedLogFlushed && !gUnifiedLogPath.empty())
		{
			std::ofstream file(gUnifiedLogPath, std::ios::app);
			if (file)
			{
				file.write(line.data(), static_cast<std::streamsize>(line.size()));
				file.flush();
			}
		}
		else
		{
			gBufferedLines.push_back(line);
		}
	}

	// Flushes buffered EarlyLoader messages into the single unified log file next to the main DLL
	// (never creating a separate .log beside the .asi). Closes immediately so the DLL can append cleanly.
	void FlushLogNextToDll(const std::filesystem::path& dllPath)
	{
		if (dllPath.empty())
			return;
		std::scoped_lock lock(gLogMutex);
		if (gUnifiedLogFlushed)
			return;
		gUnifiedLogPath = dllPath;
		gUnifiedLogPath.replace_extension(L".log");

		// A process outside the helper list (an unknown launcher or reporter) must not wipe the log of the game that
		// owns the .session marker and is still running: append to it instead.
		auto markerPath = dllPath;
		markerPath.replace_extension(L".session");
		const auto marker = EarlyLoader::ReadSessionMarker(markerPath);
		const auto decision = EarlyLoader::DecideSession(marker.pid, EarlyLoader::IsMarkerOwnerAlive(marker),
			GetCurrentProcessId(), /*logAlreadyStarted=*/false);
		std::ofstream file(gUnifiedLogPath, decision.truncateLog ? std::ios::trunc : std::ios::app);
		if (file)
		{
			for (const auto& line : gBufferedLines)
				file.write(line.data(), static_cast<std::streamsize>(line.size()));
			file.flush();
			SetEnvironmentVariableW(L"XEFG_EARLY_LOG_INITIALIZED", L"1");
		}
		gBufferedLines.clear();
		gUnifiedLogFlushed = true;
	}

	bool EnvironmentOff(const wchar_t* name)
	{
		wchar_t value[32]{};
		const DWORD length = GetEnvironmentVariableW(name, value, static_cast<DWORD>(std::size(value)));
		if (length == 0 || length >= std::size(value))
			return false;

		return _wcsicmp(value, L"0") == 0 ||
			_wcsicmp(value, L"false") == 0 ||
			_wcsicmp(value, L"off") == 0 ||
			_wcsicmp(value, L"no") == 0;
	}

	bool EnvironmentOn(const wchar_t* name)
	{
		wchar_t value[32]{};
		const DWORD length = GetEnvironmentVariableW(name, value, static_cast<DWORD>(std::size(value)));
		if (length == 0 || length >= std::size(value))
			return false;

		return _wcsicmp(value, L"1") == 0 ||
			_wcsicmp(value, L"true") == 0 ||
			_wcsicmp(value, L"on") == 0 ||
			_wcsicmp(value, L"yes") == 0;
	}

	void EnableDredEarly()
	{
		HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
		if (!d3d12)
			d3d12 = LoadLibraryW(L"d3d12.dll");
		if (!d3d12)
			return;

		using GetDebugIfaceFn = HRESULT(WINAPI*)(REFIID, void**);
		auto getDebugIface = reinterpret_cast<GetDebugIfaceFn>(GetProcAddress(d3d12, "D3D12GetDebugInterface"));
		if (!getDebugIface)
			return;

		Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> dred1;
		if (SUCCEEDED(getDebugIface(IID_PPV_ARGS(&dred1))) && dred1)
		{
			dred1->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			dred1->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			dred1->SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			WriteLog("info", "ForceDRED enabled early before D3D12CreateDevice (AutoBreadcrumbs + PageFault + Context)");
			return;
		}

		Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred0;
		if (SUCCEEDED(getDebugIface(IID_PPV_ARGS(&dred0))) && dred0)
		{
			dred0->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			dred0->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
			WriteLog("info", "ForceDRED enabled early before D3D12CreateDevice (AutoBreadcrumbs + PageFault)");
		}
	}

	std::once_flag gBootstrapOnce;

	void BootstrapImpl()
	{
		// Crash reporters and web helpers beside the game's exe load OptiScaler too. They get nothing: no hand-off to
		// the main DLL, no log, no .session marker.
		if (EarlyLoader::IsCurrentProcessHelper())
			return;

		const auto self = LoaderModulePath();
		WriteLog("info", std::format("EarlyLoader starting from {}", self.string()));

		const auto optiRoot = self.parent_path().parent_path();
		const auto iniPath = optiRoot / L"OptiScaler.ini";
		const auto ini = EarlyLoader::ParseOptiScalerIni(iniPath);
		const auto selected = EarlyLoader::SelectMainDll(optiRoot, ini);

		if ((ini.hasForceDred && ini.forceDred) || EnvironmentOn(L"XEFG_FORCE_DRED"))
			EnableDredEarly();

		if (EnvironmentOff(L"XEFG_VULKAN_HOOKS"))
		{
			WriteLog("info", "XEFG_VULKAN_HOOKS is disabled; skipping early Vulkan bootstrap");
			FlushLogNextToDll(selected.path);
			return;
		}

		if (ini.found && ini.hasVulkanHooks && !ini.vulkanHooks)
		{
			WriteLog("info", "VulkanHooks=false in OptiScaler.ini; skipping early Vulkan bootstrap");
			FlushLogNextToDll(selected.path);
			return;
		}

		if (selected.path.empty())
		{
			WriteLog("warn", std::format("No XeFG main DLL found under {}\\custom\\*\\dlssg", optiRoot.string()));
			return;
		}

		if (selected.totalMatches > 1)
		{
			WriteLog("warn", std::format("Found {} matching XeFG DLLs under custom\\*\\dlssg; selected {} ({})",
				selected.totalMatches, selected.path.string(), selected.reason));
		}
		else
		{
			WriteLog("info", std::format("Selected main DLL {} ({})", selected.path.string(), selected.reason));
		}

		HMODULE module = LoadLibraryW(selected.path.c_str());
		if (!module)
		{
			WriteLog("error", std::format("LoadLibraryW failed for {} (error {})",
				selected.path.string(), GetLastError()));
			FlushLogNextToDll(selected.path);
			return;
		}

		using BootstrapFn = void (*)(const char*);
		if (const auto bootstrap = reinterpret_cast<BootstrapFn>(
				GetProcAddress(module, EarlyLoader::kBootstrapExportName)))
		{
			WriteLog("info", std::format("Handing off early Vulkan bootstrap to {}", selected.path.string()));
			FlushLogNextToDll(selected.path);
			bootstrap("EarlyLoader.asi");
			return;
		}

		using LegacyAsiFn = void (*)();
		if (const auto legacy = reinterpret_cast<LegacyAsiFn>(
				GetProcAddress(module, EarlyLoader::kLegacyAsiExportName)))
		{
			WriteLog("warn", std::format("{} has no {}; falling back to legacy InitializeASI",
				selected.path.string(), EarlyLoader::kBootstrapExportName));
			FlushLogNextToDll(selected.path);
			legacy();
			return;
		}

		WriteLog("error", std::format("{} exports neither {} nor {}",
			selected.path.string(), EarlyLoader::kBootstrapExportName, EarlyLoader::kLegacyAsiExportName));
		FlushLogNextToDll(selected.path);
	}

	void BootstrapOnce()
	{
		std::call_once(gBootstrapOnce, BootstrapImpl);
	}
} // namespace

// OptiScaler ASI entry point (invoked early before the game creates its Vulkan instance/device).
extern "C" __declspec(dllexport) void InitializeASI()
{
	try
	{
		BootstrapOnce();
	}
	catch (const std::exception& error)
	{
		WriteLog("error", std::format("InitializeASI exception: {}", error.what()));
	}
	catch (...)
	{
		WriteLog("error", "InitializeASI failed with unknown exception");
	}
}

// OptiScaler v10+ may invoke OptiPluginInit instead of InitializeASI on .asi files that export it.
// Trigger the early bootstrap once, then return false so the loader never registers a duplicate UI tab.
extern "C" __declspec(dllexport) bool OptiPluginInit(const OptiPluginApi*, OptiPlugin*)
{
	try
	{
		BootstrapOnce();
	}
	catch (const std::exception& error)
	{
		WriteLog("error", std::format("OptiPluginInit exception: {}", error.what()));
	}
	catch (...)
	{
		WriteLog("error", "OptiPluginInit failed with unknown exception");
	}

	return false;
}
