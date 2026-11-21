#include "EarlyLoaderCore.h"
#include "../optiscaler_plugin.h"

#include <Windows.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>

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

	void WriteLog(std::string_view level, std::string_view message)
	{
		static std::mutex mutex;
		static std::ofstream file(LoaderModulePath().replace_extension(L".log"), std::ios::trunc);

		SYSTEMTIME now{};
		GetLocalTime(&now);
		const auto line = std::format("[{:02}:{:02}:{:02}.{:03}] [{}] {}\n",
			now.wHour, now.wMinute, now.wSecond, now.wMilliseconds, level, message);

		std::scoped_lock lock(mutex);
		if (IsDebuggerPresent())
			OutputDebugStringA(line.c_str());

		if (file)
		{
			file.write(line.data(), static_cast<std::streamsize>(line.size()));
			file.flush();
		}
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

	std::once_flag gBootstrapOnce;

	void BootstrapImpl()
	{
		const auto self = LoaderModulePath();
		WriteLog("info", std::format("EarlyLoader starting from {}", self.string()));

		if (EnvironmentOff(L"XEFG_VULKAN_HOOKS"))
		{
			WriteLog("info", "XEFG_VULKAN_HOOKS is disabled; skipping early Vulkan bootstrap");
			return;
		}

		const auto optiRoot = self.parent_path().parent_path();
		const auto iniPath = optiRoot / L"OptiScaler.ini";
		const auto ini = EarlyLoader::ParseOptiScalerIni(iniPath);

		if (ini.found && ini.hasVulkanHooks && !ini.vulkanHooks)
		{
			WriteLog("info", "VulkanHooks=false in OptiScaler.ini; skipping early Vulkan bootstrap");
			return;
		}

		const auto selected = EarlyLoader::SelectMainDll(optiRoot, ini);
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
			return;
		}

		using BootstrapFn = void (*)(const char*);
		if (const auto bootstrap = reinterpret_cast<BootstrapFn>(
				GetProcAddress(module, EarlyLoader::kBootstrapExportName)))
		{
			WriteLog("info", std::format("Handing off early Vulkan bootstrap to {}", selected.path.string()));
			bootstrap("EarlyLoader.asi");
			return;
		}

		using LegacyAsiFn = void (*)();
		if (const auto legacy = reinterpret_cast<LegacyAsiFn>(
				GetProcAddress(module, EarlyLoader::kLegacyAsiExportName)))
		{
			WriteLog("warn", std::format("{} has no {}; falling back to legacy InitializeASI",
				selected.path.string(), EarlyLoader::kBootstrapExportName));
			legacy();
			return;
		}

		WriteLog("error", std::format("{} exports neither {} nor {}",
			selected.path.string(), EarlyLoader::kBootstrapExportName, EarlyLoader::kLegacyAsiExportName));
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
