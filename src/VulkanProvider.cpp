#include "VulkanProvider.h"
#include "Log.h"
#include "VulkanRuntime.h"

#include <Windows.h>
#include <psapi.h>

#include <atomic>
#include <mutex>
#include <vector>

#pragma comment(lib, "psapi.lib")

namespace
{
	using GetApiFn = const VulkanProvider::Api* (*)(uint32_t version);
	using ModelNameFn = const char* (*)();

	// Each pass over the loaded modules costs a GetProcAddress per module, so a missing asi isn't searched for more
	// often than this. The menu asks every frame
	constexpr ULONGLONG SearchEveryMs = 2000;
	// Not asked again sooner while the provider can't run, the answer only changes with the asi's state
	constexpr ULONGLONG RequeryEveryMs = 30000;

	// Found once and never replaced, the asi stays loaded for the life of the process
	std::atomic<const VulkanProvider::Api*> provider = nullptr;

	std::mutex mutex;
	ModelNameFn modelName = nullptr;
	std::string moduleUtf8;
	ULONGLONG searchedAt = 0;

	VulkanProvider::Info cached;
	ULONGLONG queriedAt = 0;
	bool queried = false;
	bool lastLogged = false;
	VulkanProvider::State lastLoggedState = VulkanProvider::State::NotLoaded;

	std::string Utf8(const wchar_t* text)
	{
		const int length = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
		if (length <= 1)
			return {};

		std::string result(length - 1, '\0');
		WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), length, nullptr, nullptr);
		return result;
	}

	bool LoadedModules(std::vector<HMODULE>& modules)
	{
		modules.assign(256, nullptr);

		// The list can grow between the two calls, a few tries are enough
		for (int attempt = 0; attempt < 4; attempt++)
		{
			DWORD needed = 0;
			if (!EnumProcessModules(GetCurrentProcess(), modules.data(), (DWORD)(modules.size() * sizeof(HMODULE)),
				&needed))
				return false;

			const size_t count = needed / sizeof(HMODULE);
			if (count <= modules.size())
			{
				modules.resize(count);
				return true;
			}

			modules.assign(count + 32, nullptr);
		}

		return false;
	}

	// The first module that exports the provider. The asi can be loaded under any name, it is found by its export like
	// MFG-Unlocker's own shim does. Called with the mutex held
	void Search()
	{
		std::vector<HMODULE> modules;
		if (!LoadedModules(modules))
			return;

		for (HMODULE module : modules)
		{
			auto getApi = reinterpret_cast<GetApiFn>(GetProcAddress(module, "MfgXeMFGProviderGetApi"));
			if (!getApi)
				continue;

			wchar_t path[MAX_PATH]{};
			GetModuleFileNameW(module, path, MAX_PATH);

			const VulkanProvider::Api* table = getApi(VulkanProvider::ApiVersion);
			if (!VulkanProvider::Usable(table))
			{
				Log::Warn("{} exports MfgXeMFGProviderGetApi, but its table isn't one this dll can use (size {}, version {})",
					Utf8(path), table ? table->size : 0, table ? table->version : 0);
				continue;
			}

			moduleUtf8 = Utf8(path);
			modelName = reinterpret_cast<ModelNameFn>(GetProcAddress(module, "XeFGModelName"));
			Log::Info("Vulkan XeMFG provider found in {}", moduleUtf8);

			provider.store(table, std::memory_order_release);
			return;
		}
	}

	// Called with the mutex held
	void Refresh(const VulkanProvider::Api* api)
	{
		using VulkanProvider::State;

		const bool builtIn = VulkanRuntime::Built();

		VulkanProvider::Info next;
		next.module = builtIn ? "built into this dll" : moduleUtf8;

		const uint32_t frames = api->supported();

		char reason[256]{};
		const bool blocked = VulkanProvider::HasReason(api) && api->reason(reason, sizeof(reason) - 1) != 0;

		if (blocked || frames == 0)
		{
			next.state = State::Blocked;
			next.reason = reason;

			if (next.reason.empty())
				next.reason = "XeMFG can't run here";
		}
		else
		{
			next.state = State::Ready;
			next.maxFrames = frames;
		}

		// The built-in runtime not being ready before any game has given it a Vulkan device is no news, most games never
		// do (it says why itself when one does). An asi is only there for Vulkan games, so what it says always is
		const bool noNews = builtIn && next.state != State::Ready && !VulkanProvider::initialized.load();

		if (!noNews && (!lastLogged || next.state != lastLoggedState))
		{
			if (next.state == State::Ready)
				Log::Info("Vulkan XeMFG {} is ready, up to {} generated frames per real frame",
					builtIn ? "runtime" : "provider", next.maxFrames);
			else
				Log::Warn("Vulkan XeMFG {} can't run: {}", builtIn ? "runtime" : "provider", next.reason);

			lastLogged = true;
			lastLoggedState = next.state;
		}

		cached = std::move(next);
		queried = true;
		queriedAt = GetTickCount64();
	}
} // namespace

const VulkanProvider::Api* VulkanProvider::Get()
{
	// The runtime built into this dll is the provider, no asi is looked for
	if (const auto* builtIn = VulkanRuntime::Table())
		return builtIn;

	if (auto found = provider.load(std::memory_order_acquire))
		return found;

	std::scoped_lock lock(mutex);

	if (auto found = provider.load(std::memory_order_relaxed))
		return found;

	const ULONGLONG now = GetTickCount64();
	if (searchedAt != 0 && now - searchedAt < SearchEveryMs)
		return nullptr;

	searchedAt = now;
	Search();

	return provider.load(std::memory_order_relaxed);
}

VulkanProvider::Info VulkanProvider::Query()
{
	const Api* api = Get();

	std::scoped_lock lock(mutex);

	if (!api)
		return {};

	// The built-in runtime changes its answer as the game goes through NGX's init and the first feature, and it is asked
	// from memory, so it is never cached
	if (VulkanRuntime::Built() || !queried ||
		(cached.state != State::Ready && GetTickCount64() - queriedAt >= RequeryEveryMs))
		Refresh(api);

	return cached;
}

std::string VulkanProvider::ModelName()
{
	if (VulkanRuntime::Built())
		return VulkanRuntime::ModelName();

	ModelNameFn name;
	{
		std::scoped_lock lock(mutex);
		name = modelName;
	}

	if (!name)
		return {};

	const char* text = name();
	return text ? text : "";
}

void VulkanProvider::DeviceChanged()
{
	std::scoped_lock lock(mutex);
	queried = false;
}
