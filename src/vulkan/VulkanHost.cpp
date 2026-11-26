#include "VulkanHost.h"
#include "VulkanCoopMat.h"

#include "../Log.h"
#include "../Plugin.h"
#include "../WineCompat.h"
#include "VulkanPolicy.h"
#include "VulkanFeatureChain.h"
#include "VulkanTensor.h"

#include <MinHook.h>
#ifdef _MSC_VER
#include <intrin.h>
#define XEFG_RETURN_ADDRESS() _ReturnAddress()
#else
#define XEFG_RETURN_ADDRESS() __builtin_return_address(0)
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cwchar>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

// See VulkanHost.h. The hooks are MinHook inline detours on vulkan-1.dll's own vkCreateInstance and vkCreateDevice, so
// they catch the game however it got those functions: linked, GetProcAddress, vkGetInstanceProcAddr, a cached pointer.
// An overlay or a layer that hooks the same functions chains with them, MinHook follows the jump an earlier hook left.

namespace
{
	constexpr size_t MaxInstances = 16;
	constexpr size_t MaxDevices = 64;
	constexpr size_t MaxSwapchains = 64;
	constexpr wchar_t LoaderName[] = L"vulkan-1.dll";
	constexpr DWORD HighResTimerFlag = 0x00000002u; // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION

	struct State
	{
		// The maps and the strings, held briefly and never while calling out
		std::shared_mutex instanceLifecycleMutex;
		std::mutex mutex;
		std::vector<std::pair<VkInstance, uint32_t>> instances; // live instances in creation order (newest checked first)
		std::unordered_set<VkInstance> ignoredInstances; // skipped translation-layer / internal WMMA bridge instances
		std::map<VkDevice, VulkanHost::DeviceRecord> devices;
		std::map<VkSwapchainKHR, VkPresentModeKHR> swapchains;
		std::map<VkPhysicalDevice, std::string> refused;
		std::string detail;
		std::string lastRefusal;
		VulkanRuntime::Hooks hooks = VulkanRuntime::Hooks::Off;

		// Start and the load hook installing the hooks. loaderHooked is also read by the load hook without the lock
		std::mutex install;
		bool started = false;
		std::atomic_bool loaderHooked = false;

		std::atomic_bool enabled = true;
		std::atomic_uint32_t instanceCalls = 0;
		std::atomic_uint32_t deviceCalls = 0;
		std::atomic_uint32_t prepared = 0;

		using DestroyInstanceFn = void(VKAPI_PTR*)(VkInstance, const VkAllocationCallbacks*);

		PFN_vkGetInstanceProcAddr loaderGipa = nullptr;
		PFN_vkGetDeviceProcAddr loaderGdpa = nullptr;
		PFN_vkCreateInstance createInstance = nullptr;
		DestroyInstanceFn destroyInstance = nullptr;
		PFN_vkCreateDevice createDevice = nullptr;
		PFN_vkCreateSwapchainKHR createSwapchain = nullptr;
		PFN_vkDestroySwapchainKHR destroySwapchain = nullptr;
		PFN_vkQueuePresentKHR queuePresent = nullptr;
		void* rawCreateInstance = nullptr;
		void* rawDestroyInstance = nullptr;
		void* rawCreateDevice = nullptr;
		void* rawGetDeviceProcAddr = nullptr;
		void* rawCreateSwapchain = nullptr;
		void* rawDestroySwapchain = nullptr;
		void* rawQueuePresent = nullptr;
		std::atomic<PFN_vkCreateInstance> gipaCreateInstance = nullptr;
		std::atomic<DestroyInstanceFn> gipaDestroyInstance = nullptr;
		std::atomic<PFN_vkCreateDevice> gipaCreateDevice = nullptr;
		std::atomic<PFN_vkGetDeviceProcAddr> gipaGetDeviceProcAddr = nullptr;
		std::atomic<PFN_vkCreateSwapchainKHR> gipaCreateSwapchain = nullptr;
		std::atomic<PFN_vkDestroySwapchainKHR> gipaDestroySwapchain = nullptr;
		std::atomic<PFN_vkQueuePresentKHR> gipaQueuePresent = nullptr;
		std::atomic<PFN_vkCreateSwapchainKHR> gdpaCreateSwapchain = nullptr;
		std::atomic<PFN_vkDestroySwapchainKHR> gdpaDestroySwapchain = nullptr;
		std::atomic<PFN_vkQueuePresentKHR> gdpaQueuePresent = nullptr;
		std::atomic<PFN_vkCreateSwapchainKHR> icdCreateSwapchain = nullptr;
		std::atomic<PFN_vkDestroySwapchainKHR> icdDestroySwapchain = nullptr;
		std::atomic<PFN_vkQueuePresentKHR> icdQueuePresent = nullptr;
		void* hookedIcdCreateSwap = nullptr;
		void* hookedIcdDestroySwap = nullptr;
		void* hookedIcdQueuePresent = nullptr;
		std::atomic_bool icdCreateSwapActive = false;
		std::atomic_bool icdDestroySwapActive = false;
		std::atomic_bool icdQueuePresentActive = false;
		std::atomic<PFN_vkGetInstanceProcAddr> icdGipaTramp = nullptr;
		std::atomic<PFN_vkGetDeviceProcAddr> icdGdpaTramp = nullptr;
		void* hookedIcdGipa = nullptr;
		void* hookedIcdGdpa = nullptr;
		std::atomic<void*> capturedIcdCreateSwap = nullptr;
		std::atomic<void*> capturedIcdDestroySwap = nullptr;
		std::atomic<void*> capturedIcdQueuePresent = nullptr;
		std::atomic_bool icdCreateSwapIsNative = false;
		HMODULE loaderModule = nullptr;
		bool gipaHooked = false;
		bool gdpaHooked = false;
		std::atomic_bool presentHooked = false;
		std::atomic_bool gdpaPresentLogged = false;
		HMODULE(WINAPI* loadLibraryExW)(LPCWSTR, HANDLE, DWORD) = nullptr;

		// Dedicated lock for low-overhead present pacing
		std::mutex pacerMutex;
		VulkanPolicy::PresentPacer pacer;
		std::atomic_bool timerPeriodRaised = false;
		std::atomic_bool hwPacingActive = false;
		std::atomic_bool pacingLogged = false;
		std::atomic_bool autoSleepSpin = false;
		std::atomic_uint32_t cpuOverrunStreak = 0;
		std::atomic_uint32_t cpuCleanStreak = 0;
		std::atomic_bool translationSkipLogged = false;
		std::atomic_bool selfSkipLogged = false;

		// Lightweight present-pacing diagnostics (atomic counters; logged once + every 5s up to 12 times)
		std::atomic_bool firstPresentLogged = false;
		std::atomic_uint64_t diagPresentCalls = 0;
		std::atomic_uint64_t diagPresentPaced = 0;
		std::atomic_uint64_t diagPresentFifo = 0;
		std::atomic_uint64_t diagPresentWaitSumUs = 0;
		std::atomic_uint64_t diagLastLogNs = 0;
		std::atomic_uint32_t diagLogCount = 0;
		std::atomic_bool diagPreFgLogged = false;
		std::atomic_bool diagFgWindowStarted = false;
	};

	// Leaked on purpose: a game's thread can call the hooks while the process exits, after the static destructors
	State& S()
	{
		static State* state = new State;
		return *state;
	}

	template <class T> T* FindStruct(const void* next, VkStructureType type)
	{
		for (unsigned links = 0; next && links < 64; ++links)
		{
			auto* base = static_cast<const VkBaseInStructure*>(next);

			if (base->sType == type)
				return reinterpret_cast<T*>(const_cast<VkBaseInStructure*>(base));

			next = base->pNext;
		}

		return nullptr;
	}

	void SetHookState(VulkanRuntime::Hooks hooks, std::string detail)
	{
		State& s = S();
		std::lock_guard lock(s.mutex);
		s.hooks = hooks;
		s.detail = std::move(detail);
	}

	enum class PacingMode { Off, Auto, Force };

	PacingMode GetPacingMode() noexcept
	{
		static const PacingMode mode = [] {
			wchar_t value[16]{};
			const DWORD len = GetEnvironmentVariableW(L"XEFG_VK_PACING", value, 16);
			if (len && len < 16)
			{
				if (value[0] == L'0' || _wcsicmp(value, L"off") == 0 || _wcsicmp(value, L"false") == 0)
					return PacingMode::Off;
				if (_wcsicmp(value, L"force") == 0 || _wcsicmp(value, L"always") == 0)
					return PacingMode::Force;
			}
			return PacingMode::Auto;
		}();
		return mode;
	}

	bool PacingActive(const State& s) noexcept
	{
		const PacingMode mode = GetPacingMode();
		if (mode == PacingMode::Off)
			return false;
		if (mode == PacingMode::Force)
			return true;
		return !s.hwPacingActive.load(std::memory_order_relaxed);
	}

	uint32_t GetSwapchainBufferTarget() noexcept
	{
		static const uint32_t target = [] {
			wchar_t value[32]{};
			const DWORD len = GetEnvironmentVariableW(L"XEFG_VK_SWAPCHAIN_BUFFERS", value, 32);
			if (len && len < 32)
				return VulkanPolicy::ParseSwapchainBufferTarget(std::wstring_view(value, len), 8u);
			return 8u;
		}();
		return target;
	}

	uint64_t NowNs() noexcept
	{
		static const uint64_t freq = [] {
			LARGE_INTEGER f{};
			QueryPerformanceFrequency(&f);
			return f.QuadPart > 0 ? static_cast<uint64_t>(f.QuadPart) : 10'000'000ull;
		}();
		LARGE_INTEGER counter{};
		QueryPerformanceCounter(&counter);
		const uint64_t ticks = counter.QuadPart > 0 ? static_cast<uint64_t>(counter.QuadPart) : 0ull;
		const uint64_t sec = ticks / freq;
		const uint64_t rem = ticks % freq;
		return sec * 1'000'000'000ull + (rem * 1'000'000'000ull) / freq;
	}

	// Sub-millisecond wait using a per-thread Windows 10/11 high-resolution waitable timer (avoiding winmm.dll
	// which OptiScaler frequently proxies) plus a short QPC spin-wait tail.
	void WaitPreciseNs(uint64_t waitNs) noexcept
	{
		if (waitNs == 0)
			return;

		const uint64_t deadline = NowNs() + waitNs;
		static const bool lowCoreCpu = []() noexcept {
			const unsigned hc = std::thread::hardware_concurrency();
			return hc > 0u && hc <= 6u;
		}();
		// Alt-Tab / minimized window guard: skip pacer sleep/spin while the game window is minimized.
		{
			using IsIconicFn = BOOL(WINAPI*)(HWND);
			using GetWindowThreadProcessIdFn = DWORD(WINAPI*)(HWND, DWORD*);
			static const auto isIconicFn = [] {
				const HMODULE u32 = GetModuleHandleW(L"user32.dll");
				return u32 ? reinterpret_cast<IsIconicFn>(GetProcAddress(u32, "IsIconic")) : nullptr;
			}();
			static const auto getWndPidFn = [] {
				const HMODULE u32 = GetModuleHandleW(L"user32.dll");
				return u32 ? reinterpret_cast<GetWindowThreadProcessIdFn>(GetProcAddress(u32, "GetWindowThreadProcessId")) : nullptr;
			}();
			static std::atomic<HWND> sGameHwnd{ nullptr };
			static const DWORD sSelfPid = GetCurrentProcessId();
			if (isIconicFn && getWndPidFn)
			{
				if (HWND fg = GetForegroundWindow())
				{
					DWORD fgPid = 0;
					getWndPidFn(fg, &fgPid);
					if (fgPid == sSelfPid)
						sGameHwnd.store(fg, std::memory_order_relaxed);
				}
				if (HWND gameHwnd = sGameHwnd.load(std::memory_order_relaxed))
				{
					if (isIconicFn(gameHwnd))
						return;
				}
			}
		}
		State& s = S();
		const bool underWine = WineCompat::IsWineOrProton();
		const bool userSleepSpin = Plugin::options.pacingSpinlockSleep.load(std::memory_order_relaxed);
		const bool sleepSpin = userSleepSpin || lowCoreCpu || underWine || s.autoSleepSpin.load(std::memory_order_relaxed);
		const uint64_t SpinTailNs = sleepSpin ? 250'000ull : 1'200'000ull; // 0.25 ms sleep-friendly tail vs 1.2 ms QPC spin tail

		using CreateWaitableTimerExWFn = HANDLE(WINAPI*)(void*, LPCWSTR, DWORD, DWORD);
		static const auto createTimerEx = [] {
			const HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
			return k32 ? reinterpret_cast<CreateWaitableTimerExWFn>(GetProcAddress(k32, "CreateWaitableTimerExW")) : nullptr;
		}();

		struct ThreadTimer
		{
			HANDLE handle = nullptr;
			~ThreadTimer() { if (handle) CloseHandle(handle); }
		};
		static thread_local ThreadTimer timer;
		static thread_local bool timerAttempted = false;
		if (!timerAttempted)
		{
			timerAttempted = true;
			if (createTimerEx)
				timer.handle = createTimerEx(nullptr, nullptr, HighResTimerFlag, 0x001F0003u /* TIMER_ALL_ACCESS */);
		}

		uint64_t now = NowNs();
		if (now + SpinTailNs < deadline)
		{
			const uint64_t sleepNs = (deadline - now) - SpinTailNs;
			if (timer.handle)
			{
				LARGE_INTEGER due{};
				due.QuadPart = -static_cast<int64_t>(sleepNs / 100ull); // relative 100 ns intervals
				if (due.QuadPart < 0 && SetWaitableTimer(timer.handle, &due, 0, nullptr, nullptr, 0))
					WaitForSingleObject(timer.handle, 50u);
			}
			else if (sleepNs >= 2'000'000ull)
			{
				Sleep(static_cast<DWORD>(sleepNs / 1'000'000ull));
			}
		}

		while (true)
		{
			const uint64_t cur = NowNs();
			if (cur >= deadline)
				break;
			if (sleepSpin && (deadline - cur) > 100'000ull)
				SwitchToThread();
			else
				YieldProcessor();
		}

		// Auto-adaptive CPU-bound detection: if wait overshoots deadline by >0.75 ms for 6 consecutive presents
		// (OS scheduler starvation), automatically engage sleep-friendly 0.25 ms tail to yield CPU back to the game.
		if (!userSleepSpin && !lowCoreCpu && !underWine)
		{
			const uint64_t endNs = NowNs();
			if (endNs > deadline + 750'000ull)
			{
				s.cpuCleanStreak.store(0, std::memory_order_relaxed);
				if (s.cpuOverrunStreak.fetch_add(1, std::memory_order_relaxed) + 1u >= 6u)
					s.autoSleepSpin.store(true, std::memory_order_relaxed);
			}
			else if (endNs <= deadline + 250'000ull)
			{
				s.cpuOverrunStreak.store(0, std::memory_order_relaxed);
				if (s.autoSleepSpin.load(std::memory_order_relaxed) &&
					s.cpuCleanStreak.fetch_add(1, std::memory_order_relaxed) + 1u >= 60u)
				{
					s.autoSleepSpin.store(false, std::memory_order_relaxed);
					s.cpuCleanStreak.store(0, std::memory_order_relaxed);
				}
			}
		}
	}

	// ----------------------------------------------------------------------------------------------- instance

	struct InstanceEdit
	{
		VkInstanceCreateInfo info{};
		VkApplicationInfo application{};
	};

	uint32_t RequestedVersion(const VkInstanceCreateInfo* info)
	{
		return info->pApplicationInfo && info->pApplicationInfo->apiVersion ? info->pApplicationInfo->apiVersion
		                                                                  : VK_API_VERSION_1_0;
	}

	bool ContainsInsensitive(const char* haystack, const char* needle) noexcept
	{
		if (!haystack || !needle || !*needle)
			return false;
		for (const char* p = haystack; *p; ++p)
		{
			const char* a = p;
			const char* b = needle;
			while (*a && *b)
			{
				const unsigned char ca = static_cast<unsigned char>(*a);
				const unsigned char cb = static_cast<unsigned char>(*b);
				const unsigned char la = (ca >= 'A' && ca <= 'Z') ? static_cast<unsigned char>(ca + 32) : ca;
				const unsigned char lb = (cb >= 'A' && cb <= 'Z') ? static_cast<unsigned char>(cb + 32) : cb;
				if (la != lb)
					break;
				++a;
				++b;
			}
			if (!*b)
				return true;
		}
		return false;
	}

	bool ContainsInsensitiveW(const wchar_t* haystack, const wchar_t* needle) noexcept
	{
		if (!haystack || !needle || !*needle)
			return false;
		for (const wchar_t* p = haystack; *p; ++p)
		{
			const wchar_t* a = p;
			const wchar_t* b = needle;
			while (*a && *b)
			{
				wchar_t la = (*a >= L'A' && *a <= L'Z') ? static_cast<wchar_t>(*a + 32) : *a;
				wchar_t lb = (*b >= L'A' && *b <= L'Z') ? static_cast<wchar_t>(*b + 32) : *b;
				if (la != lb)
					break;
				++a;
				++b;
			}
			if (!*b)
				return true;
		}
		return false;
	}

	// On Wine / Proton (Linux), D3D12/D3D11 games run through VKD3D-Proton or DXVK, which create internal
	// Vulkan instances and devices inside d3d12.dll / d3d12core.dll / dxgi.dll / d3d11.dll.
	// Mutating those internal instances/devices (extensions, pNext chains, swapchains) crashes VKD3D-Proton.
	thread_local int gInternalBridgeBypassDepth = 0;

	bool IsInternalBridgeBypassed() noexcept
	{
		return gInternalBridgeBypassDepth > 0;
	}

	HMODULE SelfModule() noexcept
	{
		HMODULE mod = nullptr;
		GetModuleHandleExW(
			GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			reinterpret_cast<LPCWSTR>(&SelfModule), &mod);
		return mod;
	}

	bool IsSelfModuleCaller(const void* caller) noexcept
	{
		if (!caller)
			return false;
		static const HMODULE self = SelfModule();
		if (!self)
			return false;
		HMODULE mod = nullptr;
		if (!GetModuleHandleExW(
				GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(caller), &mod) || !mod)
			return false;
		return mod == self;
	}

	std::string DescribeModuleOfAddress(const void* addr) noexcept
	{
		if (!addr)
			return "null";
		HMODULE mod = nullptr;
		if (!GetModuleHandleExW(
				GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(addr), &mod) || !mod)
			return "unknown";
		wchar_t path[MAX_PATH]{};
		const DWORD len = GetModuleFileNameW(mod, path, MAX_PATH);
		if (len == 0 || len >= MAX_PATH)
			return "unknown";
		const wchar_t* base = path;
		for (DWORD i = 0; i < len; ++i)
		{
			if (path[i] == L'\\' || path[i] == L'/')
				base = path + i + 1;
		}
		std::string out;
		out.reserve(64);
		for (const wchar_t* p = base; *p; ++p)
			out.push_back((*p >= 0x20 && *p < 0x7F) ? static_cast<char>(*p) : '?');
		return out.empty() ? std::string("unknown") : out;
	}

	bool IsReadableMemory(const void* ptr, size_t bytes) noexcept
	{
		if (!ptr || bytes == 0)
			return false;
		MEMORY_BASIC_INFORMATION mbi{};
		if (VirtualQuery(ptr, &mbi, sizeof(mbi)) != sizeof(mbi))
			return false;
		if (mbi.State != MEM_COMMIT)
			return false;
		if ((mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0)
			return false;
		const DWORD baseProt = mbi.Protect & 0xFFu;
		const bool readable =
			baseProt == PAGE_READONLY ||
			baseProt == PAGE_READWRITE ||
			baseProt == PAGE_WRITECOPY ||
			baseProt == PAGE_EXECUTE_READ ||
			baseProt == PAGE_EXECUTE_READWRITE ||
			baseProt == PAGE_EXECUTE_WRITECOPY;
		if (!readable)
			return false;
		const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
		const uintptr_t regionEnd = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
		return start <= regionEnd && (regionEnd - start) >= bytes;
	}

	bool IsUnsafeShortStub(const void* proc) noexcept
	{
		if (!IsReadableMemory(proc, 14u))
			return true;
		const auto* b = reinterpret_cast<const uint8_t*>(proc);
		// Already detoured by OptiScaler/overlay (JMP rel32, JMP rel8, or JMP [rip+disp32]).
		if (b[0] == 0xE9 || b[0] == 0xEB || (b[0] == 0xFF && b[1] == 0x25))
			return true;
		// Short loader device-dispatch trampoline: mov rax, [rcx] (48 8B 01) + jmp [rax+disp8/disp32] (FF 60 / FF A0).
		// Safe only when followed by INT3/NOP padding out to >= 14 bytes; tightly packed stubs risk corrupting
		// adjacent functions if MinHook falls back to an absolute jump (e.g. Longshot / custom loaders).
		if (b[0] == 0x48 && b[1] == 0x8B && b[2] == 0x01 && b[3] == 0xFF && (b[4] == 0x60 || b[4] == 0xA0))
		{
			const size_t stubLen = (b[4] == 0x60) ? 6u : 9u;
			for (size_t i = stubLen; i < 14u; ++i)
			{
				if (b[i] != 0xCC && b[i] != 0x90)
					return true;
			}
		}
		return false;
	}

	// Reads VkLayerDispatchTable::GetDeviceProcAddr (slot 0 of the dispatch table stored at *(void***)device).
	// This bypasses OptiScaler's Detours hooks on vulkan-1.dll exports so device-level hooks attach downstream of
	// OptiScaler and Streamline (sl.dlss_g), directly on the ICD / bottom Vulkan layer.
	PFN_vkGetDeviceProcAddr ResolveDispatchTableGdpa(const State& s, VkDevice device, std::string& moduleOut) noexcept
	{
		if (!device || !IsReadableMemory(device, sizeof(void*)))
			return nullptr;
		void* const* dispatchTable = *reinterpret_cast<void* const* const*>(device);
		if (!dispatchTable || !IsReadableMemory(dispatchTable, sizeof(void*)))
			return nullptr;
		void* fn0 = dispatchTable[0];
		if (!fn0 || fn0 == s.rawGetDeviceProcAddr || IsSelfModuleCaller(fn0))
			return nullptr;
		HMODULE mod = nullptr;
		if (!GetModuleHandleExW(
				GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(fn0), &mod) || !mod)
			return nullptr;
		auto candidate = reinterpret_cast<PFN_vkGetDeviceProcAddr>(fn0);
		// Sanity-probe with mandatory core device command vkDestroyDevice before trusting slot 0.
		if (!candidate(device, "vkDestroyDevice"))
			return nullptr;
		moduleOut = DescribeModuleOfAddress(fn0);
		return candidate;
	}

	bool IsInternalBridgeInstance(const VkInstanceCreateInfo* info) noexcept
	{
		if (!info || !info->pApplicationInfo)
			return false;
		const char* app = info->pApplicationInfo->pApplicationName;
		return app && std::strcmp(app, "dlssg-to-xefg-wmma") == 0;
	}

	bool IsTranslationLayerInstance(const VkInstanceCreateInfo* info) noexcept
	{
		if (!info || !info->pApplicationInfo)
			return false;
		const char* engine = info->pApplicationInfo->pEngineName;
		const char* app = info->pApplicationInfo->pApplicationName;
		return ContainsInsensitive(engine, "dxvk") || ContainsInsensitive(engine, "vkd3d") ||
		       ContainsInsensitive(app, "dxvk") || ContainsInsensitive(app, "vkd3d");
	}

	bool IsTranslationLayerCaller(const void* caller) noexcept
	{
		if (!caller)
			return false;
		HMODULE mod = nullptr;
		if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		        reinterpret_cast<LPCWSTR>(caller), &mod) || !mod)
			return false;
		wchar_t path[MAX_PATH]{};
		const DWORD len = GetModuleFileNameW(mod, path, MAX_PATH);
		if (!len || len >= MAX_PATH)
			return false;
		const wchar_t* leaf = wcsrchr(path, L'\\');
		leaf = leaf ? (leaf + 1) : path;
		return _wcsicmp(leaf, L"d3d12.dll") == 0 ||
		       _wcsicmp(leaf, L"d3d12core.dll") == 0 ||
		       _wcsicmp(leaf, L"dxgi.dll") == 0 ||
		       _wcsicmp(leaf, L"d3d11.dll") == 0 ||
		       _wcsicmp(leaf, L"d3d9.dll") == 0 ||
		       _wcsicmp(leaf, L"d3d8.dll") == 0 ||
		       ContainsInsensitiveW(leaf, L"vkd3d") ||
		       ContainsInsensitiveW(leaf, L"dxvk");
	}

	// XeFG's programs need Vulkan 1.3. A game that asks for less gets 1.3 when the loader has it: the physical device
	// decides what the device can use, the instance only has to allow it
	const VkInstanceCreateInfo* PrepareInstance(const VkInstanceCreateInfo* info, InstanceEdit& edit) noexcept
	{
		State& s = S();

		if (!info || !s.loaderGipa)
			return info;

		try
		{
			const uint32_t requested = RequestedVersion(info);

			if (requested >= xvk::kRequiredApiVersion)
				return info;

			auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
				s.loaderGipa(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
			uint32_t loaderVersion = VK_API_VERSION_1_0;

			if (!enumerate || enumerate(&loaderVersion) != VK_SUCCESS || loaderVersion < xvk::kRequiredApiVersion)
			{
				Log::Warn("Vulkan loader is older than 1.3, the game's instance stays at {}.{}",
					VK_API_VERSION_MAJOR(requested), VK_API_VERSION_MINOR(requested));
				return info;
			}

			edit.info = *info;

			if (info->pApplicationInfo)
				edit.application = *info->pApplicationInfo;
			else
				edit.application = { VK_STRUCTURE_TYPE_APPLICATION_INFO };

			edit.application.apiVersion = xvk::kRequiredApiVersion;
			edit.info.pApplicationInfo = &edit.application;

			Log::Info("Vulkan instance: the game asks for {}.{}, created as 1.3", VK_API_VERSION_MAJOR(requested),
				VK_API_VERSION_MINOR(requested));

			return &edit.info;
		}
		catch (...)
		{
			return info;
		}
	}

	bool IsKnownIcdName(const wchar_t* name) noexcept
	{
		if (!name)
			return false;
		const wchar_t* leaf = name;
		for (const wchar_t* p = name; *p; ++p)
		{
			if (*p == L'\\' || *p == L'/')
				leaf = p + 1;
		}
		return _wcsicmp(leaf, L"nvoglv64.dll") == 0 ||
		       _wcsicmp(leaf, L"amdvlk64.dll") == 0 ||
		       _wcsicmp(leaf, L"igvk64.dll") == 0 ||
		       _wcsicmp(leaf, L"xe_vk64.dll") == 0 ||
		       _wcsicmp(leaf, L"igxelpicd64.dll") == 0 ||
		       _wcsicmp(leaf, L"winevulkan.dll") == 0;
	}

	VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookIcdGetDeviceProcAddr(VkDevice device, const char* pName)
	{
		State& s = S();
		PFN_vkGetDeviceProcAddr next = s.icdGdpaTramp.load(std::memory_order_acquire);
		PFN_vkVoidFunction fn = next ? next(device, pName) : nullptr;
		if (fn && pName)
		{
			if (std::strcmp(pName, "vkCreateSwapchainKHR") == 0)
				s.capturedIcdCreateSwap.store(reinterpret_cast<void*>(fn), std::memory_order_release);
			else if (std::strcmp(pName, "vkDestroySwapchainKHR") == 0)
				s.capturedIcdDestroySwap.store(reinterpret_cast<void*>(fn), std::memory_order_release);
			else if (std::strcmp(pName, "vkQueuePresentKHR") == 0)
				s.capturedIcdQueuePresent.store(reinterpret_cast<void*>(fn), std::memory_order_release);
		}
		return fn;
	}

	void TryHookIcdGdpaLocked(State& s, void* target) noexcept
	{
		if (!target || s.hookedIcdGdpa != nullptr)
			return;
		PFN_vkGetDeviceProcAddr tramp = nullptr;
		if (MH_CreateHook(target, reinterpret_cast<void*>(&HookIcdGetDeviceProcAddr),
		        reinterpret_cast<void**>(&tramp)) != MH_OK || !tramp)
			return;
		s.icdGdpaTramp.store(tramp, std::memory_order_release);
		if (MH_EnableHook(target) != MH_OK)
		{
			s.icdGdpaTramp.store(nullptr, std::memory_order_release);
			MH_RemoveHook(target);
			return;
		}
		s.hookedIcdGdpa = target;
	}

	VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookIcdGetInstanceProcAddr(VkInstance instance, const char* pName)
	{
		State& s = S();
		PFN_vkGetInstanceProcAddr next = s.icdGipaTramp.load(std::memory_order_acquire);
		PFN_vkVoidFunction fn = next ? next(instance, pName) : nullptr;
		if (fn && pName)
		{
			if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0)
			{
				if (!s.icdGdpaTramp.load(std::memory_order_acquire))
				{
					std::lock_guard lock(s.install);
					TryHookIcdGdpaLocked(s, reinterpret_cast<void*>(fn));
				}
			}
			else if (std::strcmp(pName, "vkCreateSwapchainKHR") == 0)
			{
				void* expected = nullptr;
				s.capturedIcdCreateSwap.compare_exchange_strong(expected, reinterpret_cast<void*>(fn), std::memory_order_acq_rel);
			}
			else if (std::strcmp(pName, "vkDestroySwapchainKHR") == 0)
			{
				void* expected = nullptr;
				s.capturedIcdDestroySwap.compare_exchange_strong(expected, reinterpret_cast<void*>(fn), std::memory_order_acq_rel);
			}
			else if (std::strcmp(pName, "vkQueuePresentKHR") == 0)
			{
				void* expected = nullptr;
				s.capturedIcdQueuePresent.compare_exchange_strong(expected, reinterpret_cast<void*>(fn), std::memory_order_acq_rel);
			}
		}
		return fn;
	}

	// Attach a lightweight observer hook to the loaded Vulkan GPU ICD's vk_icdGetInstanceProcAddr so we capture
	// the ICD's real vkCreateSwapchainKHR / vkDestroySwapchainKHR / vkQueuePresentKHR pointers directly — sitting
	// downstream of RTSS, Steam Overlay, OptiScaler, and Streamline (sl.dlss_g).
	void TryHookLoadedIcd(State& s) noexcept
	{
		if (s.icdGipaTramp.load(std::memory_order_acquire))
			return;
		try
		{
			std::lock_guard lock(s.install);
			if (s.hookedIcdGipa != nullptr)
				return;

			constexpr const wchar_t* kIcdModules[] = {
				L"nvoglv64.dll",
				L"amdvlk64.dll",
				L"igvk64.dll",
				L"xe_vk64.dll",
				L"igxelpicd64.dll",
				L"winevulkan.dll",
			};

			for (const wchar_t* modName : kIcdModules)
			{
				HMODULE mod = GetModuleHandleW(modName);
				if (!mod)
					continue;
				void* target = reinterpret_cast<void*>(GetProcAddress(mod, "vk_icdGetInstanceProcAddr"));
				if (!target)
					continue;

				PFN_vkGetInstanceProcAddr tramp = nullptr;
				if (MH_CreateHook(target, reinterpret_cast<void*>(&HookIcdGetInstanceProcAddr),
				        reinterpret_cast<void**>(&tramp)) != MH_OK || !tramp)
					continue;

				s.icdGipaTramp.store(tramp, std::memory_order_release);
				if (MH_EnableHook(target) != MH_OK)
				{
					s.icdGipaTramp.store(nullptr, std::memory_order_release);
					MH_RemoveHook(target);
					continue;
				}

				s.hookedIcdGipa = target;
				Log::Info("Vulkan hooks: attached ICD proc capture on {} (vk_icdGetInstanceProcAddr)",
					DescribeModuleOfAddress(target));

				// Spec-safe probe with VK_NULL_HANDLE: if the ICD exposes vkGetDeviceProcAddr globally,
				// hook it immediately so even a single-instance game's first vkCreateDevice is captured.
				if (PFN_vkVoidFunction gdpaFn = tramp(VK_NULL_HANDLE, "vkGetDeviceProcAddr"))
					TryHookIcdGdpaLocked(s, reinterpret_cast<void*>(gdpaFn));
				return;
			}
		}
		catch (...)
		{
		}
	}

	void InstanceCreated(VkInstance instance, const VkInstanceCreateInfo* used) noexcept
	{
		try
		{
			State& s = S();
			std::lock_guard lock(s.mutex);

			s.instances.erase(
				std::remove_if(s.instances.begin(), s.instances.end(),
					[instance](const auto& entry) { return entry.first == instance; }),
				s.instances.end());
			s.ignoredInstances.erase(instance);

			if (s.instances.size() >= MaxInstances)
				s.instances.erase(s.instances.begin());

			s.instances.emplace_back(instance, RequestedVersion(used));
		}
		catch (...)
		{
		}
		TryHookLoadedIcd(S());
	}

	void InstanceDestroyed(VkInstance instance) noexcept
	{
		if (!instance)
			return;
		try
		{
			State& s = S();
			std::lock_guard lock(s.mutex);
			s.instances.erase(
				std::remove_if(s.instances.begin(), s.instances.end(),
					[instance](const auto& entry) { return entry.first == instance; }),
				s.instances.end());
			s.ignoredInstances.erase(instance);
		}
		catch (...)
		{
		}
	}

	// Capture a pre-existing VkInstance that was created before our hooks were installed (common in emulators like
	// Longshot that create VkInstance before loading version.dll/OptiScaler, then resolve vkCreateDevice via GIPA).
	// Stored with apiVersion == 0 so PrepareDevice validates Vulkan 1.3 support directly against the physical device.
	void TrackLateInstance(VkInstance instance, const void* caller) noexcept
	{
		if (!instance || IsInternalBridgeBypassed() || IsTranslationLayerCaller(caller) || IsSelfModuleCaller(caller))
			return;
		try
		{
			State& s = S();
			bool added = false;
			{
				std::lock_guard lock(s.mutex);
				if (s.ignoredInstances.count(instance) != 0)
					return;
				for (const auto& entry : s.instances)
				{
					if (entry.first == instance)
						return;
				}
				if (s.instances.size() >= MaxInstances)
					s.instances.erase(s.instances.begin());
				s.instances.emplace_back(instance, 0u);
				added = true;
			}
			if (added)
				Log::Info("Vulkan hooks: adopted pre-existing VkInstance {} via vkGetInstanceProcAddr",
					static_cast<const void*>(instance));
		}
		catch (...)
		{
		}
	}

	struct ReentryGuard
	{
		bool& flag;
		explicit ReentryGuard(bool& f) : flag(f) { flag = true; }
		~ReentryGuard() { flag = false; }
	};

	VkResult CreateInstanceImpl(PFN_vkCreateInstance next, const VkInstanceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkInstance* instance, const void* caller)
	{
		if (!next)
			return VK_ERROR_INITIALIZATION_FAILED;

		static thread_local bool inCreateInstance = false;
		if (inCreateInstance)
			return next(info, allocator, instance);

		ReentryGuard guard(inCreateInstance);

		State& s = S();
		const bool internalBridge = IsInternalBridgeBypassed() || IsInternalBridgeInstance(info) || IsSelfModuleCaller(caller);
		if (internalBridge || IsTranslationLayerInstance(info) || IsTranslationLayerCaller(caller))
		{
			if (internalBridge)
			{
				if (!s.selfSkipLogged.exchange(true, std::memory_order_relaxed))
					Log::Info("Vulkan hooks: bypassing internal D3D12->Vulkan WMMA bridge instance/device");
			}
			else if (!s.translationSkipLogged.exchange(true, std::memory_order_relaxed))
			{
				Log::Info("Vulkan hooks: skipping DXVK/VKD3D translation-layer instance/device (leaving D3D-on-Vulkan untouched)");
			}
			const VkResult result = next(info, allocator, instance);
			if (result == VK_SUCCESS && instance && *instance)
			{
				try
				{
					std::lock_guard lock(s.mutex);
					if (s.ignoredInstances.size() >= MaxInstances * 2)
						s.ignoredInstances.clear();
					s.ignoredInstances.insert(*instance);
				}
				catch (...)
				{
				}
			}
			return result;
		}

		s.instanceCalls++;

		InstanceEdit edit;
		const VkInstanceCreateInfo* used = s.enabled ? PrepareInstance(info, edit) : info;

		VkResult result = next(used, allocator, instance);

		if (result != VK_SUCCESS && used != info)
		{
			Log::Warn("The loader refused the instance at 1.3 ({}), created as the game asked", static_cast<int>(result));
			used = info;
			result = next(info, allocator, instance);
		}

		if (result == VK_SUCCESS && instance && *instance)
			InstanceCreated(*instance, used);

		return result;
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateInstance(const VkInstanceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkInstance* instance)
	{
		return CreateInstanceImpl(S().createInstance, info, allocator, instance, XEFG_RETURN_ADDRESS());
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateInstanceViaGipa(const VkInstanceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkInstance* instance)
	{
		State& s = S();
		PFN_vkCreateInstance next = s.gipaCreateInstance.load(std::memory_order_acquire);
		return CreateInstanceImpl(next ? next : s.createInstance, info, allocator, instance, XEFG_RETURN_ADDRESS());
	}

	void DestroyInstanceImpl(State::DestroyInstanceFn next, VkInstance instance, const VkAllocationCallbacks* allocator)
	{
		static thread_local bool inDestroyInstance = false;
		if (inDestroyInstance)
		{
			if (next)
				next(instance, allocator);
			return;
		}

		ReentryGuard guard(inDestroyInstance);
		// Erase instance and wait for any in-flight vkEnumeratePhysicalDevices probes to finish
		// before calling vkDestroyInstance so probes can never use a destroyed VkInstance.
		{
			std::unique_lock lifecycleLock(S().instanceLifecycleMutex);
			InstanceDestroyed(instance);
		}
		if (next)
			next(instance, allocator);
	}

	VKAPI_ATTR void VKAPI_CALL HookDestroyInstance(VkInstance instance, const VkAllocationCallbacks* allocator)
	{
		DestroyInstanceImpl(S().destroyInstance, instance, allocator);
	}

	VKAPI_ATTR void VKAPI_CALL HookDestroyInstanceViaGipa(VkInstance instance, const VkAllocationCallbacks* allocator)
	{
		State& s = S();
		State::DestroyInstanceFn next = s.gipaDestroyInstance.load(std::memory_order_acquire);
		DestroyInstanceImpl(next ? next : s.destroyInstance, instance, allocator);
	}


	// ------------------------------------------------------------------------------------------------ device

	struct DeviceEdit
	{
		VkDeviceCreateInfo info{};
		std::vector<const char*> extensions;
		VkPhysicalDeviceFeatures features10{};
		VkPhysicalDeviceTimelineSemaphoreFeatures timeline{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
		VkPhysicalDeviceSynchronization2Features sync2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES };
		VkPhysicalDeviceMaintenance4Features maintenance4{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES };
		VkPhysicalDeviceShaderIntegerDotProductFeatures dot{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES
		};
		VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
		};

		VulkanHost::FeatureChain chain;
		VulkanHost::DeviceRecord record;

		// How many of the last entries of `extensions` are the tensor-core route's NVX or cooperative-matrix extensions
		// this edit added (none the game asked for itself): the part of the edit XeFG can do without
		size_t tensorAdded = 0;

		// The device the driver takes without the NVX / cooperative-matrix extensions: XeFG's features and push descriptors
		// stay, the hardware matrix route doesn't exist and the SPIR-V programs run every layer
		void DropTensorExtensions()
		{
			extensions.resize(extensions.size() - std::min(tensorAdded, extensions.size()));
			tensorAdded = 0;
			record.tensorExtensions = false;
			record.coopMatrixExtensions = false;
			if (info.pNext == &coop)
				info.pNext = coop.pNext;
			info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
			info.ppEnabledExtensionNames = extensions.data();
		}

		void Raise(VkBool32& field)
		{
			// An already enabled field may belong to the borrowed tail: do not write it.
			if (field != VK_TRUE) field = VK_TRUE;
		}

	};

	std::unique_ptr<DeviceEdit> Refuse(VkPhysicalDevice physical, const char* code) noexcept
	{
		try
		{
			State& s = S();
			{
				std::lock_guard lock(s.mutex);
				s.lastRefusal = code;
				s.refused[physical] = code;
			}

			Log::Warn("Vulkan device {} left alone: {}", static_cast<const void*>(physical), VulkanPolicy::Describe(code));
		}
		catch (...)
		{
		}

		return nullptr;
	}

	bool ProbeInstanceHasPhysical(PFN_vkGetInstanceProcAddr gipaFn, VkInstance inst, VkPhysicalDevice target) noexcept
	{
		if (!gipaFn || !inst || !target)
			return false;
#ifdef _MSC_VER
		__try
		{
#endif
			auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
				gipaFn(inst, "vkEnumeratePhysicalDevices"));
			uint32_t count = 0;
			if (!enumerate || enumerate(inst, &count, nullptr) != VK_SUCCESS || !count || count > 64)
				return false;
			VkPhysicalDevice list[64]{};
			if (enumerate(inst, &count, list) != VK_SUCCESS)
				return false;
			const uint32_t n = count < 64u ? count : 64u;
			for (uint32_t i = 0; i < n; ++i)
				if (list[i] == target)
					return true;
			return false;
#ifdef _MSC_VER
		}
		__except (EXCEPTION_EXECUTE_HANDLER)
		{
			return false;
		}
#endif
	}

	bool IsIgnoredPhysicalDevice(State& s, VkPhysicalDevice physical) noexcept
	{
		if (!physical || !s.loaderGipa)
			return false;
		std::shared_lock lifecycleLock(s.instanceLifecycleMutex);
		std::vector<VkInstance> ignored;
		{
			std::lock_guard lock(s.mutex);
			if (s.ignoredInstances.empty())
				return false;
			ignored.assign(s.ignoredInstances.begin(), s.ignoredInstances.end());
		}
		for (VkInstance inst : ignored)
		{
			if (ProbeInstanceHasPhysical(s.loaderGipa, inst, physical))
				return true;
		}
		return false;
	}

	// What XeFG's programs need from a device, and the game's device edited to have it. Null (the game's request goes
	// through as it is) when the device can't have it
	std::unique_ptr<DeviceEdit> PrepareDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info) noexcept
	{
		State& s = S();

		if (!physical || !info || !s.loaderGipa)
			return nullptr;

		try
		{
			auto edit = std::make_unique<DeviceEdit>();

			// The instance that enumerated this physical device and the API version it was created with
			uint32_t instanceVersion = 0;
			{
				std::shared_lock lifecycleLock(s.instanceLifecycleMutex);
				std::vector<std::pair<VkInstance, uint32_t>> known;
				{
					std::lock_guard lock(s.mutex);
					known.assign(s.instances.begin(), s.instances.end());
				}

				// Check newest live instances first so a stale probe instance can never shadow the active game instance
				for (auto it = known.rbegin(); it != known.rend(); ++it)
				{
					if (!ProbeInstanceHasPhysical(s.loaderGipa, it->first, physical))
						continue;
					edit->record.instance = it->first;
					instanceVersion = it->second;
					break;
				}
			}

			if (!edit->record.instance)
				return Refuse(physical, "instance_unknown");

			// instanceVersion == 0 means the VkInstance was created before our hooks loaded and adopted via GIPA.
			// Let the Vulkan 1.1+ function lookups and physical device 1.3 feature checks below validate it; if the
			// driver rejects the edited VkDeviceCreateInfo, CreateDeviceImpl falls back safely to the unedited request.
			if (instanceVersion != 0 && instanceVersion < xvk::kRequiredApiVersion)
				return Refuse(physical, "instance_api_version");

			const auto gipa = s.loaderGipa;
			const auto instance = edit->record.instance;

			auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gipa(instance, "vkGetPhysicalDeviceProperties2"));
			auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(gipa(instance, "vkGetPhysicalDeviceFeatures2"));
			auto familiesFn = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
				gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
			auto extensionsFn = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
				gipa(instance, "vkEnumerateDeviceExtensionProperties"));

			if (!properties2 || !features2 || !familiesFn || !extensionsFn)
				return Refuse(physical, "instance_functions");

			VkPhysicalDeviceVulkan11Properties v11Properties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES };
			VkPhysicalDeviceProperties2 properties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &v11Properties };
			properties2(physical, &properties);

			edit->record.vendor = properties.properties.vendorID;

			if (properties.properties.apiVersion < xvk::kRequiredApiVersion)
				return Refuse(physical, "device_api_version");

			// XeFG's inverted depth motion splat reduces with wave min/max: subgroup arithmetic in compute shaders
			constexpr VkSubgroupFeatureFlags SubgroupOps = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;

			if (!(v11Properties.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
				(v11Properties.subgroupSupportedOperations & SubgroupOps) != SubgroupOps)
				return Refuse(physical, "subgroup_operations");

			// What the layer and the program set use
			VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopSupported{
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
			};
			VkPhysicalDeviceVulkan13Features v13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &coopSupported };
			VkPhysicalDeviceVulkan12Features v12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13 };
			VkPhysicalDeviceFeatures2 supported{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12 };
			features2(physical, &supported);

			if (!v12.timelineSemaphore || !v13.synchronization2 || !v13.maintenance4 || !v13.shaderIntegerDotProduct ||
				!supported.features.shaderStorageImageExtendedFormats)
				return Refuse(physical, "device_features");

			uint32_t count = 0;
			extensionsFn(physical, nullptr, &count, nullptr);
			std::vector<VkExtensionProperties> available(count);
			extensionsFn(physical, nullptr, &count, available.data());

			for (const auto& extension : available)
			{
				if (!strcmp(extension.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
					edit->record.coopMatrixSpec = extension.specVersion;
			}

			// Intel GPUs are refused on Vulkan unless cooperative-matrix acceleration is available and allowed
			if (edit->record.vendor == 0x8086 &&
				(!edit->record.coopMatrixSpec || !coopSupported.cooperativeMatrix || !VulkanCoopMat::AllowIntelAdapter()))
				return Refuse(physical, "intel_adapter");

			auto offered = [&](const char* name)
				{
					return std::any_of(available.begin(), available.end(),
						[&](const VkExtensionProperties& extension) { return !strcmp(extension.extensionName, name); });
				};

			auto enabled = [&](const char* name)
				{
					for (uint32_t i = 0; i < info->enabledExtensionCount; ++i)
					{
						if (info->ppEnabledExtensionNames[i] && !strcmp(info->ppEnabledExtensionNames[i], name))
							return true;
					}

					return false;
				};

			if (!offered(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
				return Refuse(physical, "push_descriptor");

			// XeFG's dispatches are recorded into the game's own command buffers, so the family is the game's graphics one
			familiesFn(physical, &count, nullptr);
			std::vector<VkQueueFamilyProperties> families(count);
			familiesFn(physical, &count, families.data());

			bool family = false;

			for (uint32_t i = 0; i < info->queueCreateInfoCount && !family; ++i)
			{
				const auto& queue = info->pQueueCreateInfos[i];

				if (queue.queueFamilyIndex < families.size() &&
					(families[queue.queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT))
				{
					edit->record.family = queue.queueFamilyIndex;
					family = true;
				}
			}

			if (!family)
				return Refuse(physical, "no_graphics_family");

			// Only push descriptors (plus optional tensor/coopmat extensions) are added to the extensions, nothing is taken away
			if (info->enabledExtensionCount)
				edit->extensions.assign(info->ppEnabledExtensionNames, info->ppEnabledExtensionNames + info->enabledExtensionCount);

			if (!enabled(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
				edit->extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

			// The tensor-core route launches CUDA kernels in the game's command buffers (VK_NVX_binary_import) and gets the
			// handles of the images they read and write (VK_NVX_image_view_handle).
			if (edit->record.vendor == 0x10DE)
			{
				for (const auto& extension : available)
				{
					if (!strcmp(extension.extensionName, VK_NVX_BINARY_IMPORT_EXTENSION_NAME))
						edit->record.binaryImportSpec = extension.specVersion;

					if (!strcmp(extension.extensionName, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME))
						edit->record.imageViewHandleSpec = extension.specVersion;
				}

				if (edit->record.binaryImportSpec && edit->record.imageViewHandleSpec && VulkanTensor::Wanted())
				{
					for (const char* name : { VK_NVX_BINARY_IMPORT_EXTENSION_NAME, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME })
					{
						if (!enabled(name))
						{
							edit->extensions.push_back(name);
							edit->tensorAdded++;
						}
					}

					edit->record.tensorExtensions = true;
				}
			}

			// VK_KHR_cooperative_matrix for AMD RDNA3/RDNA4 (WMMA) and Intel Arc (XMX), or NVIDIA fallback when NVX is off
			const bool wantCoopMat = edit->record.coopMatrixSpec && coopSupported.cooperativeMatrix &&
				VulkanCoopMat::Wanted() && !edit->record.tensorExtensions;

			if (wantCoopMat)
			{
				if (!enabled(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
				{
					edit->extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
					edit->tensorAdded++;
				}

				edit->record.coopMatrixExtensions = true;
			}

			// Only treat VK_NV_present_metering as hardware flip metering on Blackwell+ GPUs (deviceID >= 0x2B00).
			// Do NOT treat VK_NV_low_latency2 (NVIDIA Reflex) or pre-Blackwell VK_NV_present_metering exposure as
			// hardware MFG pacing, or the CPU present pacer would be falsely bypassed on RTX 20/30/40 Vulkan games.
			const bool isBlackwellOrNewer = (edit->record.vendor == 0x10DE && properties.properties.deviceID >= 0x2B00u);
			edit->record.hwPacing = isBlackwellOrNewer &&
				offered("VK_NV_present_metering") && enabled("VK_NV_present_metering");

			edit->info = *info;
			edit->info.enabledExtensionCount = static_cast<uint32_t>(edit->extensions.size());
			edit->info.ppEnabledExtensionNames = edit->extensions.data();

			// Features: raise bits in an owned copy of the game's chain (never in its const storage); the individual structures it doesn't pass are added in front of the chain. A request that
			// holds a Vulkan 1.2 or 1.3 features structure must not also hold the single feature structures it covers
			if (!edit->chain.Copy(info->pNext))
				return Refuse(physical, "feature_chain_unsupported");
			const void* chain = edit->chain.Head();

			if (auto* f2 = FindStruct<VkPhysicalDeviceFeatures2>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2))
			{
				edit->Raise(f2->features.shaderStorageImageExtendedFormats);
				if (supported.features.shaderStorageImageReadWithoutFormat)
					edit->Raise(f2->features.shaderStorageImageReadWithoutFormat);
				if (supported.features.shaderStorageImageWriteWithoutFormat)
					edit->Raise(f2->features.shaderStorageImageWriteWithoutFormat);
			}
			else
			{
				if (info->pEnabledFeatures)
					edit->features10 = *info->pEnabledFeatures;

				edit->features10.shaderStorageImageExtendedFormats = VK_TRUE;
				if (supported.features.shaderStorageImageReadWithoutFormat)
					edit->features10.shaderStorageImageReadWithoutFormat = VK_TRUE;
				if (supported.features.shaderStorageImageWriteWithoutFormat)
					edit->features10.shaderStorageImageWriteWithoutFormat = VK_TRUE;
				edit->info.pEnabledFeatures = &edit->features10;
			}

			void* head = const_cast<void*>(chain);

			auto prepend = [&head](auto& structure)
				{
					structure.pNext = head;
					head = &structure;
				};

			if (auto* f12 = FindStruct<VkPhysicalDeviceVulkan12Features>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES))
				edit->Raise(f12->timelineSemaphore);
			else if (auto* t = FindStruct<VkPhysicalDeviceTimelineSemaphoreFeatures>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES))
				edit->Raise(t->timelineSemaphore);
			else
			{
				edit->timeline.timelineSemaphore = VK_TRUE;
				prepend(edit->timeline);
			}

			if (auto* f13 = FindStruct<VkPhysicalDeviceVulkan13Features>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES))
			{
				edit->Raise(f13->synchronization2);
				edit->Raise(f13->maintenance4);
				edit->Raise(f13->shaderIntegerDotProduct);
			}
			else
			{
				if (auto* sync = FindStruct<VkPhysicalDeviceSynchronization2Features>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES))
					edit->Raise(sync->synchronization2);
				else
				{
					edit->sync2.synchronization2 = VK_TRUE;
					prepend(edit->sync2);
				}

				if (auto* m = FindStruct<VkPhysicalDeviceMaintenance4Features>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES))
					edit->Raise(m->maintenance4);
				else
				{
					edit->maintenance4.maintenance4 = VK_TRUE;
					prepend(edit->maintenance4);
				}

				if (auto* d = FindStruct<VkPhysicalDeviceShaderIntegerDotProductFeatures>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES))
					edit->Raise(d->shaderIntegerDotProduct);
				else
				{
					edit->dot.shaderIntegerDotProduct = VK_TRUE;
					prepend(edit->dot);
				}
			}

			if (edit->record.coopMatrixExtensions &&
				!FindStruct<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR))
			{
				edit->coop.cooperativeMatrix = VK_TRUE;
				prepend(edit->coop);
			}

			edit->info.pNext = head;
			edit->record.physical = physical;
			edit->record.gipa = gipa;

			return edit;
		}
		catch (...)
		{
			return Refuse(physical, "prepare_exception");
		}
	}

	void InstallDeviceLevelHooks(State& s, const VulkanHost::DeviceRecord& record, VkDevice device) noexcept;

	void DeviceCreated(const DeviceEdit& edit, VkDevice device) noexcept
	{
		try
		{
			State& s = S();
			{
				std::lock_guard lock(s.mutex);

				if (s.devices.size() >= MaxDevices)
					s.devices.clear();

				s.devices[device] = edit.record;
				s.refused.erase(edit.record.physical);
			}

			if (edit.record.hwPacing)
				s.hwPacingActive.store(true, std::memory_order_relaxed);

			InstallDeviceLevelHooks(s, edit.record, device);

			s.prepared++;
			Log::Info("Vulkan device {} prepared for XeFG: graphics family {}, vendor {:#06x}{}{}{}",
				static_cast<const void*>(device), edit.record.family, edit.record.vendor,
				edit.record.tensorExtensions ? ", tensor-core extensions enabled"
				: edit.record.vendor == 0x10DE ? (edit.record.binaryImportSpec && edit.record.imageViewHandleSpec
					? ", tensor-core extensions not enabled (XEFG_TENSOR_CORES is off or the driver refused them)"
					: ", the driver doesn't offer the tensor-core extensions")
				: "",
				edit.record.coopMatrixExtensions ? ", VK_KHR_cooperative_matrix enabled" : "",
				edit.record.hwPacing ? ", hardware pacing detected (CPU present pacer auto-bypassed)" : "");
		}
		catch (...)
		{
		}
	}

	VkResult CreateDeviceImpl(PFN_vkCreateDevice next, VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkDevice* device, const void* caller)
	{
		if (!next)
			return VK_ERROR_INITIALIZATION_FAILED;

		static thread_local bool inCreateDevice = false;
		if (inCreateDevice)
			return next(physical, info, allocator, device);

		ReentryGuard guard(inCreateDevice);

		State& s = S();
		if (IsInternalBridgeBypassed() || IsSelfModuleCaller(caller))
		{
			if (!s.selfSkipLogged.exchange(true, std::memory_order_relaxed))
				Log::Info("Vulkan hooks: bypassing internal D3D12->Vulkan WMMA bridge instance/device");
			return next(physical, info, allocator, device);
		}
		if (IsTranslationLayerCaller(caller) || IsIgnoredPhysicalDevice(s, physical))
		{
			if (!s.translationSkipLogged.exchange(true, std::memory_order_relaxed))
				Log::Info("Vulkan hooks: skipping DXVK/VKD3D translation-layer instance/device (leaving D3D-on-Vulkan untouched)");
			return next(physical, info, allocator, device);
		}

		{
			std::lock_guard lock(s.mutex);
			if (s.instances.empty())
			{
				Log::Warn("Vulkan hooks: vkCreateDevice called for physical device {} with no tracked VkInstance",
					static_cast<const void*>(physical));
				return next(physical, info, allocator, device);
			}
		}

		s.deviceCalls++;

		std::unique_ptr<DeviceEdit> edit;

		if (s.enabled)
			edit = PrepareDevice(physical, info);

		TryHookLoadedIcd(s);
		VkResult result = next(physical, edit ? &edit->info : info, allocator, device);

		if (result != VK_SUCCESS && edit && edit->tensorAdded)
		{
			// The NVX / cooperative-matrix extensions are the part of the edit XeFG can do without: keep the rest and let
			// the SPIR-V programs run every layer
			Log::Warn("The driver refused the device with the tensor/cooperative-matrix extensions ({}), trying again without them",
				static_cast<int>(result));
			edit->DropTensorExtensions();
			result = next(physical, &edit->info, allocator, device);
		}

		if (result != VK_SUCCESS && edit)
		{
			// The driver took the game's request but not the edited one: the game gets the device it asked for, without
			// the bits that were raised in its own structures
			Log::Warn("The driver refused the device with XeFG's features ({}), created as the game asked",
				static_cast<int>(result));
			edit.reset();
			Refuse(physical, "device_features");
			result = next(physical, info, allocator, device);
		}

		if (result == VK_SUCCESS && edit && device && *device)
			DeviceCreated(*edit, *device);

		return result;
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkDevice* device)
	{
		return CreateDeviceImpl(S().createDevice, physical, info, allocator, device, XEFG_RETURN_ADDRESS());
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateDeviceViaGipa(VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkDevice* device)
	{
		State& s = S();
		PFN_vkCreateDevice next = s.gipaCreateDevice.load(std::memory_order_acquire);
		return CreateDeviceImpl(next ? next : s.createDevice, physical, info, allocator, device, XEFG_RETURN_ADDRESS());
	}

	// OptiScaler hooks vulkan-1.dll before ASI plugins run and returns its own hkvkCreateDevice from
	// vkGetInstanceProcAddr. Wrapping vkGetInstanceProcAddr catches games that resolve vkCreateDevice via GIPA
	// and call OptiScaler's hook directly (whose saved trampoline would otherwise bypass our loader prologue hook).
	// --------------------------------------------------------------------------------------- swapchain / present pacing

	bool IsFifoMode(VkPresentModeKHR mode) noexcept
	{
		return mode == VK_PRESENT_MODE_FIFO_KHR || mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR;
	}

	uint32_t QueryDisplayRefreshHz() noexcept
	{
		DEVMODEW dm{};
		dm.dmSize = sizeof(dm);
		if (HWND fg = GetForegroundWindow())
		{
			if (HMONITOR mon = MonitorFromWindow(fg, MONITOR_DEFAULTTONEAREST))
			{
				MONITORINFOEXW mi{};
				mi.cbSize = sizeof(mi);
				if (GetMonitorInfoW(mon, &mi) &&
					EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm) &&
					dm.dmDisplayFrequency >= 48u && dm.dmDisplayFrequency <= 1000u)
				{
					return dm.dmDisplayFrequency;
				}
			}
		}
		if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &dm) && dm.dmDisplayFrequency >= 48u && dm.dmDisplayFrequency <= 1000u)
			return dm.dmDisplayFrequency;
		return 0;
	}

	VkResult CreateSwapchainImpl(PFN_vkCreateSwapchainKHR next, VkDevice device,
		const VkSwapchainCreateInfoKHR* info, const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain,
		const void* caller, bool isNativeIcd = false)
	{
		if (!next)
			return VK_ERROR_INITIALIZATION_FAILED;

		static thread_local bool inCreateSwapchain = false;
		if (inCreateSwapchain)
			return next(device, info, allocator, swapchain);

		ReentryGuard guard(inCreateSwapchain);

		const VkSwapchainCreateInfoKHR* usedInfo = info;
		VkSwapchainCreateInfoKHR edited{};
		uint32_t capMin = 0, capMax = 0;
		bool boosted = false;

		const uint32_t targetBuffers = GetSwapchainBufferTarget();
		const bool devicePrepared = S().prepared.load(std::memory_order_relaxed) > 0;
		// Never raise minImageCount downstream of Streamline (sl.dlss_g.dll): Streamline already chose its own
		// swapchain image count (e.g. 6) and sized its internal dlfgSwapchain arrays accordingly. Creating 8 images
		// behind Streamline's back causes an out-of-bounds crash on vkAcquireNextImageKHR.
		const bool streamlineLoaded = (GetModuleHandleW(L"sl.dlss_g.dll") != nullptr);
		if (info && info->surface && devicePrepared && !streamlineLoaded)
		{
			if (isNativeIcd)
			{
				// At the ICD boundary, info->surface is already unwrapped by the Vulkan loader; do not pass it
				// back into the loader's vkGetPhysicalDeviceSurfaceCapabilitiesKHR. Rely on RecommendSwapchainImageCount
				// plus the driver-refusal retry fallback below.
				capMin = info->minImageCount;
				capMax = 0u;
				if (targetBuffers >= 6u && info->minImageCount >= 6u && info->minImageCount < targetBuffers)
				{
					const uint32_t recommended = VulkanPolicy::RecommendSwapchainImageCount(
						info->minImageCount, capMin, capMax, targetBuffers);
					if (recommended > info->minImageCount)
					{
						edited = *info;
						edited.minImageCount = recommended;
						usedInfo = &edited;
						boosted = true;
					}
				}
			}
			else
			{
				VulkanHost::DeviceRecord record{};
				bool foundDevice = false;
				{
					State& s = S();
					std::lock_guard lock(s.mutex);
					auto it = s.devices.find(device);
					if (it == s.devices.end() && s.devices.size() == 1)
						it = s.devices.begin();
					if (it != s.devices.end())
					{
						record = it->second;
						foundDevice = true;
					}
				}
				if (foundDevice && record.gipa && record.instance && record.physical)
				{
					auto getCaps = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
						record.gipa(record.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
					VkSurfaceCapabilitiesKHR caps{};
					if (getCaps && getCaps(record.physical, info->surface, &caps) == VK_SUCCESS)
					{
						capMin = caps.minImageCount;
						capMax = caps.maxImageCount;
						if (targetBuffers >= 6u && info->minImageCount >= 6u && info->minImageCount < targetBuffers)
						{
							const uint32_t recommended = VulkanPolicy::RecommendSwapchainImageCount(
								info->minImageCount, capMin, capMax, targetBuffers);
							if (recommended > info->minImageCount)
							{
								edited = *info;
								edited.minImageCount = recommended;
								usedInfo = &edited;
								boosted = true;
							}
						}
					}
				}
			}
		}

		VkResult result = next(device, usedInfo, allocator, swapchain);

		if (boosted)
		{
			if (result == VK_SUCCESS)
			{
				Log::Info("Vulkan swapchain: raised minImageCount {} -> {} ({}x{}, surface caps {}..{}, presentMode {}, caller {})",
					info->minImageCount, edited.minImageCount, info->imageExtent.width, info->imageExtent.height,
					capMin, capMax, static_cast<int>(info->presentMode), DescribeModuleOfAddress(caller));
			}
			else
			{
				Log::Warn("Vulkan swapchain: driver refused minImageCount {} ({}), retrying with original {}",
					edited.minImageCount, static_cast<int>(result), info->minImageCount);
				result = next(device, info, allocator, swapchain);
			}
		}
		else if (result == VK_SUCCESS && info && devicePrepared)
		{
			if (streamlineLoaded && targetBuffers > info->minImageCount)
			{
				Log::Info("Vulkan swapchain: created {}x{} preserving Streamline minImageCount={} (skipped downstream boost to {} to prevent sl.dlss_g mismatch crash, presentMode {}, caller {})",
					info->imageExtent.width, info->imageExtent.height, info->minImageCount, targetBuffers,
					static_cast<int>(info->presentMode), DescribeModuleOfAddress(caller));
			}
			else
			{
				Log::Info("Vulkan swapchain: created {}x{} minImageCount={} (target {}, surface caps {}..{}, presentMode {}, caller {})",
					info->imageExtent.width, info->imageExtent.height, info->minImageCount, targetBuffers,
					capMin, capMax, static_cast<int>(info->presentMode), DescribeModuleOfAddress(caller));
			}
		}

		if (result == VK_SUCCESS && info && swapchain && *swapchain && devicePrepared)
		{
			State& s = S();
			std::lock_guard lock(s.mutex);
			if (s.swapchains.size() >= MaxSwapchains)
				s.swapchains.clear();
			s.swapchains[*swapchain] = info->presentMode;
			const uint32_t hz = QueryDisplayRefreshHz();
			if (hz)
			{
				std::lock_guard pLock(s.pacerMutex);
				s.pacer.SetDisplayRefreshHz(hz);
			}
		}

		return result;
	}

	void DestroySwapchainImpl(PFN_vkDestroySwapchainKHR next, VkDevice device, VkSwapchainKHR swapchain,
		const VkAllocationCallbacks* allocator)
	{
		if (!next)
			return;

		static thread_local bool inDestroySwapchain = false;
		if (inDestroySwapchain)
		{
			next(device, swapchain, allocator);
			return;
		}

		ReentryGuard guard(inDestroySwapchain);
		if (swapchain)
		{
			State& s = S();
			std::lock_guard lock(s.mutex);
			s.swapchains.erase(swapchain);
		}
		next(device, swapchain, allocator);
	}

	VkResult QueuePresentImpl(PFN_vkQueuePresentKHR next, VkQueue queue, const VkPresentInfoKHR* info, const void* caller)
	{
		if (!next)
			return VK_ERROR_INITIALIZATION_FAILED;

		static thread_local bool inQueuePresent = false;
		if (inQueuePresent)
			return next(queue, info);

		ReentryGuard guard(inQueuePresent);

		State& s = S();
		const bool prepared = s.prepared.load(std::memory_order_relaxed) > 0;
		const bool pacingActive = prepared && PacingActive(s);
		bool vsyncFifo = false;
		uint64_t waitNs = 0;
		uint64_t diagTargetStepNs = 0;
		uint32_t diagGenCount = 0;

		if (pacingActive)
		{
			if (info && info->swapchainCount > 0 && info->pSwapchains)
			{
				std::lock_guard lock(s.mutex);
				const auto it = s.swapchains.find(info->pSwapchains[0]);
				if (it != s.swapchains.end())
					vsyncFifo = IsFifoMode(it->second);
			}

			{
				std::lock_guard lock(s.pacerMutex);
				const bool autoVrr = Plugin::options.autoVrrPacing.load(std::memory_order_relaxed);
				s.pacer.SetAutoVrrEnabled(autoVrr);
				if (autoVrr && s.pacer.displayRefreshHz == 0)
					s.pacer.SetDisplayRefreshHz(QueryDisplayRefreshHz());
				waitNs = s.pacer.ComputePresentWaitNs(NowNs(), vsyncFifo);
				diagTargetStepNs = s.pacer.targetStepNs;
				diagGenCount = s.pacer.generatedCount;
			}

			if (waitNs > 0)
			{
				if (!s.pacingLogged.exchange(true, std::memory_order_relaxed))
					Log::Info("Vulkan present pacer active (V-Sync FIFO off, smoothing multi-frame burst presents)");
				WaitPreciseNs(waitNs);
			}
		}

		if (prepared)
		{
			s.diagPresentCalls.fetch_add(1, std::memory_order_relaxed);
			if (vsyncFifo)
				s.diagPresentFifo.fetch_add(1, std::memory_order_relaxed);
			if (waitNs > 0)
			{
				s.diagPresentPaced.fetch_add(1, std::memory_order_relaxed);
				s.diagPresentWaitSumUs.fetch_add(waitNs / 1000ull, std::memory_order_relaxed);
			}

			if (!s.firstPresentLogged.exchange(true, std::memory_order_relaxed))
			{
				Log::Info("Vulkan present hook: first call intercepted (tid={}, caller={}, swapchains={}, vsyncFifo={}, pacingActive={})",
					GetCurrentThreadId(), DescribeModuleOfAddress(caller),
					info ? info->swapchainCount : 0u, vsyncFifo, pacingActive);
			}

			const bool fgActiveNow = diagGenCount > 0;
			if (fgActiveNow && !s.diagFgWindowStarted.exchange(true, std::memory_order_relaxed))
			{
				// Reset pre-FG counters when Frame Generation first starts so the 12 x 5s diagnostic
				// windows capture real FG pacing rather than loading-screen / menu idle presents.
				s.diagPresentCalls.store(0, std::memory_order_relaxed);
				s.diagPresentPaced.store(0, std::memory_order_relaxed);
				s.diagPresentFifo.store(0, std::memory_order_relaxed);
				s.diagPresentWaitSumUs.store(0, std::memory_order_relaxed);
				s.diagLastLogNs.store(NowNs(), std::memory_order_relaxed);
			}

			const bool wantDiagWindow = fgActiveNow
				? (s.diagLogCount.load(std::memory_order_relaxed) < 12u)
				: !s.diagPreFgLogged.load(std::memory_order_relaxed);
			if (wantDiagWindow)
			{
				const uint64_t nowDiagNs = NowNs();
				uint64_t lastNs = s.diagLastLogNs.load(std::memory_order_relaxed);
				if (lastNs == 0)
				{
					s.diagLastLogNs.compare_exchange_strong(lastNs, nowDiagNs, std::memory_order_relaxed);
				}
				else if (nowDiagNs >= lastNs + 5'000'000'000ull &&
					s.diagLastLogNs.compare_exchange_strong(lastNs, nowDiagNs, std::memory_order_relaxed))
				{
					const uint64_t calls = s.diagPresentCalls.exchange(0, std::memory_order_relaxed);
					const uint64_t paced = s.diagPresentPaced.exchange(0, std::memory_order_relaxed);
					const uint64_t fifo = s.diagPresentFifo.exchange(0, std::memory_order_relaxed);
					const uint64_t waitSumUs = s.diagPresentWaitSumUs.exchange(0, std::memory_order_relaxed);
					const uint64_t avgWaitUs = paced ? (waitSumUs / paced) : 0ull;
					if (fgActiveNow)
						s.diagLogCount.fetch_add(1, std::memory_order_relaxed);
					else
						s.diagPreFgLogged.store(true, std::memory_order_relaxed);
					Log::Info("Vulkan present diag [5s]: calls={}, paced={}, fifoSkipped={}, avgWaitUs={}, targetStepUs={}, genCount={}, tid={}, caller={}",
						calls, paced, fifo, avgWaitUs, diagTargetStepNs / 1000ull, diagGenCount,
						GetCurrentThreadId(), DescribeModuleOfAddress(caller));
				}
			}
		}

		return next(queue, info);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info,
		const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain)
	{
		State& s = S();
		if (s.icdCreateSwapActive.load(std::memory_order_acquire))
			return s.createSwapchain ? s.createSwapchain(device, info, allocator, swapchain) : VK_ERROR_INITIALIZATION_FAILED;
		return CreateSwapchainImpl(s.createSwapchain, device, info, allocator, swapchain, XEFG_RETURN_ADDRESS());
	}

	VKAPI_ATTR void VKAPI_CALL HookDestroySwapchain(VkDevice device, VkSwapchainKHR swapchain,
		const VkAllocationCallbacks* allocator)
	{
		State& s = S();
		if (s.icdDestroySwapActive.load(std::memory_order_acquire))
		{
			if (s.destroySwapchain)
				s.destroySwapchain(device, swapchain, allocator);
			return;
		}
		DestroySwapchainImpl(s.destroySwapchain, device, swapchain, allocator);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookQueuePresent(VkQueue queue, const VkPresentInfoKHR* info)
	{
		State& s = S();
		if (s.icdQueuePresentActive.load(std::memory_order_acquire))
			return s.queuePresent ? s.queuePresent(queue, info) : VK_ERROR_INITIALIZATION_FAILED;
		return QueuePresentImpl(s.queuePresent, queue, info, XEFG_RETURN_ADDRESS());
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateSwapchainViaGdpa(VkDevice device, const VkSwapchainCreateInfoKHR* info,
		const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain)
	{
		State& s = S();
		PFN_vkCreateSwapchainKHR next = s.gdpaCreateSwapchain.load(std::memory_order_acquire);
		PFN_vkCreateSwapchainKHR target = next ? next : s.createSwapchain;
		if (s.icdCreateSwapActive.load(std::memory_order_acquire))
			return target ? target(device, info, allocator, swapchain) : VK_ERROR_INITIALIZATION_FAILED;
		return CreateSwapchainImpl(target, device, info, allocator, swapchain, XEFG_RETURN_ADDRESS());
	}

	VKAPI_ATTR void VKAPI_CALL HookDestroySwapchainViaGdpa(VkDevice device, VkSwapchainKHR swapchain,
		const VkAllocationCallbacks* allocator)
	{
		State& s = S();
		PFN_vkDestroySwapchainKHR next = s.gdpaDestroySwapchain.load(std::memory_order_acquire);
		PFN_vkDestroySwapchainKHR target = next ? next : s.destroySwapchain;
		if (s.icdDestroySwapActive.load(std::memory_order_acquire))
		{
			if (target)
				target(device, swapchain, allocator);
			return;
		}
		DestroySwapchainImpl(target, device, swapchain, allocator);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookQueuePresentViaGdpa(VkQueue queue, const VkPresentInfoKHR* info)
	{
		State& s = S();
		PFN_vkQueuePresentKHR next = s.gdpaQueuePresent.load(std::memory_order_acquire);
		PFN_vkQueuePresentKHR target = next ? next : s.queuePresent;
		if (s.icdQueuePresentActive.load(std::memory_order_acquire))
			return target ? target(queue, info) : VK_ERROR_INITIALIZATION_FAILED;
		return QueuePresentImpl(target, queue, info, XEFG_RETURN_ADDRESS());
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateSwapchainViaGipa(VkDevice device, const VkSwapchainCreateInfoKHR* info,
		const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain)
	{
		State& s = S();
		PFN_vkCreateSwapchainKHR next = s.gipaCreateSwapchain.load(std::memory_order_acquire);
		PFN_vkCreateSwapchainKHR target = next ? next : s.createSwapchain;
		if (s.icdCreateSwapActive.load(std::memory_order_acquire))
			return target ? target(device, info, allocator, swapchain) : VK_ERROR_INITIALIZATION_FAILED;
		return CreateSwapchainImpl(target, device, info, allocator, swapchain, XEFG_RETURN_ADDRESS());
	}

	VKAPI_ATTR void VKAPI_CALL HookDestroySwapchainViaGipa(VkDevice device, VkSwapchainKHR swapchain,
		const VkAllocationCallbacks* allocator)
	{
		State& s = S();
		PFN_vkDestroySwapchainKHR next = s.gipaDestroySwapchain.load(std::memory_order_acquire);
		PFN_vkDestroySwapchainKHR target = next ? next : s.destroySwapchain;
		if (s.icdDestroySwapActive.load(std::memory_order_acquire))
		{
			if (target)
				target(device, swapchain, allocator);
			return;
		}
		DestroySwapchainImpl(target, device, swapchain, allocator);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookQueuePresentViaGipa(VkQueue queue, const VkPresentInfoKHR* info)
	{
		State& s = S();
		PFN_vkQueuePresentKHR next = s.gipaQueuePresent.load(std::memory_order_acquire);
		PFN_vkQueuePresentKHR target = next ? next : s.queuePresent;
		if (s.icdQueuePresentActive.load(std::memory_order_acquire))
			return target ? target(queue, info) : VK_ERROR_INITIALIZATION_FAILED;
		return QueuePresentImpl(target, queue, info, XEFG_RETURN_ADDRESS());
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateSwapchainViaIcd(VkDevice device, const VkSwapchainCreateInfoKHR* info,
		const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain)
	{
		State& s = S();
		PFN_vkCreateSwapchainKHR next = s.icdCreateSwapchain.load(std::memory_order_acquire);
		return CreateSwapchainImpl(next ? next : s.createSwapchain, device, info, allocator, swapchain,
			XEFG_RETURN_ADDRESS(), s.icdCreateSwapIsNative.load(std::memory_order_acquire));
	}

	VKAPI_ATTR void VKAPI_CALL HookDestroySwapchainViaIcd(VkDevice device, VkSwapchainKHR swapchain,
		const VkAllocationCallbacks* allocator)
	{
		State& s = S();
		PFN_vkDestroySwapchainKHR next = s.icdDestroySwapchain.load(std::memory_order_acquire);
		DestroySwapchainImpl(next ? next : s.destroySwapchain, device, swapchain, allocator);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookQueuePresentViaIcd(VkQueue queue, const VkPresentInfoKHR* info)
	{
		State& s = S();
		PFN_vkQueuePresentKHR next = s.icdQueuePresent.load(std::memory_order_acquire);
		return QueuePresentImpl(next ? next : s.queuePresent, queue, info, XEFG_RETURN_ADDRESS());
	}

	// When vulkan-1.dll's short dispatch stubs (vkGetDeviceProcAddr, vkCreateSwapchainKHR, vkQueuePresentKHR) were
	// already detoured by OptiScaler/overlays and skipped during InstallLoader, Streamline (sl.dlss_g) resolves the
	// ICD/layer device functions directly and bypasses our upstream GIPA/GDPA wrappers. Reading slot 0 of VkDevice's
	// VkLayerDispatchTable bypasses OptiScaler's detoured vkGetDeviceProcAddr so we hook the true downstream ICD /
	// bottom-layer functions, intercepting Streamline's 6-buffer swapchain creation (raising 6->8) and every
	// generated/real vkQueuePresentKHR from sl.dlss_g's pacer thread.
	void InstallDeviceLevelHooks(State& s, const VulkanHost::DeviceRecord& record, VkDevice device) noexcept
	{
		try
		{
			if (!device)
				return;

			std::lock_guard lock(s.install);
			if (s.hookedIcdCreateSwap && s.hookedIcdDestroySwap && s.hookedIcdQueuePresent)
				return;

			std::string gdpaMod;
			const char* gdpaSource = "dispatch-table";
			PFN_vkGetDeviceProcAddr gdpa = ResolveDispatchTableGdpa(s, device, gdpaMod);
			if (!gdpa)
			{
				gdpaSource = "fallback";
				gdpa = s.loaderGdpa;
				if (!gdpa)
					gdpa = s.gipaGetDeviceProcAddr.load(std::memory_order_acquire);
				if (!gdpa && record.gipa && record.instance)
					gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(record.gipa(record.instance, "vkGetDeviceProcAddr"));
				if (!gdpa && s.loaderModule)
					gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(s.loaderModule, "vkGetDeviceProcAddr"));
				if (gdpa)
					gdpaMod = DescribeModuleOfAddress(reinterpret_cast<const void*>(gdpa));
			}
			const bool hasCapturedIcd =
				s.capturedIcdCreateSwap.load(std::memory_order_acquire) != nullptr ||
				s.capturedIcdQueuePresent.load(std::memory_order_acquire) != nullptr;
			if (!gdpa && !hasCapturedIcd)
				return;

			if (hasCapturedIcd)
				Log::Info("Vulkan hooks: resolved device procs via icd-capture (fallback {} [{}])", gdpaSource, gdpaMod);
			else
				Log::Info("Vulkan hooks: resolved device proc resolver via {} ({})", gdpaSource, gdpaMod);

			auto isLoaderAddr = [&](void* addr) -> bool {
				if (!addr || !s.loaderModule)
					return false;
				HMODULE mod = nullptr;
				if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					reinterpret_cast<LPCWSTR>(addr), &mod))
					return false;
				return mod == s.loaderModule;
			};

			auto isOwnHook = [](void* addr) -> bool {
				return addr == reinterpret_cast<void*>(&HookCreateSwapchain) ||
					addr == reinterpret_cast<void*>(&HookCreateSwapchainViaGdpa) ||
					addr == reinterpret_cast<void*>(&HookCreateSwapchainViaGipa) ||
					addr == reinterpret_cast<void*>(&HookCreateSwapchainViaIcd) ||
					addr == reinterpret_cast<void*>(&HookDestroySwapchain) ||
					addr == reinterpret_cast<void*>(&HookDestroySwapchainViaGdpa) ||
					addr == reinterpret_cast<void*>(&HookDestroySwapchainViaGipa) ||
					addr == reinterpret_cast<void*>(&HookDestroySwapchainViaIcd) ||
					addr == reinterpret_cast<void*>(&HookQueuePresent) ||
					addr == reinterpret_cast<void*>(&HookQueuePresentViaGdpa) ||
					addr == reinterpret_cast<void*>(&HookQueuePresentViaGipa) ||
					addr == reinterpret_cast<void*>(&HookQueuePresentViaIcd);
			};

			auto hookIcdFn = [&](const char* name, void* capturedNativeTarget, void* detour, void* rawExport, void*& hookedSlot,
				auto& atomicTramp, std::atomic_bool& activeFlag, std::string& modOut, bool* usedNativeOut) -> bool {
				if (hookedSlot != nullptr)
				{
					modOut = DescribeModuleOfAddress(hookedSlot);
					return true;
				}
				bool usedNative = false;
				void* target = capturedNativeTarget;
				if (target && target != rawExport && !isOwnHook(target))
				{
					usedNative = true;
				}
				else
				{
					PFN_vkVoidFunction fn = gdpa ? gdpa(device, name) : nullptr;
					target = reinterpret_cast<void*>(fn);
				}
				if (!target || target == rawExport || isOwnHook(target))
					return false;
				if (isLoaderAddr(target) && IsUnsafeShortStub(target))
					return false;
				using FnType = decltype(atomicTramp.load());
				FnType tramp = nullptr;
				if (MH_CreateHook(target, detour, reinterpret_cast<void**>(&tramp)) != MH_OK || !tramp)
					return false;
				atomicTramp.store(tramp, std::memory_order_release);
				if (MH_EnableHook(target) != MH_OK)
				{
					atomicTramp.store(nullptr, std::memory_order_release);
					MH_RemoveHook(target);
					return false;
				}
				hookedSlot = target;
				activeFlag.store(true, std::memory_order_release);
				if (usedNativeOut)
					*usedNativeOut = usedNative;
				modOut = DescribeModuleOfAddress(target);
				return true;
			};

			std::string createMod = "none";
			std::string destroyMod = "none";
			std::string presentMod = "none";
			bool createUsedNative = false;

			const bool hookedCreate = hookIcdFn("vkCreateSwapchainKHR", s.capturedIcdCreateSwap.load(std::memory_order_acquire),
				reinterpret_cast<void*>(&HookCreateSwapchainViaIcd),
				s.rawCreateSwapchain, s.hookedIcdCreateSwap, s.icdCreateSwapchain, s.icdCreateSwapActive, createMod, &createUsedNative);
			if (hookedCreate && createUsedNative)
				s.icdCreateSwapIsNative.store(true, std::memory_order_release);
			const bool hookedDestroy = hookIcdFn("vkDestroySwapchainKHR", s.capturedIcdDestroySwap.load(std::memory_order_acquire),
				reinterpret_cast<void*>(&HookDestroySwapchainViaIcd),
				s.rawDestroySwapchain, s.hookedIcdDestroySwap, s.icdDestroySwapchain, s.icdDestroySwapActive, destroyMod, nullptr);
			const bool hookedPresent = hookIcdFn("vkQueuePresentKHR", s.capturedIcdQueuePresent.load(std::memory_order_acquire),
				reinterpret_cast<void*>(&HookQueuePresentViaIcd),
				s.rawQueuePresent, s.hookedIcdQueuePresent, s.icdQueuePresent, s.icdQueuePresentActive, presentMod, nullptr);

			if (hookedPresent)
				s.presentHooked.store(true, std::memory_order_relaxed);

			if (s.hookedIcdCreateSwap || s.hookedIcdDestroySwap || s.hookedIcdQueuePresent)
			{
				Log::Info("Vulkan hooks: installed device-level hooks (createSwapchain={} [{}], destroySwapchain={} [{}], queuePresent={} [{}]) for 8-buffer MFG & present pacing",
					hookedCreate, createMod, hookedDestroy, destroyMod, hookedPresent, presentMod);
			}
		}
		catch (...)
		{
		}
	}

	// Safe per-device swapchain/present interception via vkGetDeviceProcAddr:
	// When OptiScaler/Detours has already patched vulkan-1.dll's short device-dispatch stubs, inline hooking
	// vkCreateSwapchainKHR/vkQueuePresentKHR is skipped for safety. Wrapping them here for prepared game devices
	// restores 6->8 MFG swapchain buffering and CPU present pacing without touching DXVK/VKD3D or bridge devices.
	PFN_vkVoidFunction GetDeviceProcAddrImpl(PFN_vkGetDeviceProcAddr next, VkDevice device, const char* pName)
	{
		if (!next)
			return nullptr;

		static thread_local bool inGdpa = false;
		if (inGdpa)
			return next(device, pName);

		ReentryGuard guard(inGdpa);
		PFN_vkVoidFunction fn = next(device, pName);

		if (!fn || !pName || !device)
			return fn;

		State& s = S();
		if (!s.enabled || s.prepared.load(std::memory_order_relaxed) == 0)
			return fn;

		const bool isCreateSwap = (std::strcmp(pName, "vkCreateSwapchainKHR") == 0);
		const bool isDestroySwap = (!isCreateSwap && std::strcmp(pName, "vkDestroySwapchainKHR") == 0);
		const bool isPresent = (!isCreateSwap && !isDestroySwap && std::strcmp(pName, "vkQueuePresentKHR") == 0);
		if (!isCreateSwap && !isDestroySwap && !isPresent)
			return fn;

		bool trackedDevice = false;
		{
			std::lock_guard lock(s.mutex);
			trackedDevice = (s.devices.find(device) != s.devices.end());
		}
		if (!trackedDevice)
			return fn;

		void* addr = reinterpret_cast<void*>(fn);
		if (isCreateSwap)
		{
			if (s.icdCreateSwapActive.load(std::memory_order_acquire))
				return fn;
			if (addr != s.rawCreateSwapchain &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchain) &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchainViaGdpa) &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchainViaGipa) &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchainViaIcd))
			{
				s.gdpaCreateSwapchain.store(reinterpret_cast<PFN_vkCreateSwapchainKHR>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookCreateSwapchainViaGdpa);
			}
		}
		else if (isDestroySwap)
		{
			if (s.icdDestroySwapActive.load(std::memory_order_acquire))
				return fn;
			if (addr != s.rawDestroySwapchain &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchain) &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchainViaGdpa) &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchainViaGipa) &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchainViaIcd))
			{
				s.gdpaDestroySwapchain.store(reinterpret_cast<PFN_vkDestroySwapchainKHR>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookDestroySwapchainViaGdpa);
			}
		}
		else if (isPresent)
		{
			if (s.icdQueuePresentActive.load(std::memory_order_acquire))
				return fn;
			if (addr != s.rawQueuePresent &&
				addr != reinterpret_cast<void*>(&HookQueuePresent) &&
				addr != reinterpret_cast<void*>(&HookQueuePresentViaGdpa) &&
				addr != reinterpret_cast<void*>(&HookQueuePresentViaGipa) &&
				addr != reinterpret_cast<void*>(&HookQueuePresentViaIcd))
			{
				s.gdpaQueuePresent.store(reinterpret_cast<PFN_vkQueuePresentKHR>(fn), std::memory_order_release);
				s.presentHooked.store(true, std::memory_order_relaxed);
				if (!s.gdpaPresentLogged.exchange(true, std::memory_order_relaxed))
					Log::Info("Vulkan hooks: wrapped swapchain/present via vkGetDeviceProcAddr (pacing & 8-buffer MFG ready)");
				return reinterpret_cast<PFN_vkVoidFunction>(&HookQueuePresentViaGdpa);
			}
		}

		return fn;
	}

	VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookGetDeviceProcAddr(VkDevice device, const char* pName)
	{
		return GetDeviceProcAddrImpl(S().loaderGdpa, device, pName);
	}

	VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookGetDeviceProcAddrViaGipa(VkDevice device, const char* pName)
	{
		State& s = S();
		PFN_vkGetDeviceProcAddr next = s.gipaGetDeviceProcAddr.load(std::memory_order_acquire);
		return GetDeviceProcAddrImpl(next ? next : s.loaderGdpa, device, pName);
	}

	// OptiScaler hooks vulkan-1.dll before ASI plugins run and returns its own hkvkCreateDevice from
	// vkGetInstanceProcAddr. Wrapping vkGetInstanceProcAddr catches games and Streamline plugins that resolve
	// instance/device procs (including vkGetDeviceProcAddr and swapchain/present) via GIPA.
	VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookGetInstanceProcAddr(VkInstance instance, const char* pName)
	{
		State& s = S();
		if (instance)
			TrackLateInstance(instance, XEFG_RETURN_ADDRESS());

		PFN_vkVoidFunction fn = s.loaderGipa ? s.loaderGipa(instance, pName) : nullptr;

		if (!fn || !pName)
			return fn;

		if (std::strcmp(pName, "vkCreateDevice") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawCreateDevice &&
				addr != reinterpret_cast<void*>(&HookCreateDevice) &&
				addr != reinterpret_cast<void*>(&HookCreateDeviceViaGipa))
			{
				s.gipaCreateDevice.store(reinterpret_cast<PFN_vkCreateDevice>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookCreateDeviceViaGipa);
			}
		}
		else if (std::strcmp(pName, "vkCreateInstance") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawCreateInstance &&
				addr != reinterpret_cast<void*>(&HookCreateInstance) &&
				addr != reinterpret_cast<void*>(&HookCreateInstanceViaGipa))
			{
				s.gipaCreateInstance.store(reinterpret_cast<PFN_vkCreateInstance>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookCreateInstanceViaGipa);
			}
		}
		else if (std::strcmp(pName, "vkDestroyInstance") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawDestroyInstance &&
				addr != reinterpret_cast<void*>(&HookDestroyInstance) &&
				addr != reinterpret_cast<void*>(&HookDestroyInstanceViaGipa))
			{
				s.gipaDestroyInstance.store(reinterpret_cast<State::DestroyInstanceFn>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookDestroyInstanceViaGipa);
			}
		}
		else if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawGetDeviceProcAddr &&
				addr != reinterpret_cast<void*>(&HookGetDeviceProcAddr) &&
				addr != reinterpret_cast<void*>(&HookGetDeviceProcAddrViaGipa))
			{
				s.gipaGetDeviceProcAddr.store(reinterpret_cast<PFN_vkGetDeviceProcAddr>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookGetDeviceProcAddrViaGipa);
			}
		}
		else if (std::strcmp(pName, "vkCreateSwapchainKHR") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawCreateSwapchain &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchain) &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchainViaGdpa) &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchainViaGipa) &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchainViaIcd))
			{
				s.gipaCreateSwapchain.store(reinterpret_cast<PFN_vkCreateSwapchainKHR>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookCreateSwapchainViaGipa);
			}
		}
		else if (std::strcmp(pName, "vkDestroySwapchainKHR") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawDestroySwapchain &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchain) &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchainViaGdpa) &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchainViaGipa) &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchainViaIcd))
			{
				s.gipaDestroySwapchain.store(reinterpret_cast<PFN_vkDestroySwapchainKHR>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookDestroySwapchainViaGipa);
			}
		}
		else if (std::strcmp(pName, "vkQueuePresentKHR") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawQueuePresent &&
				addr != reinterpret_cast<void*>(&HookQueuePresent) &&
				addr != reinterpret_cast<void*>(&HookQueuePresentViaGdpa) &&
				addr != reinterpret_cast<void*>(&HookQueuePresentViaGipa) &&
				addr != reinterpret_cast<void*>(&HookQueuePresentViaIcd))
			{
				s.gipaQueuePresent.store(reinterpret_cast<PFN_vkQueuePresentKHR>(fn), std::memory_order_release);
				s.presentHooked.store(true, std::memory_order_relaxed);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookQueuePresentViaGipa);
			}
		}

		return fn;
	}

	// ------------------------------------------------------------------------------------------------ install

	// With the install lock held
	bool InstallLoader(HMODULE loader, std::string& why)
	{
		State& s = S();

		auto create = GetProcAddress(loader, "vkCreateInstance");
		auto createDevice = GetProcAddress(loader, "vkCreateDevice");
		auto gipa = GetProcAddress(loader, "vkGetInstanceProcAddr");

		if (!create || !createDevice || !gipa)
		{
			why = "vulkan-1.dll has no vkCreateInstance, vkCreateDevice or vkGetInstanceProcAddr";
			return false;
		}

		if (const auto status = MH_Initialize(); status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED)
		{
			why = std::string("MinHook can't start: ") + MH_StatusToString(status);
			return false;
		}

		s.loaderModule = loader;
		s.rawCreateInstance = reinterpret_cast<void*>(create);
		s.rawCreateDevice = reinterpret_cast<void*>(createDevice);
		s.loaderGipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(gipa);

		auto fail = [&why](const char* what, MH_STATUS status)
			{
				why = std::string(what) + ": " + MH_StatusToString(status);
				return false;
			};

		if (const auto status = MH_CreateHook(reinterpret_cast<void*>(create), reinterpret_cast<void*>(&HookCreateInstance),
				reinterpret_cast<void**>(&s.createInstance)); status != MH_OK)
			return fail("Can't hook vkCreateInstance", status);

		if (const auto status = MH_CreateHook(reinterpret_cast<void*>(createDevice), reinterpret_cast<void*>(&HookCreateDevice),
				reinterpret_cast<void**>(&s.createDevice)); status != MH_OK)
		{
			MH_RemoveHook(reinterpret_cast<void*>(create));
			return fail("Can't hook vkCreateDevice", status);
		}

		if (const auto status = MH_EnableHook(reinterpret_cast<void*>(create)); status != MH_OK)
		{
			MH_RemoveHook(reinterpret_cast<void*>(create));
			MH_RemoveHook(reinterpret_cast<void*>(createDevice));
			return fail("Can't enable the vkCreateInstance hook", status);
		}

		if (const auto status = MH_EnableHook(reinterpret_cast<void*>(createDevice)); status != MH_OK)
		{
			MH_DisableHook(reinterpret_cast<void*>(create));
			MH_RemoveHook(reinterpret_cast<void*>(create));
			MH_RemoveHook(reinterpret_cast<void*>(createDevice));
			return fail("Can't enable the vkCreateDevice hook", status);
		}

		PFN_vkGetInstanceProcAddr gipaTrampoline = nullptr;
		if (MH_CreateHook(reinterpret_cast<void*>(gipa), reinterpret_cast<void*>(&HookGetInstanceProcAddr),
				reinterpret_cast<void**>(&gipaTrampoline)) == MH_OK)
		{
			s.loaderGipa = gipaTrampoline;
			if (MH_EnableHook(reinterpret_cast<void*>(gipa)) == MH_OK)
			{
				s.gipaHooked = true;
			}
			else
			{
				s.loaderGipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(gipa);
				MH_RemoveHook(reinterpret_cast<void*>(gipa));
				Log::Warn("Vulkan hooks: couldn't enable vkGetInstanceProcAddr hook, using loader detours only");
			}
		}
		else
		{
			Log::Warn("Vulkan hooks: couldn't hook vkGetInstanceProcAddr, using loader detours only");
		}

		// Optional present/swapchain hooks for recorded-mode frame pacing (non-fatal if unavailable).
		// Device-level Vulkan loader exports (vkCreateSwapchainKHR, vkDestroySwapchainKHR, vkQueuePresentKHR) are tiny
		// dispatch stubs (mov rax, [rcx]; jmp [rax+off]). If OptiScaler/Detours already patched a stub (0xE9/0xEB/0xFF 0x25),
		// layering MinHook over the detoured short stub can corrupt the trampoline and crash during swapchain creation.
		auto tryHook = [&](HMODULE mod, const char* name, void* detour, void** rawOut, auto& trampolineOut, bool skipUnsafeStub) -> bool {
			FARPROC proc = GetProcAddress(mod, name);
			if (!proc)
				return false;
			*rawOut = reinterpret_cast<void*>(proc);
			if (skipUnsafeStub && IsUnsafeShortStub(reinterpret_cast<const void*>(proc)))
			{
				Log::Info("Vulkan hooks: skipping inline hook on {} (short/already-detoured dispatch stub)", name);
				return false;
			}
			std::remove_reference_t<decltype(trampolineOut)> tramp = nullptr;
			if (MH_CreateHook(reinterpret_cast<void*>(proc), detour, reinterpret_cast<void**>(&tramp)) != MH_OK)
				return false;
			trampolineOut = tramp;
			if (MH_EnableHook(reinterpret_cast<void*>(proc)) != MH_OK)
			{
				trampolineOut = nullptr;
				MH_RemoveHook(reinterpret_cast<void*>(proc));
				return false;
			}
			return true;
		};

		tryHook(loader, "vkDestroyInstance", reinterpret_cast<void*>(&HookDestroyInstance), &s.rawDestroyInstance, s.destroyInstance, false);
		s.gdpaHooked = tryHook(loader, "vkGetDeviceProcAddr", reinterpret_cast<void*>(&HookGetDeviceProcAddr), &s.rawGetDeviceProcAddr, s.loaderGdpa, true);
		tryHook(loader, "vkCreateSwapchainKHR", reinterpret_cast<void*>(&HookCreateSwapchain), &s.rawCreateSwapchain, s.createSwapchain, true);
		tryHook(loader, "vkDestroySwapchainKHR", reinterpret_cast<void*>(&HookDestroySwapchain), &s.rawDestroySwapchain, s.destroySwapchain, true);
		s.presentHooked.store(tryHook(loader, "vkQueuePresentKHR", reinterpret_cast<void*>(&HookQueuePresent), &s.rawQueuePresent, s.queuePresent, true), std::memory_order_relaxed);

		s.loaderHooked = true;
		return true;
	}

	void InstallFromLoad(HMODULE loader) noexcept
	{
		try
		{
			State& s = S();
			std::lock_guard lock(s.install);

			if (s.loaderHooked)
				return;

			std::string why;

			if (InstallLoader(loader, why))
			{
				SetHookState(VulkanRuntime::Hooks::Installed, {});
				Log::Info("Vulkan hooks installed on vkCreateInstance and vkCreateDevice, vulkan-1.dll was just loaded");
			}
			else
			{
				SetHookState(VulkanRuntime::Hooks::Failed, why);
				Log::Error("Vulkan hooks: {}", why);
			}
		}
		catch (...)
		{
		}
	}

	bool IsLoaderName(const wchar_t* name)
	{
		const wchar_t* leaf = name;

		for (const wchar_t* p = name; *p; ++p)
		{
			if (*p == L'\\' || *p == L'/')
				leaf = p + 1;
		}

		return _wcsicmp(leaf, L"vulkan-1.dll") == 0 || _wcsicmp(leaf, L"vulkan-1") == 0;
	}

	HMODULE WINAPI HookLoadLibraryExW(LPCWSTR name, HANDLE file, DWORD flags)
	{
		State& s = S();
		const HMODULE module = s.loadLibraryExW(name, file, flags);

		constexpr DWORD NotCode = LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE;
		static thread_local bool inside = false;

		if (module && name && !(flags & NotCode) && !inside)
		{
			if (!s.loaderHooked && IsLoaderName(name))
			{
				inside = true;
				InstallFromLoad(module);
				inside = false;
			}
			else if (!s.icdGipaTramp.load(std::memory_order_acquire) && IsKnownIcdName(name))
			{
				inside = true;
				TryHookLoadedIcd(s);
				inside = false;
			}
		}

		return module;
	}

	// With the install lock held: vulkan-1.dll isn't loaded, catch its loading. This hook is never taken out again, not
	// even once vulkan-1.dll is hooked: another hook (OptiScaler's own LoadLibrary hooks, an overlay's) may have been put
	// over this one since, and restoring the function's first bytes would cut it out. It only passes calls on from then on
	bool HookLoading(std::string& why)
	{
		State& s = S();

		if (const auto status = MH_Initialize(); status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED)
		{
			why = std::string("MinHook can't start: ") + MH_StatusToString(status);
			return false;
		}

		LPVOID target = nullptr;

		if (const auto status = MH_CreateHookApiEx(L"kernelbase", "LoadLibraryExW",
				reinterpret_cast<LPVOID>(&HookLoadLibraryExW), reinterpret_cast<LPVOID*>(&s.loadLibraryExW), &target);
			status != MH_OK)
		{
			why = std::string("Can't hook LoadLibraryExW: ") + MH_StatusToString(status);
			return false;
		}

		if (const auto status = MH_EnableHook(target); status != MH_OK)
		{
			MH_RemoveHook(target);
			s.loadLibraryExW = nullptr;
			why = std::string("Can't enable the LoadLibraryExW hook: ") + MH_StatusToString(status);
			return false;
		}

		return true;
	}
} // namespace

void VulkanHost::Start(const char* caller)
{
	State& s = S();
	std::lock_guard lock(s.install);

	if (s.started)
		return;

	// Process-wide guard: even if two copies of this DLL are ever loaded from different folders in the same
	// process, only the first one may install MinHook detours on vulkan-1.dll.
	const std::wstring guardName = L"Local\\XeFgVulkanHostHookGuard_" + std::to_wstring(GetCurrentProcessId());
	HANDLE guardMutex = CreateMutexW(nullptr, FALSE, guardName.c_str());
	if (guardMutex && GetLastError() == ERROR_ALREADY_EXISTS)
	{
		CloseHandle(guardMutex);
		s.started = true;
		Log::Info("Vulkan hooks are already owned by another instance in this process ({}); skipping duplicate install", caller);
		return;
	}
	// Intentionally kept open for the lifetime of the process when this instance owns the hooks.

	s.started = true;

	if (const HMODULE loader = GetModuleHandleW(LoaderName))
	{
		std::string why;

		if (InstallLoader(loader, why))
		{
			SetHookState(VulkanRuntime::Hooks::Installed, {});
			Log::Info("Vulkan hooks installed on vkCreateInstance and vkCreateDevice ({}). A game that already has its "
				"Vulkan device isn't covered, see the VulkanHooks notes in the readme", caller);
		}
		else
		{
			SetHookState(VulkanRuntime::Hooks::Failed, why);
			Log::Error("Vulkan hooks: {}", why);
		}

		return;
	}

	std::string why;

	if (!HookLoading(why))
	{
		SetHookState(VulkanRuntime::Hooks::Failed, why);
		Log::Error("Vulkan hooks: {}", why);
		return;
	}

	SetHookState(VulkanRuntime::Hooks::Waiting, {});
	Log::Info("vulkan-1.dll isn't loaded yet ({}), the Vulkan hooks go in when it is", caller);

	// It may have been loaded between the check and the hook
	if (const HMODULE loader = GetModuleHandleW(LoaderName); loader && !s.loaderHooked)
	{
		std::string late;

		if (InstallLoader(loader, late))
		{
			SetHookState(VulkanRuntime::Hooks::Installed, {});
			Log::Info("Vulkan hooks installed on vkCreateInstance and vkCreateDevice");
		}
		else
		{
			SetHookState(VulkanRuntime::Hooks::Failed, late);
			Log::Error("Vulkan hooks: {}", late);
		}
	}
}

void VulkanHost::SetEnabled(bool enabled)
{
	State& s = S();

	if (s.enabled.exchange(enabled) != enabled)
		Log::Info("Vulkan hooks {}", enabled ? "on" : "off, they only pass the game's calls on");
}

bool VulkanHost::Find(VkDevice device, DeviceRecord& out)
{
	State& s = S();
	std::lock_guard lock(s.mutex);

	const auto found = s.devices.find(device);

	if (found == s.devices.end())
		return false;

	out = found->second;
	return true;
}

std::string VulkanHost::RefusalFor(VkPhysicalDevice physical)
{
	State& s = S();
	std::lock_guard lock(s.mutex);

	const auto found = s.refused.find(physical);
	return found == s.refused.end() ? std::string() : found->second;
}

void VulkanHost::SetInternalBridgeBypass(bool bypass) noexcept
{
	if (bypass)
		++gInternalBridgeBypassDepth;
	else if (gInternalBridgeBypassDepth > 0)
		--gInternalBridgeBypassDepth;
}

namespace
{
	// Hold a 1 ms Windows system timer while Vulkan FG is actively generating (f5low-fg timerraise::Hold)
	// so Sleep(1) calls in Streamline/game threads never round up to 15.6 ms and drop 4x-6x generated frames.
	// Skipped under Wine/Proton; uses GetProcAddress on winmm.dll so winmm proxy DLLs forward cleanly.
	void SetSystemTimerRaised(State& s, bool on) noexcept
	{
		if (WineCompat::IsWineOrProton())
			return;
		if (s.timerPeriodRaised.load(std::memory_order_relaxed) == on)
			return;
		using TimerPeriodFn = UINT(WINAPI*)(UINT);
		static const HMODULE winmm = LoadLibraryW(L"winmm.dll");
		static const auto beginPeriod = winmm ? reinterpret_cast<TimerPeriodFn>(GetProcAddress(winmm, "timeBeginPeriod")) : nullptr;
		static const auto endPeriod = winmm ? reinterpret_cast<TimerPeriodFn>(GetProcAddress(winmm, "timeEndPeriod")) : nullptr;
		if (!beginPeriod || !endPeriod)
			return;
		if (on)
		{
			if (!s.timerPeriodRaised.exchange(true, std::memory_order_acq_rel))
			{
				if (beginPeriod(1u) != 0u)
					s.timerPeriodRaised.store(false, std::memory_order_release);
			}
		}
		else
		{
			if (s.timerPeriodRaised.exchange(false, std::memory_order_acq_rel))
				endPeriod(1u);
		}
	}
}

void VulkanHost::NotifyBatchStart(uint32_t generatedCount) noexcept
{
	State& s = S();
	if (!PacingActive(s))
		return;
	SetSystemTimerRaised(s, true);
	std::lock_guard lock(s.pacerMutex);
	s.pacer.OnBatchStart(NowNs(), generatedCount);
}

void VulkanHost::NotifyPaused() noexcept
{
	State& s = S();
	SetSystemTimerRaised(s, false);
	std::lock_guard lock(s.pacerMutex);
	s.pacer.OnPaused();
}

VulkanRuntime::Status VulkanHost::Snapshot()
{
	State& s = S();
	VulkanRuntime::Status status;

	{
		std::lock_guard lock(s.mutex);
		status.hooks = s.hooks;
		status.detail = s.detail;
		status.refusal = s.lastRefusal;
	}

	status.enabled = s.enabled;
	status.instances = s.instanceCalls;
	status.devices = s.deviceCalls;
	status.prepared = s.prepared;

	return status;
}
