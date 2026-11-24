// Runs the real VulkanProvider.cpp and ExportsVulkan.cpp on Linux the way a build with the Vulkan runtime runs them: the
// provider is the runtime built into the dll, here a fake that behaves like src/vulkan/VulkanRuntime.cpp does at its
// table (Init records the device, supported() and reason() say what keeps it from running, models come with the first
// evaluation). vk_test.cpp covers the other case, a build without the runtime and a MFGUnlocker.asi in the process.
#include "Windows.h"
#include "psapi.h"

#include "Log.h"
#include "NGX.h"
#include "VulkanProvider.h"
#include "VulkanRuntime.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------
// The exports under test (extern "C", no dllexport on Linux)
extern "C"
{
	NGXResult NVSDK_NGX_VULKAN_Init(unsigned long long, const wchar_t*, void*, void*, void*, void*, void*, const void*,
		uint32_t);
	NGXResult NVSDK_NGX_VULKAN_Init_Ext(unsigned long long, const wchar_t*, void*, void*, void*, uint32_t, const void*);
	NGXResult NVSDK_NGX_VULKAN_Init_Ext2(unsigned long long, const wchar_t*, void*, void*, void*, void*, void*, uint32_t,
		const void*);
	NGXResult NVSDK_NGX_VULKAN_Shutdown();
	NGXResult NVSDK_NGX_VULKAN_GetFeatureRequirements(void*, void*, const void*, NGXFeatureRequirement*);
	NGXResult NVSDK_NGX_VULKAN_CreateFeature(void*, uint32_t, NGXParameter*, NGXHandle**);
	NGXResult NVSDK_NGX_VULKAN_CreateFeature1(void*, void*, uint32_t, NGXParameter*, NGXHandle**);
	NGXResult NVSDK_NGX_VULKAN_ReleaseFeature(NGXHandle*);
	NGXResult NVSDK_NGX_VULKAN_EvaluateFeature(void*, const NGXHandle*, NGXParameter*, void*);
	NGXResult NVSDK_NGX_VULKAN_PopulateParameters_Impl(NGXParameter*);
}

// ---------------------------------------------------------------------------------------------------------------
static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                                                                   \
	do                                                                                                                \
	{                                                                                                                 \
		checks++;                                                                                                     \
		if (!(cond))                                                                                                  \
		{                                                                                                             \
			failures++;                                                                                               \
			std::printf("FAIL line %d: %s\n", __LINE__, #cond);                                                       \
		}                                                                                                             \
	} while (0)

// ---------------------------------------------------------------------------------------------------------------
static std::vector<std::string> logLines;

bool Log::IsEnabled() noexcept { return true; }
void Log::WarnForeignFgMods(bool, bool) noexcept {}
void Log::Write(std::string_view level, std::string_view message)
{
	logLines.push_back(std::string(level) + ": " + std::string(message));
}

static int Logged(const char* needle)
{
	int count = 0;
	for (const auto& line : logLines)
		if (line.find(needle) != std::string::npos)
			count++;
	return count;
}

// ---------------------------------------------------------------------------------------------------------------
struct FakeParams : NGXParameter
{
	std::map<std::string, long long> ints;
	std::map<std::string, void*> ptrs;

	void Set(const char* n, unsigned long long v) override { ints[n] = (long long)v; }
	void Set(const char*, float) override {}
	void Set(const char*, double) override {}
	void Set(const char* n, unsigned int v) override { ints[n] = v; }
	void Set(const char* n, int v) override { ints[n] = v; }
	void Set(const char*, ID3D11Resource*) override {}
	void Set(const char*, ID3D12Resource*) override {}
	void Set(const char* n, void* v) override { ptrs[n] = v; }

	NGXResult Get(const char*, unsigned long long*) const override { return NGX_FAIL; }
	NGXResult Get(const char*, float*) const override { return NGX_FAIL; }
	NGXResult Get(const char*, double*) const override { return NGX_FAIL; }
	NGXResult Get(const char*, unsigned int*) const override { return NGX_FAIL; }
	NGXResult Get(const char*, int*) const override { return NGX_FAIL; }
	NGXResult Get(const char*, ID3D11Resource**) const override { return NGX_FAIL; }
	NGXResult Get(const char*, ID3D12Resource**) const override { return NGX_FAIL; }
	NGXResult Get(const char*, void**) const override { return NGX_FAIL; }
	void Reset() override {}
};

// ---------------------------------------------------------------------------------------------------------------
// The fake runtime
struct Runtime
{
	std::vector<std::string> started;
	bool blockedBeforeDevice = false; // hooks off, models missing...
	std::string blockedReason = "the Vulkan hooks are off";
	bool blockedAfterInit = false;    // the device wasn't prepared...
	std::string deviceReason = "the game created its VkDevice before this dll's hooks were installed";
	NGXResult initResult = NGX_SUCCESS;

	bool deviceKnown = false;
	void* instance = nullptr;
	void* physical = nullptr;
	void* device = nullptr;
	void* gipa = nullptr;
	void* gdpa = nullptr;
	int initCalls = 0, createCalls = 0, evaluateCalls = 0, releaseCalls = 0, shutdownCalls = 0;
	std::string model;
	void* releasedImpl = nullptr;
	void* evalImpl = nullptr;
	NGXResult evaluateResult = NGX_SUCCESS;
} runtime;

static char implStorage[16][8];
static int implCount = 0;

static bool RuntimeBlocked(std::string& why)
{
	if (runtime.deviceKnown && runtime.blockedAfterInit)
	{
		why = runtime.deviceReason;
		return true;
	}

	if (!runtime.deviceKnown && runtime.blockedBeforeDevice)
	{
		why = runtime.blockedReason;
		return true;
	}

	return false;
}

static NGXResult RtInit(void* instance, void* physical, void* device, void* gipa, void* gdpa)
{
	runtime.initCalls++;

	if (runtime.initResult != NGX_SUCCESS)
		return runtime.initResult;

	runtime.deviceKnown = true;
	runtime.instance = instance;
	runtime.physical = physical;
	runtime.device = device;
	runtime.gipa = gipa;
	runtime.gdpa = gdpa;

	return NGX_SUCCESS;
}

static NGXResult RtCreate(void*, void*, NGXParameter*, void** impl)
{
	runtime.createCalls++;
	*impl = implStorage[implCount++ % 16];
	return NGX_SUCCESS;
}

static NGXResult RtEvaluate(void*, void* impl, NGXParameter*)
{
	runtime.evaluateCalls++;
	runtime.evalImpl = impl;

	if (runtime.evaluateResult == NGX_SUCCESS)
		runtime.model = "port";

	return runtime.evaluateResult;
}

static NGXResult RtRelease(void* impl)
{
	runtime.releaseCalls++;
	runtime.releasedImpl = impl;
	return NGX_SUCCESS;
}

static NGXResult RtShutdown()
{
	runtime.shutdownCalls++;
	return NGX_SUCCESS;
}

static uint32_t RtSupported()
{
	std::string why;
	return RuntimeBlocked(why) ? 0 : 5;
}

static int RtReason(char* buffer, uint32_t size)
{
	std::string why;
	const bool blocked = RuntimeBlocked(why);

	if (buffer && size)
	{
		std::strncpy(buffer, blocked ? why.c_str() : "", size - 1);
		buffer[size - 1] = 0;
	}

	return blocked ? 1 : 0;
}

static const VulkanProvider::Api table{ sizeof(VulkanProvider::Api), 1, &RtInit, &RtCreate, &RtEvaluate, &RtRelease,
	&RtShutdown, &RtSupported, &RtReason };

bool VulkanRuntime::Built() { return true; }
const VulkanProvider::Api* VulkanRuntime::Table() { return &table; }
void VulkanRuntime::Start(const char* caller) { runtime.started.push_back(caller); }
void VulkanRuntime::SetHooksEnabled(bool) {}
VulkanRuntime::Status VulkanRuntime::GetStatus() { return {}; }
std::string VulkanRuntime::ModelName() { return runtime.model; }

// ---------------------------------------------------------------------------------------------------------------
// The process: nothing in it may be searched, the runtime is the provider
static int enumCalls = 0;
static int getProcCalls = 0;
static ULONGLONG now = 100000;

HANDLE GetCurrentProcess() { return reinterpret_cast<HANDLE>(1); }
ULONGLONG GetTickCount64() { return now; }

BOOL EnumProcessModules(HANDLE, HMODULE*, DWORD, DWORD* needed)
{
	enumCalls++;
	*needed = 0;
	return 1;
}

FARPROC GetProcAddress(HMODULE, const char*)
{
	getProcCalls++;
	return nullptr;
}

DWORD GetModuleFileNameW(HMODULE, wchar_t*, DWORD) { return 0; }

int WideCharToMultiByte(UINT, DWORD, const wchar_t*, int, char*, int, const char*, BOOL*) { return 0; }

// ---------------------------------------------------------------------------------------------------------------
static void* const deviceA = reinterpret_cast<void*>(0xA000);
static void* const instance = reinterpret_cast<void*>(0x1000);
static void* const physical = reinterpret_cast<void*>(0x2000);
static void* const gipa = reinterpret_cast<void*>(0x3000);
static void* const gdpa = reinterpret_cast<void*>(0x4000);
static void* const cmdBuf = reinterpret_cast<void*>(0xC000);

static void Reset()
{
	runtime = Runtime{};
	logLines.clear();
	VulkanProvider::DeviceChanged();
	VulkanProvider::initialized = false;
}

int main()
{
	// ---- the provider is the runtime, nothing is searched for
	{
		Reset();
		CHECK(VulkanProvider::Get() == &table);
		CHECK(VulkanProvider::Usable(VulkanProvider::Get()));
		CHECK(VulkanProvider::HasReason(VulkanProvider::Get()));
		CHECK(enumCalls == 0 && getProcCalls == 0);

		const auto info = VulkanProvider::Query();
		CHECK(info.state == VulkanProvider::State::Ready);
		CHECK(info.maxFrames == 5);
		CHECK(info.module == "built into this dll");
		CHECK(enumCalls == 0 && getProcCalls == 0);
	}

	// ---- Init: the hooks are started, then the device goes to the runtime
	{
		Reset();
		CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) == NGX_SUCCESS);
		CHECK(runtime.started.size() == 1 && runtime.started[0] == "NVSDK_NGX_VULKAN_Init");
		CHECK(runtime.initCalls == 1);
		CHECK(runtime.instance == instance && runtime.physical == physical && runtime.device == deviceA);
		CHECK(runtime.gipa == gipa && runtime.gdpa == gdpa);
		CHECK(VulkanProvider::initialized.load());
		CHECK(Logged("built in") == 1);

		// Init_Ext and Init_Ext2 reach it too, the first without the loader's functions
		CHECK(NVSDK_NGX_VULKAN_Init_Ext(1, L"", instance, physical, deviceA, 0, nullptr) == NGX_SUCCESS);
		CHECK(runtime.initCalls == 2 && runtime.gipa == nullptr && runtime.gdpa == nullptr);
		CHECK(NVSDK_NGX_VULKAN_Init_Ext2(1, L"", instance, physical, deviceA, gipa, gdpa, 0, nullptr) == NGX_SUCCESS);
		CHECK(runtime.initCalls == 3 && runtime.gipa == gipa);
		CHECK(runtime.started.size() == 3);
	}

	// ---- Init: the runtime refuses the device
	{
		Reset();
		runtime.initResult = NGX_FAIL_PLATFORM_ERROR;
		CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) ==
			NGX_FAIL_PLATFORM_ERROR);
		CHECK(!VulkanProvider::initialized.load());
		CHECK(Logged("refused the device") == 1);
	}

	// ---- Init: the device can't run XeMFG (the hooks didn't prepare it): refused now, and said so, with the device known
	{
		Reset();
		runtime.blockedAfterInit = true;
		CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) ==
			NGX_FAIL_FEATURE_NOT_SUPPORTED);
		CHECK(runtime.initCalls == 1);
		// A game gave NGX a device, the menu shows what is wrong
		CHECK(VulkanProvider::initialized.load());

		const auto info = VulkanProvider::Query();
		CHECK(info.state == VulkanProvider::State::Blocked);
		CHECK(info.reason == runtime.deviceReason);
		CHECK(Logged("can't run: the game created its VkDevice") == 1);
		CHECK(Logged("can't run on this device") == 1);

		// Asked again, said once
		CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) ==
			NGX_FAIL_FEATURE_NOT_SUPPORTED);
		CHECK(Logged("can't run on this device") == 1);
		CHECK(Logged("can't run: the game created its VkDevice") == 1);
	}

	// ---- What keeps it from running before any game gave it a device is no news in the log, the answer is not cached
	{
		Reset();
		runtime.blockedBeforeDevice = true;
		auto info = VulkanProvider::Query();
		CHECK(info.state == VulkanProvider::State::Blocked);
		CHECK(info.reason == runtime.blockedReason);
		CHECK(Logged("can't run") == 0);

		runtime.blockedBeforeDevice = false;
		info = VulkanProvider::Query();
		CHECK(info.state == VulkanProvider::State::Ready);
		CHECK(Logged("is ready") == 1);

		// The other way round, after a game gave a device
		Reset();
		VulkanProvider::initialized = true;
		runtime.blockedBeforeDevice = true;
		CHECK(VulkanProvider::Query().state == VulkanProvider::State::Blocked);
		CHECK(Logged("Vulkan XeMFG runtime can't run") == 1);
	}

	// ---- Init when the build's runtime says it can't run before it has a device (models missing, hooks failed)
	{
		Reset();
		runtime.blockedBeforeDevice = true;
		// The runtime still takes the device: its answer afterwards is what counts
		CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) == NGX_SUCCESS);
		CHECK(runtime.initCalls == 1);
	}

	// ---- Features
	{
		Reset();
		FakeParams params;
		NGXHandle* handle = nullptr;
		CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) == NGX_SUCCESS);

		CHECK(NVSDK_NGX_VULKAN_CreateFeature1(deviceA, cmdBuf, NGX_FEATURE_FRAME_GENERATION, &params, &handle) ==
			NGX_SUCCESS);
		CHECK(handle && IsVulkanHandle(handle));
		CHECK(runtime.createCalls == 1);
		CHECK(VulkanProvider::liveFeatures.load() == 1);

		// Another feature than frame generation is not ours
		NGXHandle* other = nullptr;
		CHECK(NVSDK_NGX_VULKAN_CreateFeature(cmdBuf, 1, &params, &other) == NGX_FAIL_FEATURE_NOT_SUPPORTED);
		CHECK(!other && runtime.createCalls == 1);

		CHECK(runtime.model.empty() && VulkanProvider::ModelName().empty());
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, handle, &params, nullptr) == NGX_SUCCESS);
		CHECK(runtime.evaluateCalls == 1 && runtime.evalImpl != nullptr);
		CHECK(VulkanProvider::ModelName() == "port");
		CHECK(Logged("Vulkan: XeFG model port") == 1);

		// The model is logged once
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, handle, &params, nullptr) == NGX_SUCCESS);
		CHECK(Logged("Vulkan: XeFG model") == 1);

		// A failed evaluation is the runtime's result as it is
		runtime.evaluateResult = NGX_FAIL;
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, handle, &params, nullptr) == NGX_FAIL);
		runtime.evaluateResult = NGX_SUCCESS;

		void* impl = runtime.evalImpl;
		CHECK(NVSDK_NGX_VULKAN_ReleaseFeature(handle) == NGX_SUCCESS);
		CHECK(runtime.releaseCalls == 1 && runtime.releasedImpl == impl);
		CHECK(VulkanProvider::liveFeatures.load() == 0);
	}

	// ---- Shutdown
	{
		Reset();
		CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) == NGX_SUCCESS);
		CHECK(VulkanProvider::initialized.load());
		CHECK(NVSDK_NGX_VULKAN_Shutdown() == NGX_SUCCESS);
		CHECK(runtime.shutdownCalls == 1);
		CHECK(!VulkanProvider::initialized.load());
	}

	// ---- What OptiScaler is told
	{
		Reset();
		FakeParams params;
		CHECK(NVSDK_NGX_VULKAN_PopulateParameters_Impl(&params) == NGX_SUCCESS);
		CHECK(params.ints["DLSSG.MultiFrameCountMax"] == 5);
		CHECK(params.ints["FrameGeneration.Available"] == 1);
		CHECK(params.ptrs.count("DLSSG.GetCurrentSettingsCallback") == 1);

		NGXFeatureRequirement requirement{};
		CHECK(NVSDK_NGX_VULKAN_GetFeatureRequirements(nullptr, nullptr, nullptr, &requirement) == NGX_SUCCESS);
		CHECK(requirement.FeatureSupported == 0);

		// Blocked, every answer says it isn't there
		runtime.blockedBeforeDevice = true;
		FakeParams blocked;
		CHECK(NVSDK_NGX_VULKAN_PopulateParameters_Impl(&blocked) == NGX_SUCCESS);
		CHECK(blocked.ints["DLSSG.MultiFrameCountMax"] == 0);
		CHECK(blocked.ints["FrameGeneration.Available"] == 0);
		CHECK(blocked.ptrs.count("DLSSG.GetCurrentSettingsCallback") == 0);
		CHECK(NVSDK_NGX_VULKAN_GetFeatureRequirements(nullptr, nullptr, nullptr, &requirement) == NGX_SUCCESS);
		CHECK(requirement.FeatureSupported == 0x10);
	}

	std::printf("%d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}
