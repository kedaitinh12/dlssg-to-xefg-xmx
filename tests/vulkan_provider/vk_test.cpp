// Runs the real VulkanProvider.cpp and ExportsVulkan.cpp on Linux against a fake process (Win32 stubs) and a fake
// MFGUnlocker.asi provider table that behaves like the one disassembled from 1.7.1-beta2fix.1.
#include "Windows.h"
#include "psapi.h"

#include "Log.h"
#include "NGX.h"
#include "VulkanProvider.h"

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
	NGXResult NVSDK_NGX_VULKAN_Shutdown1(void*);
	NGXResult NVSDK_NGX_VULKAN_GetScratchBufferSize(uint32_t, const NGXParameter*, size_t*);
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
// Log
static std::vector<std::string> logLines;

void Log::Write(std::string_view level, std::string_view message)
{
	logLines.push_back(std::string(level) + ": " + std::string(message));
}

static bool Logged(const char* needle)
{
	for (const auto& line : logLines)
		if (line.find(needle) != std::string::npos)
			return true;
	return false;
}

// ---------------------------------------------------------------------------------------------------------------
// A fake NVSDK_NGX_Parameter
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
// A fake MFGUnlocker.asi: the same table layout and the same behaviours as the disassembled provider
struct FakeAsi
{
	bool blocked = false;
	std::string reason = "xefg_fake_block";
	std::string model = "";

	void* device = nullptr;
	void* instance = nullptr;
	void* gipa = nullptr;
	void* gdpa = nullptr;
	int initCalls = 0, createCalls = 0, evaluateCalls = 0, releaseCalls = 0, shutdownCalls = 0, supportedCalls = 0;

	void* createDevice = reinterpret_cast<void*>(1);
	void* createCmd = nullptr;
	NGXParameter* createParams = nullptr;
	void* evalCmd = nullptr;
	void* evalImpl = nullptr;
	NGXParameter* evalParams = nullptr;
	void* releasedImpl = nullptr;

	NGXResult createResult = NGX_SUCCESS;
	bool createNullImpl = false;
} asi;

static char implStorage[16][8];
static int implCount = 0;

static NGXResult AsiInit(void* instance, void* pd, void* device, void* gipa, void* gdpa)
{
	(void)pd;
	asi.initCalls++;
	if (device && asi.device && device != asi.device)
		return NGX_FAIL | 2; // 0xBAD00002, like the real vkInit
	asi.instance = instance;
	asi.device = device;
	asi.gipa = gipa;
	asi.gdpa = gdpa;
	return NGX_SUCCESS;
}

static NGXResult AsiCreate(void* device, void* cmd, NGXParameter* params, void** impl)
{
	asi.createCalls++;
	asi.createDevice = device;
	asi.createCmd = cmd;
	asi.createParams = params;
	if (!impl)
		return NGX_FAIL_INVALID_PARAMETER;
	*impl = nullptr;
	if (asi.createResult != NGX_SUCCESS)
		return asi.createResult;
	if (!asi.createNullImpl)
		*impl = implStorage[implCount++ % 16];
	return NGX_SUCCESS;
}

static NGXResult AsiEvaluate(void* cmd, void* impl, NGXParameter* params)
{
	asi.evaluateCalls++;
	asi.evalCmd = cmd;
	asi.evalImpl = impl;
	asi.evalParams = params;
	if (!cmd || !impl || !params)
		return NGX_FAIL_INVALID_PARAMETER;
	return NGX_SUCCESS;
}

static NGXResult AsiRelease(void* impl)
{
	asi.releaseCalls++;
	asi.releasedImpl = impl;
	return impl ? NGX_SUCCESS : NGX_FAIL_INVALID_PARAMETER;
}

static NGXResult AsiShutdown()
{
	asi.shutdownCalls++;
	asi.device = nullptr;
	return NGX_SUCCESS;
}

static uint32_t AsiSupported()
{
	asi.supportedCalls++;
	return asi.blocked ? 0 : 5; // the disassembly: blocked ? 0 : 5
}

static int AsiReason(char* buffer, uint32_t size)
{
	if (asi.blocked)
	{
		if (buffer && size)
		{
			std::strncpy(buffer, asi.reason.c_str(), size - 1);
			buffer[size - 1] = 0;
		}
		return 1;
	}
	if (buffer && size)
		buffer[0] = 0;
	return 0;
}

static const char* AsiModelName() { return asi.model.c_str(); }

// The real table is 0x68 bytes, the Vulkan half is the first 0x40
struct FullTable
{
	VulkanProvider::Api api;
	void* d3d12[5];
};

static FullTable good{ { 0x68, 1, &AsiInit, &AsiCreate, &AsiEvaluate, &AsiRelease, &AsiShutdown, &AsiSupported,
	&AsiReason }, {} };

static VulkanProvider::Api tooSmall{ 0x30, 1, &AsiInit, &AsiCreate, &AsiEvaluate, &AsiRelease, &AsiShutdown,
	&AsiSupported, nullptr };
static VulkanProvider::Api wrongVersion{ 0x68, 2, &AsiInit, &AsiCreate, &AsiEvaluate, &AsiRelease, &AsiShutdown,
	&AsiSupported, &AsiReason };
static VulkanProvider::Api noEvaluate{ 0x68, 1, &AsiInit, &AsiCreate, nullptr, &AsiRelease, &AsiShutdown,
	&AsiSupported, &AsiReason };
// Accepted: the Vulkan half without reason(), the table stops at supported()
static VulkanProvider::Api noReason{ 0x38, 1, &AsiInit, &AsiCreate, &AsiEvaluate, &AsiRelease, &AsiShutdown,
	&AsiSupported, nullptr };

static const VulkanProvider::Api* GetTooSmall(uint32_t v) { return v == 1 ? &tooSmall : nullptr; }
static const VulkanProvider::Api* GetWrongVersion(uint32_t v) { return v == 1 ? &wrongVersion : nullptr; }
static const VulkanProvider::Api* GetNoEvaluate(uint32_t v) { return v == 1 ? &noEvaluate : nullptr; }
static const VulkanProvider::Api* GetNull(uint32_t) { return nullptr; }
static const VulkanProvider::Api* GetGood(uint32_t v) { return v == 1 ? &good.api : nullptr; }

// ---------------------------------------------------------------------------------------------------------------
// A fake process
struct FakeModule
{
	char handle;
	std::wstring path;
	const VulkanProvider::Api* (*getApi)(uint32_t) = nullptr;
	const char* (*modelName)() = nullptr;
};

static std::vector<FakeModule*> modules;
static ULONGLONG now = 100000;
static int enumCalls = 0;
static int getProcCalls = 0;

HANDLE GetCurrentProcess() { return reinterpret_cast<HANDLE>(1); }
ULONGLONG GetTickCount64() { return now; }

BOOL EnumProcessModules(HANDLE, HMODULE* out, DWORD bytes, DWORD* needed)
{
	enumCalls++;
	*needed = (DWORD)(modules.size() * sizeof(HMODULE));
	const size_t room = bytes / sizeof(HMODULE);
	for (size_t i = 0; i < modules.size() && i < room; i++)
		out[i] = reinterpret_cast<HMODULE>(&modules[i]->handle);
	return 1;
}

static FakeModule* Find(HMODULE module)
{
	for (auto* m : modules)
		if (reinterpret_cast<HMODULE>(&m->handle) == module)
			return m;
	return nullptr;
}

FARPROC GetProcAddress(HMODULE module, const char* name)
{
	getProcCalls++;
	const FakeModule* m = Find(module);
	if (!m)
		return nullptr;
	if (std::strcmp(name, "MfgXeMFGProviderGetApi") == 0 && m->getApi)
		return reinterpret_cast<FARPROC>(m->getApi);
	if (std::strcmp(name, "XeFGModelName") == 0 && m->modelName)
		return reinterpret_cast<FARPROC>(m->modelName);
	return nullptr;
}

DWORD GetModuleFileNameW(HMODULE module, wchar_t* path, DWORD size)
{
	const FakeModule* m = Find(module);
	if (!m)
		return 0;
	std::wcsncpy(path, m->path.c_str(), size - 1);
	return (DWORD)m->path.size();
}

int WideCharToMultiByte(UINT, DWORD, const wchar_t* wide, int wideCount, char* narrow, int narrowCount, const char*, BOOL*)
{
	// ASCII is all the tests use
	int length = 0;
	while (wideCount < 0 ? wide[length] != 0 : length < wideCount)
		length++;
	const int total = wideCount < 0 ? length + 1 : length;
	if (narrow && narrowCount >= total)
		for (int i = 0; i < total; i++)
			narrow[i] = (char)wide[i];
	return total;
}

// ---------------------------------------------------------------------------------------------------------------
static void* const deviceA = reinterpret_cast<void*>(0xA000);
static void* const deviceB = reinterpret_cast<void*>(0xB000);
static void* const instance = reinterpret_cast<void*>(0x1000);
static void* const physical = reinterpret_cast<void*>(0x2000);
static void* const gipa = reinterpret_cast<void*>(0x3000);
static void* const gdpa = reinterpret_cast<void*>(0x4000);
static void* const cmdBuf = reinterpret_cast<void*>(0xC000);

int main()
{
	using VulkanProvider::State;
	FakeParams params;
	NGXHandle* handle = nullptr;

	// ---- 1. no asi in the process: everything says "not available", nothing crashes
	static FakeModule game{ 'g', L"C:\\game\\game.exe" };
	modules.push_back(&game);

	CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) ==
		NGX_FAIL_FEATURE_NOT_SUPPORTED);
	CHECK(!VulkanProvider::initialized.load());
	CHECK(VulkanProvider::Query().state == State::NotLoaded);

	{
		FakeParams p;
		CHECK(NVSDK_NGX_VULKAN_PopulateParameters_Impl(&p) == NGX_SUCCESS);
		CHECK(p.ints["FrameGeneration.Available"] == 0);
		CHECK(p.ints["FrameGeneration.NeedsUpdatedDriver"] == 0);
		CHECK((uint32_t)p.ints["FrameGeneration.FeatureInitResult"] == 0xBAD00001u);
		CHECK(p.ints["DLSSG.MultiFrameCountMax"] == 0);
		CHECK(p.ptrs.empty());
		CHECK(NVSDK_NGX_VULKAN_PopulateParameters_Impl(nullptr) == NGX_FAIL_INVALID_PARAMETER);
	}

	{
		NGXFeatureRequirement requirement;
		std::memset(&requirement, 0xCD, sizeof(requirement));
		CHECK(NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, nullptr, &requirement) == NGX_SUCCESS);
		CHECK(requirement.FeatureSupported == 0x10);
		CHECK(requirement.MinHWArchitecture == 0);
		CHECK(requirement.MinOSVersion[0] == 0 && requirement.MinOSVersion[254] == 0);
		CHECK(NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, nullptr, nullptr) ==
			NGX_FAIL_INVALID_PARAMETER);
	}

	handle = reinterpret_cast<NGXHandle*>(0x1234);
	CHECK(NVSDK_NGX_VULKAN_CreateFeature(cmdBuf, NGX_FEATURE_FRAME_GENERATION, &params, &handle) ==
		NGX_FAIL_FEATURE_NOT_SUPPORTED);
	CHECK(handle == nullptr);
	CHECK(NVSDK_NGX_VULKAN_CreateFeature(cmdBuf, NGX_FEATURE_FRAME_GENERATION, nullptr, &handle) ==
		NGX_FAIL_INVALID_PARAMETER);
	CHECK(NVSDK_NGX_VULKAN_CreateFeature(cmdBuf, NGX_FEATURE_FRAME_GENERATION, &params, nullptr) ==
		NGX_FAIL_INVALID_PARAMETER);

	{
		NGXHandle fake{ NGX_HANDLE_VULKAN | 1 };
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, &fake, &params, nullptr) == NGX_FAIL_FEATURE_NOT_SUPPORTED);
	}

	CHECK(NVSDK_NGX_VULKAN_Shutdown() == NGX_SUCCESS);
	CHECK(NVSDK_NGX_VULKAN_Shutdown1(deviceA) == NGX_SUCCESS);
	CHECK(NVSDK_NGX_VULKAN_ReleaseFeature(nullptr) == NGX_FAIL_INVALID_PARAMETER);

	{
		size_t size = 123;
		CHECK(NVSDK_NGX_VULKAN_GetScratchBufferSize(NGX_FEATURE_FRAME_GENERATION, &params, &size) == NGX_SUCCESS);
		CHECK(size == 0);
		CHECK(NVSDK_NGX_VULKAN_GetScratchBufferSize(NGX_FEATURE_FRAME_GENERATION, &params, nullptr) ==
			NGX_FAIL_INVALID_PARAMETER);
	}

	// The search for a missing asi is rate limited: many calls, one pass over the process
	{
		const int before = enumCalls;
		for (int i = 0; i < 100; i++)
			CHECK(VulkanProvider::Get() == nullptr);
		CHECK(enumCalls - before <= 1);
	}

	// ---- 2. modules that export the provider with a table this dll can't use are skipped
	static FakeModule m1{ 'a', L"C:\\x\\small.asi", &GetTooSmall };
	static FakeModule m2{ 'b', L"C:\\x\\version2.asi", &GetWrongVersion };
	static FakeModule m3{ 'c', L"C:\\x\\noeval.asi", &GetNoEvaluate };
	static FakeModule m4{ 'd', L"C:\\x\\null.asi", &GetNull };
	modules.insert(modules.end(), { &m1, &m2, &m3, &m4 });

	now += 2500;
	CHECK(VulkanProvider::Get() == nullptr);
	CHECK(VulkanProvider::Query().state == State::NotLoaded);
	CHECK(Logged("small.asi exports MfgXeMFGProviderGetApi"));
	CHECK(Logged("noeval.asi exports MfgXeMFGProviderGetApi"));

	// ---- 3. the real asi shows up late, behind 400 modules so the module list has to grow past the first buffer
	static std::vector<FakeModule> filler(400, FakeModule{ 'f', L"C:\\windows\\filler.dll" });
	for (auto& f : filler)
		modules.push_back(&f);

	static FakeModule asiModule{ 'z', L"C:\\game\\MFGUnlocker.asi", &GetGood, &AsiModelName };
	modules.push_back(&asiModule);

	CHECK(VulkanProvider::Get() == nullptr); // searched less than 2 s ago
	now += 2100;
	const auto* found = VulkanProvider::Get();
	CHECK(found == &good.api);
	CHECK(VulkanProvider::Get() == found);

	{
		const int before = getProcCalls;
		for (int i = 0; i < 50; i++)
			VulkanProvider::Get();
		CHECK(getProcCalls == before); // found once, never searched again
	}

	{
		const auto info = VulkanProvider::Query();
		CHECK(info.state == State::Ready);
		CHECK(info.maxFrames == 5);
		CHECK(info.module == "C:\\game\\MFGUnlocker.asi");
		CHECK(info.reason.empty());
	}

	// Ready is cached, the menu asks every frame
	{
		const int before = asi.supportedCalls;
		for (int i = 0; i < 100; i++)
			VulkanProvider::Query();
		CHECK(asi.supportedCalls == before);
	}

	// ---- 4. Init
	CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) == NGX_SUCCESS);
	CHECK(VulkanProvider::initialized.load());
	CHECK(asi.device == deviceA && asi.instance == instance && asi.gipa == gipa && asi.gdpa == gdpa);

	CHECK(NVSDK_NGX_VULKAN_Init_Ext(1, L"", instance, physical, deviceA, 0, nullptr) == NGX_SUCCESS);
	CHECK(asi.gipa == nullptr && asi.gdpa == nullptr); // Init_Ext has none

	CHECK(NVSDK_NGX_VULKAN_Init_Ext2(1, L"", instance, physical, deviceA, gipa, gdpa, 0, nullptr) == NGX_SUCCESS);
	CHECK(asi.gipa == gipa && asi.gdpa == gdpa);

	CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceB, gipa, gdpa, nullptr, 0) == (NGX_FAIL | 2));

	// ---- 5. features
	{
		const uint32_t live = VulkanProvider::liveFeatures.load();
		handle = nullptr;
		CHECK(NVSDK_NGX_VULKAN_CreateFeature(cmdBuf, NGX_FEATURE_FRAME_GENERATION, &params, &handle) == NGX_SUCCESS);
		CHECK(handle != nullptr && IsVulkanHandle(handle));
		CHECK(asi.createDevice == nullptr && asi.createCmd == cmdBuf && asi.createParams == &params);
		CHECK(VulkanProvider::liveFeatures.load() == live + 1);

		NGXHandle* second = nullptr;
		CHECK(NVSDK_NGX_VULKAN_CreateFeature1(deviceA, cmdBuf, NGX_FEATURE_FRAME_GENERATION, &params, &second) ==
			NGX_SUCCESS);
		CHECK(second && second != handle && second->Id != handle->Id && IsVulkanHandle(second));
		CHECK(asi.createDevice == deviceA);

		// A handle that is the Vulkan one: evaluate passes the provider's own impl, not ours
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, handle, &params, nullptr) == NGX_SUCCESS);
		CHECK(asi.evalCmd == cmdBuf && asi.evalParams == &params);
		CHECK(asi.evalImpl != nullptr && asi.evalImpl != handle);

		void* impl = asi.evalImpl;

		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(nullptr, handle, &params, nullptr) == NGX_FAIL_INVALID_PARAMETER);
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, nullptr, &params, nullptr) == NGX_FAIL_INVALID_PARAMETER);
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, handle, nullptr, nullptr) == NGX_FAIL_INVALID_PARAMETER);

		// A D3D12 handle (no Vulkan bit) given to the Vulkan exports isn't touched
		NGXHandle d3d12{ 7 };
		const int evaluated = asi.evaluateCalls, released = asi.releaseCalls;
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, &d3d12, &params, nullptr) == NGX_FAIL_FEATURE_NOT_FOUND);
		CHECK(NVSDK_NGX_VULKAN_ReleaseFeature(&d3d12) == NGX_FAIL_FEATURE_NOT_FOUND);
		CHECK(asi.evaluateCalls == evaluated && asi.releaseCalls == released);

		// The model is logged once the asi names it
		CHECK(!Logged("XeFG model"));
		asi.model = "port";
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, handle, &params, nullptr) == NGX_SUCCESS);
		CHECK(Logged("Vulkan: XeFG model port"));
		const size_t lines = logLines.size();
		CHECK(NVSDK_NGX_VULKAN_EvaluateFeature(cmdBuf, handle, &params, nullptr) == NGX_SUCCESS);
		CHECK(logLines.size() == lines);
		CHECK(VulkanProvider::ModelName() == "port");

		CHECK(NVSDK_NGX_VULKAN_ReleaseFeature(handle) == NGX_SUCCESS);
		CHECK(asi.releasedImpl == impl);
		CHECK(NVSDK_NGX_VULKAN_ReleaseFeature(second) == NGX_SUCCESS);
		CHECK(VulkanProvider::liveFeatures.load() == live);
	}

	// Other features and a provider that fails or hands back nothing
	handle = reinterpret_cast<NGXHandle*>(0x1234);
	CHECK(NVSDK_NGX_VULKAN_CreateFeature(cmdBuf, 0, &params, &handle) == NGX_FAIL_FEATURE_NOT_SUPPORTED);
	CHECK(handle == nullptr);

	asi.createResult = NGX_FAIL | 2;
	CHECK(NVSDK_NGX_VULKAN_CreateFeature(cmdBuf, NGX_FEATURE_FRAME_GENERATION, &params, &handle) == (NGX_FAIL | 2));
	CHECK(handle == nullptr);
	asi.createResult = NGX_SUCCESS;

	asi.createNullImpl = true;
	CHECK(NVSDK_NGX_VULKAN_CreateFeature(cmdBuf, NGX_FEATURE_FRAME_GENERATION, &params, &handle) == NGX_FAIL);
	CHECK(handle == nullptr);
	asi.createNullImpl = false;

	// ---- 6. what the caller sees when it can run
	{
		FakeParams p;
		CHECK(NVSDK_NGX_VULKAN_PopulateParameters_Impl(&p) == NGX_SUCCESS);
		CHECK(p.ints["FrameGeneration.Available"] == 1);
		CHECK(p.ints["FrameGeneration.NeedsUpdatedDriver"] == 0);
		CHECK(p.ints["FrameGeneration.FeatureInitResult"] == 1);
		CHECK(p.ints["DLSSG.MultiFrameCountMax"] == 5);
		CHECK(p.ptrs["DLSSG.GetCurrentSettingsCallback"] != nullptr);
		CHECK(p.ptrs["DLSSG.EstimateVRAMCallback"] != nullptr);

		using Settings = NGXResult (*)(NGXHandle*, NGXParameter*);
		using Estimate = NGXResult (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
			uint32_t, size_t*);

		FakeParams q;
		NGXHandle h{ NGX_HANDLE_VULKAN | 3 };
		auto settings = reinterpret_cast<Settings>(p.ptrs["DLSSG.GetCurrentSettingsCallback"]);
		CHECK(settings(&h, &q) == NGX_SUCCESS);
		CHECK(q.ints["DLSSG.MustCallEval"] == 1 && q.ints["DLSSG.BurstCaptureRunning"] == 0);
		CHECK(settings(nullptr, &q) == NGX_FAIL_INVALID_PARAMETER);
		CHECK(settings(&h, nullptr) == NGX_FAIL_INVALID_PARAMETER);

		size_t vram = 0;
		auto estimate = reinterpret_cast<Estimate>(p.ptrs["DLSSG.EstimateVRAMCallback"]);
		CHECK(estimate(1, 2, 3, 4, 5, 6, 7, 8, 9, &vram) == NGX_SUCCESS);
		CHECK(vram == 0x28000000);
		CHECK(estimate(1, 2, 3, 4, 5, 6, 7, 8, 9, nullptr) == NGX_SUCCESS);

		NGXFeatureRequirement requirement;
		std::memset(&requirement, 0xCD, sizeof(requirement));
		CHECK(NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, nullptr, &requirement) == NGX_SUCCESS);
		CHECK(requirement.FeatureSupported == 0);
	}

	// ---- 7. the asi says it can't run (MFG-Unlocker's own gate): reason() is shown, Init refuses, nothing is offered
	asi.blocked = true;
	VulkanProvider::DeviceChanged();
	{
		const auto info = VulkanProvider::Query();
		CHECK(info.state == State::Blocked);
		CHECK(info.maxFrames == 0);
		CHECK(info.reason == "xefg_fake_block");
		CHECK(Logged("provider can't run: xefg_fake_block"));
	}

	{
		const int inits = asi.initCalls;
		CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) ==
			NGX_FAIL_FEATURE_NOT_SUPPORTED);
		CHECK(asi.initCalls == inits);

		FakeParams p;
		CHECK(NVSDK_NGX_VULKAN_PopulateParameters_Impl(&p) == NGX_SUCCESS);
		CHECK(p.ints["FrameGeneration.Available"] == 0);
		CHECK(p.ints["DLSSG.MultiFrameCountMax"] == 0);
		CHECK((uint32_t)p.ints["FrameGeneration.FeatureInitResult"] == 0xBAD00001u);
		CHECK(p.ptrs.empty());

		NGXFeatureRequirement requirement{};
		CHECK(NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, nullptr, &requirement) == NGX_SUCCESS);
		CHECK(requirement.FeatureSupported == 0x10);
	}

	// Blocked is asked again only now and then, the menu draws every frame
	{
		const int before = asi.supportedCalls;
		for (int i = 0; i < 100; i++)
			VulkanProvider::Query();
		CHECK(asi.supportedCalls == before);

		asi.blocked = false;
		CHECK(VulkanProvider::Query().state == State::Blocked); // still the answer from before
		now += 31000;
		const auto info = VulkanProvider::Query();
		CHECK(info.state == State::Ready && info.maxFrames == 5);
		CHECK(Logged("provider is ready"));
	}

	// Init asks again itself, so an earlier "blocked" can't outlive the asi becoming ready
	asi.blocked = true;
	VulkanProvider::DeviceChanged();
	CHECK(VulkanProvider::Query().state == State::Blocked);
	asi.blocked = false;
	CHECK(NVSDK_NGX_VULKAN_Init(1, L"", instance, physical, deviceA, gipa, gdpa, nullptr, 0) == NGX_SUCCESS);

	// ---- 8. shutdown reaches the provider and clears the flag
	{
		const int before = asi.shutdownCalls;
		CHECK(NVSDK_NGX_VULKAN_Shutdown1(deviceA) == NGX_SUCCESS);
		CHECK(asi.shutdownCalls == before + 1);
		CHECK(!VulkanProvider::initialized.load());
	}

	// ---- 9. the table check on its own
	{
		using namespace VulkanProvider;
		CHECK(Usable(&good.api) && HasReason(&good.api));
		CHECK(Usable(&noReason) && !HasReason(&noReason));
		CHECK(!Usable(&tooSmall) && !Usable(&wrongVersion) && !Usable(&noEvaluate) && !Usable(nullptr));
		static_assert(sizeof(FullTable) == 0x68);
		static_assert(sizeof(NGXFeatureRequirement) == 0x108);
	}

	std::printf("%d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}
