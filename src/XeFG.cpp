#include "XeFG.h"
#include "Log.h"
#include "Util.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace
{
	using PfnCreateContext = xefgd_result_t(*)(ID3D12Device*, xefgd_handle_t*);
	using PfnInit = xefgd_result_t(*)(xefgd_handle_t, const xefgd_init_params_t*);
	using PfnExecute = xefgd_result_t(*)(xefgd_handle_t, ID3D12GraphicsCommandList*, const xefgd_exec_params_t*);
	using PfnDestroy = xefgd_result_t(*)(xefgd_handle_t);
	using PfnSetLoggingCallback = xefgd_result_t(*)(xefgd_handle_t, uint32_t, xefgd_log_callback_t, void*);
	using PfnSettingsInstance = uint8_t * (*) ();

	struct SettingsLayout;

	struct Api
	{
		PfnCreateContext createContext = nullptr;
		PfnInit init = nullptr;
		PfnExecute execute = nullptr;
		PfnDestroy destroy = nullptr;
		PfnSetLoggingCallback setLoggingCallback = nullptr;
		PfnSettingsInstance settingsInstance = nullptr;

		HMODULE module = nullptr;

		// XefgSettings singleton
		uint8_t* settings = nullptr;
		const SettingsLayout* layout = nullptr;

		// 22-byte interpolation step calculation inside XeFG::Execute (for DLSSG.InterpolationTime)
		uint8_t* interpolationTimeSite = nullptr;
		bool interpolationTimePatched = false;
	} api;

	struct InterpolationTimeOverride
	{
		const void* outputBasesKey = nullptr; // matches [r14 + 0x238] (xefgd_exec_params_t::pOutputBases)
		float step = 0.0f;                    // interpolationTime / (firstFrameIndex + 1)
	};
	static_assert(offsetof(InterpolationTimeOverride, outputBasesKey) == 0);
	static_assert(offsetof(InterpolationTimeOverride, step) == 8);

	InterpolationTimeOverride gInterpTimeOverride{};
	std::mutex gInterpTimePatchMutex;

	bool PatchInterpolationTime()
	{
		if (api.interpolationTimePatched)
			return true;

		std::lock_guard lock(gInterpTimePatchMutex);
		if (api.interpolationTimePatched)
			return true;
		if (!api.interpolationTimeSite)
			return false;

		constexpr size_t kSiteBytes = 0x16; // 22 bytes

		// Trampoline replays the original 1.0f / (numInterpolatedFrames + 1) computation using an immediate 1.0f
		// (avoiding RIP-relative displacement), then overrides xmm7 if [r14 + 0x238] matches outputBasesKey.
		std::vector<uint8_t> tramp = {
			0xFF, 0xC1,                         // inc ecx
			0x0F, 0x57, 0xC0,                   // xorps xmm0, xmm0
			0xF3, 0x48, 0x0F, 0x2A, 0xC1,       // cvtsi2ss xmm0, rcx
			0xB8, 0x00, 0x00, 0x80, 0x3F,       // mov eax, 0x3f800000 (1.0f)
			0x66, 0x0F, 0x6E, 0xF8,             // movd xmm7, eax
			0xF3, 0x0F, 0x5E, 0xF8,             // divss xmm7, xmm0
			0x48, 0xB8,                         // movabs rax, &gInterpTimeOverride
		};
		const auto appendU64 = [](std::vector<uint8_t>& buf, uint64_t val) {
			const auto* bytes = reinterpret_cast<const uint8_t*>(&val);
			buf.insert(buf.end(), bytes, bytes + sizeof(val));
		};
		appendU64(tramp, reinterpret_cast<uint64_t>(&gInterpTimeOverride));
		const uint8_t tail[] = {
			0x49, 0x8B, 0x8E, 0x38, 0x02, 0x00, 0x00, // mov rcx, [r14 + 0x238] (params.pOutputBases)
			0x48, 0x3B, 0x08,                         // cmp rcx, [rax]
			0x75, 0x05,                               // jne +5
			0xF3, 0x0F, 0x10, 0x78, 0x08,             // movss xmm7, [rax + 8]
			0xFF, 0x25, 0x00, 0x00, 0x00, 0x00,       // jmp qword ptr [rip + 0]
		};
		tramp.insert(tramp.end(), std::begin(tail), std::end(tail));
		appendU64(tramp, reinterpret_cast<uint64_t>(api.interpolationTimeSite + kSiteBytes));

		void* mem = VirtualAlloc(nullptr, tramp.size(), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
		if (!mem)
			return false;
		memcpy(mem, tramp.data(), tramp.size());

		std::vector<uint8_t> detour = { 0xFF, 0x25, 0x00, 0x00, 0x00, 0x00 };
		appendU64(detour, reinterpret_cast<uint64_t>(mem));
		detour.resize(kSiteBytes, 0x90);

		DWORD oldProtect = 0;
		if (!VirtualProtect(api.interpolationTimeSite, kSiteBytes, PAGE_EXECUTE_READWRITE, &oldProtect))
		{
			VirtualFree(mem, 0, MEM_RELEASE);
			return false;
		}
		memcpy(api.interpolationTimeSite, detour.data(), detour.size());
		VirtualProtect(api.interpolationTimeSite, kSiteBytes, oldProtect, &oldProtect);
		FlushInstructionCache(GetCurrentProcess(), api.interpolationTimeSite, kSiteBytes);

		api.interpolationTimePatched = true;
		Log::Info("Patched XeFG's interpolation time");
		return true;
	}

	// Offsets into the XefgSettings singleton
	struct SettingsLayout
	{
		size_t allowModelOverride; // byte, the library never sets it itself
		size_t modelOverrideSet;   // byte
		size_t modelOverride;      // int32
		size_t sanityFloat;        // float, initialized to 0.7, the scene change threshold
		size_t debugLines;         // uint32, read by the output shaders' constants
		size_t maxFramesSet;       // byte
		size_t maxFrames;          // uint32, only used when above the default of 3
	};

	constexpr SettingsLayout Layout_1_3_1{ 77, 100, 104, 112, 120, 360, 364 };

	// A std::string was added at 64, so most fields moved by 32. The model override flag didn't: it's at 110, 109 is new
	constexpr SettingsLayout Layout_1_3_4{ 110, 132, 136, 144, 152, 392, 396 };

	constexpr const SettingsLayout* Layouts[] = { &Layout_1_3_4, &Layout_1_3_1 };

	struct KnownBuild
	{
		uint32_t timeDateStamp;
		uint32_t sizeOfImage;
		uint32_t createContext, init, execute, destroy, setLoggingCallback, settingsInstance;
		const SettingsLayout* layout;
	};

	const KnownBuild knownBuilds[] = {
		// libxess_fg.dll 1.3.1 from XeSS SDK 3.0.2
		{ 0x69CB0F4D, 0x015ED000, 0x1853B0, 0x1871C0, 0x185800, 0x1874C0, 0x188060, 0x1A4610, &Layout_1_3_1 },
		// libxess_fg.dll 1.3.4.95 (22,866,056 bytes, SHA-256 752b1491350a2983...); tools/CheckXeFgBuild.py checks a file
		// against this row, and against everything else this dll relies on
		{ 0x6A95ED0F, 0x015D7000, 0x1856A0, 0x1874B0, 0x185AF0, 0x1877B0, 0x188350, 0x1A4EA0, &Layout_1_3_4 },
	};

	// Builds other than the known ones are scanned for, only accepted when the struct layouts are likely unchanged
	constexpr uint16_t CompatibleMajor = 1;
	constexpr uint16_t CompatibleMinor = 3;

	struct Range
	{
		const uint8_t* begin;
		const uint8_t* end;
	};

	struct Image
	{
		const uint8_t* base;
		const IMAGE_NT_HEADERS64* nt;
		Range text;
		std::vector<Range> data; // readable, non executable sections
	};

	Image ParseImage(HMODULE module)
	{
		Image image{};
		image.base = reinterpret_cast<const uint8_t*>(module);
		image.nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(
			image.base + reinterpret_cast<const IMAGE_DOS_HEADER*>(image.base)->e_lfanew);

		auto section = IMAGE_FIRST_SECTION(image.nt);

		for (WORD i = 0; i < image.nt->FileHeader.NumberOfSections; i++, section++)
		{
			Range range{ image.base + section->VirtualAddress,
						  image.base + section->VirtualAddress + section->Misc.VirtualSize };

			if (section->Characteristics & IMAGE_SCN_MEM_EXECUTE)
			{
				if (!image.text.begin)
					image.text = range;
			}
			else if ((section->Characteristics & IMAGE_SCN_MEM_READ) &&
				(section->Characteristics & IMAGE_SCN_CNT_INITIALIZED_DATA))
			{
				image.data.push_back(range);
			}
		}

		return image;
	}

	// IDA style pattern, "48 8B ?? 18"
	std::vector<const uint8_t*> FindPattern(Range range, const char* pattern)
	{
		std::vector<int> bytes;

		for (const char* c = pattern; *c;)
		{
			if (*c == ' ')
			{
				c++;
			}
			else if (*c == '?')
			{
				bytes.push_back(-1);
				c += c[1] == '?' ? 2 : 1;
			}
			else
			{
				bytes.push_back(static_cast<int>(strtoul(c, nullptr, 16)));
				c += 2;
			}
		}

		std::vector<const uint8_t*> hits;

		for (const uint8_t* p = range.begin; p + bytes.size() <= range.end; p++)
		{
			size_t i = 0;
			while (i < bytes.size() && (bytes[i] < 0 || p[i] == bytes[i]))
				i++;

			if (i == bytes.size())
				hits.push_back(p);
		}

		return hits;
	}

	// Start of the function containing address, from the image's .pdata. Chained unwind entries (a function's cold
	// parts) are followed back to the primary one
	const uint8_t* FunctionStart(const uint8_t* address)
	{
		DWORD64 imageBase = 0;
		auto function = RtlLookupFunctionEntry(reinterpret_cast<DWORD64>(address), &imageBase, nullptr);

		while (function)
		{
			auto unwind = reinterpret_cast<const uint8_t*>(imageBase + function->UnwindData);

			if (((unwind[0] >> 3) & UNW_FLAG_CHAININFO) == 0)
				return reinterpret_cast<const uint8_t*>(imageBase + function->BeginAddress);

			uint8_t codeCount = unwind[2];
			function = reinterpret_cast<PRUNTIME_FUNCTION>(const_cast<uint8_t*>(unwind) + 4 + ((codeCount + 1) & ~1) * 2);
		}

		return nullptr;
	}

	// Functions with a RIP relative reference to the NUL terminated string text
	std::vector<const uint8_t*> FunctionsReferencing(const Image& image, const char* text)
	{
		std::vector<const uint8_t*> functions;
		const size_t size = strlen(text) + 1;

		for (const Range& range : image.data)
		{
			for (const uint8_t* string = range.begin; string + size <= range.end; string++)
			{
				if (memcmp(string, text, size) != 0)
					continue;

				// A disp32 that ends its instruction
				for (const uint8_t* p = image.text.begin; p + 4 <= image.text.end; p++)
				{
					int32_t displacement;
					memcpy(&displacement, p, 4);

					if (p + 4 + displacement != string)
						continue;

					const uint8_t* function = FunctionStart(p);

					if (function && std::ranges::find(functions, function) == functions.end())
						functions.push_back(function);
				}
			}
		}

		return functions;
	}

	const uint8_t* UniqueFunctionByPattern(const Image& image, const char* pattern)
	{
		auto hits = FindPattern(image.text, pattern);
		return hits.size() == 1 ? FunctionStart(hits[0]) : nullptr;
	}

	template <class T> T As(const uint8_t* address) { return reinterpret_cast<T>(const_cast<uint8_t*>(address)); }

	// The hidden C API lives in one translation unit, so functions that need disambiguation are picked relative to
	// CreateContext
	void Scan(const Image& image)
	{
		auto create = FunctionsReferencing(image, "phContext is null.");

		if (create.size() != 1)
			return;

		const uint8_t* anchor = create[0];
		api.createContext = As<PfnCreateContext>(anchor);

		auto nearAnchor = [anchor](std::vector<const uint8_t*> functions)
			{
				std::erase_if(functions, [anchor](const uint8_t* f) { return llabs(f - anchor) >= 0x10000; });
				return functions;
			};

		if (auto execute = nearAnchor(FunctionsReferencing(image, "pCommandList is null.")); execute.size() == 1)
			api.execute = As<PfnExecute>(execute[0]);

		// Referenced by xefgD3D12GetInitParams and xefgD3D12Init, the sources are alphabetical so Init comes last
		if (auto init = nearAnchor(FunctionsReferencing(image, "pInitParams is null.")); !init.empty())
			api.init = As<PfnInit>(*std::ranges::max_element(init));

		auto logging = nearAnchor(FunctionsReferencing(image, "Unsupported loggingLevel ({}) value."));
		if (!logging.empty())
		{
			api.setLoggingCallback = As<PfnSetLoggingCallback>(*std::ranges::min_element(
				logging, {}, [anchor](const uint8_t* f) { return llabs(f - anchor); }));
		}

		// vtable->shutdown(); vtable->deleting_dtor(1); free(handle, 0x10)
		api.destroy = As<PfnDestroy>(UniqueFunctionByPattern(
			image, "48 8B 41 18 48 8B CA FF 15 ?? ?? ?? ?? 48 8B CB E8 ?? ?? ?? ?? 48 8B C8 48 85 C0 74 ?? "
			"48 8B 10 48 8B 02 BA 01 00 00 00 FF 15 ?? ?? ?? ?? BA 10 00 00 00 48 8B CB E8"));

		// The settings getter is the magic static that runs XefgSettings' constructor, the only function referencing
		// "Inside libxess_fg.dll". It's called directly or through a jmp thunk:
		//   lea rbx, [singleton]; mov rcx, rbx; call constructor_or_thunk
		auto constructors = FunctionsReferencing(image, "Inside libxess_fg.dll");

		if (constructors.size() != 1)
			return;

		std::vector<const uint8_t*> targets{ constructors[0] };

		for (const uint8_t* p : FindPattern(image.text, "E9 ?? ?? ?? ??"))
		{
			int32_t relative;
			memcpy(&relative, p + 1, 4);

			if (p + 5 + relative == constructors[0])
				targets.push_back(p);
		}

		for (const uint8_t* p : FindPattern(image.text, "48 8D 1D ?? ?? ?? ?? 48 8B CB E8"))
		{
			int32_t relative;
			memcpy(&relative, p + 11, 4);

			if (std::ranges::find(targets, p + 15 + relative) != targets.end())
			{
				api.settingsInstance = As<PfnSettingsInstance>(FunctionStart(p));
				break;
			}
		}
	}

	// Fills library's version, false when there's none
	bool GetVersion(HMODULE module, XeFG::LibraryInfo& library)
	{
		struct Version
		{
			uint16_t major, minor, patch, reserved;
		} version{};

		using PfnGetVersion = int32_t(*)(Version*);
		auto getVersion = reinterpret_cast<PfnGetVersion>(GetProcAddress(module, "xefgSwapChainGetVersion"));

		if (!getVersion || getVersion(&version) != XEFGD_RESULT_SUCCESS)
		{
			Log::Warn("Couldn't get the XeFG version");
			return false;
		}

		Log::Info("XeFG version {}.{}.{}", version.major, version.minor, version.patch);
		library.major = version.major;
		library.minor = version.minor;
		library.patch = version.patch;
		return true;
	}

	XeFG::LibraryInfo library;
} // namespace

const XeFG::LibraryInfo& XeFG::Library() { return library; }

bool XeFG::Resolve(HMODULE module)
{
	api = {};
	api.module = module;
	library = { ModulePath(module) };

	const bool hasVersion = GetVersion(module, library);
	const Image image = ParseImage(module);
	const KnownBuild* known = nullptr;

	for (const KnownBuild& build : knownBuilds)
	{
		if (build.timeDateStamp == image.nt->FileHeader.TimeDateStamp &&
			build.sizeOfImage == image.nt->OptionalHeader.SizeOfImage)
		{
			known = &build;
		}
	}

	if (known)
	{
		api.createContext = As<PfnCreateContext>(image.base + known->createContext);
		api.init = As<PfnInit>(image.base + known->init);
		api.execute = As<PfnExecute>(image.base + known->execute);
		api.destroy = As<PfnDestroy>(image.base + known->destroy);
		api.setLoggingCallback = As<PfnSetLoggingCallback>(image.base + known->setLoggingCallback);
		api.settingsInstance = As<PfnSettingsInstance>(image.base + known->settingsInstance);
		api.layout = known->layout;
		library.knownBuild = true;
	}
	else
	{
		Log::Info("Unknown libxess_fg.dll build {:08X}, scanning for the API", image.nt->FileHeader.TimeDateStamp);

		if (!hasVersion || library.major != CompatibleMajor || library.minor != CompatibleMinor)
			return false;

		Scan(image);
	}

	// Without the settings there's no multi frame generation, the default model has no multi frame variant.
	// A scanned build gets the first layout whose scene change threshold is where it should be
	if (api.settingsInstance)
	{
		uint8_t* settings = api.settingsInstance();

		for (const SettingsLayout* layout : Layouts)
		{
			float sanity;
			memcpy(&sanity, settings + layout->sanityFloat, sizeof(sanity));

			if ((!api.layout || api.layout == layout) && sanity == 0.7f)
			{
				api.settings = settings;
				api.layout = layout;
				break;
			}
		}
	}

	// Locate the 22-byte interpolation step calculation inside XeFG::Execute for DLSSG.InterpolationTime
	// (OptiScaler FlipMeter TargetFps + VRR fractional timestamps).
	if (auto timeHits = FindPattern(image.text,
			"41 8B 8E 40 02 00 00 44 8B C1 89 4C 24 ?? 41 8B 86 64 02 00 00 85 C0 74 ?? "
			"41 8B 96 60 02 00 00 89 54 24 ?? 44 8D 04 10 44 89 44 24 ?? "
			"FF C1 0F 57 C0 F3 48 0F 2A C1 F3 0F 10 3D ?? ?? ?? ?? F3 0F 5E F8");
		timeHits.size() == 1)
	{
		api.interpolationTimeSite = const_cast<uint8_t*>(timeHits[0] + 0x2d);
	}

	auto rva = [&image](const void* function)
		{ return function ? static_cast<uint32_t>(static_cast<const uint8_t*>(function) - image.base) : 0; };

	Log::Info("XeFG API ({}): create {:X}, init {:X}, execute {:X}, destroy {:X}, logging {:X}, settings {:X}, time {:X}",
		known ? "known build" : "scanned", rva(api.createContext), rva(api.init), rva(api.execute),
		rva(api.destroy), rva(api.setLoggingCallback), rva(api.settingsInstance), rva(api.interpolationTimeSite));

	return api.createContext && api.init && api.execute && api.destroy && api.settings;
}

void XeFG::SetMaxInterpolatedFrames(uint32_t maxFrames)
{
	api.settings[api.layout->maxFramesSet] = 1;
	memcpy(api.settings + api.layout->maxFrames, &maxFrames, sizeof(maxFrames));
}

void XeFG::OverrideModel(int32_t modelId)
{
	api.settings[api.layout->allowModelOverride] = 1;
	api.settings[api.layout->modelOverrideSet] = 1;
	memcpy(api.settings + api.layout->modelOverride, &modelId, sizeof(modelId));
}

int32_t XeFG::AutoModel(ID3D12Device* device)
{
	// XefgContext::model, the same in 1.3.1 and 1.3.4. CreateContext only stores its pick once the device passed
	// XeFG's checks, one of them failing Intel GPUs with driver extensions but without XMX
	constexpr size_t ContextModel = 76;

	api.settings[api.layout->modelOverrideSet] = 0;

	xefgd_handle_t probe = nullptr;
	api.createContext(device, &probe);

	if (!probe)
		return 0;

	const uint8_t* context = *reinterpret_cast<uint8_t* const*>(reinterpret_cast<const uint8_t*>(probe) + 8);
	int32_t model;
	memcpy(&model, context + ContextModel, sizeof(model));

	api.destroy(probe);
	return model;
}

void XeFG::SetDebugLines(bool enable)
{
	uint32_t value = enable ? 1 : 0;
	memcpy(api.settings + api.layout->debugLines, &value, sizeof(value));
}

xefgd_result_t XeFG::CreateContext(ID3D12Device* device, xefgd_handle_t* context)
{
	return api.createContext(device, context);
}

void XeFG::SetLoggingCallback(xefgd_handle_t context, xefgd_logging_level_t level, xefgd_log_callback_t callback)
{
	if (api.setLoggingCallback)
		api.setLoggingCallback(context, level, callback, nullptr);
}

xefgd_result_t XeFG::Init(xefgd_handle_t context, const xefgd_init_params_t& params)
{
	return api.init(context, &params);
}

bool XeFG::SupportsInterpolationTime() { return api.interpolationTimeSite != nullptr; }

xefgd_result_t XeFG::Execute(xefgd_handle_t context, ID3D12GraphicsCommandList* cmdList,
	const xefgd_exec_params_t& params, float interpolationTime)
{
	if (interpolationTime > 0.0f && interpolationTime < 1.0f && PatchInterpolationTime())
	{
		gInterpTimeOverride.outputBasesKey = params.pOutputBases;
		gInterpTimeOverride.step = interpolationTime / static_cast<float>(params.firstFrameIndex + 1);
		const auto result = api.execute(context, cmdList, &params);
		gInterpTimeOverride.outputBasesKey = nullptr;
		return result;
	}
	return api.execute(context, cmdList, &params);
}

xefgd_result_t XeFG::Destroy(xefgd_handle_t context) { return api.destroy(context); }

HMODULE XeFG::Module() { return api.module; }
