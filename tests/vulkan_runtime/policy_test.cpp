// Checks src/vulkan/VulkanPolicy.h, the decisions of the Vulkan runtime that need neither Vulkan nor Windows: when a batch
// of generated frames starts, which outputs an evaluation records, how the game's formats and camera map to what XeFG
// reads, and what each reason code means. The values it expects are written out here from MFG-Unlocker's own policy
// (xess/ngx_provider_policy.h and xess/standalone_policy.h) and from the formats' registry values, not computed from the
// header's code.
#include "vulkan/VulkanPolicy.h"

#include <cmath>
#include "XeFgEnabler.h"
#include "HwFlipQueue.h"
#include "DredAttribution.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <string>
#include <tuple>

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

static bool Near(float a, float b, float tolerance = 1e-5f)
{
	return std::fabs(a - b) <= tolerance * std::fmax(1.0f, std::fabs(b));
}

using VulkanPolicy::OutputRange;

int main()
{
	// Color copies must not reinterpret BGRA as RGBA or packed HDR as FP16.
	CHECK(VulkanPolicy::CopyCompatible(37, 43));
	CHECK(VulkanPolicy::CopyCompatible(50, 44));
	CHECK(VulkanPolicy::CopyCompatible(97, 97));
	CHECK(!VulkanPolicy::CopyCompatible(37, 44));
	CHECK(!VulkanPolicy::CopyCompatible(64, 97));
	CHECK(!VulkanPolicy::CopyCompatible(0, 0));
	CHECK(!VulkanPolicy::CopyCompatible(126, 126));
	CHECK(VulkanPolicy::SampledAspect(1));
	CHECK(VulkanPolicy::SampledAspect(2));
	CHECK(!VulkanPolicy::SampledAspect(0));
	CHECK(!VulkanPolicy::SampledAspect(4));
	CHECK(!VulkanPolicy::SampledAspect(6));
	// ---- batches
	CHECK(VulkanPolicy::MaxGenerated == 5);

	// index 1 always starts one, so does the very first call whatever its index
	CHECK(VulkanPolicy::NewBatch(1, 0, 5, 5, 10));
	CHECK(VulkanPolicy::NewBatch(1, 5, 5, 5, 10));
	CHECK(VulkanPolicy::NewBatch(3, 0, 5, 0, 0));
	CHECK(VulkanPolicy::NewBatch(2, 1, 1, 1, 0));
	// the next index of the same batch doesn't
	CHECK(!VulkanPolicy::NewBatch(2, 1, 5, 5, 10));
	CHECK(!VulkanPolicy::NewBatch(5, 4, 5, 5, 1));
	CHECK(!VulkanPolicy::NewBatch(3, 1, 5, 5, 1)); // a skipped index within the batch is still its batch
	// an index that doesn't go on from the last one, or another count, does
	CHECK(VulkanPolicy::NewBatch(2, 2, 5, 5, 10));
	CHECK(VulkanPolicy::NewBatch(2, 4, 5, 5, 10));
	CHECK(VulkanPolicy::NewBatch(2, 1, 3, 5, 10));
	CHECK(VulkanPolicy::NewBatch(2, 1, 5, 3, 10));

	// ---- which outputs an evaluation records
	const OutputRange none{};
	CHECK(VulkanPolicy::ExecuteRange(true, 0, 5, false) == none);
	CHECK(VulkanPolicy::ExecuteRange(true, 6, 5, false) == none);
	CHECK(VulkanPolicy::ExecuteRange(false, 0, 5, false) == none);
	CHECK(VulkanPolicy::ExecuteRange(true, 1, 0, false) == none);

	// the batch's first evaluation records the pair's shared work with output 0 and every output up to its own index
	CHECK((VulkanPolicy::ExecuteRange(true, 1, 5, false) == OutputRange{ true, 0, 1, 1 }));
	CHECK((VulkanPolicy::ExecuteRange(true, 3, 5, false) == OutputRange{ true, 0, 3, 3 })); // a skipped index 1
	CHECK((VulkanPolicy::ExecuteRange(true, 5, 5, false) == OutputRange{ true, 0, 5, 5 }));
	// each later one only its own
	CHECK((VulkanPolicy::ExecuteRange(false, 2, 5, false) == OutputRange{ true, 1, 1, 2 }));
	CHECK((VulkanPolicy::ExecuteRange(false, 5, 5, false) == OutputRange{ true, 4, 1, 5 }));
	// the whole batch at once
	CHECK((VulkanPolicy::ExecuteRange(true, 1, 5, true) == OutputRange{ true, 0, 0, 5 }));
	CHECK((VulkanPolicy::ExecuteRange(true, 4, 5, true) == OutputRange{ true, 0, 0, 5 }));
	CHECK(VulkanPolicy::ExecuteRange(false, 2, 5, true) == none);

	// every output of a 6x batch is recorded exactly once, in order, whichever index the batch starts at
	for (uint32_t count = 1; count <= VulkanPolicy::MaxGenerated; ++count)
	{
		for (uint32_t first = 1; first <= count; ++first)
		{
			uint32_t through = 0;
			bool ok = true;

			for (uint32_t index = first; index <= count; ++index)
			{
				const auto range = VulkanPolicy::ExecuteRange(index == first, index, count, false);
				ok &= range.record && range.through == index;
				// the part of the batch recorded so far plus this call's frames is what is ready
				ok &= range.first + range.count == index;
				ok &= range.first == (index == first ? 0u : index - 1);
				ok &= range.first <= through;
				through = range.through;
			}

			CHECK(ok && through == count);
		}
	}

	// ---- DLSSG.MultiFrameCount and DLSSG.MultiFrameIndex
	CHECK(VulkanPolicy::ClampCount(0) == 1);
	CHECK(VulkanPolicy::ClampCount(1) == 1);
	CHECK(VulkanPolicy::ClampCount(3) == 3);
	CHECK(VulkanPolicy::ClampCount(5) == 5);
	CHECK(VulkanPolicy::ClampCount(6) == 5);
	CHECK(VulkanPolicy::ClampCount(0xFFFFFFFFu) == 5);
	CHECK(VulkanPolicy::ClampIndex(0, 5) == 1);
	CHECK(VulkanPolicy::ClampIndex(3, 5) == 3);
	CHECK(VulkanPolicy::ClampIndex(7, 5) == 5);
	CHECK(VulkanPolicy::ClampIndex(2, 1) == 1);

	// ---- the motion vector scale: the game's (a pixel scale of its extent) carried to the size they are resampled to
	{
		auto [x, y] = VulkanPolicy::MotionScale(1.0f, 1.0f, 1920, 1080, 1920, 1080);
		CHECK(Near(x, 1.0f) && Near(y, 1.0f));
		std::tie(x, y) = VulkanPolicy::MotionScale(1.0f, 1.0f, 960, 540, 1920, 1080);
		CHECK(Near(x, 2.0f) && Near(y, 2.0f));
		std::tie(x, y) = VulkanPolicy::MotionScale(1920.0f, 1080.0f, 1920, 1080, 1280, 720);
		CHECK(Near(x, 1280.0f) && Near(y, 720.0f));
		std::tie(x, y) = VulkanPolicy::MotionScale(-1.0f, 0.5f, 100, 200, 300, 100);
		CHECK(Near(x, -3.0f) && Near(y, 0.25f));
		// an empty extent keeps the scale
		std::tie(x, y) = VulkanPolicy::MotionScale(2.0f, 3.0f, 0, 0, 1920, 1080);
		CHECK(Near(x, 2.0f) && Near(y, 3.0f));
		std::tie(x, y) = VulkanPolicy::MotionScale(2.0f, 3.0f, 0, 10, 1920, 20);
		CHECK(Near(x, 2.0f) && Near(y, 6.0f));
	}

	// ---- the projection: left handed, row major
	{
		float m[16];
		const float half = std::tan(0.5f * 1.0471976f);

		// nothing known: 60 degrees, near 0.1, far 10000, 16:9
		VulkanPolicy::Projection(16.0f / 9.0f, 0, 0, 0, false, m);
		CHECK(Near(m[5], 1.0f / half));
		CHECK(Near(m[0], m[5] / (16.0f / 9.0f)));
		CHECK(Near(m[10], 10000.0f / (10000.0f - 0.1f)));
		CHECK(Near(m[14], -0.1f * m[10]));
		CHECK(m[11] == 1.0f && m[15] == 0.0f);
		for (int i : { 1, 2, 3, 4, 6, 7, 8, 9, 12, 13 })
			CHECK(m[i] == 0.0f);

		// given: 90 degrees, near 1, far 101
		VulkanPolicy::Projection(2.0f, 1.0f, 101.0f, 1.5707964f, false, m);
		CHECK(Near(m[5], 1.0f, 1e-4f) && Near(m[0], 0.5f, 1e-4f));
		CHECK(Near(m[10], 1.01f, 1e-4f) && Near(m[14], -1.01f, 1e-4f));

		// inverted depth swaps them
		VulkanPolicy::Projection(2.0f, 1.0f, 101.0f, 1.5707964f, true, m);
		CHECK(Near(m[10], -0.01f, 1e-4f));
		CHECK(Near(m[14], -101.0f * m[10], 1e-3f));

		// an infinite far plane is taken to its limit, so is an infinite near plane (inverted)
		const float infinity = std::numeric_limits<float>::infinity();
		VulkanPolicy::Projection(1.0f, 0.5f, infinity, 1.0f, false, m);
		CHECK(Near(m[10], 1.0f) && Near(m[14], -0.5f));
		VulkanPolicy::Projection(1.0f, 0.5f, infinity, 1.0f, true, m);
		CHECK(Near(m[10], 0.0f) && Near(m[14], 0.5f));

		// out of range values are not taken
		VulkanPolicy::Projection(1.0f, 1.0f, 1.0f, 1.0f, false, m); // near == far
		CHECK(Near(m[10], 10000.0f / (10000.0f - 0.1f)));
		VulkanPolicy::Projection(1.0f, 1.0f, 100.0f, 3.2f, false, m); // field of view over 180 degrees
		CHECK(Near(m[5], 1.0f / half));
		VulkanPolicy::Projection(0.0f, 0, 0, 0, false, m); // no aspect ratio
		CHECK(Near(m[0], m[5]));
	}

	// ---- formats: the registry values the header spells out
	namespace Vk = VulkanPolicy::Vk;
	namespace Dxgi = VulkanPolicy::Dxgi;
	CHECK(Vk::R8G8B8A8Unorm == 37 && Vk::R8G8B8A8Srgb == 43 && Vk::B8G8R8A8Unorm == 44 && Vk::B8G8R8A8Srgb == 50);
	CHECK(Vk::A2B10G10R10UnormPack32 == 64 && Vk::R16G16Sfloat == 83 && Vk::R16G16B16A16Sfloat == 97);
	CHECK(Vk::R32G32Sfloat == 103 && Vk::R32G32B32A32Sfloat == 109 && Vk::B10G11R11UfloatPack32 == 122);
	CHECK(Dxgi::Unknown == 0 && Dxgi::R16G16B16A16Float == 10 && Dxgi::R32G32Float == 16 && Dxgi::R10G10B10A2Unorm == 24);
	CHECK(Dxgi::R11G11B10Float == 26 && Dxgi::R8G8B8A8Unorm == 28 && Dxgi::R16G16Float == 34 && Dxgi::R32Float == 41);
	CHECK(Dxgi::B8G8R8A8Unorm == 87);

	// the colors XeFG takes as a back buffer, an sRGB one as its UNORM twin
	struct Row
	{
		uint32_t vk;
		uint32_t dxgi;
		bool srgb;
	};

	for (const Row& row : { Row{ 37, 28, false }, Row{ 43, 28, true }, Row{ 44, 87, false }, Row{ 50, 87, true },
		     Row{ 64, 24, false }, Row{ 97, 10, false }, Row{ 122, 26, false } })
	{
		const auto color = VulkanPolicy::ColorFormat(row.vk);
		CHECK(color.dxgi == row.dxgi && color.srgb == row.srgb);
	}

	// every other format is one XeFG's swap chain doesn't take
	{
		std::set<uint32_t> taken{ 37, 43, 44, 50, 64, 97, 122 };
		bool ok = true;

		for (uint32_t format = 0; format < 300; ++format)
		{
			if (!taken.count(format))
				ok &= VulkanPolicy::ColorFormat(format).dxgi == Dxgi::Unknown && !VulkanPolicy::ColorFormat(format).srgb;
		}

		CHECK(ok);
	}

	// the motion vectors: two 32-bit floats when the game's are, else two 16-bit ones, each with its copy shader
	CHECK(VulkanPolicy::MotionFormat(Vk::R32G32Sfloat) == Dxgi::R32G32Float);
	CHECK(VulkanPolicy::MotionFormat(Vk::R32G32B32A32Sfloat) == Dxgi::R32G32Float);
	CHECK(VulkanPolicy::MotionFormat(Vk::R16G16Sfloat) == Dxgi::R16G16Float);
	CHECK(VulkanPolicy::MotionFormat(Vk::R16G16B16A16Sfloat) == Dxgi::R16G16Float);
	CHECK(VulkanPolicy::MotionFormat(0) == Dxgi::R16G16Float);
	CHECK(VulkanPolicy::MotionCopyShader(Dxgi::R16G16Float) == 1);
	CHECK(VulkanPolicy::MotionCopyShader(Dxgi::R32G32Float) == 2);

	// ---- the reasons
	{
		const char* codes[] = { "not_initialized", "pending", "hooks_off", "hooks_failed", "no_hooks", "no_loader",
			"device_not_prepared", "instance_api_version", "instance_unknown", "instance_functions", "device_api_version",
			"subgroup_operations", "device_features", "push_descriptor", "no_graphics_family", "intel_adapter",
			"device_functions", "prepare_exception", "no_xefg", "no_vulkan_models", "xvk_device", "copy_pipelines" };
		std::set<std::string> texts;

		for (const char* code : codes)
		{
			const auto text = VulkanPolicy::Describe(code);
			CHECK(!text.empty() && text != code);
			CHECK(text.find(' ') != std::string::npos); // a sentence, not a code
			CHECK(text.front() != ' ' && text.back() != ' ' && text.back() != '.');
			CHECK(texts.insert(text).second); // every code has its own text
		}

		CHECK(VulkanPolicy::Describe("xefg_resolve") == "XeFG's internal entry points couldn't be used (resolve)");
		CHECK(VulkanPolicy::Describe("something_else") == "something_else");
		CHECK(VulkanPolicy::Describe("") == "");
		CHECK(VulkanPolicy::Describe("xefg").empty() == false && VulkanPolicy::Describe("xefg") == "xefg");
	}

	// ---- the tensor-core route: its switches, its layer mask, when it replaces the layers, what its reasons mean
	{
		using VulkanPolicy::IsOffWord;
		using VulkanPolicy::IsOnWord;

		// XEFG_TENSOR_CORES: 0, off and false turn it off in any case, nothing else does
		for (const wchar_t* word : { L"0", L"off", L"OFF", L"Off", L"false", L"FALSE", L"fAlSe" })
			CHECK(IsOffWord(word) && !IsOnWord(word));

		for (const wchar_t* word : { L"", L"1", L"on", L"true", L"no", L"disabled", L"of", L"offf", L" off", L"off ", L"00",
			     L"-0", L"fals" })
			CHECK(!IsOffWord(word));

		// XEFG_TENSOR_CORES_VULKAN_BARRIERS: 1, on and true turn it on
		for (const wchar_t* word : { L"1", L"on", L"ON", L"On", L"true", L"TRUE", L"tRuE" })
			CHECK(IsOnWord(word) && !IsOffWord(word));

		for (const wchar_t* word : { L"", L"0", L"off", L"yes", L"y", L"enabled", L"11", L" 1", L"1 ", L"tru", L"truee" })
			CHECK(!IsOnWord(word));

		// XEFG_TENSOR_CORES_MASK: hexadecimal, a bit per layer of the table of 17, anything else is every layer
		constexpr uint32_t all = (1u << 17) - 1;
		using VulkanPolicy::ParseLayerMask;

		CHECK(all == 0x1FFFF);
		CHECK(ParseLayerMask(L"", all) == all);
		CHECK(ParseLayerMask(L"   ", all) == all);
		CHECK(ParseLayerMask(L"1ffff", all) == all);
		CHECK(ParseLayerMask(L"1FFFF", all) == all);
		CHECK(ParseLayerMask(L"0x1ffff", all) == all);
		CHECK(ParseLayerMask(L"0X1FfFf", all) == all);
		CHECK(ParseLayerMask(L"0", all) == 0); // no layer at all, the route stays installed and launches nothing
		CHECK(ParseLayerMask(L"0x0", all) == 0);
		CHECK(ParseLayerMask(L"1", all) == 1);
		CHECK(ParseLayerMask(L"3", all) == 3);
		CHECK(ParseLayerMask(L"a", all) == 10);
		CHECK(ParseLayerMask(L"0x10", all) == 0x10);
		CHECK(ParseLayerMask(L"00000007", all) == 7);
		CHECK(ParseLayerMask(L"  0x10\t", all) == 0x10);
		CHECK(ParseLayerMask(L"\t1000 ", all) == 0x1000);
		CHECK(ParseLayerMask(L"10000", all) == 0x10000); // the 17th layer is the last
		CHECK(ParseLayerMask(L"20000", all) == all);     // a bit past the table
		CHECK(ParseLayerMask(L"ffffffff", all) == all);
		CHECK(ParseLayerMask(L"123456789", all) == all); // longer than 8 digits, even with leading zeros it is too long
		CHECK(ParseLayerMask(L"000000001", all) == all);
		CHECK(ParseLayerMask(L"0x", all) == all);
		CHECK(ParseLayerMask(L"x1", all) == all);
		CHECK(ParseLayerMask(L"zz", all) == all);
		CHECK(ParseLayerMask(L"12 34", all) == all);
		CHECK(ParseLayerMask(L"-1", all) == all);
		CHECK(ParseLayerMask(L"+1", all) == all);
		CHECK(ParseLayerMask(L"1.0", all) == all);
		CHECK(ParseLayerMask(L"0b11", all) == 0xB11); // b is a hexadecimal digit, not a base prefix
		CHECK(ParseLayerMask(L"ff", 0x7F) == 0x7F);    // `all` is the table's size, not 17 bits
		CHECK(ParseLayerMask(L"80", 0x7F) == 0x7F);
		CHECK(ParseLayerMask(L"7f", 0x7F) == 0x7F);
		CHECK(ParseLayerMask(L"7e", 0x7F) == 0x7E);

		// which network paths put the ported model's layers on the tensor cores: the CUDA path, and XMX (native) which is
		// that on a GPU without XeFG's own XMX network, as every Vulkan GPU is. Not XMX Port, not DP4a's own model
		using VulkanPolicy::UseTensorCores;
		CHECK(UseTensorCores(true, true, false));   // CUDA
		CHECK(UseTensorCores(true, false, true));   // XMX (native)
		CHECK(UseTensorCores(true, true, true));
		CHECK(!UseTensorCores(true, false, false)); // XMX Port
		CHECK(!UseTensorCores(false, true, false)); // XeFG's own DP4a model has none of the layers
		CHECK(!UseTensorCores(false, false, true));
		CHECK(!UseTensorCores(false, true, true));
		CHECK(!UseTensorCores(false, false, false));

		// the reasons, the codes of MFG-Unlocker's xvk_tensor.cpp and the ones this dll gives
		const char* codes[] = { "environment_off", "no_port", "no_fatbin", "unvalidated_v0_combination", "already_attempted",
			"no_extensions", "device_inputs", "instance_functions", "not_nvidia", "adapter_luid", "compute_capability_unknown",
			"architecture_unsupported", "texturing_spec_old", "device_functions", "kernel_missing", "module_bytes",
			"module_create", "module_create_fault", "function_create", "function_create_fault", "exception", "driver_fault" };
		std::set<std::string> texts;

		for (const char* code : codes)
		{
			const auto text = VulkanPolicy::DescribeTensor(code);
			CHECK(!text.empty() && text != code);
			CHECK(text.find(' ') != std::string::npos); // a sentence, not a code
			CHECK(text.front() != ' ' && text.back() != ' ' && text.back() != '.');
			CHECK(texts.insert(text).second); // every code has its own text
		}

		CHECK(VulkanPolicy::DescribeTensor("environment_off") == "XEFG_TENSOR_CORES is off in the environment");
		CHECK(VulkanPolicy::DescribeTensor("not_nvidia") == "the GPU isn't an NVIDIA one");
		CHECK(VulkanPolicy::DescribeTensor("not_attempted") == "not_attempted"); // no device yet: nothing to say
		CHECK(VulkanPolicy::DescribeTensor("something_else") == "something_else");
		CHECK(VulkanPolicy::DescribeTensor("") == "");
		CHECK(VulkanPolicy::DescribeTensor("dispatch_shape") == "dispatch_shape"); // a layer's, not the route's
	}

	// ---- the VK_KHR_cooperative_matrix route: when it replaces the layers and what its reasons mean
	{
		using VulkanPolicy::UseCoopMatrix;
		using VulkanPolicy::IsAmdRdna4Device;
		CHECK(UseCoopMatrix(true, true, false, false));         // XMX Port
		CHECK(UseCoopMatrix(true, false, true, false));         // XMX (native)
		CHECK(UseCoopMatrix(true, false, false, true));         // CUDA fallback on AMD/Intel
		CHECK(UseCoopMatrix(true, false, false, false, true));  // WMMA model path
		CHECK(!UseCoopMatrix(true, false, false, false, false));
		CHECK(!UseCoopMatrix(false, false, false, false, true));// DP4a model ignores WMMA
		CHECK(!UseCoopMatrix(false, true, false, false));       // DP4a model
		CHECK(IsAmdRdna4Device(0x1002, 0x7550, "AMD Radeon RX 9070 XT"));
		CHECK(!IsAmdRdna4Device(0x1002, 0x744C, "AMD Radeon RX 7900 XTX"));
		CHECK(!IsAmdRdna4Device(0x10DE, 0x7550, "NVIDIA GeForce RTX 5060"));

		const char* coopCodes[] = { "environment_off", "no_port", "already_attempted", "no_extensions", "device_inputs",
			"instance_functions", "device_functions", "no_int8_tile", "layout_create", "shader_dir_missing", "exception",
			"driver_fault" };
		std::set<std::string> coopTexts;

		for (const char* code : coopCodes)
		{
			const auto text = VulkanPolicy::DescribeCoopMatrix(code);
			CHECK(!text.empty() && text != code);
			CHECK(text.find(' ') != std::string::npos);
			CHECK(text.front() != ' ' && text.back() != ' ' && text.back() != '.');
			CHECK(coopTexts.insert(text).second);
		}
	}


	// ---- recorded-mode Vulkan PresentPacer: burst smoothing, V-Sync FIFO bypass, late-frame no-debt, pause reset
	{
		VulkanPolicy::PresentPacer pacer;
		constexpr uint64_t ms = 1'000'000ull;

		// Before two batches establish cadence, never stall
		pacer.OnBatchStart(100 * ms, 3);
		CHECK(pacer.ComputePresentWaitNs(101 * ms, false) == 0);

		// Second batch arrives 20 ms later (50 base FPS, 4x FG -> 200 FPS, target step = 20ms * 0.92 / 4 = 4.60 ms)
		pacer.OnBatchStart(120 * ms, 3);
		// G1 arrives on time at 120 ms (19 ms since last present at 101 ms -> no wait)
		CHECK(pacer.ComputePresentWaitNs(120 * ms, false) == 0);
		// G2 bursts 0.2 ms later -> should wait ~4.40 ms
		const uint64_t waitG2 = pacer.ComputePresentWaitNs(120 * ms + 200'000ull, false);
		CHECK(waitG2 == 4'400'000ull);
		// G3 bursts 0.2 ms after G2's paced time -> should wait ~4.40 ms
		const uint64_t waitG3 = pacer.ComputePresentWaitNs(124'600'000ull + 200'000ull, false);
		CHECK(waitG3 == 4'400'000ull);
		// Real frame arrives late (10 ms after G3) -> must NOT wait or accumulate debt
		CHECK(pacer.ComputePresentWaitNs(140 * ms, false) == 0);

		// RDR2 Vulkan regression: steady 50 FPS (20 ms) 3x FG hit by a 36 ms hitch followed by a 10 ms catch-up frame
		// must NOT inflate emaBatchNs or cascade waits into subsequent batches.
		VulkanPolicy::PresentPacer rdr2;
		uint64_t t = 1000 * ms;
		for (int i = 0; i < 6; ++i)
		{
			rdr2.OnBatchStart(t, 2);
			(void)rdr2.ComputePresentWaitNs(t, false);
			(void)rdr2.ComputePresentWaitNs(t + 6 * ms, false);
			(void)rdr2.ComputePresentWaitNs(t + 12 * ms, false);
			t += 20 * ms;
		}
		const uint64_t steadyStep = rdr2.targetStepNs;
		CHECK(steadyStep == (20 * ms * 92ull) / (3ull * 100ull));

		// Single 36 ms asset-streaming hitch -> drains immediately (0 wait) and preserves steady cadence
		t += 16 * ms; // total delta = 36 ms
		rdr2.OnBatchStart(t, 2);
		CHECK(rdr2.ComputePresentWaitNs(t, false) == 0);
		CHECK(rdr2.ComputePresentWaitNs(t + 200'000ull, false) == 0);
		CHECK(rdr2.ComputePresentWaitNs(t + 400'000ull, false) == 0);
		CHECK(rdr2.targetStepNs == steadyStep);

		// Catch-up frame arrives 10 ms later -> G1 must NOT block behind stale debt, and step stays bounded <= steadyStep
		t += 10 * ms;
		rdr2.OnBatchStart(t, 2);
		CHECK(rdr2.targetStepNs <= steadyStep);
		CHECK(rdr2.ComputePresentWaitNs(t, false) == 0);

		// V-Sync FIFO active -> always returns 0
		pacer.OnBatchStart(140 * ms, 3);
		CHECK(pacer.ComputePresentWaitNs(140 * ms + 100'000ull, true) == 0);

		// Pause disengages pacer immediately
		pacer.OnPaused();
		CHECK(pacer.ComputePresentWaitNs(141 * ms, false) == 0);

		// Inactivity beyond 250 ms auto-disengages
		pacer.OnBatchStart(200 * ms, 1);
		pacer.OnBatchStart(220 * ms, 1);
		CHECK(pacer.ComputePresentWaitNs(500 * ms, false) == 0);

		// Adaptive GPU-Bound Burst Release hysteresis (f5low-kit DriverPacing=2 / FlipBurst=1):
		// Sustained queue pressure for >= 2s engages gpuBoundBurst; smooth cadence for >= 2s releases it.
		VulkanPolicy::PresentPacer burst{};
		uint64_t bt = 1'000'000'000ull;
		for (int i = 0; i < 4; ++i)
		{
			burst.OnBatchStart(bt, 2);
			burst.ComputePresentWaitNs(bt, false);
			burst.ComputePresentWaitNs(bt + 6 * ms, false);
			burst.ComputePresentWaitNs(bt + 12 * ms, false);
			bt += 20 * ms;
		}
		CHECK(!burst.gpuBoundBurst);

		// Sustain incomplete batches (queue pressure) for > 2 seconds (110 batches * 20 ms = 2.2 s)
		for (int i = 0; i < 110; ++i)
		{
			burst.OnBatchStart(bt, 2);
			burst.ComputePresentWaitNs(bt, false); // only 1 of 3 presents completed -> prevIncomplete
			bt += 20 * ms;
		}
		CHECK(burst.gpuBoundBurst);
		CHECK(burst.ComputePresentWaitNs(bt + 100'000ull, false) == 0);

		// Restore smooth complete batches for > 2 seconds (110 batches * 20 ms = 2.2 s) -> burst releases
		for (int i = 0; i < 110; ++i)
		{
			burst.OnBatchStart(bt, 2);
			burst.ComputePresentWaitNs(bt, false);
			burst.ComputePresentWaitNs(bt + 6 * ms, false);
			burst.ComputePresentWaitNs(bt + 12 * ms, false);
			bt += 20 * ms;
		}
		CHECK(!burst.gpuBoundBurst);

		// RDR2 Vulkan 6x MFG (5 generated + 1 real = 6 frames) pipelined pacing regression:
		// Render thread starts the next batch while 1 present of the previous batch is still in flight (5/6 done)
		// and G1 arrives 6 ms after OnBatchStart. Pacer must NOT falsely trip gpuBoundBurst and must space G2..G5 evenly.
		VulkanPolicy::PresentPacer mfg6{};
		uint64_t mt = 2'000'000'000ull;
		for (int i = 0; i < 120; ++i)
		{
			mfg6.OnBatchStart(mt, 5);
			uint64_t p0 = mt + 6 * ms; // G1 arrives 6 ms after render-thread OnBatchStart
			(void)mfg6.ComputePresentWaitNs(p0, false);
			uint64_t cur = p0;
			for (int g = 1; g < 5; ++g) // 5 of 6 presents complete before next OnBatchStart (normal pipelining)
			{
				const uint64_t w = mfg6.ComputePresentWaitNs(cur + 200'000ull, false);
				if (i >= 5)
					CHECK(w > 2'000'000ull); // must pace smoothly (~3.5-4.3 ms), never burst at 0
				cur += 200'000ull + w;
			}
			mt += 26 * ms; // ~38.5 base FPS -> ~230 FPS at 6x MFG
		}
		CHECK(!mfg6.gpuBoundBurst);
		CHECK(!mfg6.skipBatch);

		// Post-Hitch Median Window Protection: back-to-back hitch spikes must not pollute recentDeltas or inflate emaBatchNs
		VulkanPolicy::PresentPacer hp{};
		uint64_t ht = 3'000'000'000ull;
		for (int i = 0; i < 6; ++i)
		{
			hp.OnBatchStart(ht, 2);
			ht += 16 * ms;
		}
		const uint64_t steadyEma = hp.emaBatchNs;
		// Two consecutive 45 ms hitches (> 130% of 16 ms)
		hp.OnBatchStart(ht + 29 * ms, 2); ht += 45 * ms;
		CHECK(hp.skipBatch);
		hp.OnBatchStart(ht, 2); ht += 45 * ms;
		CHECK(hp.skipBatch);
		// Returning to steady 16 ms must stay at steadyEma (not dragged upward by hitch spikes)
		hp.OnBatchStart(ht, 2); ht += 16 * ms;
		hp.OnBatchStart(ht, 2); ht += 16 * ms;
		CHECK(hp.emaBatchNs == steadyEma);

		// Selective X3+ Source Cadence Alternation Smoothing: stays off on steady X3+ and on X2,
		// engages on persistent short/long alternation at X3+ (<= +4% budget), and relaxes when steady.
		VulkanPolicy::PresentPacer alt{};
		uint64_t at = 4'000'000'000ull;
		for (int i = 0; i < 6; ++i) { alt.OnBatchStart(at, 2); at += 16 * ms; }
		CHECK(!alt.alternationActive);
		// Persistent 14 ms / 19 ms alternation (> 15% swing, < 30% hitch threshold)
		for (int i = 0; i < 8; ++i)
		{
			alt.OnBatchStart(at, 2);
			at += (i % 2 == 0 ? 14 : 19) * ms;
		}
		CHECK(alt.alternationActive);
		const uint64_t maxAllowedStep = ((alt.emaBatchNs * 104ull / 100ull) * 92ull) / (3ull * 100ull);
		CHECK(alt.targetStepNs <= maxAllowedStep);
		// Steady frames disengage alternation automatically
		for (int i = 0; i < 10; ++i) { alt.OnBatchStart(at, 2); at += 16 * ms; }
		CHECK(!alt.alternationActive);
		// X2 (count == 1) must never engage alternation smoothing
		VulkanPolicy::PresentPacer alt2{};
		uint64_t a2 = 5'000'000'000ull;
		for (int i = 0; i < 10; ++i) { alt2.OnBatchStart(a2, 1); a2 += (i % 2 == 0 ? 13 : 20) * ms; }
		CHECK(!alt2.alternationActive);
	}

	// ---- Vulkan MFG swapchain image count policy (6 -> 8 double-buffered 4x MFG)
	{
		using VulkanPolicy::ParseSwapchainBufferTarget;
		using VulkanPolicy::RecommendSwapchainImageCount;

		CHECK(ParseSwapchainBufferTarget(L"") == 8u);
		CHECK(ParseSwapchainBufferTarget(L"  ") == 8u);
		CHECK(ParseSwapchainBufferTarget(L"0") == 0u);
		CHECK(ParseSwapchainBufferTarget(L"off") == 0u);
		CHECK(ParseSwapchainBufferTarget(L"FALSE") == 0u);
		CHECK(ParseSwapchainBufferTarget(L"6") == 6u);
		CHECK(ParseSwapchainBufferTarget(L"8") == 8u);
		CHECK(ParseSwapchainBufferTarget(L"10") == 10u);
		CHECK(ParseSwapchainBufferTarget(L"12") == 12u);
		CHECK(ParseSwapchainBufferTarget(L"5") == 8u);
		CHECK(ParseSwapchainBufferTarget(L"16") == 8u);
		CHECK(ParseSwapchainBufferTarget(L"abc") == 8u);

		// Standard Streamline 6-buffer swapchain boosted to 8 when driver allows (capMax == 0 unbounded or >= 8)
		CHECK(RecommendSwapchainImageCount(6u, 2u, 0u, 8u) == 8u);
		CHECK(RecommendSwapchainImageCount(6u, 2u, 8u, 8u) == 8u);
		CHECK(RecommendSwapchainImageCount(6u, 2u, 16u, 8u) == 8u);
		// Clamped to driver maxImageCount
		CHECK(RecommendSwapchainImageCount(6u, 2u, 7u, 8u) == 7u);
		CHECK(RecommendSwapchainImageCount(6u, 2u, 6u, 8u) == 6u);
		// Non-MFG game swapchains (< 6 buffers) are never touched
		CHECK(RecommendSwapchainImageCount(3u, 2u, 8u, 8u) == 3u);
		CHECK(RecommendSwapchainImageCount(4u, 2u, 8u, 8u) == 4u);
		// Disabled or already >= target
		CHECK(RecommendSwapchainImageCount(6u, 2u, 8u, 0u) == 6u);
		CHECK(RecommendSwapchainImageCount(8u, 2u, 16u, 8u) == 8u);
		CHECK(RecommendSwapchainImageCount(10u, 2u, 16u, 8u) == 10u);
	}

	// Sustained FPS drop (120 FPS menu -> 60 FPS gameplay): 1st slow frame drains as hitch,
	// 2nd consecutive slow frame adapts EMA immediately instead of staying trapped in isHitch.
	{
		VulkanPolicy::PresentPacer p{};
		uint64_t t = 1'000'000'000ull;
		for (int i = 0; i < 6; ++i)
		{
			p.OnBatchStart(t, 3u);
			for (uint32_t k = 0; k < 4u; ++k)
				p.ComputePresentWaitNs(t + k * 2'000'000ull, false);
			t += 8'333'333ull; // 120 FPS
		}
		CHECK(p.emaBatchNs >= 8'000'000ull && p.emaBatchNs <= 8'700'000ull);
		// 1st slow frame (16.67 ms): single hitch -> drains without inflating EMA
		t += 8'333'333ull; // delta = 16.67 ms
		p.OnBatchStart(t, 3u);
		for (uint32_t k = 0; k < 4u; ++k)
			p.ComputePresentWaitNs(t + k * 2'000'000ull, false);
		CHECK(p.skipBatch);
		CHECK(p.emaBatchNs < 10'000'000ull);
		// 2nd consecutive slow frame (16.67 ms): real regime shift -> adopts 16.67 ms & resumes pacing
		t += 16'666'667ull;
		p.OnBatchStart(t, 3u);
		CHECK(!p.skipBatch);
		CHECK(p.emaBatchNs >= 16'000'000ull && p.emaBatchNs <= 17'000'000ull);
	}

	// CameraMotionScale + DX12/Vulkan reset-decision loop regression tests:
	// 1) Isolated teleport (1000) followed by 100 normal steps (1.0) must NOT poison threshold (stays <= 12.0).
	// 2) Consistent cm-scale movement (12.0/frame for 100 frames) must bootstrap in 5 frames instead of 100 resets.
	{
		auto simulateLoop = [](const float* steps, size_t count, uint32_t& outResets, float& outCutThresh) {
			XeFgEnabler::CameraMotionScale scale;
			uint32_t quiet = 0u;
			outResets = 0u;
			for (size_t i = 0; i < count; ++i)
			{
				bool reset = false;
				if (quiet > 0u)
					--quiet;
				const float dist = steps[i];
				const float distSq = dist * dist;
				const float cutSq = scale.CutThresholdSq();
				const float hardSq = scale.HardTeleportThresholdSq();
				if (distSq > hardSq || (quiet == 0u && distSq > cutSq))
				{
					reset = true;
					quiet = 4u;
					++outResets;
				}
				scale.ObserveCandidate(dist, reset);
			}
			outCutThresh = std::sqrt(scale.CutThresholdSq());
		};

		float seqJump[101]{};
		seqJump[0] = 1000.0f;
		for (size_t i = 1; i < 101; ++i) seqJump[i] = 1.0f;
		uint32_t resetsJump = 0;
		float cutJump = 0.0f;
		simulateLoop(seqJump, 101, resetsJump, cutJump);
		CHECK(resetsJump == 1u);
		CHECK(cutJump <= 12.01f);

		float seqCm[100]{};
		for (size_t i = 0; i < 100; ++i) seqCm[i] = 12.0f;
		uint32_t resetsCm = 0;
		float cutCm = 0.0f;
		simulateLoop(seqCm, 100, resetsCm, cutCm);
		CHECK(resetsCm == 5u);
		CHECK(cutCm >= 143.9f && cutCm <= 144.1f);
	}

	// Stale-present cancellation (NSS-OFG): start with an actively paced batch (skipBatch == false, wait > 0),
	// then exceed emaBatchNs and prove skipBatch flips false -> true and drains immediately with 0 wait
	{
		VulkanPolicy::PresentPacer p;
		const uint64_t step = 16'000'000ull; // 16 ms base (~62.5 FPS)
		p.OnBatchStart(step * 1, 3);
		p.OnBatchStart(step * 2, 3);
		p.OnBatchStart(step * 3, 3);
		CHECK(!p.skipBatch);
		(void)p.ComputePresentWaitNs(step * 3 + 500'000ull, false);
		const uint64_t normalWait = p.ComputePresentWaitNs(step * 3 + 1'000'000ull, false);
		CHECK(normalWait > 0);
		CHECK(!p.skipBatch);
		const uint64_t staleWait = p.ComputePresentWaitNs(step * 3 + 20'000'000ull, false);
		CHECK(staleWait == 0);
		CHECK(p.skipBatch);
		const uint64_t followWait = p.ComputePresentWaitNs(step * 3 + 20'500'000ull, false);
		CHECK(followWait == 0);
	}

	// Intel Arc Battlemage (0x8086) 4 KiB alignment guard for pooled compute buffers in [4 KiB, 64 KiB]
	{
		CHECK(VulkanPolicy::SafeBufferAlignment(0x8086u, 0xE20Bu, 4096ull, 64ull) == 4096ull);
		CHECK(VulkanPolicy::SafeBufferAlignment(0x8086u, 0xE20Bu, 32768ull, 256ull) == 4096ull);
		CHECK(VulkanPolicy::SafeBufferAlignment(0x8086u, 0xE20Bu, 65536ull, 64ull) == 4096ull);
		CHECK(VulkanPolicy::SafeBufferAlignment(0x8086u, 0xE20Bu, 65536ull, 65536ull) == 65536ull);
		CHECK(VulkanPolicy::SafeBufferAlignment(0x8086u, 0xE20Bu, 2048ull, 64ull) == 64ull);
		CHECK(VulkanPolicy::SafeBufferAlignment(0x8086u, 0xE20Bu, 131072ull, 64ull) == 64ull);
		CHECK(VulkanPolicy::SafeBufferAlignment(0x1002u, 0xE20Bu, 16384ull, 64ull) == 64ull);
		CHECK(VulkanPolicy::SafeBufferAlignment(0x10DEu, 0xE20Bu, 16384ull, 64ull) == 64ull);
		CHECK(VulkanPolicy::SafeBufferAlignment(0x8086u, 0x56A0u, 8192ull, 64ull) == 64ull);
	}

	// Coarse optical-flow 4K+ recommendation policy
	{
		CHECK(VulkanPolicy::ShouldUseCoarseFlow(3840u, 2160u));
		CHECK(VulkanPolicy::ShouldUseCoarseFlow(5120u, 2160u));
		CHECK(!VulkanPolicy::ShouldUseCoarseFlow(2560u, 1440u));
		CHECK(!VulkanPolicy::ShouldUseCoarseFlow(1920u, 1080u));
	}

	// CameraTurnGuard: sustained same-direction yaw flick suppresses false GhostBuster reset; instant cut still resets
	{
		XeFgEnabler::CameraTurnGuard guard;
		const float f0[3] = { 0.0f, 0.0f, 1.0f };
		const float f1[3] = { 0.766044f, 0.0f, 0.642788f }; // +50 deg yaw (cos=0.643 < 0.707)
		const float f2[3] = { 0.984808f, 0.0f, -0.173648f }; // +50 deg more yaw in same direction
		const bool t1 = guard.ObserveAndCheckSustainedTurn(f1, f0, 0.642788f);
		CHECK(!t1); // first frame of turn isn't yet a sustained streak
		const bool t2 = guard.ObserveAndCheckSustainedTurn(f2, f1, 0.642788f);
		CHECK(t2);  // second consecutive same-axis frame IS a sustained mouse turn
		// With sustainedTurn=true, 50 deg flick (cos=0.643 >= 0.342) does NOT false-reset GhostBuster:
		CHECK(!XeFgEnabler::ShouldGhostBusterReset(true, true, true, 0.642788f, 1.0f, 1.0f, true, true, 0.0f, 4.0f, t2));
		// Instant >70 deg cut (cos=0.20 < 0.342) STILL resets even if sustainedTurn=true:
		CHECK(XeFgEnabler::ShouldGhostBusterReset(true, true, true, 0.20f, 1.0f, 1.0f, true, true, 0.0f, 4.0f, true));
		// FOV jump STILL resets even during a sustained turn:
		CHECK(XeFgEnabler::ShouldGhostBusterReset(true, true, true, 0.642788f, 1.4f, 1.0f, true, true, 0.0f, 4.0f, true));
	}

	// HwFlipQueue state helper
	{
		CHECK(std::string_view(HwFlipQueue::StateName(HwFlipQueue::State::Enabled)) == "enabled");
		CHECK(std::string_view(HwFlipQueue::StateName(HwFlipQueue::State::Disabled)) == "disabled");
		CHECK(std::string_view(HwFlipQueue::StateName(HwFlipQueue::State::Unknown)) == "unknown");
	}

	// DRED stage attribution (before / inside / after / no_xefg)
	{
		using namespace DredAttribution;
		const auto r0 = Classify(5, 10, nullptr, 0);
		CHECK(r0.location == Location::NoXefg && !r0.hasXefgSpan);

		const ContextEntry gameOnly[] = {
			{ 2, L"Game: ShadowPass", nullptr },
			{ 7, L"Game: PostProcess", nullptr },
		};
		const auto r1 = Classify(5, 10, gameOnly, 2);
		CHECK(r1.location == Location::NoXefg && !r1.hasXefgSpan);

		const ContextEntry entries[] = {
			{ 1,  L"Game: GBuffer", nullptr },
			{ 5,  L"dlssg-to-xefg: wmma-batch", nullptr },
			{ 9,  L"dlssg-to-xefg: execute", nullptr },
			{ 15, L"Game: PresentUI", nullptr },
		};

		const auto before = Classify(3, 20, entries, 4);
		CHECK(before.location == Location::BeforeXefg);
		CHECK(before.hasXefgSpan && before.xefgBegin == 5 && before.xefgEnd == 14);
		CHECK(before.activeStageW == nullptr);

		const auto inside1 = Classify(6, 20, entries, 4);
		CHECK(inside1.location == Location::InsideXefg);
		CHECK(inside1.activeStageW != nullptr && std::wstring_view(inside1.activeStageW) == L"dlssg-to-xefg: wmma-batch");
		CHECK(inside1.activeStageIndex == 5);

		const auto inside2 = Classify(14, 20, entries, 4);
		CHECK(inside2.location == Location::InsideXefg);
		CHECK(inside2.activeStageW != nullptr && std::wstring_view(inside2.activeStageW) == L"dlssg-to-xefg: execute");
		CHECK(inside2.activeStageIndex == 9);

		const auto after = Classify(15, 20, entries, 4);
		CHECK(after.location == Location::AfterXefg);
		CHECK(after.xefgBegin == 5 && after.xefgEnd == 14);

		const ContextEntry unsorted[] = {
			{ 12, nullptr, "dlssg-to-xefg: wmma-copy" },
			{ 4,  nullptr, "" },
			{ 8,  nullptr, "dlssg-to-xefg: execute-dot4" },
		};
		const auto rt = Classify(16, 20, unsorted, 3);
		CHECK(rt.location == Location::InsideXefg);
		CHECK(rt.xefgBegin == 8 && rt.xefgEnd == 19);
		CHECK(rt.activeStageA != nullptr && std::string_view(rt.activeStageA) == "dlssg-to-xefg: wmma-copy");
		CHECK(std::string_view(LocationName(Location::InsideXefg)) == "inside_xefg");
	}

	// Vulkan HUD storage image view format resolution (BGRA8 -> RGBA8 zero-copy mutable alias fallback)
	{
		using namespace VulkanPolicy;
		CHECK(ColorStorageViewFormat(Dxgi::R10G10B10A2Unorm, true) == Vk::A2B10G10R10UnormPack32);
		CHECK(ColorStorageViewFormat(Dxgi::R10G10B10A2Unorm, false) == Vk::A2B10G10R10UnormPack32);
		CHECK(ColorStorageViewFormat(Dxgi::R8G8B8A8Unorm, true) == Vk::R8G8B8A8Unorm);
		CHECK(ColorStorageViewFormat(Dxgi::R8G8B8A8Unorm, false) == Vk::R8G8B8A8Unorm);
		CHECK(ColorStorageViewFormat(Dxgi::B8G8R8A8Unorm, true) == Vk::B8G8R8A8Unorm);
		CHECK(ColorStorageViewFormat(Dxgi::B8G8R8A8Unorm, false) == Vk::R8G8B8A8Unorm);
		CHECK(ColorStorageViewFormat(Dxgi::Unknown, false) == Vk::R8G8B8A8Unorm);

		// One-time HDR10 bootstrap reinit (1 -> maxFrames) policy
		CHECK(NeedsHdrBootstrapReinit(Dxgi::R10G10B10A2Unorm, 5u, false));
		CHECK(NeedsHdrBootstrapReinit(Dxgi::R10G10B10A2Unorm, 2u, false));
		CHECK(!NeedsHdrBootstrapReinit(Dxgi::R10G10B10A2Unorm, 1u, false));
		CHECK(!NeedsHdrBootstrapReinit(Dxgi::R10G10B10A2Unorm, 5u, true));
		CHECK(!NeedsHdrBootstrapReinit(Dxgi::R8G8B8A8Unorm, 5u, false));
		CHECK(!NeedsHdrBootstrapReinit(Dxgi::B8G8R8A8Unorm, 5u, false));
		CHECK(!NeedsHdrBootstrapReinit(Dxgi::R16G16B16A16Float, 5u, false));

		// Opt-in VK_AMD_anti_lag device-create guard (feature bit, single-GPU, game-owned ext/struct)
		CHECK(CanEnableAmdAntiLag(true, 0x1002u, true, true, 1u, false, false));
		CHECK(CanEnableAmdAntiLag(true, 0x1002u, true, true, 0u, false, false));
		CHECK(!CanEnableAmdAntiLag(false, 0x1002u, true, true, 1u, false, false));
		CHECK(!CanEnableAmdAntiLag(true, 0x10DEu, true, true, 1u, false, false));
		CHECK(!CanEnableAmdAntiLag(true, 0x1002u, false, true, 1u, false, false));
		CHECK(!CanEnableAmdAntiLag(true, 0x1002u, true, false, 1u, false, false));
		CHECK(!CanEnableAmdAntiLag(true, 0x1002u, true, true, 2u, false, false));
		CHECK(!CanEnableAmdAntiLag(true, 0x1002u, true, true, 1u, true, false));
		CHECK(!CanEnableAmdAntiLag(true, 0x1002u, true, true, 1u, false, true));

		// Opt-in VK_AMD_anti_lag dispatch policy (OFF by default, AMD only, skips if game already enabled it)
		CHECK(ShouldDispatchAmdAntiLag(true, 0x1002u, true, false, false));
		CHECK(!ShouldDispatchAmdAntiLag(false, 0x1002u, true, false, false));
		CHECK(!ShouldDispatchAmdAntiLag(true, 0x10DEu, true, false, false));
		CHECK(!ShouldDispatchAmdAntiLag(true, 0x8086u, true, false, false));
		CHECK(!ShouldDispatchAmdAntiLag(true, 0x1002u, false, false, false));
		CHECK(!ShouldDispatchAmdAntiLag(true, 0x1002u, true, true, false));
		CHECK(!ShouldDispatchAmdAntiLag(true, 0x1002u, true, false, true));

		// Anti-Lag swapchain ownership + MFG real-frame gating
		CHECK(!IsRealFramePresentForAntiLag(false, 0u, 1u)); // unregistered / foreign swapchain: never fire
		CHECK(!IsRealFramePresentForAntiLag(false, 2u, 3u)); // unregistered swapchain even on real frame: never fire
		CHECK(IsRealFramePresentForAntiLag(true, 0u, 1u));   // FG paused/off: every present is a real frame
		CHECK(!IsRealFramePresentForAntiLag(true, 1u, 1u));  // 2x MFG: 1st present is interpolated -> skip
		CHECK(IsRealFramePresentForAntiLag(true, 1u, 2u));   // 2x MFG: 2nd present is real -> fire
		CHECK(!IsRealFramePresentForAntiLag(true, 2u, 1u));  // 3x MFG: interp 1 -> skip
		CHECK(!IsRealFramePresentForAntiLag(true, 2u, 2u));  // 3x MFG: interp 2 -> skip
		CHECK(IsRealFramePresentForAntiLag(true, 2u, 3u));   // 3x MFG: real frame 3 -> fire once
		CHECK(!IsRealFramePresentForAntiLag(true, 2u, 4u));  // 3x MFG: extra present before next batch -> skip
		CHECK(!IsRealFramePresentForAntiLag(true, 3u, 1u));  // 4x MFG: interp 1 -> skip
		CHECK(!IsRealFramePresentForAntiLag(true, 3u, 3u));  // 4x MFG: interp 3 -> skip
		CHECK(IsRealFramePresentForAntiLag(true, 3u, 4u));   // 4x MFG: real frame 4 -> fire once
		CHECK(!IsRealFramePresentForAntiLag(true, 3u, 5u));  // 4x MFG: extra present before next batch -> skip

		// D3D12->Vulkan WMMA context-init failure backoff & ResetContext preservation (FH6 menu toggle storm)
		CHECK(!CanRetryFailedWmmaContext(1u, 0ull, true));       // same settings, <10s cooldown -> skip
		CHECK(!CanRetryFailedWmmaContext(1u, 9999ull, true));    // still inside 10s cooldown -> skip
		CHECK(CanRetryFailedWmmaContext(1u, 10000ull, true));    // cooldown elapsed, attempt < 3 -> retry
		CHECK(CanRetryFailedWmmaContext(2u, 15000ull, true));    // cooldown elapsed, attempt < 3 -> retry
		CHECK(!CanRetryFailedWmmaContext(3u, 60000ull, true));   // 3 failures reached -> stop retrying same settings
		CHECK(CanRetryFailedWmmaContext(3u, 0ull, false));       // settings changed -> retry immediately
		CHECK(!ShouldResetWmmaFailedState(false));               // menu toggle without live context -> keep backoff
		CHECK(ShouldResetWmmaFailedState(true));                 // tearing down working context -> reset backoff

		// Standalone SPIR-V override validation (splat_motion_aware*.spv)
		CHECK(IsValidSpirvBlob(20u, 0x07230203u));
		CHECK(IsValidSpirvBlob(4096u, 0x07230203u));
		CHECK(!IsValidSpirvBlob(16u, 0x07230203u));              // truncated header
		CHECK(!IsValidSpirvBlob(22u, 0x07230203u));              // not 4-byte word aligned
		CHECK(!IsValidSpirvBlob(20u, 0x43425844u));              // DXBC magic, not SPIR-V
		CHECK(!IsValidSpirvBlob(2u * 1024u * 1024u, 0x07230203u)); // > 1 MiB rejected

		// Camera near/far normalization for BOTH depthInverted=true and depthInverted=false
		{
			float n = 0.15f, f = 10000.0f; bool inv = false;
			CHECK(!NormalizeCameraPlanes(n, f, inv) && Near(n, 0.15f) && Near(f, 10000.0f) && !inv);
			float nInf = 0.15f, fInf = 0.0f; bool invInf = true;
			CHECK(!NormalizeCameraPlanes(nInf, fInf, invInf) && Near(nInf, 0.15f) && Near(fInf, 0.0f) && invInf);
			float nSwapT = 10000.0f, fSwapT = 0.15f; bool invT = true;
			CHECK(NormalizeCameraPlanes(nSwapT, fSwapT, invT) && Near(nSwapT, 0.15f) && Near(fSwapT, 10000.0f) && invT);
			float nSwapF = 10000.0f, fSwapF = 0.15f; bool invF = false;
			CHECK(NormalizeCameraPlanes(nSwapF, fSwapF, invF) && Near(nSwapF, 0.15f) && Near(fSwapF, 10000.0f) && invF);
			float mRefInv[16], mSwappedT[16], mSwappedF[16];
			Projection(2.0f, 1.0f, 101.0f, 1.5707964f, true, mRefInv);
			Projection(2.0f, 101.0f, 1.0f, 1.5707964f, true, mSwappedT);
			Projection(2.0f, 101.0f, 1.0f, 1.5707964f, false, mSwappedF);
			CHECK(Near(mRefInv[10], mSwappedT[10], 1e-5f) && Near(mRefInv[14], mSwappedT[14], 1e-4f));
			CHECK(Near(mRefInv[10], mSwappedF[10], 1e-5f) && Near(mRefInv[14], mSwappedF[14], 1e-4f));
		}

		// Hardened rolling VRAM drift diagnostic (120-frame warmup, 3 consecutive rising windows, budget-% threshold)
		{
			constexpr uint32_t kBudget16G = 16384u; // stepMin = 128 MiB, totalMin = 512 MiB
			VramDriftTracker t;
			CHECK(!t.Sample(0u, kBudget16G, 8u).ready);
			// 15 polls * 8 = 120 warmup frames skipped (startup surge ignored)
			for (int i = 0; i < 15; ++i)
				CHECK(!t.Sample(6000u + uint32_t(i * 100), kBudget16G, 8u).ready);
			// First post-warmup poll captures baseline = 8000 MiB
			CHECK(!t.Sample(8000u, kBudget16G, 8u).ready);
			// Window 1: rises +200 MiB (8200) -> logs baseline Info (not Warn: only 1 rising window)
			VramDriftReport r1{};
			for (int i = 0; i < 30; ++i)
				r1 = t.Sample(8200u, kBudget16G, 8u);
			CHECK(r1.ready && !r1.warn && r1.baselineMiB == 8000u && r1.deltaMiB == 200 && r1.risingStreak == 1u);
			// Window 2: rises +200 MiB (8400) -> silent (only 2 rising windows)
			VramDriftReport r2{};
			for (int i = 0; i < 30; ++i)
				r2 = t.Sample(8400u, kBudget16G, 8u);
			CHECK(!r2.ready);
			// Window 3: rises +200 MiB (8600, total +600 >= 512) -> 3 consecutive rising windows -> WARN
			VramDriftReport r3{};
			for (int i = 0; i < 30; ++i)
				r3 = t.Sample(8600u, kBudget16G, 8u);
			CHECK(r3.ready && r3.warn && r3.deltaMiB == 600 && r3.risingStreak == 3u);
			t.Reset();
			CHECK(t.baselineMiB == 0u && t.warmupFrames == 0u && !t.baselineLogged);
		}

		// Shared Adaptive VRAM Guard state machine (D3D12 + Vulkan: gradual step-down, dwell, budget-% margins)
		{
			constexpr uint64_t kMiB = 1024ull * 1024ull;
			// 8 GB vs 24 GB margin scaling
			CHECK(VramGuardState::SafetyMarginBytes(8192ull * kMiB) > 200ull * kMiB);
			CHECK(VramGuardState::SafetyMarginBytes(24576ull * kMiB) > VramGuardState::SafetyMarginBytes(8192ull * kMiB));
			CHECK(VramGuardState::ReleaseMarginBytes(24576ull * kMiB) == 1546188226ull); // 6% of 24 GiB
			CHECK(VramGuardState::ReleaseMarginBytes(32768ull * kMiB) == 1536ull * kMiB); // upper clamp

			VramGuardState g;
			const uint64_t budget = 16384ull * kMiB;
			const uint64_t tightUsage = budget - VramGuardState::SafetyMarginBytes(budget) + 16ull * kMiB;
			const uint64_t comfortUsage = budget - VramGuardState::ReleaseMarginBytes(budget) - 64ull * kMiB;
			// Requested 3 generated frames (4x): tight steps down 3 -> 2 immediately
			CHECK(g.Update(tightUsage, budget, 3u, 1000ull) == VramGuardState::Transition::Engaged);
			CHECK(g.active && g.stepDown == 1u && g.EffectiveCount(3u) == 2u);
			// Crossing 2 -> 1 requires 2 consecutive tight polls (avoids 112 ms rebuild thrash)
			CHECK(g.Update(tightUsage, budget, 3u, 2000ull) == VramGuardState::Transition::None);
			CHECK(g.EffectiveCount(3u) == 2u);
			CHECK(g.Update(tightUsage, budget, 3u, 3000ull) == VramGuardState::Transition::Engaged);
			CHECK(g.EffectiveCount(3u) == 1u);
			// Stepping back up 1 -> 2 requires 10s dwell
			CHECK(g.Update(comfortUsage, budget, 3u, 4000ull) == VramGuardState::Transition::None);
			CHECK(g.Update(comfortUsage, budget, 3u, 13000ull) == VramGuardState::Transition::None);
			CHECK(g.Update(comfortUsage, budget, 3u, 14000ull) == VramGuardState::Transition::Easing);
			CHECK(g.EffectiveCount(3u) == 2u);
			// Stepping back up 2 -> 3 requires 3s dwell
			CHECK(g.Update(comfortUsage, budget, 3u, 15000ull) == VramGuardState::Transition::None);
			CHECK(g.Update(comfortUsage, budget, 3u, 18000ull) == VramGuardState::Transition::Released);
			CHECK(!g.active && g.stepDown == 0u && g.EffectiveCount(3u) == 3u);
			// Adaptive WMMA wait timeout: clamp(1.5 * realFrameEma, 8 ms, 50 ms)
			CHECK(WmmaAdaptiveWaitTimeoutNs(0u) == 25'000'000ull);       // warmup default 25 ms
			CHECK(WmmaAdaptiveWaitTimeoutNs(4'000u) == 8'000'000ull);    // 250 fps -> 8 ms floor
			CHECK(WmmaAdaptiveWaitTimeoutNs(16'667u) == 25'000'500ull);  // 60 fps -> 25 ms
			CHECK(WmmaAdaptiveWaitTimeoutNs(25'000u) == 37'500'000ull);  // 40 fps -> 37.5 ms
			CHECK(WmmaAdaptiveWaitTimeoutNs(40'000u) == 50'000'000ull);  // 25 fps -> 50 ms cap
			CHECK(!ShouldReclaimClampedOutputs(false, 100u));
			CHECK(!ShouldReclaimClampedOutputs(true, 23u));
			CHECK(ShouldReclaimClampedOutputs(true, 24u));

			// Slow-leak detection (10-window cumulative baseline check + warn cooldown)
			VramDriftTracker slow;
			for (int i = 0; i < 15; ++i) slow.Sample(8000u, 16384u, 8u);
			slow.Sample(8000u, 16384u, 8u); // baseline = 8000 MiB
			VramDriftReport slowRep{};
			for (int w = 1; w <= 10; ++w)
			{
				for (int i = 0; i < 30; ++i)
					slowRep = slow.Sample(8000u + uint32_t(w * 60), 16384u, 8u); // +60 MiB/window (<128 fast threshold, +600 total >= 512)
				if (w < 10) CHECK(!slowRep.warn);
			}
			CHECK(slowRep.ready && slowRep.warn && slowRep.deltaMiB == 600 && slowRep.windowIndex == 10u);
			// Cooldown prevents immediate re-warning on the next 3 windows
			VramDriftReport cdRep{};
			for (int w = 1; w <= 3; ++w)
				for (int i = 0; i < 30; ++i)
					cdRep = slow.Sample(8600u + uint32_t(w * 200), 16384u, 8u);
			CHECK(!cdRep.warn);

			// One-time +700 MiB map load then flat for 100 windows: must NOT repeat warnings after window 10
			VramDriftTracker mapLoad;
			for (int i = 0; i < 15; ++i) mapLoad.Sample(8000u, 16384u, 8u);
			mapLoad.Sample(8000u, 16384u, 8u); // baseline = 8000 MiB
			int warnCount = 0;
			for (int w = 1; w <= 100; ++w)
			{
				VramDriftReport r{};
				for (int i = 0; i < 30; ++i)
					r = mapLoad.Sample(8700u, 16384u, 8u); // jumped +700 once, then flat forever
				if (r.warn)
					++warnCount;
			}
			CHECK(warnCount == 1); // fires at window 10 once, then 0 warnings across windows 11..100
		}
	}

	// Vulkan ICD direct-bypass policy helpers (v15r8)
	{
		CHECK(VulkanPolicy::IsKnownVulkanIcdName(L"amdvlk64.dll"));
		CHECK(VulkanPolicy::IsKnownVulkanIcdName(L"C:\\Windows\\System32\\DriverStore\\FileRepository\\u0412\\AMDVLK64.DLL"));
		CHECK(VulkanPolicy::IsKnownVulkanIcdName(L"C:/Windows/System32/nvoglv64.dll"));
		CHECK(VulkanPolicy::IsKnownVulkanIcdName(L"igvk64.dll"));
		CHECK(VulkanPolicy::IsKnownVulkanIcdName(L"xe_vk64.dll"));
		CHECK(VulkanPolicy::IsKnownVulkanIcdName(L"igxelpicd64.dll"));
		CHECK(VulkanPolicy::IsKnownVulkanIcdName(L"winevulkan.dll"));
		CHECK(!VulkanPolicy::IsKnownVulkanIcdName(L""));
		CHECK(!VulkanPolicy::IsKnownVulkanIcdName(L"vulkan-1.dll"));
		CHECK(!VulkanPolicy::IsKnownVulkanIcdName(L"C:\\Games\\OptiScaler.dll"));
		CHECK(!VulkanPolicy::IsKnownVulkanIcdName(L"RTSSHooks64.dll"));

		CHECK(!VulkanPolicy::ShouldBypassWmmaDeviceProc(""));
		CHECK(!VulkanPolicy::ShouldBypassWmmaDeviceProc("vkDestroyDevice"));
		CHECK(VulkanPolicy::ShouldBypassWmmaDeviceProc("vkQueueSubmit2"));
		CHECK(VulkanPolicy::ShouldBypassWmmaDeviceProc("vkCreateCommandPool"));
		CHECK(VulkanPolicy::ShouldBypassWmmaDeviceProc("vkCmdDispatch"));

		// Pure DirectIcdResolver::Init decision tests (v15r9): no layer, wrapping layer, unknown ICD, gdpa wrong module
		int dummyIcdMod = 0, dummyOtherMod = 0, dummyLoaderFn = 0, dummyLayerFn = 0, dummyGdpaFn = 0;
		VulkanPolicy::DirectIcdInitProbe validProbe{};
		validProbe.allowBypass = true;
		validProbe.loaderProbeFn = &dummyLoaderFn;
		validProbe.probeModule = &dummyIcdMod;
		validProbe.probeModulePath = L"C:\\Windows\\System32\\amdvlk64.dll";
		validProbe.icdGdpaFn = &dummyGdpaFn;
		validProbe.icdGdpaModule = &dummyIcdMod;
		validProbe.icdProbeFn = &dummyLoaderFn;
		CHECK(VulkanPolicy::ShouldEnableDirectIcdBypass(validProbe));

		// Wrapping layer present: loader vkGetImageMemoryRequirements != ICD vkGetImageMemoryRequirements
		auto layerProbe = validProbe;
		layerProbe.icdProbeFn = &dummyLayerFn;
		CHECK(!VulkanPolicy::ShouldEnableDirectIcdBypass(layerProbe));

		// Unknown ICD DLL name
		auto unknownIcdProbe = validProbe;
		unknownIcdProbe.probeModulePath = L"C:\\Program Files\\SomeLayer\\vk_layer_custom.dll";
		CHECK(!VulkanPolicy::ShouldEnableDirectIcdBypass(unknownIcdProbe));

		// icdGdpa points outside probeModule (wrong module / hooked gdpa)
		auto wrongModProbe = validProbe;
		wrongModProbe.icdGdpaModule = &dummyOtherMod;
		CHECK(!VulkanPolicy::ShouldEnableDirectIcdBypass(wrongModProbe));

		// Killswitch disabled and null-pointer guards
		auto disabledProbe = validProbe;
		disabledProbe.allowBypass = false;
		CHECK(!VulkanPolicy::ShouldEnableDirectIcdBypass(disabledProbe));

		auto nullLoaderProbe = validProbe;
		nullLoaderProbe.loaderProbeFn = nullptr;
		CHECK(!VulkanPolicy::ShouldEnableDirectIcdBypass(nullLoaderProbe));

		auto nullGdpaProbe = validProbe;
		nullGdpaProbe.icdGdpaFn = nullptr;
		CHECK(!VulkanPolicy::ShouldEnableDirectIcdBypass(nullGdpaProbe));

		auto nullIcdProbe = validProbe;
		nullIcdProbe.icdProbeFn = nullptr;
		CHECK(!VulkanPolicy::ShouldEnableDirectIcdBypass(nullIcdProbe));

		// Pure DirectIcdResolver::Resolve decision tests
		int dummyDirectFn = 0;
		CHECK(VulkanPolicy::ShouldUseDirectIcdProc("vkQueueSubmit2", &dummyLayerFn, &dummyOtherMod, &dummyDirectFn, &dummyIcdMod, &dummyIcdMod));
		CHECK(!VulkanPolicy::ShouldUseDirectIcdProc("vkQueueSubmit2", &dummyLoaderFn, &dummyIcdMod, &dummyDirectFn, &dummyIcdMod, &dummyIcdMod)); // already direct
		CHECK(!VulkanPolicy::ShouldUseDirectIcdProc("vkDestroyDevice", &dummyLayerFn, &dummyOtherMod, &dummyDirectFn, &dummyIcdMod, &dummyIcdMod)); // always loader
		CHECK(!VulkanPolicy::ShouldUseDirectIcdProc("vkQueueSubmit2", &dummyLayerFn, &dummyOtherMod, &dummyDirectFn, &dummyOtherMod, &dummyIcdMod)); // direct outside ICD
		CHECK(!VulkanPolicy::ShouldUseDirectIcdProc("vkQueueSubmit2", &dummyLayerFn, &dummyOtherMod, nullptr, &dummyIcdMod, &dummyIcdMod)); // null direct
		CHECK(VulkanPolicy::ShouldUseDirectIcdProc("vkCreateCommandPool", nullptr, nullptr, &dummyDirectFn, &dummyIcdMod, &dummyIcdMod)); // loader null, direct valid
	}

	// In-game WMMA GPU timestamp benchmark helpers (v15r10)
	{
		// Normal delta: 200,000 ticks * 10 ns/tick = 2.0 ms
		const float ms1 = VulkanPolicy::ComputeGpuTimestampMs(1000000ull, 1200000ull, 64u, 10.0f);
		CHECK(ms1 > 1.999f && ms1 < 2.001f);

		// 36-bit wrap-around: (end - start) & ((1<<36)-1) = 100,000 ticks * 10 ns = 1.0 ms
		const uint64_t start36 = (1ull << 36) - 40000ull;
		const uint64_t end36 = 60000ull;
		const float msWrap36 = VulkanPolicy::ComputeGpuTimestampMs(start36, end36, 36u, 10.0f);
		CHECK(msWrap36 > 0.999f && msWrap36 < 1.001f);

		// 64-bit wrap-around: 50,000 ticks * 40 ns = 2.0 ms
		const float msWrap64 = VulkanPolicy::ComputeGpuTimestampMs(~0ull - 9999ull, 40000ull, 64u, 40.0f);
		CHECK(msWrap64 > 1.999f && msWrap64 < 2.001f);

		// Invalid inputs / outlier > 1000 ms rejected
		CHECK(VulkanPolicy::ComputeGpuTimestampMs(0ull, 100ull, 0u, 10.0f) < 0.0f);
		CHECK(VulkanPolicy::ComputeGpuTimestampMs(0ull, 100ull, 65u, 10.0f) < 0.0f);
		CHECK(VulkanPolicy::ComputeGpuTimestampMs(0ull, 100ull, 64u, 0.0f) < 0.0f);
		CHECK(VulkanPolicy::ComputeGpuTimestampMs(0ull, 200000000ull, 64u, 10.0f) < 0.0f); // 2000 ms > 1000 ms

		// GpuBenchWindow: first report at 60 batches, second report at 300 batches
		VulkanPolicy::GpuBenchWindow win;
		VulkanPolicy::GpuBenchReport rep{};
		for (uint32_t i = 1; i <= 59; ++i)
		{
			rep = win.Sample(1.5f, 3u);
			CHECK(!rep.ready);
		}
		rep = win.Sample(1.5f, 3u);
		CHECK(rep.ready);
		CHECK(rep.batches == 60u);
		CHECK(rep.genFrames == 3u);
		CHECK(rep.batchAvgMs > 1.499f && rep.batchAvgMs < 1.501f);
		CHECK(rep.perGenFrameAvgMs > 0.499f && rep.perGenFrameAvgMs < 0.501f);

		// Invalid samples ignored
		CHECK(!win.Sample(-1.0f, 3u).ready);
		CHECK(!win.Sample(1.5f, 0u).ready);

		// Second window requires 300 samples and computes percentiles accurately
		for (uint32_t i = 1; i <= 299; ++i)
		{
			rep = win.Sample(static_cast<float>(i) * 0.01f, 2u);
			CHECK(!rep.ready);
		}
		rep = win.Sample(3.00f, 2u);
		CHECK(rep.ready);
		CHECK(rep.batches == 300u);
		CHECK(rep.batchMinMs > 0.009f && rep.batchMinMs < 0.011f);
		CHECK(rep.batchMaxMs > 2.999f && rep.batchMaxMs < 3.001f);
		CHECK(rep.batchP50Ms > 1.49f && rep.batchP50Ms < 1.51f);
		CHECK(rep.batchP95Ms > 2.84f && rep.batchP95Ms < 2.86f);

		// Live WMMA LDS stride mode selector (v15r12)
		VulkanPolicy::SetWmmaLdsStrideMode(0);
		CHECK(VulkanPolicy::GetWmmaLdsStrideMode() == 0);
		CHECK(VulkanPolicy::WmmaLdsStrideValue(0) == 4);
		VulkanPolicy::SetWmmaLdsStrideMode(1);
		CHECK(VulkanPolicy::GetWmmaLdsStrideMode() == 1);
		CHECK(VulkanPolicy::WmmaLdsStrideValue(1) == 5);
		VulkanPolicy::SetWmmaLdsStrideMode(99); // clamps to default 0
		CHECK(VulkanPolicy::GetWmmaLdsStrideMode() == 0);
		CHECK(VulkanPolicy::WmmaLdsStrideValue(99) == 4);

		// Actual executed per-batch stride classification (v15r14)
		CHECK(VulkanPolicy::ClassifyBatchStride(0u, 0u) == VulkanPolicy::BatchStrideKind::None);
		CHECK(VulkanPolicy::ClassifyBatchStride(17u, 0u) == VulkanPolicy::BatchStrideKind::Stride4);
		CHECK(VulkanPolicy::ClassifyBatchStride(0u, 17u) == VulkanPolicy::BatchStrideKind::Stride5);
		CHECK(VulkanPolicy::ClassifyBatchStride(10u, 7u) == VulkanPolicy::BatchStrideKind::Mixed);
		CHECK(std::string(VulkanPolicy::BatchStrideTag(VulkanPolicy::BatchStrideKind::None)) == "none");
		CHECK(std::string(VulkanPolicy::BatchStrideTag(VulkanPolicy::BatchStrideKind::Stride4)) == "4");
		CHECK(std::string(VulkanPolicy::BatchStrideTag(VulkanPolicy::BatchStrideKind::Stride5)) == "5");
		CHECK(std::string(VulkanPolicy::BatchStrideTag(VulkanPolicy::BatchStrideKind::Mixed)) == "mixed");
		CHECK(VulkanPolicy::ClassifyBatchStride(16u, 0u, 1u) == VulkanPolicy::BatchStrideKind::Fallback);
		CHECK(VulkanPolicy::ClassifyBatchStride(0u, 16u, 1u) == VulkanPolicy::BatchStrideKind::Fallback);
		CHECK(VulkanPolicy::ClassifyBatchStride(8u, 8u, 1u) == VulkanPolicy::BatchStrideKind::Fallback);
		CHECK(std::string(VulkanPolicy::BatchStrideTag(VulkanPolicy::BatchStrideKind::Fallback)) == "fallback");

		// Unified AMD architecture classifier shared by D3D12 (wmma_route.cpp) and Vulkan (xvk_coopmat.cpp) (v15r16)
		using VulkanPolicy::AmdArch;
		using VulkanPolicy::ClassifyAmdArch;
		// RDNA 4 discrete
		CHECK(ClassifyAmdArch(0x1002u, 0x7550u, "AMD Radeon RX 9070 XT") == AmdArch::Rdna4);
		CHECK(ClassifyAmdArch(0x1002u, 0x7590u, "AMD Radeon RX 9060 XT") == AmdArch::Rdna4);
		CHECK(ClassifyAmdArch(0x1002u, 0x0000u, "AMD Radeon AI PRO R9700") == AmdArch::Rdna4);
		// RDNA 3 / 3.5 discrete & APUs (ROG Ally / Legion Go 780M, Strix Point 890M, Strix Halo 8060S, Krackan 860M)
		CHECK(ClassifyAmdArch(0x1002u, 0x744Cu, "AMD Radeon RX 7900 XTX") == AmdArch::Rdna3);
		CHECK(ClassifyAmdArch(0x1002u, 0x15BFu, "AMD Radeon 780M Graphics") == AmdArch::Rdna3);
		CHECK(ClassifyAmdArch(0x1002u, 0x15C8u, "AMD Radeon 760M Graphics") == AmdArch::Rdna3);
		CHECK(ClassifyAmdArch(0x1002u, 0x150Eu, "AMD Radeon 890M Graphics") == AmdArch::Rdna3);
		CHECK(ClassifyAmdArch(0x1002u, 0x1586u, "AMD Radeon 8060S Graphics") == AmdArch::Rdna3);
		CHECK(ClassifyAmdArch(0x1002u, 0x1114u, "AMD Radeon 860M Graphics") == AmdArch::Rdna3);
		// RADV uppercase device strings (case-insensitive matching)
		CHECK(ClassifyAmdArch(0x1002u, 0x0000u, "AMD Radeon Graphics (RADV GFX1103_R1)") == AmdArch::Rdna3);
		CHECK(ClassifyAmdArch(0x1002u, 0x0000u, "AMD Radeon PRO W7900") == AmdArch::Rdna3);
		CHECK(ClassifyAmdArch(0x1002u, 0x0000u, "AMD Custom GPU 0932 (RADV VANGOGH)") == AmdArch::Rdna2);
		// Legacy GCN 1 HD 7900 / HD 7700 must NOT misclassify as RDNA 3
		CHECK(ClassifyAmdArch(0x1002u, 0x6798u, "AMD Radeon HD 7900 Series") == AmdArch::Unknown);
		CHECK(ClassifyAmdArch(0x1002u, 0x683Du, "AMD Radeon HD 7700 Series") == AmdArch::Unknown);
		// Capability fallback: unrecognized AMD device with int8 16x16x16 Wave32 coopmat is at least RDNA 3
		CHECK(ClassifyAmdArch(0x1002u, 0x1999u, "Future AMD APU", true) == AmdArch::Rdna3);
		CHECK(ClassifyAmdArch(0x1002u, 0x1999u, "Future AMD APU", false) == AmdArch::Unknown);
		// RDNA 2 discrete (incl. Navi 24 RX 6500 XT / 6400 0x743F) & APUs (Steam Deck LCD 0x163F & OLED Sephiroth 0x1435)
		CHECK(ClassifyAmdArch(0x1002u, 0x73BFu, "AMD Radeon RX 6800 XT") == AmdArch::Rdna2);
		CHECK(ClassifyAmdArch(0x1002u, 0x743Fu, "AMD Radeon RX 6500 XT") == AmdArch::Rdna2);
		CHECK(ClassifyAmdArch(0x1002u, 0x7422u, "AMD Radeon RX 6500M") == AmdArch::Rdna2);
		CHECK(ClassifyAmdArch(0x1002u, 0x163Fu, "AMD Custom GPU 0405") == AmdArch::Rdna2);
		CHECK(ClassifyAmdArch(0x1002u, 0x1435u, "AMD Custom GPU 0405") == AmdArch::Rdna2);
		CHECK(ClassifyAmdArch(0x1002u, 0x1681u, "AMD Radeon 680M Graphics") == AmdArch::Rdna2);
		// Wide-string overload parity for D3D12 DXGI adapter descriptions
		CHECK(ClassifyAmdArch(0x1002u, 0x15BFu, std::wstring_view(L"AMD Radeon 780M Graphics")) == AmdArch::Rdna3);
		CHECK(ClassifyAmdArch(0x1002u, 0x743Fu, std::wstring_view(L"AMD Radeon RX 6500 XT")) == AmdArch::Rdna2);
		CHECK(ClassifyAmdArch(0x1002u, 0x1435u, std::wstring_view(L"AMD Custom GPU 0405")) == AmdArch::Rdna2);
		// Non-AMD vendors always return Unknown even if coopmat flag is set
		CHECK(ClassifyAmdArch(0x10DEu, 0x2684u, "NVIDIA GeForce RTX 4090", true) == AmdArch::Unknown);
		CHECK(ClassifyAmdArch(0x8086u, 0xE20Bu, "Intel Arc B580", true) == AmdArch::Unknown);
	}

	std::printf("%d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}
