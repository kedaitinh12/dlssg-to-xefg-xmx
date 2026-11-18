// Checks src/vulkan/VulkanPolicy.h, the decisions of the Vulkan runtime that need neither Vulkan nor Windows: when a batch
// of generated frames starts, which outputs an evaluation records, how the game's formats and camera map to what XeFG
// reads, and what each reason code means. The values it expects are written out here from MFG-Unlocker's own policy
// (xess/ngx_provider_policy.h and xess/standalone_policy.h) and from the formats' registry values, not computed from the
// header's code.
#include "vulkan/VulkanPolicy.h"

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

	std::printf("%d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}
