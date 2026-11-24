// The decisions of the Vulkan runtime that need neither Vulkan nor Windows: when a batch of generated frames starts,
// which outputs an evaluation records, how a game's formats and camera map to what XeFG reads, and what each reason the
// runtime can't run means. Header only, tests/vulkan_runtime checks it on any machine.
//
// NewBatch, ExecuteRange, the motion vector scale and the projection are MFG-Unlocker's (xess/ngx_provider_policy.h and
// xess/standalone_policy.h, MIT, Copyright (c) 2026 Michael Robles, see NOTICE.md) so that a game gets the same frames
// from this dll as from MFGUnlocker.asi. The rest is this repository's.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

namespace VulkanPolicy
{
	// Generated frames per real frame: 6x
	constexpr uint32_t MaxGenerated = 5;

	// DLSS-G asks for the generated frames of a real frame in turn, DLSSG.MultiFrameIndex 1..count, one evaluation each.
	// A batch starts at index 1. An index that doesn't follow the previous one, or a new count, starts one too, so a
	// skipped first request never reuses the last real frame's batch
	inline bool NewBatch(uint32_t index, uint32_t lastIndex, uint32_t count, uint32_t batchCount, uint64_t batches) noexcept
	{
		return index == 1 || index <= lastIndex || count != batchCount || batches == 0;
	}

	// The outputs an evaluation has XeFG record (xefgd_exec_params_t::firstFrameIndex and frameCount). XeFG's swap chain
	// spreads a batch over its presents and so does this: the batch's first evaluation records the pair's shared work
	// with output 0 and every output up to its own index (a skipped index 1 included), each later one only its own.
	// `through` is how many outputs are ready once this evaluation's are recorded. wholeBatch has the first evaluation
	// record them all (frameCount 0) and the later ones nothing
	struct OutputRange
	{
		bool record = false;
		uint32_t first = 0;
		uint32_t count = 0;
		uint32_t through = 0;

		bool operator==(const OutputRange&) const = default;
	};

	inline OutputRange ExecuteRange(bool newBatch, uint32_t index, uint32_t batchCount, bool wholeBatch) noexcept
	{
		if (!index || index > batchCount)
			return {};

		if (wholeBatch)
			return newBatch ? OutputRange{ true, 0, 0, batchCount } : OutputRange{};

		if (newBatch)
			return { true, 0, index, index };

		return { true, index - 1, 1, index };
	}

	// DLSSG.MultiFrameCount and DLSSG.MultiFrameIndex as the evaluation uses them: the count within 1..MaxGenerated, the
	// index within 1..count
	inline uint32_t ClampCount(uint32_t count) noexcept
	{
		return std::clamp<uint32_t>(count, 1u, MaxGenerated);
	}

	inline uint32_t ClampIndex(uint32_t index, uint32_t count) noexcept
	{
		return std::clamp<uint32_t>(index, 1u, count);
	}

	// NGX DLSS-G's DLSSG.MvecScaleX/Y is already a pixel scale of the motion vectors' extent (1.0 for pixel motion
	// vectors, the extent for normalized ones), unlike Streamline's. XeFG wants pixels of the copy it reads, which is the
	// depth's size, so only the resampling applies. An empty extent keeps the scale
	inline std::pair<float, float> MotionScale(float scaleX, float scaleY, uint32_t motionWidth, uint32_t motionHeight,
		uint32_t targetWidth, uint32_t targetHeight) noexcept
	{
		return { motionWidth ? scaleX * float(targetWidth) / float(motionWidth) : scaleX,
			motionHeight ? scaleY * float(targetHeight) / float(motionHeight) : scaleY };
	}

	// XeFG's camera from inputs without matrices: identity view and a left-handed perspective projection over the output
	// aspect (row major). DLSS-G gives near, far and the vertical field of view; without them it is 60 degrees, near 0.1
	// and far 10000. Near and far swap for inverted depth, an infinite plane is taken to its limit
	inline void Projection(float aspect, float cameraNear, float cameraFar, float verticalFov, bool depthInverted,
		float out[16]) noexcept
	{
		const bool known = cameraNear > 0 && cameraFar > 0 && verticalFov > 0.0001f && verticalFov < 3.13f &&
			cameraNear != cameraFar;

		float nearPlane = known ? cameraNear : 0.1f;
		float farPlane = known ? cameraFar : 10000.0f;

		if (depthInverted)
			std::swap(nearPlane, farPlane);

		const float y = 1.0f / std::tan(0.5f * (known ? verticalFov : 1.0471976f));
		const float x = y / (aspect > 0 ? aspect : 1.0f);

		float depth = 0;
		float offset = 0;

		if (std::isinf(farPlane))
		{
			depth = 1.0f;
			offset = -nearPlane;
		}
		else if (std::isinf(nearPlane))
		{
			depth = 0.0f;
			offset = farPlane;
		}
		else
		{
			depth = farPlane / (farPlane - nearPlane);
			offset = -nearPlane * depth;
		}

		const float matrix[16] = { x, 0, 0, 0, 0, y, 0, 0, 0, 0, depth, 1, 0, 0, offset, 0 };
		std::memcpy(out, matrix, sizeof(matrix));
	}

	// VkFormat and DXGI_FORMAT values, so this header needs neither header. VulkanRuntime.cpp static_asserts that these are
	// the real enum values
	namespace Vk
	{
		constexpr uint32_t R8G8B8A8Unorm = 37;
		constexpr uint32_t R8G8B8A8Srgb = 43;
		constexpr uint32_t B8G8R8A8Unorm = 44;
		constexpr uint32_t B8G8R8A8Srgb = 50;
		constexpr uint32_t A2B10G10R10UnormPack32 = 64;
		constexpr uint32_t R16G16Sfloat = 83;
		constexpr uint32_t R16G16B16A16Sfloat = 97;
		constexpr uint32_t R32G32Sfloat = 103;
		constexpr uint32_t R32G32B32A32Sfloat = 109;
		constexpr uint32_t B10G11R11UfloatPack32 = 122;
	} // namespace Vk

	namespace Dxgi
	{
		constexpr uint32_t Unknown = 0;
		constexpr uint32_t R16G16B16A16Float = 10;
		constexpr uint32_t R32G32Float = 16;
		constexpr uint32_t R10G10B10A2Unorm = 24;
		constexpr uint32_t R11G11B10Float = 26;
		constexpr uint32_t R8G8B8A8Unorm = 28;
		constexpr uint32_t R16G16Float = 34;
		constexpr uint32_t R32Float = 41;
		constexpr uint32_t B8G8R8A8Unorm = 87;
	} // namespace Dxgi

	// The format XeFG reads and writes a game's color image as. XeFG's outputs are typed UAVs, so an sRGB back buffer is
	// seen as its UNORM twin (the bits are the same). Unknown for a format XeFG's swap chain doesn't take
	struct Color
	{
		uint32_t dxgi = Dxgi::Unknown;
		bool srgb = false;
	};

	inline Color ColorFormat(uint32_t vkFormat) noexcept
	{
		switch (vkFormat)
		{
		case Vk::R8G8B8A8Srgb:
			return { Dxgi::R8G8B8A8Unorm, true };
		case Vk::R8G8B8A8Unorm:
			return { Dxgi::R8G8B8A8Unorm, false };
		case Vk::B8G8R8A8Srgb:
			return { Dxgi::B8G8R8A8Unorm, true };
		case Vk::B8G8R8A8Unorm:
			return { Dxgi::B8G8R8A8Unorm, false };
		case Vk::A2B10G10R10UnormPack32:
			return { Dxgi::R10G10B10A2Unorm, false };
		case Vk::R16G16B16A16Sfloat:
			return { Dxgi::R16G16B16A16Float, false };
		case Vk::B10G11R11UfloatPack32:
			return { Dxgi::R11G11B10Float, false };
		default:
			return {};
		}
	}

	// The copy path only supports the color formats understood by XeFG. sRGB and
	// UNORM twins have the same texel layout; RGBA/BGRA and packed HDR do not.
	inline bool CopyCompatible(uint32_t source, uint32_t target) noexcept
	{
		const auto a = ColorFormat(source).dxgi;
		return a != Dxgi::Unknown && a == ColorFormat(target).dxgi;
	}

	// Combined depth/stencil image views cannot be sampled by our float sampler.
	// Callers must supply a single depth or color aspect (Vulkan bits 2 and 1).
	inline bool SampledAspect(uint32_t aspect) noexcept { return aspect == 1 || aspect == 2; }

	// The format of the motion vector copy: two 32-bit floats when the game's are, else two 16-bit ones
	inline uint32_t MotionFormat(uint32_t vkFormat) noexcept
	{
		return vkFormat == Vk::R32G32Sfloat || vkFormat == Vk::R32G32B32A32Sfloat ? Dxgi::R32G32Float : Dxgi::R16G16Float;
	}

	// Which of the three input copy shaders (VulkanInputCopy.h) copies the motion vectors: 1 rg16f, 2 rg32f. The depth
	// copy is 0, r32f
	inline uint32_t MotionCopyShader(uint32_t dxgiMotionFormat) noexcept
	{
		return dxgiMotionFormat == Dxgi::R32G32Float ? 2 : 1;
	}

	// ------------------------------------------------------------------------------------------ tensor cores

	namespace Detail
	{
		// ASCII, any case: the environment's words are never anything else
		inline bool SameWord(std::wstring_view value, std::wstring_view word) noexcept
		{
			if (value.size() != word.size())
				return false;

			for (size_t i = 0; i < value.size(); ++i)
			{
				wchar_t c = value[i];

				if (c >= L'A' && c <= L'Z')
					c = static_cast<wchar_t>(c - L'A' + L'a');

				if (c != word[i])
					return false;
			}

			return true;
		}
	} // namespace Detail

	// A switch of the environment turned off: 0, off or false in any case. An empty value, or any other word, is not
	inline bool IsOffWord(std::wstring_view value) noexcept
	{
		return Detail::SameWord(value, L"0") || Detail::SameWord(value, L"off") || Detail::SameWord(value, L"false");
	}

	// A switch of the environment turned on: 1, on or true in any case
	inline bool IsOnWord(std::wstring_view value) noexcept
	{
		return Detail::SameWord(value, L"1") || Detail::SameWord(value, L"on") || Detail::SameWord(value, L"true");
	}

	// XEFG_TENSOR_CORES_MASK, the layers the route may launch: a hexadecimal number (0x allowed) with a bit per layer of
	// the table, the lowest for the first. Empty, not hexadecimal or with a bit past the table is every layer (`all`), the
	// same reading MFG-Unlocker's XeFGTensorCoresMask has
	inline uint32_t ParseLayerMask(std::wstring_view value, uint32_t all) noexcept
	{
		while (!value.empty() && (value.front() == L' ' || value.front() == L'\t'))
			value.remove_prefix(1);

		while (!value.empty() && (value.back() == L' ' || value.back() == L'\t'))
			value.remove_suffix(1);

		if (value.size() > 2 && value[0] == L'0' && (value[1] == L'x' || value[1] == L'X'))
			value.remove_prefix(2);

		if (value.empty() || value.size() > 8)
			return all;

		uint64_t mask = 0;

		for (const wchar_t c : value)
		{
			uint32_t digit = 0;

			if (c >= L'0' && c <= L'9')
				digit = static_cast<uint32_t>(c - L'0');
			else if (c >= L'a' && c <= L'f')
				digit = static_cast<uint32_t>(c - L'a') + 10;
			else if (c >= L'A' && c <= L'F')
				digit = static_cast<uint32_t>(c - L'A') + 10;
			else
				return all;

			mask = mask * 16 + digit;
		}

		return mask <= all ? static_cast<uint32_t>(mask) : all;
	}

	// The tensor-core route replaces the int8 layers of the ported model, so it runs for the CUDA network path (and for
	// XMX (native), which the D3D12 route also treats as CUDA on a GPU without XeFG's own XMX network, as no Vulkan GPU
	// has it). Fallback (DP4a) runs XeFG's own model and XMX Port runs the layers as SPIR-V programs
	inline bool UseTensorCores(bool portModel, bool cudaPath, bool nativePath) noexcept
	{
		return portModel && (cudaPath || nativePath);
	}

	// The VK_KHR_cooperative_matrix route (AMD RDNA3/RDNA4 WMMA and Intel Arc/Battlemage XMX) replaces the int8 layers of
	// the ported MFG model when the WMMA, XMX Port, XMX (native), or CUDA path is selected on a GPU where the NVIDIA CUDA
	// cubin route isn't active. Fallback (DP4a) keeps XeFG's stock model.
	inline bool UseCoopMatrix(bool portModel, bool xmxPortPath, bool xmxNativePath, bool cudaPath, bool wmmaPath = false) noexcept
	{
		return portModel && (xmxPortPath || xmxNativePath || cudaPath || wmmaPath);
	}

	// Identifies AMD RDNA 4 (gfx1200/gfx1201: RX 9070 XT / 9070 / 9060 XT, PCI DeviceID 0x7550..0x75FF), where 3rd-gen
	// Matrix Cores deliver 2048 INT8 ops/clock/CU (4x DP4a's 512 ops/clock/CU).
	inline bool IsAmdRdna4Device(uint32_t vendorId, uint32_t deviceId, std::string_view deviceName = {}) noexcept
	{
		if (vendorId != 0x1002)
			return false;

		if (deviceId >= 0x7550 && deviceId <= 0x75FF)
			return true;

		return deviceName.find("RX 90") != std::string_view::npos ||
		       deviceName.find("RDNA 4") != std::string_view::npos ||
		       deviceName.find("gfx12") != std::string_view::npos;
	}

	// What a reason code of the cooperative-matrix route means, for the log and the menu
	inline std::string DescribeCoopMatrix(std::string_view reason)
	{
		struct Row
		{
			std::string_view code;
			std::string_view text;
		};

		static constexpr Row CoopMatrixRows[] = {
			{ "environment_off", "XEFG_COOP_MATRIX is off in the environment" },
			{ "no_port", "the ported MFG model couldn't be installed in libxess_fg.dll" },
			{ "already_attempted", "the cooperative-matrix route was already installed on this device" },
			{ "no_extensions", "the game's device has no VK_KHR_cooperative_matrix enabled" },
			{ "device_inputs", "the layer's device lacks what the cooperative-matrix route needs" },
			{ "instance_functions", "the Vulkan loader lacks vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR" },
			{ "device_functions", "the device lacks compute pipeline functions for cooperative matrices" },
			{ "no_int8_tile", "the GPU offers no subgroup SINT8 x SINT8 -> SINT32 cooperative matrix tile" },
			{ "no_s8_matrix_shape", "the GPU offers no subgroup SINT8 x SINT8 -> SINT32 cooperative matrix shape" },
			{ "layout_create", "the cooperative-matrix descriptor or pipeline layout couldn't be created" },
			{ "shader_dir_missing", "the SPV_KHR_cooperative_matrix shaders aren't installed in xefg_mfg_programs\\vulkan\\coopmat" },
			{ "exception", "the cooperative-matrix route failed while installing" },
			{ "driver_fault", "the driver faulted while launching a cooperative-matrix shader, the route turned itself off" },
		};

		for (const auto& row : CoopMatrixRows)
		{
			if (row.code == reason)
				return std::string(row.text);
		}

		return std::string(reason);
	}

	// What a reason code of the tensor-core route means, for the log and the menu: why it isn't installed. The codes are
	// MFG-Unlocker's (xess/vulkan/xvk_tensor.cpp), except the two this dll gives itself. A code this doesn't know is
	// returned as it is
	inline std::string DescribeTensor(std::string_view reason)
	{
		struct Row
		{
			std::string_view code;
			std::string_view text;
		};

		static constexpr Row TensorRows[] = {
			{ "environment_off", "XEFG_TENSOR_CORES is off in the environment" },
			{ "no_port", "the ported MFG model couldn't be installed in libxess_fg.dll, and its int8 layers are what the route "
				"replaces" },
			{ "no_fatbin", "xefg_cuda_layers.fatbin isn't in xefg_mfg_programs, or can't be read" },
			{ "unvalidated_v0_combination", "the route was asked for a combination of options it hasn't been validated with" },
			{ "already_attempted", "the route was already installed on this device" },
			{ "no_extensions", "the game's device has no VK_NVX_binary_import and VK_NVX_image_view_handle: the driver doesn't "
				"offer them, or the device was created before this dll's hooks" },
			{ "device_inputs", "the layer's device lacks what the route needs" },
			{ "instance_functions", "the Vulkan loader lacks vkGetPhysicalDeviceProperties2" },
			{ "not_nvidia", "the GPU isn't an NVIDIA one" },
			{ "adapter_luid", "the driver gives no adapter LUID, so the CUDA device of the GPU can't be found" },
			{ "compute_capability_unknown", "the CUDA driver (nvcuda.dll) doesn't know this GPU, or can't tell it from another" },
			{ "architecture_unsupported", "the GPU isn't an RTX 20, 30, 40 or 50 series one (compute capability 7.5 RTX, 8.6, 8.9 "
				"or 12.0)" },
			{ "texturing_spec_old", "the driver's VK_NVX_binary_import or VK_NVX_image_view_handle is too old (versions 2 and 3 "
				"are needed)" },
			{ "device_functions", "the driver lacks the VK_NVX_binary_import functions" },
			{ "kernel_missing", "xefg_cuda_layers.fatbin lacks a kernel the route launches, it isn't the pinned build" },
			{ "module_bytes", "xefg_cuda_layers.fatbin is empty" },
			{ "module_create", "the driver couldn't import the kernels" },
			{ "module_create_fault", "the driver crashed importing the kernels" },
			{ "function_create", "the driver couldn't create a kernel function" },
			{ "function_create_fault", "the driver crashed creating a kernel function" },
			{ "exception", "the route failed while installing" },
			{ "driver_fault", "the driver crashed while launching a kernel, the route turned itself off" },
		};

		for (const auto& row : TensorRows)
		{
			if (row.code == reason)
				return std::string(row.text);
		}

		return std::string(reason);
	}

	// What a reason code of the runtime means, for the log and the menu. The codes are MFG-Unlocker's where they are its
	// own (a device that can't run XeMFG), the rest are this dll's. A code this doesn't know is returned as it is
	inline std::string Describe(std::string_view reason)
	{
		struct Row
		{
			std::string_view code;
			std::string_view text;
		};

		static constexpr Row Rows[] = {
			{ "not_initialized", "the game hasn't initialized NGX on a Vulkan device yet" },
			{ "pending", "the game's device is known, nothing was asked of it yet" },
			{ "hooks_off", "the Vulkan hooks are off (XEFG_VULKAN_HOOKS=0 or VulkanHooks=0 in OptiScaler.ini)" },
			{ "no_hooks", "this build has no Vulkan runtime, it was built without the Vulkan headers" },
			{ "hooks_failed", "the Vulkan hooks couldn't be installed (see the log)" },
			{ "no_loader", "vulkan-1.dll isn't loaded, the game isn't using Vulkan (yet)" },
			{ "device_not_prepared", "the game created its VkDevice before this dll's hooks were installed, or on a GPU the hooks "
				"refused (see the log); Vulkan features can only be added when the device is created" },
			{ "instance_api_version", "the game's Vulkan instance is older than 1.3 and the loader couldn't raise it" },
			{ "instance_unknown", "the game's VkInstance wasn't seen: it was created before this dll's hooks were installed" },
			{ "instance_functions", "the Vulkan loader lacks the functions the hooks need" },
			{ "device_api_version", "the GPU's driver doesn't offer Vulkan 1.3" },
			{ "subgroup_operations", "the GPU can't do subgroup arithmetic in compute shaders" },
			{ "device_features", "the GPU lacks timeline semaphores, synchronization2, maintenance4, shader integer dot product or "
				"extended storage image formats" },
			{ "push_descriptor", "the GPU lacks VK_KHR_push_descriptor" },
			{ "no_graphics_family", "the game's device has no graphics queue family" },
			{ "intel_adapter", "Intel GPUs aren't supported on Vulkan: XeFG's own network needs Intel's D3D12 driver path" },
			{ "device_functions", "the device lacks functions the runtime needs" },
			{ "feature_chain_unsupported", "a feature request needs editing behind an unknown pNext structure; the game keeps its original device request" },
			{ "prepare_exception", "the hooks failed while preparing the device" },
			{ "no_xefg", "no compatible libxess_fg.dll was found" },
			{ "no_vulkan_models", "the Vulkan models aren't installed or don't check out (xefg_mfg_programs\\vulkan)" },
			{ "xvk_device", "the D3D12-on-Vulkan layer couldn't be created on the game's device" },
			{ "copy_pipelines", "the input copy pipelines couldn't be created" },
		};

		for (const auto& row : Rows)
		{
			if (row.code == reason)
				return std::string(row.text);
		}

		// xefg_ carries what failed after it
		if (reason.starts_with("xefg_"))
			return "XeFG's internal entry points couldn't be used (" + std::string(reason.substr(5)) + ")";

		return std::string(reason);
	}

	// XEFG_VK_SWAPCHAIN_BUFFERS: target minimum swapchain image count for Multi-Frame Generation swapchains
	// (requested >= 6, e.g. Streamline's 6-buffer Vulkan swapchain). Raising 6 -> 8 allows 4x MFG (4 frames/batch)
	// to double-buffer cleanly (8 / 4 = 2) instead of hitting wrap-around phase stalls every alternating batch.
	// Empty or invalid falls back to defaultTarget (8); 0/off/false disables boosting; valid explicit values are 6..12.
	inline uint32_t ParseSwapchainBufferTarget(std::wstring_view value, uint32_t defaultTarget = 8u) noexcept
	{
		while (!value.empty() && (value.front() == L' ' || value.front() == L'\t'))
			value.remove_prefix(1);
		while (!value.empty() && (value.back() == L' ' || value.back() == L'\t'))
			value.remove_suffix(1);
		if (value.empty())
			return defaultTarget;
		if (IsOffWord(value))
			return 0u;
		uint32_t parsed = 0;
		for (const wchar_t c : value)
		{
			if (c < L'0' || c > L'9')
				return defaultTarget;
			parsed = parsed * 10u + static_cast<uint32_t>(c - L'0');
			if (parsed > 12u)
				return defaultTarget;
		}
		return (parsed >= 6u && parsed <= 12u) ? parsed : defaultTarget;
	}

	// Computes the safe boosted swapchain minImageCount within surface capabilities (capMax == 0 means unbounded).
	// Only boosts MFG-sized swapchains (requested >= 6) so non-FG or 2-3 buffer game swapchains stay untouched.
	inline uint32_t RecommendSwapchainImageCount(uint32_t requested, uint32_t capMin, uint32_t capMax,
		uint32_t targetMin = 8u) noexcept
	{
		if (targetMin < 6u || requested < 6u || requested >= targetMin)
			return requested;
		uint32_t desired = std::clamp<uint32_t>(targetMin, 6u, 12u);
		if (capMax > 0u && desired > capMax)
			desired = capMax;
		if (desired < capMin)
			desired = capMin;
		return desired > requested ? desired : requested;
	}

	// CPU-side Vulkan present pacer for recorded-mode Multi-Frame Generation (e.g. Streamline/OptiScaler on GPUs
	// without hardware flip metering and with V-Sync FIFO off).
	// Tracks real-frame batch cadence with a 5-sample median-filtered EMA (rejecting single-frame asset-streaming
	// hitches so they cannot inflate targetStepNs into a multi-frame stutter cascade), drains immediately after a
	// hitch, and bounds each present to its batch-relative slot deadline so timing debt never spills across batches.
	struct PresentPacer
	{
		static constexpr uint64_t MinBatchNs = 2'000'000ull;       // 2 ms (500 base FPS)
		static constexpr uint64_t MaxBatchNs = 250'000'000ull;     // 250 ms (4 base FPS)
		static constexpr uint64_t ActiveWindowNs = 250'000'000ull; // auto-disengage after 250 ms idle
		static constexpr uint64_t MaxWaitNs = 25'000'000ull;       // 25 ms hard safety cap per present
		static constexpr uint64_t TargetRatioNum = 92ull;          // 92% of nominal interval leaves scanout & jitter headroom
		static constexpr uint64_t TargetRatioDen = 100ull;
		static constexpr uint64_t HitchThresholdNum = 130ull;      // >30% over steady cadence = hitch spike, drain immediately
		static constexpr uint64_t HitchThresholdDen = 100ull;
		static constexpr uint32_t MedianWindow = 5u;

		uint64_t lastBatchNs = 0;
		uint64_t batchStartNs = 0;
		uint64_t batchPresentOriginNs = 0;
		uint64_t emaBatchNs = 0;
		uint64_t targetStepNs = 0;
		uint64_t lastPresentNs = 0;
		uint64_t activeUntilNs = 0;
		uint64_t recentDeltas[MedianWindow]{};
		uint32_t recentCount = 0;
		uint32_t recentWrite = 0;
		uint32_t generatedCount = 0;
		uint32_t presentsInBatch = 0;
		bool skipBatch = false;
		// Adaptive GPU-Bound Burst Release hysteresis (f5low-kit DriverPacing=2 / FlipBurst=1):
		// When sustained queue pressure lasts >= 2s, bypass CPU pacing completely until smooth headroom holds for >= 2s.
		static constexpr uint64_t BurstEngageNs = 2'000'000'000ull;
		static constexpr uint64_t BurstReleaseNs = 2'000'000'000ull;

		// Selective Source Cadence Alternation Smoothing for XeFG X3+ (RTX-Encore xefgSourcePacing):
		// Observes source intervals without limiting steady frames; only engages on persistent short/long
		// alternation (>= 4 consecutive sign flips > 15% swing) and caps adjustment to <= 4% above EMA.
		static constexpr uint32_t AltEngageStreak = 4u;
		static constexpr uint64_t AltSwingNum = 15ull;
		static constexpr uint64_t AltSwingDen = 100ull;
		static constexpr uint64_t AltMaxBudgetNum = 104ull;
		static constexpr uint64_t AltMaxBudgetDen = 100ull;

		// Auto VRR (Hz - 3) cap & Queue-Pressure Guard state
		uint32_t displayRefreshHz = 0;
		bool autoVrrEnabled = false;
		uint32_t pressureStreak = 0;
		bool gpuBoundBurst = false;
		uint64_t burstEngageSinceNs = 0;
		uint64_t burstReleaseSinceNs = 0;
		uint64_t prevDeltaNs = 0;
		uint32_t slowShiftStreak = 0;
		uint32_t fastShiftStreak = 0;
		uint64_t fastShiftMaxNs = 0;
		int8_t lastSwingSign = 0;
		uint32_t altStreak = 0;
		bool alternationActive = false;

		void SetDisplayRefreshHz(uint32_t hz) noexcept
		{
			displayRefreshHz = hz;
		}

		void SetAutoVrrEnabled(bool enabled) noexcept
		{
			autoVrrEnabled = enabled;
		}

		void Reset() noexcept
		{
			lastBatchNs = 0;
			batchStartNs = 0;
			batchPresentOriginNs = 0;
			emaBatchNs = 0;
			targetStepNs = 0;
			lastPresentNs = 0;
			activeUntilNs = 0;
			std::memset(recentDeltas, 0, sizeof(recentDeltas));
			recentCount = 0;
			recentWrite = 0;
			generatedCount = 0;
			presentsInBatch = 0;
			skipBatch = false;
			pressureStreak = 0;
			gpuBoundBurst = false;
			burstEngageSinceNs = 0;
			burstReleaseSinceNs = 0;
			prevDeltaNs = 0;
			slowShiftStreak = 0;
			fastShiftStreak = 0;
			fastShiftMaxNs = 0;
			lastSwingSign = 0;
			altStreak = 0;
			alternationActive = false;
		}

		void OnPaused() noexcept
		{
			lastBatchNs = 0;
			batchStartNs = 0;
			batchPresentOriginNs = 0;
			targetStepNs = 0;
			activeUntilNs = 0;
			generatedCount = 0;
			presentsInBatch = 0;
			skipBatch = false;
			pressureStreak = 0;
			gpuBoundBurst = false;
			burstEngageSinceNs = 0;
			burstReleaseSinceNs = 0;
			prevDeltaNs = 0;
			slowShiftStreak = 0;
			fastShiftStreak = 0;
			fastShiftMaxNs = 0;
			lastSwingSign = 0;
			altStreak = 0;
			alternationActive = false;
		}

		void OnBatchStart(uint64_t nowNs, uint32_t count) noexcept
		{
			count = ClampCount(count);
			const uint32_t prevTotal = generatedCount ? (generatedCount + 1u) : 0u;
			const uint32_t prevPresents = presentsInBatch;
			const bool prevIncomplete = lastBatchNs != 0 && targetStepNs != 0 && prevTotal != 0 && prevPresents < prevTotal;

			generatedCount = count;
			presentsInBatch = 0;
			batchStartNs = nowNs;
			batchPresentOriginNs = 0;
			activeUntilNs = nowNs + ActiveWindowNs;
			skipBatch = false;

			if (lastBatchNs != 0 && nowNs > lastBatchNs)
			{
				const uint64_t delta = nowNs - lastBatchNs;
				if (delta >= MinBatchNs && delta <= MaxBatchNs)
				{
					const bool aboveHitch = emaBatchNs != 0 &&
						delta * HitchThresholdDen > emaBatchNs * HitchThresholdNum;
					const bool belowFast = emaBatchNs != 0 &&
						delta * HitchThresholdNum < emaBatchNs * HitchThresholdDen;

					// Asymmetric regime-shift detection (f5low-fg ReadyInterval synergy):
					// - 1 slow frame > 1.3x EMA: treated as a transient hitch (drains without polluting median/EMA).
					// - 2 consecutive slow frames > 1.3x EMA: sustained FPS drop (e.g. 120 FPS menu -> 60 FPS play);
					//   immediately adopt min(prevDeltaNs, delta) so pacing never gets stuck permanently in isHitch.
					// - 3 consecutive fast frames < EMA / 1.3: sustained FPS rise; adopt max of the 3 so 1-2 frame
					//   post-stall catch-up bursts never shrink targetStepNs prematurely.
					bool sustainedSlowShift = false;
					bool sustainedFastShift = false;
					if (aboveHitch)
					{
						++slowShiftStreak;
						fastShiftStreak = 0;
						fastShiftMaxNs = 0;
						if (slowShiftStreak >= 2u && prevDeltaNs != 0)
							sustainedSlowShift = true;
					}
					else
					{
						slowShiftStreak = 0;
						if (belowFast)
						{
							++fastShiftStreak;
							fastShiftMaxNs = std::max(fastShiftMaxNs, delta);
							if (fastShiftStreak >= 3u)
								sustainedFastShift = true;
						}
						else
						{
							fastShiftStreak = 0;
							fastShiftMaxNs = 0;
						}
					}

					const bool isHitch = aboveHitch && !sustainedSlowShift;

					if (sustainedSlowShift)
					{
						const uint64_t shifted = std::min(prevDeltaNs, delta);
						emaBatchNs = shifted;
						for (uint32_t i = 0; i < MedianWindow; ++i)
							recentDeltas[i] = shifted;
						recentCount = MedianWindow;
						recentWrite = 0;
						slowShiftStreak = 0;
						pressureStreak = 0;
					}
					else if (sustainedFastShift)
					{
						const uint64_t shifted = fastShiftMaxNs;
						emaBatchNs = shifted;
						for (uint32_t i = 0; i < MedianWindow; ++i)
							recentDeltas[i] = shifted;
						recentCount = MedianWindow;
						recentWrite = 0;
						fastShiftStreak = 0;
						fastShiftMaxNs = 0;
					}
					else
					{
						// Post-Hitch Median Window Protection: never let a single hitch spike pollute recentDeltas
						// once steady cadence is established, so transient hitches cannot inflate medianDelta.
						if (!isHitch)
						{
							recentDeltas[recentWrite] = delta;
							recentWrite = (recentWrite + 1u) % MedianWindow;
							if (recentCount < MedianWindow)
								++recentCount;
						}

						uint64_t sorted[MedianWindow]{};
						for (uint32_t i = 0; i < recentCount; ++i)
							sorted[i] = recentDeltas[i];
						std::sort(sorted, sorted + recentCount);
						const uint64_t medianDelta = recentCount ? sorted[recentCount / 2u] : delta;

						if (emaBatchNs == 0)
						{
							emaBatchNs = medianDelta;
						}
						else if (!isHitch)
						{
							if (medianDelta < emaBatchNs || medianDelta > emaBatchNs * 2ull || medianDelta * 2ull < emaBatchNs)
							{
								// Adapt quickly downward (higher FPS) or on large framerate shifts
								emaBatchNs = (emaBatchNs + medianDelta) / 2ull;
							}
							else
							{
								// Smooth 0.25 EMA on the median-filtered cadence
								emaBatchNs = (emaBatchNs * 3ull + medianDelta) / 4ull;
							}
						}
					}

					// Selective Source Cadence Alternation Smoothing (X3+ only: count >= 2).
					// Detects persistent short/long frame alternation and uses the pair-average (capped at +4%)
					// so generated frames keep even spacing; stays completely idle on steady frames or X2.
					uint64_t effectiveBatchNs = emaBatchNs;
					if (count >= 2u && !isHitch && prevDeltaNs != 0 && emaBatchNs != 0)
					{
						const uint64_t diff = (delta > prevDeltaNs) ? (delta - prevDeltaNs) : (prevDeltaNs - delta);
						const int8_t sign = (delta > prevDeltaNs) ? 1 : ((delta < prevDeltaNs) ? -1 : 0);
						if (sign != 0 && diff * AltSwingDen >= emaBatchNs * AltSwingNum && (lastSwingSign == 0 || sign == -lastSwingSign))
						{
							if (altStreak < 16u)
								++altStreak;
							lastSwingSign = sign;
						}
						else
						{
							altStreak = (altStreak > 0u) ? (altStreak - 1u) : 0u;
							if (altStreak == 0u)
								lastSwingSign = 0;
						}
						alternationActive = (altStreak >= AltEngageStreak);
						if (alternationActive)
						{
							const uint64_t pairAvgNs = (prevDeltaNs + delta) / 2ull;
							const uint64_t maxBudgetNs = (emaBatchNs * AltMaxBudgetNum) / AltMaxBudgetDen;
							if (pairAvgNs > effectiveBatchNs)
								effectiveBatchNs = std::min(pairAvgNs, maxBudgetNs);
						}
					}
					else
					{
						altStreak = 0;
						lastSwingSign = 0;
						alternationActive = false;
					}
					prevDeltaNs = delta;

					const uint64_t totalFrames = static_cast<uint64_t>(count) + 1ull;
					targetStepNs = (effectiveBatchNs * TargetRatioNum) / (totalFrames * TargetRatioDen);

					// Auto VRR (Hz - 3) floor: keep generated presents inside the display's VRR window
					if (autoVrrEnabled && displayRefreshHz >= 48u)
					{
						const uint64_t vrrMinStepNs = 1'000'000'000ull / static_cast<uint64_t>(displayRefreshHz - 3u);
						if (targetStepNs < vrrMinStepNs)
							targetStepNs = std::min(vrrMinStepNs, MaxWaitNs);
					}

					// Queue-Pressure Guard + Adaptive GPU-Bound Burst Release hysteresis (2s engage / 2s release):
					// At 4x-6x MFG (prevTotal = 4..6), the render thread routinely pipelines 1-2 frames ahead of the
					// present thread; only treat incomplete batches as queue pressure when <50% of presents completed.
					const bool prevSevereIncomplete = prevIncomplete && (prevPresents * 2u < prevTotal);
					const bool queuePressure = prevSevereIncomplete || (emaBatchNs != 0 && delta * 100ull > emaBatchNs * 135ull);
					if (queuePressure)
					{
						if (pressureStreak < 8u)
							++pressureStreak;
						burstReleaseSinceNs = 0;
						if (!gpuBoundBurst)
						{
							if (burstEngageSinceNs == 0)
								burstEngageSinceNs = nowNs;
							else if (nowNs >= burstEngageSinceNs + BurstEngageNs)
								gpuBoundBurst = true;
						}
					}
					else
					{
						if (pressureStreak > 0u)
							--pressureStreak;
						burstEngageSinceNs = 0;
						if (gpuBoundBurst)
						{
							if (burstReleaseSinceNs == 0)
								burstReleaseSinceNs = nowNs;
							else if (nowNs >= burstReleaseSinceNs + BurstReleaseNs)
							{
								gpuBoundBurst = false;
								burstReleaseSinceNs = 0;
							}
						}
					}

					// On a hitch spike, sustained queue pressure, or active GPU-bound burst release, drain without CPU sleep
					if (isHitch || pressureStreak >= 2u || gpuBoundBurst)
					{
						lastPresentNs = 0;
						skipBatch = true;
					}
					else if (prevSevereIncomplete)
					{
						lastPresentNs = 0;
					}
				}
				else
				{
					// Long hitch or pause: reset cadence state until two valid batches arrive
					emaBatchNs = 0;
					targetStepNs = 0;
					lastPresentNs = 0;
					recentCount = 0;
					recentWrite = 0;
					prevDeltaNs = 0;
					slowShiftStreak = 0;
					fastShiftStreak = 0;
					fastShiftMaxNs = 0;
					lastSwingSign = 0;
					altStreak = 0;
					alternationActive = false;
				}
			}

			if (lastPresentNs > nowNs)
				lastPresentNs = nowNs;

			lastBatchNs = nowNs;
		}

		// Returns nanoseconds to wait before calling vkQueuePresentKHR and advances lastPresentNs to the target time.
		uint64_t ComputePresentWaitNs(uint64_t nowNs, bool vsyncFifo) noexcept
		{
			if (vsyncFifo)
			{
				lastPresentNs = nowNs;
				return 0;
			}

			if (skipBatch || generatedCount == 0 || targetStepNs == 0 || nowNs > activeUntilNs)
			{
				if (skipBatch && generatedCount > 0 && nowNs <= activeUntilNs)
					++presentsInBatch;
				lastPresentNs = nowNs;
				return 0;
			}

			// Guard against runaway pacing if a menu/overlay presents without new FG batches
			if (presentsInBatch >= generatedCount + 2u)
			{
				lastPresentNs = nowNs;
				return 0;
			}

			// Stale-present cancellation (NSS-OFG): if the batch has already exceeded a full real-frame interval,
			// drain remaining generated presents immediately so stale waits never add input latency.
			if (batchStartNs != 0 && emaBatchNs != 0 && nowNs > batchStartNs + emaBatchNs)
			{
				skipBatch = true;
				++presentsInBatch;
				lastPresentNs = nowNs;
				return 0;
			}

			const uint32_t slotIndex = presentsInBatch++;
			if (slotIndex == 0)
				batchPresentOriginNs = nowNs;

			if (lastPresentNs == 0 || nowNs <= lastPresentNs)
			{
				lastPresentNs = nowNs;
				return 0;
			}

			// Anchor slot deadlines to when the batch's first present actually starts (batchPresentOriginNs) rather than
			// render-thread batchStartNs, preventing 4x-6x bursts when GPU work delays G1 relative to OnBatchStart.
			uint64_t desiredNs = lastPresentNs + targetStepNs;
			const uint64_t originNs = batchPresentOriginNs ? batchPresentOriginNs : batchStartNs;
			if (originNs != 0 && slotIndex > 0)
			{
				const uint64_t slotDeadlineNs = originNs + static_cast<uint64_t>(slotIndex) * targetStepNs;
				if (desiredNs > slotDeadlineNs)
					desiredNs = slotDeadlineNs;
			}

			if (nowNs >= desiredNs)
			{
				// On time or late: present immediately, never accumulate timing debt
				lastPresentNs = nowNs;
				return 0;
			}

			uint64_t waitNs = desiredNs - nowNs;
			const uint64_t capNs = std::min(targetStepNs, MaxWaitNs);
			if (waitNs > capNs)
				waitNs = capNs;

			lastPresentNs = nowNs + waitNs;
			return waitNs;
		}
	};
} // namespace VulkanPolicy
