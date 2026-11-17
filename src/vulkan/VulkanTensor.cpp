// See VulkanTensor.h.
#include "VulkanTensor.h"

#include "../Log.h"
#include "../MfgKernels.h"
#include "../MfgPort/Cuda/cuda_layers.h"
#include "../MfgPort/Cuda/cuda_route.h"
#include "../MfgPort/Sha256.h"
#include "../VulkanPrograms.h"
#include "VulkanPolicy.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace
{
	// The fatbin the route was validated with: resource 7300 of MFG-Unlocker v1.7.0's release dll, which tools/
	// GenerateXeFgPrograms.ps1 pins and extracts, and resource 3903 of the 1.7.1 MFGUnlocker.asi builds
	constexpr char PinnedSha256[] = "2ffae52e7878ae7e374ff953b864204f079c3199dc904b2ce7ecf8b37ee130f7";
	constexpr std::streamoff MaxFatbinBytes = 16 * 1024 * 1024;
	constexpr uint32_t AllLayers = (1u << std::size(xefg_cuda::kLayers)) - 1;

	std::wstring Environment(const wchar_t* name)
	{
		wchar_t value[64]{};
		const DWORD length = GetEnvironmentVariableW(name, value, static_cast<DWORD>(std::size(value)));

		// Longer than the buffer is as good as not set, no switch is that long
		return length && length < std::size(value) ? std::wstring(value, length) : std::wstring();
	}

	struct Fatbin
	{
		std::vector<unsigned char> bytes;
		std::string sha256; // lower case hex, empty when it couldn't be computed
	};

	// Empty when the file isn't there, can't be read, is empty or bigger than any build of the kernels
	Fatbin ReadFatbin(const std::filesystem::path& path)
	{
		Fatbin fatbin;
		std::ifstream file(path, std::ios::binary);

		if (!file)
			return fatbin;

		file.seekg(0, std::ios::end);
		const std::streamoff size = file.tellg();

		if (!file || size <= 0 || size > MaxFatbinBytes)
			return fatbin;

		file.seekg(0, std::ios::beg);
		std::vector<unsigned char> bytes(static_cast<size_t>(size));
		file.read(reinterpret_cast<char*>(bytes.data()), size);

		if (!file || file.gcount() != size)
			return fatbin;

		sha256_util::Digest digest{};

		if (sha256_util::Compute(bytes.data(), bytes.size(), digest))
			fatbin.sha256 = sha256_util::LowerHex(digest);

		fatbin.bytes = std::move(bytes);

		return fatbin;
	}

	// The route logs through xefg_cuda::Report, which goes here
	void RouteLog(const char* stage, const std::string& fields) noexcept
	{
		try
		{
			Log::Info("{} {}", stage, fields);
		}
		catch (...)
		{
		}
	}

	bool Refused(std::string& reason, const char* code)
	{
		reason = code;
		xefg_cuda::ReportVulkan(xefg_cuda::State::Unavailable, 0, 0, 0, code);

		return false;
	}
} // namespace

bool VulkanTensor::Wanted()
{
	return !VulkanPolicy::IsOffWord(Environment(L"XEFG_TENSOR_CORES"));
}

bool VulkanTensor::Install(ID3D12Device* device, std::string& reason)
{
	reason.clear();

	try
	{
		xefg_cuda::SetLog(&RouteLog);

		if (!Wanted())
		{
			reason = "environment_off";
			xefg_cuda::ReportVulkan(xefg_cuda::State::Disabled, 0, 0, 0, reason.c_str());
			Log::Info("Vulkan tensor cores: off ({})", VulkanPolicy::DescribeTensor(reason));

			return false;
		}

		if (!device)
			return Refused(reason, "device_inputs");

		const auto path = MfgKernels::ProgramsDirectory() / L"xefg_cuda_layers.fatbin";
		const Fatbin fatbin = ReadFatbin(path);

		if (fatbin.bytes.empty())
		{
			Refused(reason, "no_fatbin");
			Log::Info("Vulkan tensor cores: unavailable, {}: {}", VulkanPolicy::DescribeTensor(reason),
				VulkanPrograms::PathText(path));

			return false;
		}

		if (fatbin.sha256 != PinnedSha256)
		{
			// The D3D12 route launches whatever the file holds too: a build with CudaRoot makes its own kernels
			Log::Warn("Vulkan tensor cores: {} isn't the pinned fatbin (SHA-256 {}, pinned {}), its kernels are launched all the "
				"same", VulkanPrograms::PathText(path), fatbin.sha256.empty() ? "unknown" : fatbin.sha256, PinnedSha256);
		}

		xefg_cuda::SetLayerMask(VulkanPolicy::ParseLayerMask(Environment(L"XEFG_TENSOR_CORES_MASK"), AllLayers));

		xvk::TensorOptions options;
		options.fullBarriers = VulkanPolicy::IsOnWord(Environment(L"XEFG_TENSOR_CORES_VULKAN_BARRIERS"));

		if (!xvk::InstallTensorCores(device, fatbin.bytes.data(), fatbin.bytes.size(), options, reason))
		{
			if (reason.empty())
				reason = "exception";

			xefg_cuda::ReportVulkan(xefg_cuda::State::Unavailable, 0, 0, 0, reason.c_str());
			Log::Info("Vulkan tensor cores: unavailable, {}", VulkanPolicy::DescribeTensor(reason));

			return false;
		}

		const auto status = xefg_cuda::VulkanSnapshot();
		Log::Info("Vulkan tensor cores: ready on compute capability {}.{}, {} layers, {} ({} bytes{})", status.major, status.minor,
			status.layers, VulkanPrograms::PathText(path), fatbin.bytes.size(), options.fullBarriers ? ", full barriers" : "");

		return true;
	}
	catch (...)
	{
		return Refused(reason, "exception");
	}
}
