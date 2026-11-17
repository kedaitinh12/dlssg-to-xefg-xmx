// See VulkanCoopMat.h.
#include "VulkanCoopMat.h"

#include "../Log.h"
#include "../MfgKernels.h"
#include "../MfgPort/Cuda/cuda_layers.h"
#include "../MfgPort/Cuda/cuda_route.h"
#include "../VulkanPrograms.h"
#include "VulkanPolicy.h"

#include <filesystem>
#include <iterator>
#include <string>

namespace
{
	constexpr uint32_t AllLayers = (1u << std::size(xefg_cuda::kLayers)) - 1;

	std::wstring Environment(const wchar_t* name)
	{
		wchar_t value[64]{};
		const DWORD length = GetEnvironmentVariableW(name, value, static_cast<DWORD>(std::size(value)));
		return length && length < std::size(value) ? std::wstring(value, length) : std::wstring();
	}

	bool Refused(std::string& reason, const char* code)
	{
		reason = code;
		xefg_cuda::ReportCoopMatrix(xefg_cuda::State::Unavailable, 0, 0, 0, 0, code);
		return false;
	}
} // namespace

bool VulkanCoopMat::Wanted()
{
	return !VulkanPolicy::IsOffWord(Environment(L"XEFG_COOP_MATRIX"));
}

bool VulkanCoopMat::AllowIntelAdapter()
{
	return Wanted() && !VulkanPolicy::IsOffWord(Environment(L"XEFG_INTEL_VULKAN"));
}

bool VulkanCoopMat::Install(ID3D12Device* device, std::string& reason)
{
	reason.clear();

	try
	{
		if (!Wanted())
		{
			reason = "environment_off";
			xefg_cuda::ReportCoopMatrix(xefg_cuda::State::Disabled, 0, 0, 0, 0, reason.c_str());
			Log::Info("Vulkan cooperative matrix: off ({})", VulkanPolicy::DescribeCoopMatrix(reason));
			return false;
		}

		if (!device)
			return Refused(reason, "device_inputs");

		const auto dir = VulkanPrograms::DirectoryFor(MfgKernels::ProgramsDirectory()) / L"coopmat";
		xefg_cuda::SetCoopMatrixLayerMask(VulkanPolicy::ParseLayerMask(Environment(L"XEFG_COOP_MATRIX_MASK"), AllLayers));

		xvk::CoopMatrixOptions options;
		options.fullBarriers = VulkanPolicy::IsOnWord(Environment(L"XEFG_COOP_MATRIX_VULKAN_BARRIERS"));

		if (!xvk::InstallCoopMatrix(device, dir.wstring().c_str(), options, reason))
		{
			if (reason.empty())
				reason = "exception";

			xefg_cuda::ReportCoopMatrix(xefg_cuda::State::Unavailable, 0, 0, 0, 0, reason.c_str());
			Log::Info("Vulkan cooperative matrix: unavailable, {}", VulkanPolicy::DescribeCoopMatrix(reason));
			return false;
		}

		const auto status = xefg_cuda::CoopMatrixSnapshot();
		const uint32_t tileM = static_cast<uint32_t>(status.major);
		const uint32_t tileN = static_cast<uint32_t>((status.minor >> 8) & 0xFF);
		const uint32_t tileK = static_cast<uint32_t>(status.minor & 0xFF);
		Log::Info("Vulkan cooperative matrix: ready ({}x{}x{} s8->s32), {} layer modules in {}{}",
			tileM, tileN, tileK, status.layers, VulkanPrograms::PathText(dir),
			options.fullBarriers ? ", full barriers" : "");

		return true;
	}
	catch (...)
	{
		return Refused(reason, "exception");
	}
}
