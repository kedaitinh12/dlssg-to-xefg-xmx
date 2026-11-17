// The calls src/vulkan/xvk_tensor.cpp makes into src/MfgPort/Cuda (cuda_route.h and cuda_layers.h), written out so that
// check_syntax.sh proves the declarations match them. xvk_tensor.cpp is MFG-Unlocker's file and needs the whole of its
// layer's headers, which the stubs don't have: the Windows build is the check of the file itself, this one is of what this
// repository adds underneath it. Nothing is linked or run.
#include "MfgPort/Cuda/cuda_layers.h"

#include <iterator>
#include <string>

void TensorApiCheck(const LUID& luid, UINT x, UINT y, UINT z, XefgCudaParams& params)
{
	// TensorRoute::Install
	int major{};
	int minor{};
	std::string name;
	const bool allowed = xefg_cuda::AdapterCapability(luid, major, minor, name) &&
		xefg_cuda::SupportedArchitecture(major, minor, name);

	for (const auto kernel : xefg_cuda::kKernels)
		(void)kernel;

	xefg_cuda::ReportVulkan(xefg_cuda::State::Active, major, minor, uint32_t(std::size(xefg_cuda::kLayers)), "");
	xefg_cuda::Report("cuda_route_vk", std::string("\"state\":\"active\""));

	// TensorRoute::Fault
	xefg_cuda::ReportVulkan(xefg_cuda::State::Unavailable, major, minor, 0, "driver_fault");

	// TensorRoute::TryLaunch
	const uint8_t id = xefg_cuda::LayerForSha256("");

	if (id && xefg_cuda::LayerEnabled(id))
	{
		const auto& layer = xefg_cuda::kLayers[id - 1];
		const bool decoder = layer.family == xefg_cuda::Family::Decoder;
		const bool flow = layer.family == xefg_cuda::Family::Flow;
		const uint32_t constants[12]{};
		uint32_t gridX{};
		uint32_t gridY{};

		if (xefg_cuda::DispatchShapeOk(layer, x, y, z) && (constants[0] & layer.constants) == layer.constants)
		{
			xefg_cuda::FillLaunch(layer, x, y, constants, params, gridX, gridY);
			xefg_cuda::CountVulkan(1, 0);
		}
		else
		{
			xefg_cuda::Report("cuda_vk_fallback", std::string(layer.info.program) + layer.info.kernel);
			xefg_cuda::CountVulkan(0, 1);
		}

		(void)decoder;
		(void)flow;
		(void)layer.kernel;
		(void)layer.gridZ;
	}

	(void)allowed;
}

// What VulkanRuntime.cpp and Plugin.cpp read of the Vulkan route
void TensorStatusCheck()
{
	const xefg_cuda::Status status = xefg_cuda::VulkanSnapshot();
	(void)status.state;
	(void)status.reason;
	(void)status.launches;
	(void)status.fallbacks;
	(void)xefg_cuda::VulkanActive();
	(void)xefg_cuda::VulkanLaunched();
	xefg_cuda::SetLayerMask(0x1FFFFu);
}
