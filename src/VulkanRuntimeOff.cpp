// The Vulkan runtime of a build without the Vulkan headers: there is none. VulkanRuntime.h says what the exports and the
// menu do then. dlssg-to-xefg.vcxproj builds src/vulkan/VulkanRuntime.cpp instead when it finds the headers.
#include "VulkanRuntime.h"

bool VulkanRuntime::Built()
{
	return false;
}

const VulkanProvider::Api* VulkanRuntime::Table()
{
	return nullptr;
}

void VulkanRuntime::Start(const char*)
{
}

void VulkanRuntime::SetHooksEnabled(bool)
{
}

VulkanRuntime::Status VulkanRuntime::GetStatus()
{
	return {};
}

std::string VulkanRuntime::ModelName()
{
	return {};
}

VulkanRuntime::TensorStatus VulkanRuntime::GetTensorStatus()
{
	return {};
}

VulkanRuntime::CoopMatrixStatus VulkanRuntime::GetCoopMatrixStatus()
{
	return {};
}
