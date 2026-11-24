// Stub implementation for builds without Vulkan headers (VulkanRuntime != true).
#include "D3D12VulkanWmma.h"

namespace D3D12VulkanWmma
{
	bool Install(ID3D12Device*, std::string& reason) noexcept
	{
		reason = "no_vulkan_runtime";
		return false;
	}

	bool Active() noexcept { return false; }
	bool ContextReady() noexcept { return false; }
	bool WantsSharedTextures() noexcept { return false; }
	bool CanShareTexture(const D3D12_RESOURCE_DESC&) noexcept { return false; }
	bool EnsureContext(const ContextSettings&) noexcept { return false; }
	void WarmupContextAsync(const ContextSettings&) noexcept {}
	void ResetContext() noexcept {}
	bool TryRecoverAsync(const ContextSettings&) noexcept { return false; }

	bool ScheduleBatch(ID3D12GraphicsCommandList*, const ContextSettings&, const FrameInputs&, const FrameInputs&,
		ID3D12Resource* const*, const xefgd_2d_t*, uint32_t, ID3D12Resource*, D3D12_RESOURCE_STATES, bool) noexcept
	{
		return false;
	}

	bool AppendOutputCopy(ID3D12GraphicsCommandList*, ID3D12Resource*, ID3D12Resource*, D3D12_RESOURCE_STATES) noexcept
	{
		return false;
	}

	void ForgetResource(ID3D12Resource*) noexcept {}

	Status Snapshot() noexcept
	{
		Status s;
		s.built = false;
		s.active = false;
		s.reason = "no_vulkan_runtime";
		return s;
	}
} // namespace D3D12VulkanWmma
