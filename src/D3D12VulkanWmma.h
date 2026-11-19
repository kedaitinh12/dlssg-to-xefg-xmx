// D3D12 -> Vulkan VK_KHR_cooperative_matrix (hardware WMMA INT8) bridge for AMD RDNA 4 / RDNA 3.
//
// In D3D12 games, HLSL Shader Model 6.6 only exposes DP4A (dot4add_i8packed, 512 INT8 ops/clk/CU).
// This bridge creates a private Vulkan 1.3 device on the same physical GPU (matched by adapter LUID,
// immune to DXGI spoofing), mounts xvk's D3D12-on-Vulkan runtime with VK_KHR_cooperative_matrix
// (16x16x16 SINT8xSINT8->SINT32, 2048 INT8 ops/clk/CU on RDNA 4), and shares FrameGenerator's
// input/output textures + timeline fence across D3D12 and Vulkan via VK_KHR_external_memory_win32
// and VK_KHR_external_semaphore_win32.
//
// Only 1 D3D12->Vulkan signal and 1 Vulkan->D3D12 wait occur per entire multi-frame batch (all 28
// XeFG layers and all 1..5 generated frames execute inside a single Vulkan command buffer).
//
// Nothing here includes Vulkan headers so non-Vulkan builds compile against D3D12VulkanWmmaOff.cpp.
#pragma once

#include "XeFG.h"

#include <d3d12.h>
#include <cstdint>
#include <string>

namespace D3D12VulkanWmma
{
	struct Status
	{
		bool built = false;
		bool active = false;
		uint32_t tileM = 0;
		uint32_t tileN = 0;
		uint32_t tileK = 0;
		uint32_t coopLayers = 0;
		uint64_t batches = 0;
		uint64_t fallbacks = 0;
		std::string reason;
	};

	struct FrameInputs
	{
		ID3D12Resource* backbuffer = nullptr;
		ID3D12Resource* hudless = nullptr;
		ID3D12Resource* depth = nullptr;
		ID3D12Resource* mvecs = nullptr;
		xefgd_frame_t data{};
	};

	struct ContextSettings
	{
		uint32_t width = 0;
		uint32_t height = 0;
		DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
		uint32_t initFlags = 0;
		uint32_t uiMode = XEFGD_UI_MODE_AUTO;
		int32_t model = 0;
	};

	// Initializes the private Vulkan 1.3 + VK_KHR_cooperative_matrix device matching `d3dDevice` by LUID,
	// installs xvk + VulkanCoopMat (17 hardware WMMA SPIR-V layers), creates the shared timeline fence,
	// and hooks ID3D12CommandQueue::ExecuteCommandLists. Safe to call once per device.
	bool Install(ID3D12Device* d3dDevice, std::string& reason) noexcept;

	// True when the D3D12->Vulkan hardware WMMA bridge is ready and can accelerate XeFG batches.
	bool Active() noexcept;

	// True when a texture created by FrameGenerator should include D3D12_HEAP_FLAG_SHARED so it can be
	// imported into Vulkan via VK_KHR_external_memory_win32.
	bool WantsSharedTextures() noexcept;

	// Returns true if the given D3D12 texture descriptor can be created with D3D12_HEAP_FLAG_SHARED and
	// imported into Vulkan (no ALLOW_DEPTH_STENCIL, 2D texture, supported DXGI format, single-sampled).
	bool CanShareTexture(const D3D12_RESOURCE_DESC& desc) noexcept;

	// Ensures the Vulkan-side XeFG context matches `settings` (recreates on resolution/format/flag change).
	bool EnsureContext(const ContextSettings& settings) noexcept;

	// Destroys the Vulkan-side XeFG context (e.g. when FrameGenerator recreates its context).
	void ResetContext() noexcept;

	// Schedules a full XeFG batch (all `frameCount` generated frames) on the Vulkan VK_KHR_cooperative_matrix
	// queue to run immediately when `cmdList` is submitted to ID3D12CommandQueue::ExecuteCommandLists.
	// `sharedOutputs[0..frameCount-1]` must be D3D12_HEAP_FLAG_SHARED textures; `firstGameOutput` is the
	// game's destination texture for frameIndex == 1 (copied back on D3D12 right after Vulkan signals done).
	// Returns false if any resource cannot be shared/imported, so caller falls back cleanly to D3D12.
	bool ScheduleBatch(
		ID3D12GraphicsCommandList* cmdList,
		const ContextSettings& settings,
		const FrameInputs& prev,
		const FrameInputs& next,
		ID3D12Resource* const* sharedOutputs,
		const xefgd_2d_t* outputBases,
		uint32_t frameCount,
		ID3D12Resource* firstGameOutput,
		D3D12_RESOURCE_STATES outputState) noexcept;

	// Appends a post-copy (`sharedOutput` -> `gameOutput`) for `frameIndex > 1` when OptiScaler records
	// multiple generated frames onto the same `cmdList`. Returns true if `cmdList` has an active pending
	// Vulkan batch (so caller skips its inline Copy); returns false if `cmdList` is a separate command list
	// (caller then performs normal Copy from `sharedOutput`, which was already populated by frameIndex == 1).
	bool AppendOutputCopy(
		ID3D12GraphicsCommandList* cmdList,
		ID3D12Resource* sharedOutput,
		ID3D12Resource* gameOutput,
		D3D12_RESOURCE_STATES outputState) noexcept;

	// Drops any cached Vulkan import handles for a D3D12 resource that is about to be released.
	void ForgetResource(ID3D12Resource* resource) noexcept;

	Status Snapshot() noexcept;
} // namespace D3D12VulkanWmma
