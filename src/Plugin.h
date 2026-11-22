// The dlssg-to-xefg tab of OptiScaler's menu. Its settings are kept in [Plugin.<name>] of OptiScaler.ini when this dll
// is in OptiScaler's custom\<name>\dlssg, without OptiScaler the defaults are used
#pragma once

#include <atomic>
#include <cstdint>

namespace Plugin
{
	// The selected network path
	enum class Path : int32_t
	{
		XmxPort,
		Cuda,
		Fallback, // DP4a model 18
		XmxNative, // XeFG's own XMX network (17), no port, Intel GPUs with XMX only
		Wmma, // Ported XMX model (16) with WMMA INT8 acceleration (AMD RDNA 4 / RDNA 3 & Vulkan Cooperative Matrix)
	};

	// The path's name in the menu and in the log
	const char* PathName(Path path);

	struct Options
	{
		// XeFG's own XMX network on Intel GPUs with XMX, the ported model on DXIL, CUDA tensor cores, or WMMA INT8,
		// or its DP4a model. Unsupported choices fall back the way the menu shows
		std::atomic<Path> path = Path::XmxNative;
		// Lets XeFG composite the UI from the game's HUDless color instead of interpolating it with the scene
		std::atomic_bool hudlessUi = true;
		// Detects sudden camera teleports / cutscene cuts and forces XeFG history reset to prevent smearing
		std::atomic_bool cameraCutDetection = true;
		// Monitors local VRAM via IDXGIAdapter3 and temporarily clamps high MFG multipliers to 2x near budget
		std::atomic_bool vramGuard = true;
		// Caps Vulkan present pacer cadence to (DisplayHz - 3) to stay inside the VRR/G-Sync/FreeSync window
		std::atomic_bool autoVrrPacing = false;
		// Experimental D3D12->Vulkan VK_KHR_cooperative_matrix bridge (default off: D3D12 cs_6_6 WMMA overrides run without
		// allocating a duplicate Vulkan XeFG context or triggering OptiScaler Vulkan_wDx12 hooks on DX12 games)
		std::atomic_bool d3d12VulkanWmmaBridge = false;
		// DLSS Enabler 4.10.0.26 RC1 features:
		// Skips 10 real frames when HUD-less/UI composition state toggles to prevent glitched menu transitions
		std::atomic_bool uiTransitionCooldown = true;
		// GhostBuster anti-ghosting: resets history on sharp snaps (>45 deg), sudden FOV/ADS jumps, or missing camera MV
		std::atomic_bool ghostBuster = true;
		// Allows Vulkan present pacer spinlock to sleep/yield to reduce CPU usage
		std::atomic_bool pacingSpinlockSleep = false;
		// Extended XeFG flag overrides: 0 = Auto (game), 1 = Force Off, 2 = Force On
		std::atomic<int32_t> depthInvertedOverride = 0;
		std::atomic<int32_t> jitteredMvOverride = 0;
		// Ignores game-supplied DLSSG.Reset (fixes games that spam Reset=1 every frame; internal safety resets still apply)
		std::atomic_bool ignoreGameReset = false;
		// Not saved
		std::atomic_bool debugLines = false;
	};

	inline Options options;

	enum class CudaSupport : int32_t
	{
		Unknown, // The startup probe hasn't looked at a GPU yet
		Supported,
		Unsupported,
	};

	inline std::atomic<CudaSupport> cudaSupport = CudaSupport::Unknown;

	enum class XmxSupport : int32_t
	{
		Unknown, // The startup probe hasn't looked at a GPU yet
		Supported, // XeFG picks its own XMX network on this GPU
		Unsupported,
	};

	inline std::atomic<XmxSupport> xmxSupport = XmxSupport::Unknown;

	enum class WmmaSupport : int32_t
	{
		Unknown, // The startup probe hasn't looked at a GPU yet
		Supported, // AMD WMMA INT8 or Vulkan VK_KHR_cooperative_matrix INT8 is available on this GPU
		Unsupported,
	};

	inline std::atomic<WmmaSupport> wmmaSupport = WmmaSupport::Unknown;

	// Which network is running
	enum class Network : int32_t
	{
		Unknown,
		XmxNative, // XeFG's own XMX network (17)
		XmxPort,
		Cuda,
		Wmma, // Ported XMX model (16) accelerated with WMMA INT8
		Fallback, // DP4a model 18, the ported model isn't available
	};

	// One real frame of the frame generator that ran last
	struct Status
	{
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t generatedFrames = 0; // between two real frames
		bool hudless = false;
		Network network = Network::Unknown;
		bool failed = false; // no XeFG context
		bool fp16 = false;   // 16-bit float (scRGB) back buffer, XeFG only takes 8-bit and HDR10
	};

	void Report(const Status& status);
} // namespace Plugin
