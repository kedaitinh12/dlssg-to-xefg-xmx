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

	// Preferred network path fallback order per GPU vendor:
	//   AMD (0x1002):    WMMA -> XMX Port -> Fallback (DP4a)
	//   NVIDIA (0x10DE): CUDA -> XMX Port -> Fallback (DP4a)
	//   Intel (0x8086):  XMX (native) -> XMX Port -> Fallback (DP4a)
	//   Unknown vendor:  XMX (native) -> XMX Port -> Fallback (DP4a)
	Path PreferredPathForGpu(uint32_t vendorId, bool xmxReady, bool cudaReady, bool wmmaReady, bool portReady) noexcept;

	// Safely resolves a requested path for the active GPU; if XMX (native) is requested on a GPU that cannot run it
	// (e.g. AMD/NVIDIA default or stale INI), falls back through that vendor's own priority chain.
	Path ResolveEffectivePath(Path requested, uint32_t vendorId, bool xmxReady, bool cudaReady, bool wmmaReady, bool portReady) noexcept;

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
		// Experimental D3D12->Vulkan VK_KHR_cooperative_matrix bridge (default on; disable in OptiScaler menu or via
		// XEFG_D3D12_VULKAN_WMMA=0 to fall back to D3D12 cs_6_6 WMMA overrides if a game freezes or conflicts with hooks)
		std::atomic_bool d3d12VulkanWmmaBridge = true;
		// DLSS Enabler 4.10.0.26 RC1 features (opt-in OFF by default for maximum uninterrupted smoothness & FPS):
		// Skips 10 real frames when HUD-less/UI composition state toggles to prevent glitched menu transitions
		std::atomic_bool uiTransitionCooldown = false;
		// GhostBuster anti-ghosting: resets history on sharp snaps (>45 deg), sudden FOV/ADS jumps, or missing camera MV
		std::atomic_bool ghostBuster = false;
		// Allows Vulkan present pacer spinlock to sleep/yield to reduce CPU usage
		std::atomic_bool pacingSpinlockSleep = false;
		// Extended XeFG flag overrides: 0 = Auto (game), 1 = Force Off, 2 = Force On
		std::atomic<int32_t> depthInvertedOverride = 0;
		std::atomic<int32_t> jitteredMvOverride = 0;
		// UI alpha mode for Vulkan separate UI texture: 0 = Auto, 1 = Straight, 2 = Premultiplied (RTX-Encore xefgUiAlpha)
		std::atomic<int32_t> uiAlphaMode = 0;
		// 3-Tier Adaptive HUD Protection mode: 0 = Auto (3-Tier), 1 = Tier 1 Native Only, 2 = Tier 2 HUD-Diff Composite, 3 = Tier 3 Hybrid, 4 = Off (Pure XeFG / Max FPS)
		std::atomic<int32_t> hudTierMode = 0;
		// Tier 4 Tone-Map Matcher mode: 0 = Auto (1D LUT -> 3D Bilateral Grid), 1 = 1D LUT only, 2 = 3D Bilateral Grid
		std::atomic<int32_t> toneMapMatchMode = 0;
		// Debug overlay (not saved): paints detected Tier 2 / Tier 3 HUD mask pixels magenta
		std::atomic_bool hudDebugMagenta = false;
		// Diagnostic (not saved; 0% overhead when off): hides game HUD-less input to isolate pre-FG judder vs HUD compositing
		std::atomic_bool debugHideHudless = false;
		// Debug split-screen comparison (not saved; 0% overhead when off): left = original real frame, right = generated/HUD-protected output
		std::atomic_bool hudSplitCompare = false;
		std::atomic<float> hudSplitPos = 0.5f;
		// Enables runtime file logging (dlssg_to_xefg_intel_is_better.log); turn OFF for zero disk I/O
		std::atomic_bool logging = true;
		// Opt-in DX12 DRED crash diagnostics (AutoBreadcrumbs + PageFault on TDR; 0% overhead when OFF; requires game restart)
		std::atomic_bool forceDred = false;
		// Opt-in lightweight crash minidump (.dmp via System32 dbghelp.dll on unhandled crash; 0% overhead when OFF)
		std::atomic_bool crashMiniDump = false;
		// Debug per-stage GPU time profiler (not saved; 0% overhead when off)
		std::atomic_bool gpuProfiler = false;
		// Ignores game-supplied DLSSG.Reset (fixes games that spam Reset=1 every frame; internal safety resets still apply)
		std::atomic_bool ignoreGameReset = false;
		// Not saved
		std::atomic_bool debugLines = false;
	};

	inline Options options;

	enum class ActiveApi : int32_t
	{
		Unknown = 0,
		D3D12 = 1,
		Vulkan = 2,
	};

	inline std::atomic<ActiveApi> activeApi{ ActiveApi::Unknown };

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
	inline std::atomic<uint32_t> gpuVendorId{ 0 };
	inline std::atomic<int32_t> fallbackFromPath{ -1 };
	inline std::atomic<int32_t> fallbackToPath{ -1 };

	inline void RecordPathFallback(Path requested, Path resolved) noexcept
	{
		if (requested != resolved)
		{
			fallbackFromPath.store(static_cast<int32_t>(requested), std::memory_order_relaxed);
			fallbackToPath.store(static_cast<int32_t>(resolved), std::memory_order_relaxed);
		}
	}

	inline void ClearPathFallback() noexcept
	{
		fallbackFromPath.store(-1, std::memory_order_relaxed);
		fallbackToPath.store(-1, std::memory_order_relaxed);
	}

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
		int32_t hudTier = 0; // HudTier::ActiveTier (0=Interpolated, 1=Native, 2=Diff-Comp, 3=Hybrid)
		Network network = Network::Unknown;
		bool failed = false; // no XeFG context
		bool fp16 = false;   // 16-bit float (scRGB) back buffer, XeFG only takes 8-bit and HDR10
		uint32_t requestedFrames = 0; // requested by game/Streamline before Adaptive VRAM guard clamp
		bool vramGuarded = false;     // Adaptive VRAM guard actively clamping MFG multiplier to 2x
		uint32_t vramUsageMiB = 0;    // local VRAM usage sampled by Adaptive VRAM guard
		uint32_t vramBudgetMiB = 0;   // local VRAM budget sampled by Adaptive VRAM guard
		int32_t toneMapActiveMode = 0; // 0 = Off, 1 = 1D LUT, 2 = 3D Bilateral Grid
		bool profilerActive = false;   // true when opt-in GPU profiler has valid readings
		float totalGpuMs = 0.0f;       // smoothed total Evaluate GPU time (ms)
		float interpGpuMs = 0.0f;      // smoothed XeFG interpolation GPU time (ms)
		float hudGpuMs = 0.0f;         // smoothed HUD / Tone-Map Match GPU time (ms)
	};

	void Report(const Status& status);

	// Games whose Streamline HUDLess buffer differs from Backbuffer across the 3D scene (post-process/tone-map mismatch),
	// causing XeFG's XEFGD_UI_MODE_BACKBUFFER_HUDLESS UI extractor to overwrite interpolated frames with the real backbuffer.
	bool IsKnownBrokenHudlessGame() noexcept;
	void ApplyBrokenHudlessQuirkOnce() noexcept;
} // namespace Plugin
