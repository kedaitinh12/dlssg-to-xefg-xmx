#pragma once

#include "NGX.h"
#include "XeFgEnabler.h"
#include "XeFG.h"
#include "HudTierPolicy.h"
#include "HudTierPass.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <memory>

using Microsoft::WRL::ComPtr;

// One DLSSG feature instance, driving one XeFG context
class FrameGenerator
{
public:
	// 6x
	static constexpr uint32_t MaxFrames = 5;

	explicit FrameGenerator(ID3D12Device* device) : device(device) {}

	// Called once per generated frame, DLSSG.MultiFrameIndex 1 starts a new real frame
	NGXResult Evaluate(ID3D12GraphicsCommandList* cmdList, const NGXParameter& params);

	// Probes XeFG's own XMX network and the CUDA tensor-core route once, as soon as a device is available. Fills
	// Plugin::xmxSupport and Plugin::cudaSupport so the menu can disable paths that would fail before the first FG
	// context. Called from NGX's Init and, as a fallback, from CreateContext.
	static void ProbeSupport(ID3D12Device* device);

private:
	// What the context was created for, it's recreated when any of it changes
	struct Settings
	{
		uint32_t width = 0;
		uint32_t height = 0;
		DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
		uint32_t initFlags = 0;
		uint32_t uiMode = XEFGD_UI_MODE_AUTO;
		int32_t model = 0;

		bool operator==(const Settings&) const = default;
	};

	// Copies of a real frame's inputs, XeFG reads them after the game moved on
	struct Frame
	{
		ComPtr<ID3D12Resource> backbuffer;
		ComPtr<ID3D12Resource> hudless;
		ID3D12Resource* matchedHudless = nullptr; // non-owning ping-pong texture owned by HudTierPass (Tier 4 only)
		ComPtr<ID3D12Resource> depth;
		ComPtr<ID3D12Resource> spareDepth;  // provkey.1: alternating depth format cache (Dying Light 2 / The Beast)
		ComPtr<ID3D12Resource> depthScratch;
		ComPtr<ID3D12Resource> mvecs;
		ComPtr<ID3D12Resource> spareMotion; // provkey.1: alternating motion format cache
		xefgd_frame_t data{};
		bool valid = false;
	};

	struct Context
	{
		xefgd_handle_t handle = nullptr;
		ComPtr<ID3D12Resource> outputs[MaxFrames];
		xefgd_2d_t outputBases[MaxFrames]{};

		// Ping-ponged between the previous and the current real frame
		Frame frames[2];
		uint32_t current = 0;

		// The previous and the current real frame can be interpolated
		bool pairValid = false;
		uint32_t batchGeneratedThrough = 0;
		bool directOutput = false;
		// Tracks whether cached frame/output copies were allocated with D3D12_HEAP_FLAG_SHARED for Vulkan WMMA
		bool sharedTexturesMode = false;
		// Set if D3D12 XeFG::Execute fallback wrote to shared textures during async Vulkan bridge init
		bool wmmaFallbackTouchedShared = false;

		// Camera Cut & Teleport Detector + GhostBuster state (prevents smearing/ghosting on cuts, snaps, FOV jumps)
		float prevCameraPos[3]{};
		float prevCameraFwd[3]{};
		float prevCameraFov = 0.0f;
		bool hasPrevCamera = false;
		bool hasPrevFov = false;
		// Cooldown window (kQuiet) preventing back-to-back synthetic CameraCut/GhostBuster resets on flashes/shake
		uint32_t syntheticResetQuietFrames = 0;

		// UI / HUD-less transition cooldown (DLSS Enabler XeFG: skip 10 real frames on state toggle)
		uint32_t uiTransitionCooldown = 0;
		bool hasPrevRawHudless = false;
		bool prevRawHudless = false;

		// Long-pause / Alt+Tab history reset (NSS-OFG): resets history if real-frame gap exceeds 500 ms
		uint64_t lastRealFrameTickMs = 0;
		// Throttles Tier 1 Native HudDiffGuardCS to 1/16 real frames after initial 8-frame probe (>93% less GPU cost)
		uint64_t hudDiffGuardFrameCounter = 0;

		// Stall circuit breaker & device-loss latch (f5low-fg ROBUSTNESS):
		// - Real D3D12 DEVICE_REMOVED latches pass-through permanently until feature/device recreation.
		// - Transient stalls recover only after >= 8 evaluations AND >= 300 ms since the last failure.
		uint32_t consecutiveExecuteFailures = 0;
		uint32_t stallCooldownEvaluations = 0;
		uint64_t stallCooldownUntilMs = 0;
		bool deviceLostLatched = false;

		~Context();
	};

	enum class InterpolateResult
	{
		Interpolated,
		Skipped, // no usable pair, the real frame is shown instead
		Failed,
	};

	bool CreateContext();
	bool PollVramGuard() noexcept;
	void CaptureRealFrame(ID3D12GraphicsCommandList* cmdList, const NGXParameter& params);
	InterpolateResult Interpolate(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* output,
		D3D12_RESOURCE_DESC outputDesc, uint32_t frameIndex, uint32_t frameCount,
		float interpolationTime = 0.0f, bool callerRecording = false);

	ComPtr<ID3D12Device> device;
	Settings settings;
	std::shared_ptr<Context> context;
	bool createFailed = false; // retried when the settings change
	bool portedModelInitFailed = false; // auto-fallback to DP4a model 18 if ported model 16 fails XeFG::Init (e.g. VKD3D-Proton)

	// The menu path last seen, and the route launches since the context or the path changed. A few generated frames
	// later they say whether the selected path really runs on tensor cores
	int32_t loggedPath = -1;
	uint64_t routeLaunches = 0;
	uint32_t routeFrames = 0;
	bool routeReported = false;

	// Adaptive VRAM Guard: cached IDXGIAdapter3 + hysteresis state (zero per-frame factory overhead)
	ComPtr<IDXGIAdapter3> vramAdapter;
	bool vramAdapterChecked = false;
	bool vramGuardActive = false;
	uint32_t vramPollCounter = 0;

	// 3-Tier Adaptive HUD Protection (persists across XeFG context recreations when tier changes)
	HudTier::StateMachine hudTierState;
	HudTierPass hudTierPass;
	HudTier::Decision hudDecision{};
	HudTier::ActiveTier loggedHudTier = HudTier::ActiveTier::Interpolated;
	bool hasLoggedHudTier = false;
	uint32_t vramUsageMiB = 0;
	uint32_t vramBudgetMiB = 0;

	// Opt-in Per-Stage GPU Time Profiler (lazy-allocated ONLY when Plugin::options.gpuProfiler is enabled; 0% overhead when off)
	struct GpuProfiler
	{
		static constexpr uint32_t kSlots = 4; // 4-slot ring prevents reading an in-flight GPU slot on 3+ frame DLSS-G pacer queues
		static constexpr uint32_t kQueriesPerSlot = 6; // [0]=start, [1]=afterCapture, [2]=beforeInterp, [3]=afterInterp, [4]=beforePostHud, [5]=end
		ComPtr<ID3D12QueryHeap> queryHeap;
		ComPtr<ID3D12Resource> readback;
		const uint64_t* mapped = nullptr;
		uint64_t freqHz = 0;
		uint32_t slot = 0;
		uint32_t framesRecorded = 0;
		bool slotValid[kSlots]{};
		uint64_t lastConsumedEnd[kSlots]{};
		bool initAttempted = false;
		bool ready = false;
		bool hasReading = false;
		float totalMs = 0.0f;
		float interpMs = 0.0f;
		float hudMs = 0.0f;

		~GpuProfiler()
		{
			if (readback && mapped)
				readback->Unmap(0, nullptr);
		}

		bool EnsureReady(ID3D12Device* dev) noexcept;
		void PollAndAdvance() noexcept;
		void Mark(ID3D12GraphicsCommandList* cmdList, uint32_t idx) noexcept;
		void ResolveSlot(ID3D12GraphicsCommandList* cmdList) noexcept;
	};
	GpuProfiler gpuProfiler;
};
