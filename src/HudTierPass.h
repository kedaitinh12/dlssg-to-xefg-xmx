// D3D12 GPU compute passes for the 3-Tier Adaptive HUD Protection system:
// - Tier 1 Guard: 1/16-pixel |Real - Hudless| diff counter (non-blocking ring-buffered readback, 0 GPU stall)
// - Tier 2 Composite: 3x3 dilated |Real - Hudless| > tolerance HUD composite onto generated output (xmx_hud_composite)
// - Tier 3 Hybrid (Tone-Map Match + Static HUD Mask):
//     * Lazy-allocated inline 256-bin log-luminance median LUT (MFG-Unlocker V1.7.2) for dynamic/HDR-mismatched UI
//     * Temporal per-pixel stability map + static-HUD composite for fixed UI (crosshairs, static HUD bars/text)
#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

class HudTierPass
{
public:
	HudTierPass() = default;
	~HudTierPass();

	HudTierPass(const HudTierPass&) = delete;
	HudTierPass& operator=(const HudTierPass&) = delete;

	// Initializes pipelines once per device (dynamically loads d3dcompiler_47.dll; returns false gracefully if absent).
	bool EnsureInitialized(ID3D12Device* device, uint32_t width, uint32_t height) noexcept;

	// True once compute pipelines and buffers are ready.
	bool Available() const noexcept { return pipelinesReady_ && resourcesReady_; }

	// Returns whether a resource's format can be bound directly as a D3D12 RWTexture2D<float4> UAV.
	static bool IsUavCompatibleFormat(DXGI_FORMAT format) noexcept;

	// Returns whether a backbuffer format is supported by Tier 4 Tone-Map Match (10-bit or 8-bit UNORM, non-sRGB).
	static bool IsToneMapMatchSupported(DXGI_FORMAT backbufferFormat) noexcept;

	// Polls the 2-frame-old ring-buffered readback slot (non-blocking, zero GPU flush/stall).
	// Returns true and writes fraction in [0, 1] and whether it came from Tier 4 tone-matched HUDless.
	bool PollDiffFraction(float& outFraction, bool& outFromToneMatched) noexcept;
	bool PollStaticFraction(float& outFraction) noexcept;

	// Advances the ring-buffer slot once per real frame (call at start of CaptureRealFrame).
	void BeginRealFrame() noexcept;

	// Tier 1 Guard (analyze-only): counts 1/16 sampled pixels where |Real - Hudless| > tolerance.
	// Preserves resource states (realState / hudlessState).
	void RecordHudDiffGuard(ID3D12GraphicsCommandList* cmdList,
		ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
		ID3D12Resource* hudlessTex, D3D12_RESOURCE_STATES hudlessState) noexcept;

	// Tier 2 Composite: runs 3x3 dilated |Real - Hudless| > tolerance and writes HUD pixels from Real onto Output
	// (or magenta when debugMagenta is true). Also updates the 1/16 diff counter when updateCounter is true.
	// Preserves resource states (realState / hudlessState / outputState).
	void RecordHudDiffComposite(ID3D12GraphicsCommandList* cmdList,
		ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
		ID3D12Resource* hudlessTex, D3D12_RESOURCE_STATES hudlessState,
		ID3D12Resource* outputTex, D3D12_RESOURCE_STATES outputState,
		bool updateCounter, bool debugMagenta, bool fromToneMatched = false) noexcept;

	// Tier 4 Tone-Map Match (lazy-allocated, 0 VRAM/cost when unused; async PSO compile = zero 500ms timeout):
	// toneMapMode: 0 = Auto (1D 256-bin LUT, auto-promotes to 3D Bilateral Grid on residual spatial mismatch),
	//              1 = 1D 256-bin LUT only, 2 = 3D Bilateral Grid (amortized fit every 4 real frames or on forceRefit).
	// Writes the tone-matched HUD-less texture into ping-pong slot (0 or 1), leaving it in matchedOutputState.
	// Returns the matched texture pointer on success, or nullptr if unavailable/unsupported.
	ID3D12Resource* RecordToneMapMatch(ID3D12GraphicsCommandList* cmdList,
		uint32_t frameSlot, bool wantShared,
		ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
		ID3D12Resource* hudlessTex, D3D12_RESOURCE_STATES hudlessState,
		D3D12_RESOURCE_STATES matchedOutputState,
		int32_t toneMapMode = 0, bool forceRefit = false) noexcept;

	// Active Tone-Map Matcher mode for UI telemetry: 0 = Off, 1 = 1D LUT, 2 = 3D Bilateral Grid.
	int32_t ActiveToneMapMode() const noexcept { return activeToneMapMode_; }

	// Opt-in split-screen debug comparison (0% overhead when not called):
	// Left of splitPos shows realTex (original frame), divider is cyan, right keeps outputTex.
	void RecordSplitCompare(ID3D12GraphicsCommandList* cmdList,
		ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
		ID3D12Resource* outputTex, D3D12_RESOURCE_STATES outputState,
		float splitPos) noexcept;

	// Tier 3 Update: compares CurrReal vs PrevReal on each real frame, updating the R8_UINT stability texture
	// and counting 1/16 sampled static-HUD pixels (count >= 4). Preserves resource states.
	void RecordStaticHudUpdate(ID3D12GraphicsCommandList* cmdList,
		ID3D12Resource* currRealTex, D3D12_RESOURCE_STATES currState,
		ID3D12Resource* prevRealTex, D3D12_RESOURCE_STATES prevState,
		bool resetHistory) noexcept;

	// Tier 3 Composite: copies pixels with staticCount >= 4 from CurrReal onto Output (or magenta in debug mode).
	// Preserves resource states (currState / outputState).
	void RecordStaticHudComposite(ID3D12GraphicsCommandList* cmdList,
		ID3D12Resource* currRealTex, D3D12_RESOURCE_STATES currState,
		ID3D12Resource* outputTex, D3D12_RESOURCE_STATES outputState,
		bool debugMagenta) noexcept;

	// Fused Tier 3 Hybrid Composite: combines 3x3 dilated |Real - Hudless| diff and Static-HUD Mask (staticCount >= 4)
	// in a single compute pass (1/2 VRAM bandwidth, 0 intermediate UAV barrier).
	// Software motion-vector unjitter for XMX Port / CUDA / WMMA (model 16) when game submits jittered MVs:
	// removes (prevJitter - curJitter) / mvecScale in-place on our private mvecs copy so model 16 never needs
	// unported XEFGD_INIT_FLAG_JITTERED_MV shaders (e.g. key 35112441167526893749578).
	bool CanUnjitterMotionVectors(DXGI_FORMAT format) const noexcept;
	void RecordUnjitterMotionVectors(ID3D12GraphicsCommandList* cmdList,
		ID3D12Resource* mvecsTex, D3D12_RESOURCE_STATES mvecsState,
		uint32_t baseX, uint32_t baseY, uint32_t sizeX, uint32_t sizeY,
		float deltaRawX, float deltaRawY) noexcept;

	bool HasHybridComposite() const noexcept { return hybridRootSig_.Get() != nullptr && psoHybridComposite_.Get() != nullptr; }
	void RecordHybridHudComposite(ID3D12GraphicsCommandList* cmdList,
		ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
		ID3D12Resource* hudlessTex, D3D12_RESOURCE_STATES hudlessState,
		ID3D12Resource* outputTex, D3D12_RESOURCE_STATES outputState,
		bool updateCounter, bool debugMagenta, bool fromToneMatched = false) noexcept;

private:
	static constexpr uint32_t kReadbackSlots = 4; // 4-frame readback ring prevents races on 3+ frames-in-flight DLSS-G pacer queues
	static constexpr uint32_t kDescriptorBlocks = 96; // 96-block ring * 5 descriptors = 480 descriptors (safe across 8+ frames in flight at 6x MFG)
	static constexpr uint32_t kDescriptorsPerBlock = 5; // t0, t1, [t2], u0, u1
	static constexpr uint32_t kToneMatchDescSlots = 32; // 32-slot ring * 3 descriptors = 96 descriptors (never mutates in-flight descriptors)

	bool CreatePipelines(ID3D12Device* device) noexcept;
	bool CreateResources(ID3D12Device* device, uint32_t width, uint32_t height) noexcept;
	struct ToneMatchCompileJob
	{
		Microsoft::WRL::ComPtr<ID3D12Device> device;
		Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
		Microsoft::WRL::ComPtr<ID3D12PipelineState> pso[5];
		std::atomic<bool> cancel{ false };
		std::atomic<uint8_t> state{ 0 }; // 0 = compiling, 1 = ready, 2 = failed
	};

	struct BilateralCompileJob
	{
		Microsoft::WRL::ComPtr<ID3D12Device> device;
		Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
		Microsoft::WRL::ComPtr<ID3D12PipelineState> pso[4]; // clear, accum, solve, apply
		std::atomic<bool> cancel{ false };
		std::atomic<uint8_t> state{ 0 }; // 0 = compiling, 1 = ready, 2 = failed
	};

	bool EnsureToneMapMatchReady(ID3D12Device* device, DXGI_FORMAT backbufferFormat, bool wantShared) noexcept;
	bool EnsureBilateralMatchReady(ID3D12Device* device, DXGI_FORMAT backbufferFormat, bool wantShared) noexcept;
	void StopToneMatchCompile() noexcept;
	void StopBilateralCompile() noexcept;
	void ReleaseToneMapMatchResources() noexcept;
	void ReleaseResources() noexcept;
	uint32_t AllocDescriptorBlock() noexcept;

	Microsoft::WRL::ComPtr<ID3D12Device> device_;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoDiffGuard_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoDiffComposite_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoStaticUpdate_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoStaticComposite_;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> unjitterRootSig_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoUnjitterMv2_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoUnjitterMv4_;
	bool typedUavLoadSupported_ = false;
	bool unjitterFormatSupported_[4]{}; // R16G16F, R32G32F, R16G16B16A16F, R32G32B32A32F
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoSplitCompare_;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> hybridRootSig_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoHybridComposite_;

	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvUavHeap_;
	uint32_t descSize_ = 0;
	uint32_t descBlockCursor_ = 0;

	// Tier 3 temporal stability map (R8_UINT, width x height)
	Microsoft::WRL::ComPtr<ID3D12Resource> staticCountTex_;
	// 16-byte GPU counter buffer (uint[0] = diffCount, uint[1] = staticCount)
	Microsoft::WRL::ComPtr<ID3D12Resource> counterGpu_;
	Microsoft::WRL::ComPtr<ID3D12Resource> counterZeroUpload_;
	Microsoft::WRL::ComPtr<ID3D12Resource> counterReadback_[kReadbackSlots];
	const uint32_t* mappedReadback_[kReadbackSlots]{};
	bool readbackHasDiff_[kReadbackSlots]{};
	bool readbackDiffFromToneMatch_[kReadbackSlots]{};
	bool readbackHasStatic_[kReadbackSlots]{};

	// Tier 4 Tone-Map Match lazy resources (allocated only if Tier 4 actually activates; 0 VRAM on normal games)
	Microsoft::WRL::ComPtr<ID3D12RootSignature> toneMatchRootSig_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> toneMatchPso_[5]; // clear, accumulate, median, fill, apply
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> toneMatchHeap_;
	uint32_t toneMatchDescCursor_ = 0;
	Microsoft::WRL::ComPtr<ID3D12Resource> toneMatchHistogram_;   // 3 * 256 * 1024 * 4 = 3 MiB
	Microsoft::WRL::ComPtr<ID3D12Resource> toneMatchLut_;         // 3 * 256 * 4 = 3 KiB
	Microsoft::WRL::ComPtr<ID3D12Resource> toneMatchOutput_[2];   // Ping-pong matched HUDless (slot 0 / 1)
	DXGI_FORMAT toneMatchFormat_ = DXGI_FORMAT_UNKNOWN;
	bool toneMatchShared_ = false;
	std::shared_ptr<ToneMatchCompileJob> toneMatchCompileJob_;
	std::thread toneMatchCompileThread_;
	bool toneMatchPipeAttempted_ = false;
	bool toneMatchPipeReady_ = false;
	bool toneMatchResReady_ = false;

	// Tier 4 3D Bilateral-Grid Tone-Map Matcher (async compiled, amortized fit every 4 real frames)
	Microsoft::WRL::ComPtr<ID3D12RootSignature> bilateralRootSig_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> bilateralPso_[4]; // clear, accum, solve, apply
	Microsoft::WRL::ComPtr<ID3D12Resource> bilateralStats_;       // gridW * gridH * 8 * 22 * 4 bytes (~169 KiB @ 1440p)
	Microsoft::WRL::ComPtr<ID3D12Resource> bilateralModel_;       // gridW * gridH * 8 * 12 * 4 bytes (~92 KiB @ 1440p)
	uint32_t bilateralCellsX_ = 0;
	uint32_t bilateralCellsY_ = 0;
	std::shared_ptr<BilateralCompileJob> bilateralCompileJob_;
	std::thread bilateralCompileThread_;
	bool bilateralPipeAttempted_ = false;
	bool bilateralPipeReady_ = false;
	bool bilateralResReady_ = false;
	bool bilateralHasValidModel_ = false;
	uint32_t bilateralFrameCounter_ = 0;
	uint32_t toneMatchHighResidualCount_ = 0;
	bool autoPromotedBilateral_ = false;
	int32_t activeToneMapMode_ = 0;

	uint32_t readbackSlot_ = 0;
	uint64_t realFrameCount_ = 0;
	uint64_t staticReadbackFrameCounter_ = 0;

	uint32_t width_ = 0;
	uint32_t height_ = 0;
	bool initAttempted_ = false;
	bool pipelinesReady_ = false;
	bool resourcesReady_ = false;
};
