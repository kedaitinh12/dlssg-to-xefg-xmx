// D3D12 GPU compute passes for the 4-Tier Adaptive HUD Protection system:
// - Tier 1 Guard: 1/16-pixel |Real - Hudless| diff counter (non-blocking ring-buffered readback, 0 GPU stall)
// - Tier 2 Composite: 3x3 dilated |Real - Hudless| > tolerance HUD composite onto generated output (xmx_hud_composite)
// - Tier 3 Static HUD Mask: Temporal per-pixel stability map across real frames + selective static-HUD composite
// - Tier 4 Tone-Map Match (lazy-allocated inline): 256-bin log-luminance median LUT (ported from MFG-Unlocker V1.7.2)
#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>

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

	// Tier 4 Tone-Map Match (lazy-allocated, 0 VRAM/cost when unused):
	// Builds a per-channel 256-bin log-luminance median LUT from (hudlessTex -> realTex) and writes the tone-matched
	// HUD-less texture into ping-pong slot (0 or 1), leaving it in outputState (NON_PIXEL|PIXEL_SHADER_RESOURCE).
	// Returns the matched texture pointer on success, or nullptr if unavailable/unsupported.
	ID3D12Resource* RecordToneMapMatch(ID3D12GraphicsCommandList* cmdList,
		uint32_t frameSlot, bool wantShared,
		ID3D12Resource* realTex, D3D12_RESOURCE_STATES realState,
		ID3D12Resource* hudlessTex, D3D12_RESOURCE_STATES hudlessState,
		D3D12_RESOURCE_STATES matchedOutputState) noexcept;

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

private:
	static constexpr uint32_t kReadbackSlots = 2;
	static constexpr uint32_t kDescriptorBlocks = 8; // 8 dispatches * 4 descriptors = 32 descriptors per frame
	static constexpr uint32_t kDescriptorsPerBlock = 4; // t0, t1, u0, u1

	bool CreatePipelines(ID3D12Device* device) noexcept;
	bool CreateResources(ID3D12Device* device, uint32_t width, uint32_t height) noexcept;
	bool EnsureToneMapMatchReady(ID3D12Device* device, DXGI_FORMAT backbufferFormat, bool wantShared) noexcept;
	void ReleaseToneMapMatchResources() noexcept;
	void ReleaseResources() noexcept;
	uint32_t AllocDescriptorBlock() noexcept;

	Microsoft::WRL::ComPtr<ID3D12Device> device_;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoDiffGuard_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoDiffComposite_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoStaticUpdate_;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> psoStaticComposite_;

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
	Microsoft::WRL::ComPtr<ID3D12Resource> toneMatchHistogram_;   // 3 * 256 * 1024 * 4 = 3 MiB
	Microsoft::WRL::ComPtr<ID3D12Resource> toneMatchLut_;         // 3 * 256 * 4 = 3 KiB
	Microsoft::WRL::ComPtr<ID3D12Resource> toneMatchOutput_[2];   // Ping-pong matched HUDless (slot 0 / 1)
	DXGI_FORMAT toneMatchFormat_ = DXGI_FORMAT_UNKNOWN;
	bool toneMatchShared_ = false;
	bool toneMatchPipeAttempted_ = false;
	bool toneMatchPipeReady_ = false;
	bool toneMatchResReady_ = false;
	uint32_t readbackSlot_ = 0;
	uint64_t realFrameCount_ = 0;

	uint32_t width_ = 0;
	uint32_t height_ = 0;
	bool initAttempted_ = false;
	bool pipelinesReady_ = false;
	bool resourcesReady_ = false;
};
