// 3-Tier Adaptive HUD Protection & Fallback Policy (shared across DX12, Vulkan, and unit tests).
//
// Tier 1 (NativeHudless): Uses the game's native DLSSG.HUDLess / DLSSG.UI plane through XeFG UI composition,
//   continuously monitored by a lightweight 1/16-pixel GPU Pixel-Diff Guard (zero GPU stall).
//
// Tier 2 (HudDiffComposite): XeFG interpolates clean HUD-less scene in UI_MODE_AUTO and a lightweight 3x3
//   dilated compute shader (|Real - Hudless| > tolerance) composites HUD pixels on top.
//
// Tier 3 (Hybrid: Tone-Map Matcher + Static HUD Mask):
//   Combines two complementary protection engines, each handling the UI elements it is best suited for:
//   - HUDless Tone-Map Matcher (lazy-allocated inline 256-bin log-luminance median LUT, 0 cost/VRAM when unused):
//     Handles dynamic, animated, and semi-transparent UI whenever the game provides a HUDLess buffer whose
//     color/exposure/HDR format mismatches Backbuffer (e.g. RenoDX HDR or Witcher 3 RGBA16F vs HDR10).
//   - Temporal Static HUD Mask:
//     Handles fixed UI elements (crosshairs, static health/ammo bars, minimap borders, fixed HUD text) —
//     reinforcing static elements alongside Tone-Map Matcher, and taking over completely when a game has NO
//     HUDLess buffer, a duplicate/broken HUDLess, or when Tone-Map Matcher itself fails (>38% diff for 6 frames).
#pragma once

#include <cstdint>

namespace HudTier
{
	// User-selectable mode in OptiScaler menu / OptiScaler.ini (HudTierMode).
	enum class Mode : int32_t
	{
		Auto = 0,               // Automatic 3-tier selection & escalation (default)
		Tier1NativeOnly = 1,    // Strictly Tier 1 (Native XeFG HUDless/UI) or interpolated if unavailable
		Tier2DiffComposite = 2, // Prefer Tier 2 (Custom HUD-Diff Composite), escalate to Tier 3 if HUDless breaks/missing
		Tier3Hybrid = 3,        // Force Tier 3 Hybrid (Tone-Map Match for dynamic/HDR HUD + Static Mask for fixed HUD)
		Off = 4,                // Pure XeFG fast-path: disables all HudTierPass compute/readback passes for maximum FPS
	};

	// Effective HUD protection tier active on the current frame.
	enum class ActiveTier : int32_t
	{
		Interpolated = 0,       // No separate HUD protection; UI interpolated with scene
		Tier1Native = 1,        // Tier 1: Native XeFG HUDless / UI plane (with GPU Pixel-Diff Guard)
		Tier2DiffComposite = 2, // Tier 2: Custom GPU HUD-Diff Composite (xmx_hud_composite style)
		Tier3Hybrid = 3,        // Tier 3: Hybrid fallback (Tone-Map Match + Static HUD Mask, or Static Mask alone)
	};

	inline const char* ModeName(Mode mode) noexcept
	{
		switch (mode)
		{
		case Mode::Auto: return "Auto (3-Tier)";
		case Mode::Tier1NativeOnly: return "Tier 1 (Native HUDless)";
		case Mode::Tier2DiffComposite: return "Tier 2 (HUD-Diff Composite)";
		case Mode::Tier3Hybrid: return "Tier 3 (Hybrid: Tone-Match + Static)";
		case Mode::Off: return "Off (Pure XeFG / Max FPS)";
		default: return "Auto (3-Tier)";
		}
	}

	inline const char* ActiveTierShortName(ActiveTier tier) noexcept
	{
		switch (tier)
		{
		case ActiveTier::Tier1Native: return "HUD T1: Native";
		case ActiveTier::Tier2DiffComposite: return "HUD T2: Diff-Comp";
		case ActiveTier::Tier3Hybrid: return "HUD T3: Hybrid";
		default: return "UI interpolated";
		}
	}

	struct Inputs
	{
		Mode mode = Mode::Auto;
		bool hudlessUiEnabled = true;            // Master "HUDless UI" checkbox
		bool hasGameHudless = false;             // Game supplied a non-null DLSSG.HUDLess
		bool hudlessStructurallyValid = false;   // Passed size/format/MSAA/subrect/alias/known-broken checks
		bool hudlessFormatMismatch = false;      // Same size/subrect, but HUDLess format differs from Backbuffer (e.g. RGBA16F vs HDR10)
		bool hasSeparateUiPlane = false;         // Vulkan DLSSG.UI separate plane available & valid
		bool gpuComputeAvailable = true;         // GPU compute pass available
		bool toneMapMatchAvailable = false;      // Inline DX12 256-bin Tone-Map Matcher available (false on Vulkan)
		bool preferTier2ForNativeIssues = false; // True if XeFG native UI composition previously stalled/failed
		bool preferNativeForWmma = false;        // True on active D3D12->Vulkan WMMA bridge: Auto keeps hardware WMMA
	};

	struct Decision
	{
		ActiveTier activeTier = ActiveTier::Interpolated;
		bool useXeFgNativeUi = false;         // Pass HUDless/UI to XeFG's own UI composition mode
		bool useHudlessAsSceneSource = false; // Feed clean (or tone-matched) HUDless as scene source while Diff-Composite runs
		bool runToneMapMatch = false;         // Tier 3 dynamic/HDR engine: inline 256-bin log-luminance median LUT (lazy allocated)
		bool runHudDiffGuard = false;         // Run lightweight 1/16-pixel Real-vs-Hudless diff counter
		bool runHudDiffComposite = false;     // Run 3x3 HUD-diff composite onto generated output (Tier 2 or Tier 3 Hybrid)
		bool runStaticHudUpdate = false;      // Tier 3 static engine: temporal stability map update on real frames
		bool runStaticHudComposite = false;   // Tier 3 static engine: static HUD composite onto generated output
		const char* reason = "interpolated";
	};

	class StateMachine
	{
	public:
		static constexpr float kBrokenDiffFraction = 0.38f;
		static constexpr float kFullScreenOverlayFraction = 0.60f;
		// 0.0005% (~18 px at 1440p, <=1 sampled pixel at 1/16 stride): distinguishes a truly empty/duplicate
		// HUDLess from a minimal 1 px crosshair (~30-100 px), with zero dead zone below kHealthyMinFraction.
		static constexpr float kEmptyDiffFraction = 0.000005f;
		static constexpr float kHealthyMinFraction = 0.000005f;
		// Stronger threshold (0.02%, ~737 px at 1440p) required to latch everHealthy_ so a few stray tonemap/sharpen
		// rounding pixels on a duplicate backbuffer can never permanently disable empty-HUDLess detection.
		static constexpr float kEverHealthyMinFraction = 0.0002f;
		static constexpr float kHealthyMaxFraction = 0.30f;
		static constexpr uint32_t kBrokenConfirmFrames = 6u;
		static constexpr uint32_t kHealthyOverlayConfirmFrames = 14u;
		static constexpr uint32_t kEverHealthyConfirmFrames = 4u;
		static constexpr uint32_t kEmptyConfirmFrames = 30u;
		static constexpr uint32_t kRecoveryConfirmFrames = 12u;

		// Adaptive Tier 1 diff-guard probe cadence (shared by DX12 and Vulkan):
		// every real frame during 8-frame warmup, 1/4 while recovering from fallback, 1/16 in steady state (>93% less GPU cost).
		static constexpr uint64_t kWarmupGuardFrames = 8u;
		static constexpr uint64_t kRecoveryGuardStride = 4u;
		static constexpr uint64_t kSteadyGuardStride = 16u;

		static constexpr float kStaticHudMinFraction = 0.0001f;
		static constexpr float kStaticHudMaxFraction = 0.45f;

		void Reset() noexcept
		{
			brokenStreak_ = 0;
			emptyStreak_ = 0;
			healthyStreak_ = 0;
			healthyConfirmCount_ = 0;
			matchedBrokenStreak_ = 0;
			everHealthy_ = false;
			gpuMarkedBroken_ = false;
			gpuMarkedEmpty_ = false;
			toneMapMatchFailed_ = false;
			lastDiffFraction_ = 0.0f;
			lastStaticFraction_ = 0.0f;
			hasDiffSample_ = false;
			hasStaticSample_ = false;
			ResetHudlessSourceTracking();
		}

		// Tracks the game's active HUDLess resource identity (pointer/handle, extent, format).
		// Tolerates up to 4 rotating swapchain/ring-buffer pointers of the same extent/format, and resets
		// the everHealthy_ latch whenever the game switches to a different HUDLess source (e.g. menu <-> gameplay).
		// Returns true when an active everHealthy_ latch was reset by a source change (for rate-limited logging).
		bool NotifyHudlessSource(uint64_t resourceId, uint32_t width, uint32_t height, uint32_t format) noexcept
		{
			if (resourceId == 0u)
				return false;

			bool resetEverHealthy = false;
			const bool shapeChanged = (recentHudlessCount_ > 0u) &&
				(width != lastHudlessW_ || height != lastHudlessH_ || format != lastHudlessFmt_);
			if (shapeChanged)
			{
				resetEverHealthy = everHealthy_;
				everHealthy_ = false;
				healthyConfirmCount_ = 0u;
				emptyStreak_ = 0u;
				recentHudlessCount_ = 0u;
			}

			for (uint32_t i = 0u; i < recentHudlessCount_; ++i)
			{
				if (recentHudlessIds_[i] == resourceId)
					return resetEverHealthy;
			}

			if (everHealthy_ || recentHudlessCount_ >= kMaxTrackedHudlessRing)
			{
				// Brand-new resource pointer outside the established ring -> game switched HUDLess source.
				resetEverHealthy = resetEverHealthy || everHealthy_;
				everHealthy_ = false;
				healthyConfirmCount_ = 0u;
				emptyStreak_ = 0u;
				recentHudlessCount_ = 0u;
			}

			recentHudlessIds_[recentHudlessCount_++] = resourceId;
			lastHudlessW_ = width;
			lastHudlessH_ = height;
			lastHudlessFmt_ = format;
			return resetEverHealthy;
		}

		// Returns whether the analyze-only 1/16-pixel HUD diff guard should run on this real frame.
		bool ShouldRunDiffGuard(uint64_t guardFrame) const noexcept
		{
			if (guardFrame <= kWarmupGuardFrames)
				return true;
			const uint64_t stride = (gpuMarkedBroken_ || gpuMarkedEmpty_)
				? kRecoveryGuardStride
				: kSteadyGuardStride;
			return (guardFrame % stride) == 0u;
		}

		// Feeds the non-blocking GPU readback of |Real - Hudless| > tolerance (1/16 sampled fraction in [0, 1]).
		// Pass fromToneMatched = true when the sample came from Tier 3's tone-matched HUDless output.
		// Pass skipVerdict = true on camera-cut / reset / invalid-history transition frames to record telemetry
		// without advancing escalation or recovery streaks.
		void FeedHudDiffFraction(float fraction, bool fromToneMatched = false, bool skipVerdict = false) noexcept
		{
			if (!(fraction >= 0.0f && fraction <= 1.0f))
				return;

			lastDiffFraction_ = fraction;
			hasDiffSample_ = true;
			if (skipVerdict)
				return;

			if (fromToneMatched)
			{
				if (fraction > kBrokenDiffFraction)
				{
					if (++matchedBrokenStreak_ >= kBrokenConfirmFrames)
						toneMapMatchFailed_ = true;
				}
				else if (fraction >= kHealthyMinFraction && fraction <= kHealthyMaxFraction)
				{
					matchedBrokenStreak_ = 0;
				}
				return;
			}

			if (fraction > kBrokenDiffFraction)
			{
				healthyStreak_ = 0;
				emptyStreak_ = 0;
				// Once HUDLess is proven healthy in this session, require a longer confirmation streak on
				// >=60% diff spikes (typical full-screen pause/map/inventory overlays) while still catching
				// persistent mid-session HDR/exposure mismatches.
				const uint32_t needed = (everHealthy_ && fraction >= kFullScreenOverlayFraction)
					? kHealthyOverlayConfirmFrames
					: kBrokenConfirmFrames;
				if (++brokenStreak_ >= needed)
				{
					gpuMarkedBroken_ = true;
					gpuMarkedEmpty_ = false;
				}
			}
			else if (fraction < kEmptyDiffFraction)
			{
				healthyStreak_ = 0;
				brokenStreak_ = 0;
				// Once HUDLess is proven non-empty/healthy, hidden-HUD cutscenes or photo mode (diff == 0)
				// must not demote it as a duplicate backbuffer.
				if (!everHealthy_ && ++emptyStreak_ >= kEmptyConfirmFrames)
				{
					gpuMarkedEmpty_ = true;
				}
			}
			else if (fraction >= kHealthyMinFraction && fraction <= kHealthyMaxFraction)
			{
				brokenStreak_ = 0;
				emptyStreak_ = 0;
				if (!everHealthy_)
				{
					if (fraction >= kEverHealthyMinFraction)
					{
						if (++healthyConfirmCount_ >= kEverHealthyConfirmFrames)
							everHealthy_ = true;
					}
					else
					{
						healthyConfirmCount_ = 0;
					}
				}
				if ((gpuMarkedBroken_ || gpuMarkedEmpty_) && ++healthyStreak_ >= kRecoveryConfirmFrames)
				{
					gpuMarkedBroken_ = false;
					gpuMarkedEmpty_ = false;
					toneMapMatchFailed_ = false;
					matchedBrokenStreak_ = 0;
					healthyStreak_ = 0;
				}
			}
			else
			{
				healthyStreak_ = 0;
			}
		}

		void FeedStaticHudFraction(float fraction) noexcept
		{
			if (!(fraction >= 0.0f && fraction <= 1.0f))
				return;

			lastStaticFraction_ = fraction;
			hasStaticSample_ = true;
		}

		Decision Evaluate(const Inputs& in) const noexcept
		{
			Decision d{};

			// Pure XeFG fast-path: zero HudTierPass compute passes, barriers, or readbacks (matches Nov 18 baseline speed).
			if (in.mode == Mode::Off)
			{
				if (in.hudlessUiEnabled && in.hasGameHudless && in.hudlessStructurallyValid)
				{
					d.activeTier = ActiveTier::Tier1Native;
					d.useXeFgNativeUi = true;
					d.reason = "off-pure-xefg-hudless";
				}
				else
				{
					d.activeTier = ActiveTier::Interpolated;
					d.reason = "off-pure-xefg";
				}
				return d;
			}

			const bool staticCoverageUsable = !hasStaticSample_ ||
				(lastStaticFraction_ >= kStaticHudMinFraction && lastStaticFraction_ <= kStaticHudMaxFraction);

			const bool canToneMatch = in.hasGameHudless &&
				(in.hudlessStructurallyValid || in.hudlessFormatMismatch) &&
				in.gpuComputeAvailable && in.toneMapMatchAvailable && !toneMapMatchFailed_;

			// Explicit Tier 3 Hybrid override: combines Tone-Map Matcher (when a compatible HUDLess exists)
			// for dynamic/HDR UI with Static HUD Mask for fixed UI (crosshairs, static HUD bars/text).
			if (in.mode == Mode::Tier3Hybrid)
			{
				if (!in.gpuComputeAvailable)
				{
					d.activeTier = ActiveTier::Interpolated;
					d.reason = "tier3-unavailable-no-compute";
					return d;
				}
				d.activeTier = ActiveTier::Tier3Hybrid;
				d.runStaticHudUpdate = true;
				d.runStaticHudComposite = staticCoverageUsable;
				if (in.hudlessUiEnabled && canToneMatch)
				{
					d.useHudlessAsSceneSource = true;
					d.runToneMapMatch = true;
					d.runHudDiffGuard = true;
					d.runHudDiffComposite = true;
					d.reason = "forced-tier3-hybrid-tonemap+static";
				}
				else
				{
					d.reason = toneMapMatchFailed_
						? "forced-tier3-tonemap-failed -> static-mask"
						: "forced-tier3-static-mask";
				}
				return d;
			}

			if (!in.hudlessUiEnabled)
			{
				d.activeTier = ActiveTier::Interpolated;
				d.reason = "hudless-ui-disabled";
				return d;
			}

			const bool hudlessUsable = in.hasGameHudless && in.hudlessStructurallyValid &&
				!gpuMarkedBroken_ && !gpuMarkedEmpty_;

			if (hudlessUsable)
			{
				if (in.mode == Mode::Tier2DiffComposite ||
					(in.mode == Mode::Auto && in.preferTier2ForNativeIssues && in.gpuComputeAvailable))
				{
					if (in.gpuComputeAvailable)
					{
						d.activeTier = ActiveTier::Tier2DiffComposite;
						d.useHudlessAsSceneSource = true;
						d.runHudDiffGuard = true;
						d.runHudDiffComposite = true;
						d.reason = (in.mode == Mode::Tier2DiffComposite)
							? "tier2-diff-composite"
							: "tier1-issue -> tier2-diff-composite";
						return d;
					}
				}

				d.activeTier = ActiveTier::Tier1Native;
				d.useXeFgNativeUi = true;
				d.runHudDiffGuard = in.gpuComputeAvailable;
				d.reason = "tier1-native-hudless";
				return d;
			}

			if (in.hasSeparateUiPlane)
			{
				d.activeTier = ActiveTier::Tier1Native;
				d.useXeFgNativeUi = true;
				d.reason = "tier1-separate-ui-plane";
				return d;
			}

			if (in.mode == Mode::Tier1NativeOnly)
			{
				d.activeTier = ActiveTier::Interpolated;
				if (gpuMarkedBroken_)
					d.reason = "tier1-only-hudless-broken";
				else if (gpuMarkedEmpty_)
					d.reason = "tier1-only-hudless-empty";
				else
					d.reason = "tier1-only-no-hudless";
				return d;
			}

			// On the D3D12->Vulkan WMMA bridge, Auto mode avoids escalating to post-HUD compute composite (Tier 2/3),
			// which would silently disable hardware WMMA and fall back to slow DXIL. Explicit Tier 2/3 modes still work.
			if (in.mode == Mode::Auto && in.preferNativeForWmma)
			{
				d.activeTier = ActiveTier::Interpolated;
				d.reason = "wmma-auto-keep-hw-route";
				return d;
			}

			// Tier 3 Hybrid fallback (Auto / Tier 2 when native HUDLess is missing, empty, or color/format-mismatched):
			// - When a color/format-mismatched HUDLess is present and Tone-Map Matcher is available, run BOTH:
			//   Tone-Map Matcher (protects dynamic, animated, semi-transparent, HDR-mismatched UI) +
			//   Static HUD Mask (protects fixed crosshairs, static HUD bars/text).
			// - Otherwise (no HUDLess, duplicate/empty HUDLess, Vulkan, or Tone-Map Matcher failed), run Static HUD Mask alone.
			if (in.gpuComputeAvailable)
			{
				d.activeTier = ActiveTier::Tier3Hybrid;
				d.runStaticHudUpdate = true;
				d.runStaticHudComposite = staticCoverageUsable;

				const bool toneMatchEligible = canToneMatch && (gpuMarkedBroken_ || in.hudlessFormatMismatch);
				if (toneMatchEligible)
				{
					d.useHudlessAsSceneSource = true;
					d.runToneMapMatch = true;
					d.runHudDiffGuard = true;
					d.runHudDiffComposite = true;
					d.reason = in.hudlessFormatMismatch
						? "hudless-format-mismatch -> tier3-hybrid"
						: "hudless-tonemap-mismatch -> tier3-hybrid";
					return d;
				}

				d.runHudDiffGuard = in.hasGameHudless && in.hudlessStructurallyValid;
				if (toneMapMatchFailed_)
					d.reason = "tier3-tonemap-failed -> static-mask";
				else if (gpuMarkedBroken_)
					d.reason = "hudless-pixel-diff-broken -> tier3-static-mask";
				else if (gpuMarkedEmpty_)
					d.reason = "hudless-duplicate-backbuffer -> tier3-static-mask";
				else if (in.hasGameHudless)
					d.reason = "hudless-invalid -> tier3-static-mask";
				else
					d.reason = "no-hudless -> tier3-static-mask";
				return d;
			}

			d.activeTier = ActiveTier::Interpolated;
			d.reason = "no-hudless-no-compute";
			return d;
		}

		bool EverHealthy() const noexcept { return everHealthy_; }
		bool IsGpuMarkedBroken() const noexcept { return gpuMarkedBroken_; }
		bool IsGpuMarkedEmpty() const noexcept { return gpuMarkedEmpty_; }
		bool IsToneMapMatchFailed() const noexcept { return toneMapMatchFailed_; }
		float LastDiffFraction() const noexcept { return lastDiffFraction_; }
		float LastStaticFraction() const noexcept { return lastStaticFraction_; }

	private:
		uint32_t brokenStreak_ = 0;
		uint32_t emptyStreak_ = 0;
		uint32_t healthyStreak_ = 0;
		uint32_t healthyConfirmCount_ = 0;
		uint32_t matchedBrokenStreak_ = 0;
		bool everHealthy_ = false;
		bool gpuMarkedBroken_ = false;
		bool gpuMarkedEmpty_ = false;
		bool toneMapMatchFailed_ = false;
		float lastDiffFraction_ = 0.0f;
		float lastStaticFraction_ = 0.0f;
		bool hasDiffSample_ = false;
		bool hasStaticSample_ = false;

		static constexpr uint32_t kMaxTrackedHudlessRing = 4u;
		uint64_t recentHudlessIds_[kMaxTrackedHudlessRing]{};
		uint32_t recentHudlessCount_ = 0u;
		uint32_t lastHudlessW_ = 0u;
		uint32_t lastHudlessH_ = 0u;
		uint32_t lastHudlessFmt_ = 0u;

		void ResetHudlessSourceTracking() noexcept
		{
			for (uint32_t i = 0u; i < kMaxTrackedHudlessRing; ++i)
				recentHudlessIds_[i] = 0u;
			recentHudlessCount_ = 0u;
			lastHudlessW_ = 0u;
			lastHudlessH_ = 0u;
			lastHudlessFmt_ = 0u;
		}
	};
} // namespace HudTier
