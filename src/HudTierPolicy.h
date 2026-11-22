// 4-Tier Adaptive HUD Protection & Fallback Policy (shared across DX12, Vulkan, and unit tests).
//
// Tier 1 (NativeHudless): Uses the game's native DLSSG.HUDLess / DLSSG.UI plane through XeFG UI composition,
//   continuously monitored by a lightweight 1/16-pixel GPU Pixel-Diff Guard (zero GPU stall).
//
// Tier 2 (HudDiffComposite): XeFG interpolates clean HUD-less scene in UI_MODE_AUTO and a lightweight 3x3
//   dilated compute shader (|Real - Hudless| > tolerance) composites HUD pixels on top.
//
// Tier 4 (ToneMapMatch): Conditional inline recovery tier (zero cost & zero VRAM on normal games).
//   Wakes up ONLY when a game provides a valid DLSSG.HUDLess whose color/exposure/tonemap mismatches Backbuffer
//   (e.g. RenoDX HDR or pre-tonemap HUDLess, where Pixel-Diff Guard sees >38% mismatch for 6 frames).
//   Runs an inline 256-bin per-channel log-luminance median LUT (ported from MFG-Unlocker V1.7.2 hudless_match.cpp)
//   to tone-match HUDLess into Backbuffer's color space, then composites HUD via Tier 2. If even the matched
//   HUDLess still fails the diff guard (>38% for 6 frames), automatically escalates to Tier 3 Static Mask.
//
// Tier 3 (StaticHudMask): Universal final safety net for games with NO HUD-less buffer or unrecoverable HUDless.
//   Tracks pixels unchanged across consecutive real frames (crosshair, static HUD text, health bar, minimap frame).
#pragma once

#include <cstdint>

namespace HudTier
{
	// User-selectable mode in OptiScaler menu / OptiScaler.ini (HudTierMode).
	enum class Mode : int32_t
	{
		Auto = 0,               // Automatic 4-tier selection & escalation (default)
		Tier1NativeOnly = 1,    // Strictly Tier 1 (Native XeFG HUDless/UI) or interpolated if unavailable
		Tier2DiffComposite = 2, // Prefer Tier 2 (Custom HUD-Diff Composite), escalate if HUDless breaks/missing
		Tier3StaticMask = 3,    // Force Tier 3 (Temporal Static-HUD Mask), ignores game HUD-less buffer
		Tier4ToneMapMatch = 4,  // Force Tier 4 (Inline 256-bin Tone-Map Match HUDless), falls back to Tier 3 if missing/fails
	};

	// Effective HUD protection tier active on the current frame.
	enum class ActiveTier : int32_t
	{
		Interpolated = 0,       // No separate HUD protection; UI interpolated with scene
		Tier1Native = 1,        // Tier 1: Native XeFG HUDless / UI plane (with GPU Pixel-Diff Guard)
		Tier2DiffComposite = 2, // Tier 2: Custom GPU HUD-Diff Composite (xmx_hud_composite style)
		Tier3StaticMask = 3,    // Tier 3: Temporal Static-HUD Mask (no HUDless required)
		Tier4ToneMapMatch = 4,  // Tier 4: Inline 256-bin Tone-Map Matched HUDless + Diff-Composite (RenoDX/HDR recovery)
	};

	inline const char* ModeName(Mode mode) noexcept
	{
		switch (mode)
		{
		case Mode::Auto: return "Auto (4-Tier)";
		case Mode::Tier1NativeOnly: return "Tier 1 (Native HUDless)";
		case Mode::Tier2DiffComposite: return "Tier 2 (HUD-Diff Composite)";
		case Mode::Tier3StaticMask: return "Tier 3 (Static HUD Mask)";
		case Mode::Tier4ToneMapMatch: return "Tier 4 (Tone-Map Match HUDless)";
		default: return "Auto (4-Tier)";
		}
	}

	inline const char* ActiveTierShortName(ActiveTier tier) noexcept
	{
		switch (tier)
		{
		case ActiveTier::Tier1Native: return "HUD T1: Native";
		case ActiveTier::Tier2DiffComposite: return "HUD T2: Diff-Comp";
		case ActiveTier::Tier3StaticMask: return "HUD T3: Static Mask";
		case ActiveTier::Tier4ToneMapMatch: return "HUD T4: Tone-Match";
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
	};

	struct Decision
	{
		ActiveTier activeTier = ActiveTier::Interpolated;
		bool useXeFgNativeUi = false;         // Pass HUDless/UI to XeFG's own UI composition mode
		bool useHudlessAsSceneSource = false; // Feed clean (or tone-matched) HUDless as scene source while Tier 2/4 composites HUD
		bool runToneMapMatch = false;         // Run inline 256-bin log-luminance median LUT on HUDless (lazy allocated)
		bool runHudDiffGuard = false;         // Run lightweight 1/16-pixel Real-vs-Hudless diff counter
		bool runHudDiffComposite = false;     // Run custom 3x3 HUD-diff composite onto generated output
		bool runStaticHudUpdate = false;      // Run Tier 3 temporal stability map update on real frames
		bool runStaticHudComposite = false;   // Run Tier 3 static HUD composite onto generated output
		const char* reason = "interpolated";
	};

	class StateMachine
	{
	public:
		static constexpr float kBrokenDiffFraction = 0.38f;
		static constexpr float kEmptyDiffFraction = 0.00005f;
		static constexpr float kHealthyMinFraction = 0.0002f;
		static constexpr float kHealthyMaxFraction = 0.30f;
		static constexpr uint32_t kBrokenConfirmFrames = 6u;
		static constexpr uint32_t kEmptyConfirmFrames = 30u;
		static constexpr uint32_t kRecoveryConfirmFrames = 20u;

		static constexpr float kStaticHudMinFraction = 0.0001f;
		static constexpr float kStaticHudMaxFraction = 0.45f;

		void Reset() noexcept
		{
			brokenStreak_ = 0;
			emptyStreak_ = 0;
			healthyStreak_ = 0;
			matchedBrokenStreak_ = 0;
			gpuMarkedBroken_ = false;
			gpuMarkedEmpty_ = false;
			toneMapMatchFailed_ = false;
			lastDiffFraction_ = 0.0f;
			lastStaticFraction_ = 0.0f;
			hasDiffSample_ = false;
			hasStaticSample_ = false;
		}

		// Feeds the non-blocking GPU readback of |Real - Hudless| > tolerance (1/16 sampled fraction in [0, 1]).
		// Pass fromToneMatched = true when the sample came from Tier 4's tone-matched HUDless output.
		void FeedHudDiffFraction(float fraction, bool fromToneMatched = false) noexcept
		{
			if (!(fraction >= 0.0f && fraction <= 1.0f))
				return;

			lastDiffFraction_ = fraction;
			hasDiffSample_ = true;

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
				if (++brokenStreak_ >= kBrokenConfirmFrames)
				{
					gpuMarkedBroken_ = true;
					gpuMarkedEmpty_ = false;
				}
			}
			else if (fraction < kEmptyDiffFraction)
			{
				healthyStreak_ = 0;
				brokenStreak_ = 0;
				if (++emptyStreak_ >= kEmptyConfirmFrames)
				{
					gpuMarkedEmpty_ = true;
				}
			}
			else if (fraction >= kHealthyMinFraction && fraction <= kHealthyMaxFraction)
			{
				brokenStreak_ = 0;
				emptyStreak_ = 0;
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

			const bool staticCoverageUsable = !hasStaticSample_ ||
				(lastStaticFraction_ >= kStaticHudMinFraction && lastStaticFraction_ <= kStaticHudMaxFraction);

			// Explicit Tier 3 override works even if the game's broken HUDless checkbox was turned off.
			if (in.mode == Mode::Tier3StaticMask)
			{
				if (in.gpuComputeAvailable)
				{
					d.activeTier = ActiveTier::Tier3StaticMask;
					d.runStaticHudUpdate = true;
					d.runStaticHudComposite = staticCoverageUsable;
					d.reason = "forced-tier3-static-mask";
					return d;
				}
				d.activeTier = ActiveTier::Interpolated;
				d.reason = "tier3-unavailable-no-compute";
				return d;
			}

			if (!in.hudlessUiEnabled)
			{
				d.activeTier = ActiveTier::Interpolated;
				d.reason = "hudless-ui-disabled";
				return d;
			}

			// Explicit Tier 4 (Tone-Map Match HUDless) requested by user.
			if (in.mode == Mode::Tier4ToneMapMatch)
			{
				if (in.hasGameHudless && (in.hudlessStructurallyValid || in.hudlessFormatMismatch) &&
					in.gpuComputeAvailable && in.toneMapMatchAvailable && !toneMapMatchFailed_)
				{
					d.activeTier = ActiveTier::Tier4ToneMapMatch;
					d.useHudlessAsSceneSource = true;
					d.runToneMapMatch = true;
					d.runHudDiffGuard = true;
					d.runHudDiffComposite = true;
					d.reason = "forced-tier4-tonemap-match";
					return d;
				}
				if (in.gpuComputeAvailable)
				{
					d.activeTier = ActiveTier::Tier3StaticMask;
					d.runStaticHudUpdate = true;
					d.runStaticHudComposite = staticCoverageUsable;
					d.reason = toneMapMatchFailed_ ? "tier4-match-failed -> tier3-static-mask"
					                               : "tier4-no-hudless -> tier3-static-mask";
					return d;
				}
				d.activeTier = ActiveTier::Interpolated;
				d.reason = "tier4-unavailable";
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

			// Color/tonemap-mismatched or format-mismatched HUDless (e.g. Witcher 3 / RenoDX HDR RGBA16F vs HDR10):
			// in Auto/Tier2, try conditional Tier 4 Tone-Map Match first before falling back to Tier 3 Static Mask.
			if (in.mode != Mode::Tier1NativeOnly && !toneMapMatchFailed_ && in.hasGameHudless &&
				((in.hudlessStructurallyValid && gpuMarkedBroken_) || in.hudlessFormatMismatch) &&
				in.gpuComputeAvailable && in.toneMapMatchAvailable)
			{
				d.activeTier = ActiveTier::Tier4ToneMapMatch;
				d.useHudlessAsSceneSource = true;
				d.runToneMapMatch = true;
				d.runHudDiffGuard = true;
				d.runHudDiffComposite = true;
				d.reason = in.hudlessFormatMismatch
					? "hudless-format-mismatch -> tier4-tonemap-match"
					: "hudless-tonemap-mismatch -> tier4-tonemap-match";
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

			if (in.gpuComputeAvailable)
			{
				d.activeTier = ActiveTier::Tier3StaticMask;
				d.runStaticHudUpdate = true;
				d.runStaticHudComposite = staticCoverageUsable;
				d.runHudDiffGuard = in.hasGameHudless && in.hudlessStructurallyValid;
				if (toneMapMatchFailed_)
					d.reason = "tier4-match-failed -> tier3-static-mask";
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

		bool IsGpuMarkedBroken() const noexcept { return gpuMarkedBroken_; }
		bool IsGpuMarkedEmpty() const noexcept { return gpuMarkedEmpty_; }
		bool IsToneMapMatchFailed() const noexcept { return toneMapMatchFailed_; }
		float LastDiffFraction() const noexcept { return lastDiffFraction_; }
		float LastStaticFraction() const noexcept { return lastStaticFraction_; }

	private:
		uint32_t brokenStreak_ = 0;
		uint32_t emptyStreak_ = 0;
		uint32_t healthyStreak_ = 0;
		uint32_t matchedBrokenStreak_ = 0;
		bool gpuMarkedBroken_ = false;
		bool gpuMarkedEmpty_ = false;
		bool toneMapMatchFailed_ = false;
		float lastDiffFraction_ = 0.0f;
		float lastStaticFraction_ = 0.0f;
		bool hasDiffSample_ = false;
		bool hasStaticSample_ = false;
	};
} // namespace HudTier
