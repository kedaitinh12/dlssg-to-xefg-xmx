// Unit tests for the 4-Tier Adaptive HUD Protection & Fallback Policy (src/HudTierPolicy.h).
#include "HudTierPolicy.h"

#include <cstdio>
#include <cstdlib>

namespace
{
	int gChecks = 0;
	int gFailures = 0;

	void Check(bool cond, const char* expr, int line)
	{
		++gChecks;
		if (!cond)
		{
			++gFailures;
			std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
		}
	}
} // namespace

#define CHECK(expr) Check((expr), #expr, __LINE__)

int main()
{
	using namespace HudTier;

	// 1. Clean game HUDless in Auto mode => Tier 1 Native + GPU Pixel-Diff Guard (Tier 4 stays dormant!)
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = true;
		in.hudlessStructurallyValid = true;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = true;

		const auto d = sm.Evaluate(in);
		CHECK(d.activeTier == ActiveTier::Tier1Native);
		CHECK(d.useXeFgNativeUi);
		CHECK(d.runHudDiffGuard);
		CHECK(!d.runToneMapMatch);
		CHECK(!d.runHudDiffComposite);
		CHECK(!d.runStaticHudUpdate);
	}

	// 2. Color-mismatched HUDless (RenoDX HDR, >38% diff for 6 frames):
	//    Auto tries Tier 4 Tone-Map Match first; if matched output STILL fails 6 frames => escalates to Tier 3 Static Mask!
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = true;
		in.hudlessStructurallyValid = true;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = true;

		for (int i = 0; i < 5; ++i)
			sm.FeedHudDiffFraction(0.55f);
		CHECK(!sm.IsGpuMarkedBroken());
		CHECK(sm.Evaluate(in).activeTier == ActiveTier::Tier1Native);

		// 6th frame trips Pixel-Diff Guard => wakes up Tier 4 Tone-Map Match!
		sm.FeedHudDiffFraction(0.55f);
		CHECK(sm.IsGpuMarkedBroken());
		const auto d4 = sm.Evaluate(in);
		CHECK(d4.activeTier == ActiveTier::Tier4ToneMapMatch);
		CHECK(d4.runToneMapMatch);
		CHECK(d4.useHudlessAsSceneSource);
		CHECK(d4.runHudDiffComposite);
		CHECK(!d4.useXeFgNativeUi);

		// If tone-matched HUDless is clean (5% diff), stay cleanly on Tier 4!
		for (int i = 0; i < 10; ++i)
			sm.FeedHudDiffFraction(0.05f, true);
		CHECK(!sm.IsToneMapMatchFailed());
		CHECK(sm.Evaluate(in).activeTier == ActiveTier::Tier4ToneMapMatch);

		// If even tone-matched HUDless still fails (>38% for 6 frames), escalate to Tier 3 Static Mask!
		for (int i = 0; i < 6; ++i)
			sm.FeedHudDiffFraction(0.50f, true);
		CHECK(sm.IsToneMapMatchFailed());
		const auto d3 = sm.Evaluate(in);
		CHECK(d3.activeTier == ActiveTier::Tier3StaticMask);
		CHECK(!d3.runToneMapMatch);
		CHECK(d3.runStaticHudUpdate);
		CHECK(d3.runStaticHudComposite);
	}

	// 3. Broken HUDless when Tone-Map Match is unavailable (e.g. Vulkan) => goes directly to Tier 3 Static Mask
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = true;
		in.hudlessStructurallyValid = true;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = false;

		for (int i = 0; i < 6; ++i)
			sm.FeedHudDiffFraction(0.55f);
		const auto d = sm.Evaluate(in);
		CHECK(d.activeTier == ActiveTier::Tier3StaticMask);
		CHECK(!d.runToneMapMatch);
		CHECK(d.runStaticHudUpdate);
		CHECK(d.runStaticHudComposite);

		// 20 healthy frames recover back to Tier 1
		for (int i = 0; i < 20; ++i)
			sm.FeedHudDiffFraction(0.05f);
		CHECK(!sm.IsGpuMarkedBroken());
		CHECK(sm.Evaluate(in).activeTier == ActiveTier::Tier1Native);
	}

	// 4. Duplicate/empty HUDless (<0.005% diff for 30 frames) => skips Tier 4, goes directly to Tier 3 Static Mask
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = true;
		in.hudlessStructurallyValid = true;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = true;

		for (int i = 0; i < 30; ++i)
			sm.FeedHudDiffFraction(0.00001f);
		CHECK(sm.IsGpuMarkedEmpty());
		const auto d = sm.Evaluate(in);
		CHECK(d.activeTier == ActiveTier::Tier3StaticMask);
		CHECK(!d.runToneMapMatch);
	}

	// 5. Game with NO HUDless => skips Tier 4 completely, goes directly to Tier 3 Static Mask
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = false;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = true;

		const auto d = sm.Evaluate(in);
		CHECK(d.activeTier == ActiveTier::Tier3StaticMask);
		CHECK(!d.runToneMapMatch);
		CHECK(d.runStaticHudUpdate);
		CHECK(d.runStaticHudComposite);
	}

	// 6. Static-HUD coverage guard: still camera (>45% static) skips composite, moving camera (8%) applies
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = false;
		in.gpuComputeAvailable = true;

		sm.FeedStaticHudFraction(0.72f);
		CHECK(!sm.Evaluate(in).runStaticHudComposite);
		sm.FeedStaticHudFraction(0.08f);
		CHECK(sm.Evaluate(in).runStaticHudComposite);
	}

	// 7. Explicit Tier 4 mode falls back to Tier 3 when game has no HUDless
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Tier4ToneMapMatch;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = false;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = true;

		const auto d = sm.Evaluate(in);
		CHECK(d.activeTier == ActiveTier::Tier3StaticMask);
		CHECK(!d.runToneMapMatch);
	}

	// 8. Vulkan separate UI plane stays on Tier 1 even without HUDless
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = false;
		in.hasSeparateUiPlane = true;
		in.gpuComputeAvailable = false;

		const auto d = sm.Evaluate(in);
		CHECK(d.activeTier == ActiveTier::Tier1Native);
		CHECK(d.useXeFgNativeUi);
	}

		// Format-mismatched HUDless (e.g. Witcher 3 / RenoDX HDR RGBA16F HUDLess + HDR10 Backbuffer):
	{
		HudTier::StateMachine sm;
		HudTier::Inputs in{};
		in.mode = HudTier::Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = true;
		in.hudlessStructurallyValid = false;
		in.hudlessFormatMismatch = true;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = true;

		const auto d1 = sm.Evaluate(in);
		CHECK(d1.activeTier == HudTier::ActiveTier::Tier4ToneMapMatch);
		CHECK(d1.runToneMapMatch);
		CHECK(d1.useHudlessAsSceneSource && d1.runHudDiffComposite);

		for (int i = 0; i < 6; ++i)
			sm.FeedHudDiffFraction(0.55f, true);
		const auto d2 = sm.Evaluate(in);
		CHECK(d2.activeTier == HudTier::ActiveTier::Tier3StaticMask);
		CHECK(!d2.runToneMapMatch && d2.runStaticHudUpdate);
	}

	std::printf("hud_tier_test: %d checks, %d failed\n", gChecks, gFailures);
	return gFailures == 0 ? 0 : 1;
}
