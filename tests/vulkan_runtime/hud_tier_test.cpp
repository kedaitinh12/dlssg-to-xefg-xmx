#include "HudTierPolicy.h"

#include <cstdio>

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
}

#define CHECK(expr) Check((expr), #expr, __LINE__)

int main()
{
	using namespace HudTier;

	// 1. Healthy game HUDless in Auto mode -> Tier 1 Native + lightweight Diff Guard (zero Tier 3 overhead).
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
		CHECK(!d.useHudlessAsSceneSource);
		CHECK(!d.runToneMapMatch);
		CHECK(d.runHudDiffGuard);
		CHECK(!d.runHudDiffComposite);
		CHECK(!d.runStaticHudUpdate);
		CHECK(!d.runStaticHudComposite);
	}

	// 2. Color/tonemap-mismatched HUDless (>38% diff for 6 frames, e.g. RenoDX HDR) ->
	//    Tier 3 Hybrid (Tone-Map Match for dynamic/HDR UI + Static Mask for fixed UI).
	//    If even matched output fails >38% for 6 frames -> stays Tier 3 Hybrid with Static Mask only.
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
			sm.FeedHudDiffFraction(0.55f, false);
		CHECK(sm.Evaluate(in).activeTier == ActiveTier::Tier1Native);

		sm.FeedHudDiffFraction(0.55f, false);
		const auto dHybrid = sm.Evaluate(in);
		CHECK(dHybrid.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(!dHybrid.useXeFgNativeUi);
		CHECK(dHybrid.useHudlessAsSceneSource);
		CHECK(dHybrid.runToneMapMatch);
		CHECK(dHybrid.runHudDiffComposite);
		CHECK(dHybrid.runHudDiffGuard);
		CHECK(dHybrid.runStaticHudUpdate);
		CHECK(dHybrid.runStaticHudComposite);

		// Healthy matched output keeps Tier 3 Hybrid (Tone-Match + Static) active.
		sm.FeedHudDiffFraction(0.08f, true);
		const auto dStillHybrid = sm.Evaluate(in);
		CHECK(dStillHybrid.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(dStillHybrid.runToneMapMatch);
		CHECK(dStillHybrid.runStaticHudUpdate);

		// If tone-matched output still fails >38% for 6 frames -> drop Tone-Map Match, keep Static Mask only.
		for (int i = 0; i < 6; ++i)
			sm.FeedHudDiffFraction(0.60f, true);
		const auto dStaticOnly = sm.Evaluate(in);
		CHECK(dStaticOnly.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(!dStaticOnly.runToneMapMatch);
		CHECK(!dStaticOnly.runHudDiffComposite);
		CHECK(dStaticOnly.runStaticHudUpdate);
		CHECK(dStaticOnly.runStaticHudComposite);
		CHECK(dStaticOnly.runHudDiffGuard); // keeps monitoring raw HUDless for recovery

		// Recovery: 20 healthy raw frames return to Tier 1 Native and reset Tone-Map failure state.
		for (int i = 0; i < 20; ++i)
			sm.FeedHudDiffFraction(0.05f, false);
		const auto dRec = sm.Evaluate(in);
		CHECK(dRec.activeTier == ActiveTier::Tier1Native);
		CHECK(!sm.IsToneMapMatchFailed());
	}

	// 3. Format-mismatched HUDless (e.g. Witcher 3 / RenoDX HDR RGBA16F HUDLess + HDR10 Backbuffer):
	//    Auto jumps straight to Tier 3 Hybrid (Tone-Map Match + Static Mask), drops Tone-Match if it fails.
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = true;
		in.hudlessStructurallyValid = false;
		in.hudlessFormatMismatch = true;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = true;

		const auto d1 = sm.Evaluate(in);
		CHECK(d1.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(d1.runToneMapMatch);
		CHECK(d1.useHudlessAsSceneSource);
		CHECK(d1.runHudDiffComposite);
		CHECK(d1.runStaticHudUpdate);
		CHECK(d1.runStaticHudComposite);

		for (int i = 0; i < 6; ++i)
			sm.FeedHudDiffFraction(0.55f, true);
		const auto d2 = sm.Evaluate(in);
		CHECK(d2.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(!d2.runToneMapMatch);
		CHECK(d2.runStaticHudUpdate);
		CHECK(d2.runStaticHudComposite);
	}

	// 4. Vulkan (toneMapMatchAvailable == false) with broken HUDless -> Tier 3 Hybrid (Static Mask only).
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
			sm.FeedHudDiffFraction(0.50f, false);
		const auto d = sm.Evaluate(in);
		CHECK(d.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(!d.runToneMapMatch);
		CHECK(d.runStaticHudUpdate);
		CHECK(d.runStaticHudComposite);
	}

	// 5. Game with NO HUDless -> Tier 3 Hybrid (Static Mask only), and pause-menu suppression (>45% static).
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = false;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = true;

		const auto d1 = sm.Evaluate(in);
		CHECK(d1.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(!d1.runToneMapMatch);
		CHECK(d1.runStaticHudUpdate);
		CHECK(d1.runStaticHudComposite);

		// Pause screen (>45% static pixels) suppresses static composite so the screen doesn't freeze.
		sm.FeedStaticHudFraction(0.75f);
		const auto d2 = sm.Evaluate(in);
		CHECK(d2.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(d2.runStaticHudUpdate);
		CHECK(!d2.runStaticHudComposite);
	}

	// 6. Explicit Tier 3 Hybrid mode: runs Tone-Match + Static when HUDless exists, Static-only when missing.
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Tier3Hybrid;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = true;
		in.hudlessStructurallyValid = true;
		in.gpuComputeAvailable = true;
		in.toneMapMatchAvailable = true;

		const auto d1 = sm.Evaluate(in);
		CHECK(d1.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(d1.runToneMapMatch);
		CHECK(d1.runHudDiffComposite);
		CHECK(d1.runStaticHudUpdate);
		CHECK(d1.runStaticHudComposite);

		in.hasGameHudless = false;
		const auto d2 = sm.Evaluate(in);
		CHECK(d2.activeTier == ActiveTier::Tier3Hybrid);
		CHECK(!d2.runToneMapMatch);
		CHECK(!d2.runHudDiffComposite);
		CHECK(d2.runStaticHudUpdate);
		CHECK(d2.runStaticHudComposite);
	}

	// 7. Separate Vulkan UI plane -> Tier 1 Native.
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

	std::printf("hud_tier_test: %d checks, %d failed\n", gChecks, gFailures);
	return gFailures == 0 ? 0 : 1;
}
