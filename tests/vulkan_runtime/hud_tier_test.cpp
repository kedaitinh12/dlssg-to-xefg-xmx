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

	// Hardware WMMA Auto preference: keep Tier 1 Native / Interpolated instead of silently falling back to DXIL
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = false;
		in.gpuComputeAvailable = true;
		in.preferNativeForWmma = true;
		const auto dAuto = sm.Evaluate(in);
		CHECK(dAuto.activeTier == ActiveTier::Interpolated && !dAuto.runStaticHudComposite && !dAuto.runHudDiffComposite);

		// Explicit Tier 3 user override still works even when WMMA is active
		in.mode = Mode::Tier3Hybrid;
		const auto dForced = sm.Evaluate(in);
		CHECK(dForced.activeTier == ActiveTier::Tier3Hybrid && dForced.runStaticHudUpdate);

		// With healthy HUDless, WMMA Auto stays on Tier 1 Native
		in.mode = Mode::Auto;
		in.hasGameHudless = true;
		in.hudlessStructurallyValid = true;
		in.preferTier2ForNativeIssues = false;
		const auto dT1 = sm.Evaluate(in);
		CHECK(dT1.activeTier == ActiveTier::Tier1Native && dT1.useXeFgNativeUi && !dT1.runHudDiffComposite);

		// Stability over performance: preferTier2ForNativeIssues (native UI hang) overrides preferNativeForWmma
		in.preferTier2ForNativeIssues = true;
		const auto dHangGuard = sm.Evaluate(in);
		CHECK(dHangGuard.activeTier == ActiveTier::Tier2DiffComposite && dHangGuard.runHudDiffComposite);
	}

	// 9. Transition-frame skipVerdict guard: camera cuts / resets record telemetry without advancing streaks.
	{
		StateMachine sm;
		for (int i = 0; i < 10; ++i)
			sm.FeedHudDiffFraction(0.85f, false, /*skipVerdict=*/true);
		CHECK(!sm.IsGpuMarkedBroken());
		CHECK(!sm.IsGpuMarkedEmpty());
		CHECK(sm.LastDiffFraction() > 0.80f);
	}

	// 10. Small 1px crosshair (no dead zone) + stronger everHealthy_ threshold (>=0.02% x 4) + source-change reset.
	{
		StateMachine sm;
		Inputs in{};
		in.mode = Mode::Auto;
		in.hudlessUiEnabled = true;
		in.hasGameHudless = true;
		in.hudlessStructurallyValid = true;
		in.gpuComputeAvailable = true;

		// Minimal 1px crosshair (~0.002% = ~70 px at 1440p) is NOT marked empty (zero dead zone),
		// but does NOT prematurely latch everHealthy_ (prevents stray rounding pixels on duplicate backbuffers from latching).
		for (int i = 0; i < 10; ++i)
			sm.FeedHudDiffFraction(0.00002f, false);
		CHECK(!sm.EverHealthy());
		CHECK(!sm.IsGpuMarkedEmpty());
		CHECK(sm.Evaluate(in).activeTier == ActiveTier::Tier1Native);

		// Stronger evidence (>= 0.02% for 4 consecutive samples) across a 3-buffer swapchain ring latches everHealthy_.
		sm.NotifyHudlessSource(0x1001u, 2560u, 1440u, 28u);
		sm.FeedHudDiffFraction(0.005f, false);
		sm.NotifyHudlessSource(0x1002u, 2560u, 1440u, 28u);
		sm.FeedHudDiffFraction(0.005f, false);
		sm.NotifyHudlessSource(0x1003u, 2560u, 1440u, 28u);
		sm.FeedHudDiffFraction(0.005f, false);
		CHECK(!sm.EverHealthy());
		sm.NotifyHudlessSource(0x1001u, 2560u, 1440u, 28u);
		sm.FeedHudDiffFraction(0.005f, false);
		CHECK(sm.EverHealthy());

		// Continuing within the established 3-buffer ring keeps everHealthy_ intact during hidden-HUD cutscenes.
		for (int i = 0; i < 40; ++i)
		{
			sm.NotifyHudlessSource(0x1001u + static_cast<uint64_t>(i % 3), 2560u, 1440u, 28u);
			sm.FeedHudDiffFraction(0.0f, false);
		}
		CHECK(sm.EverHealthy());
		CHECK(!sm.IsGpuMarkedEmpty());
		CHECK(sm.Evaluate(in).activeTier == ActiveTier::Tier1Native);

		// Switching to a brand-new HUDLess resource (e.g. menu -> gameplay) resets everHealthy_ and returns true for logging.
		CHECK(sm.NotifyHudlessSource(0x2001u, 2560u, 1440u, 28u));
		CHECK(!sm.NotifyHudlessSource(0x2001u, 2560u, 1440u, 28u));
		CHECK(!sm.EverHealthy());
		for (int i = 0; i < 30; ++i)
			sm.FeedHudDiffFraction(0.0f, false);
		CHECK(sm.IsGpuMarkedEmpty());
		CHECK(sm.Evaluate(in).activeTier == ActiveTier::Tier3Hybrid);

		// Format or extent change also resets everHealthy_ and returns true.
		StateMachine smFmt;
		CHECK(!smFmt.NotifyHudlessSource(0x3001u, 2560u, 1440u, 28u));
		for (int i = 0; i < 4; ++i)
			smFmt.FeedHudDiffFraction(0.01f, false);
		CHECK(smFmt.EverHealthy());
		CHECK(smFmt.NotifyHudlessSource(0x3001u, 2560u, 1440u, 10u));
		CHECK(!smFmt.EverHealthy());
	}

	// 11. Transient full-screen menu/map overlay (>=60% diff) after everHealthy_ requires extended streak (14 vs 6),
	//     while persistent mid-session mismatch still escalates cleanly.
	{
		StateMachine sm;
		for (int i = 0; i < 4; ++i)
			sm.FeedHudDiffFraction(0.03f, false);
		CHECK(sm.EverHealthy());

		// 10 frames of pause/map overlay (75% diff) must NOT falsely trip gpuMarkedBroken_.
		for (int i = 0; i < 10; ++i)
			sm.FeedHudDiffFraction(0.75f, false);
		CHECK(!sm.IsGpuMarkedBroken());

		// Returning to normal gameplay clears brokenStreak_.
		sm.FeedHudDiffFraction(0.03f, false);
		CHECK(!sm.IsGpuMarkedBroken());

		// Persistent >=60% mismatch for 14 consecutive samples still escalates.
		for (int i = 0; i < 13; ++i)
			sm.FeedHudDiffFraction(0.75f, false);
		CHECK(!sm.IsGpuMarkedBroken());
		sm.FeedHudDiffFraction(0.75f, false);
		CHECK(sm.IsGpuMarkedBroken());
	}

	// 12. Adaptive diff-guard probe cadence: warmup every frame (<=8), steady 1/16, recovery 1/4 + faster 12-frame recovery.
	{
		StateMachine sm;
		for (uint64_t f = 1; f <= 8; ++f)
			CHECK(sm.ShouldRunDiffGuard(f));
		CHECK(!sm.ShouldRunDiffGuard(9));
		CHECK(!sm.ShouldRunDiffGuard(12));
		CHECK(sm.ShouldRunDiffGuard(16));

		// Trigger broken -> cadence accelerates to 1/4 for fast recovery.
		for (int i = 0; i < 6; ++i)
			sm.FeedHudDiffFraction(0.55f, false);
		CHECK(sm.IsGpuMarkedBroken());
		CHECK(sm.ShouldRunDiffGuard(12));
		CHECK(!sm.ShouldRunDiffGuard(13));
		CHECK(sm.ShouldRunDiffGuard(16));

		// 12 healthy samples recover cleanly and return cadence to 1/16.
		for (int i = 0; i < 11; ++i)
			sm.FeedHudDiffFraction(0.04f, false);
		CHECK(sm.IsGpuMarkedBroken());
		sm.FeedHudDiffFraction(0.04f, false);
		CHECK(!sm.IsGpuMarkedBroken());
		CHECK(!sm.ShouldRunDiffGuard(12));
		CHECK(sm.ShouldRunDiffGuard(16));
	}

	std::printf("hud_tier_test: %d checks, %d failed\n", gChecks, gFailures);
	return gFailures == 0 ? 0 : 1;
}
