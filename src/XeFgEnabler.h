// Shared GhostBuster anti-ghosting and extended XeFG override helpers inspired by DLSS Enabler 4.10.0.26 RC1.
// Header-only with zero OS dependencies so both D3D12 (FrameGenerator) and Vulkan (VulkanRuntime) share identical
// behavior and can be unit-tested offline.
#pragma once

#include <cmath>
#include <cstdint>

namespace XeFgEnabler
{
	// Tri-state override matching DLSS Enabler XeFG extended settings:
	// 0 = Auto (keep game value), 1 = Force Off, 2 = Force On.
	inline bool ApplyTriStateOverride(int32_t mode, bool gameValue) noexcept
	{
		if (mode == 1)
			return false;
		if (mode == 2)
			return true;
		return gameValue;
	}

	// Adaptive per-frame camera displacement tracker (scale-invariant across meter-scale and UE centimeter-scale engines).
	// Uses a 5-sample median of recent non-zero camera steps so 1-2 frame teleports never inflate the baseline,
	// while sustained movement (e.g. 5-50 cm/frame in Unreal Engine) raises thresholds automatically.
	struct CameraMotionScale
	{
		float recentSteps[5]{};
		uint32_t sampleCount = 0;
		uint32_t writeIndex = 0;
		float baselineStep = 0.0f;

		void Reset() noexcept
		{
			for (float& s : recentSteps)
				s = 0.0f;
			sampleCount = 0;
			writeIndex = 0;
			baselineStep = 0.0f;
		}

		// Record a non-zero camera step distance and update the robust baseline.
		// Call AFTER evaluating cut/GhostBuster for the current frame so a sudden teleport
		// is judged against prior frames first.
		void Observe(float dist) noexcept
		{
			if (!std::isfinite(dist) || dist <= 1e-3f)
				return;

			recentSteps[writeIndex] = dist;
			writeIndex = (writeIndex + 1u) % 5u;
			if (sampleCount < 5u)
			{
				++sampleCount;
				// Wait until all 5 normal-motion samples are collected so a single early spike cannot poison the baseline.
				if (sampleCount < 5u)
					return;
			}

			float sorted[5];
			for (uint32_t i = 0; i < sampleCount; ++i)
				sorted[i] = recentSteps[i];
			for (uint32_t i = 1; i < sampleCount; ++i)
			{
				const float key = sorted[i];
				uint32_t j = i;
				while (j > 0 && sorted[j - 1] > key)
				{
					sorted[j] = sorted[j - 1];
					--j;
				}
				sorted[j] = key;
			}
			const float median = sorted[sampleCount / 2u];

			// Rise immediately when sustained motion reveals a larger world-unit scale (e.g. UE cm),
			// decay slowly (~0.998/moving frame) when slowing down so brief combat pauses keep the learned scale.
			if (median >= baselineStep)
				baselineStep = median;
			else
			{
				const float decayed = baselineStep * 0.998f;
				baselineStep = (decayed > median) ? decayed : median;
			}
		}

		float CutThresholdSq() const noexcept
		{
			const float scaled = baselineStep * 12.0f;
			const float thresh = (scaled > 4.5f) ? scaled : 4.5f;
			return thresh * thresh;
		}

		float HardTeleportThresholdSq() const noexcept
		{
			const float scaled = baselineStep * 25.0f;
			const float thresh = (scaled > 10.0f) ? scaled : 10.0f;
			return thresh * thresh;
		}

		float GhostBusterMoveThresholdSq() const noexcept
		{
			const float scaled = baselineStep * 6.0f;
			const float thresh = (scaled > 2.0f) ? scaled : 2.0f;
			return thresh * thresh;
		}
	};

	// GhostBuster anti-ghosting guard (DLSS Enabler [GhostBuster]):
	// Forces a clean XeFG history reset when rapid disocclusion would otherwise leave ghosting trails:
	// - Sharp single-frame camera rotation snap (> ~45 deg, cos(45 deg) ~= 0.707)
	// - Sudden FOV / zoom / ADS jump (> 0.25 rad ~= 14.3 deg)
	// - Missing camera motion in motion vectors (DLSSG.CameraMotionIncluded == 0) combined with camera movement
	inline bool ShouldGhostBusterReset(bool enabled, bool hasCamera, bool hasPrevCamera, float fwdDot,
		float curFovRad, float prevFovRad, bool hasPrevFov, bool cameraMotionIncluded, float distSq,
		float moveThresholdSq = 4.0f) noexcept
	{
		if (!enabled)
			return false;

		if (hasCamera && hasPrevCamera && fwdDot < 0.707f)
			return true;

		if (hasPrevFov && curFovRad > 0.01f && prevFovRad > 0.01f && std::abs(curFovRad - prevFovRad) > 0.25f)
			return true;

		// Require a substantial jump (> moveThreshold, default 2m / adaptive for cm engines) or sharp snap (>30 deg,
		// cos(30 deg) ~= 0.866) when CameraMotionIncluded == 0 so ordinary movement/panning never spams resets.
		if (!cameraMotionIncluded && hasCamera && hasPrevCamera && (distSq > moveThresholdSq || fwdDot < 0.866f))
			return true;

		return false;
	}
} // namespace XeFgEnabler
