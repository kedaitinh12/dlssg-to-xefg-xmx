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

	// Distinguishes sustained player mouse/stick camera turns from instant scene cuts by tracking 3D rotation-axis
	// continuity across consecutive frames (zero GPU cost, OS-independent across DX12/Vulkan and Windows/Linux).
	struct CameraTurnGuard
	{
		float prevAxis[3]{};
		bool hasPrevAxis = false;
		uint32_t turnStreak = 0;

		void Reset() noexcept
		{
			prevAxis[0] = prevAxis[1] = prevAxis[2] = 0.0f;
			hasPrevAxis = false;
			turnStreak = 0;
		}

		// Returns true when the current frame continues a smooth same-direction camera rotation (e.g. fast mouse flick)
		// with <= ~70 deg per frame (fwdDot >= 0.342), so GhostBuster does not falsely drop FG mid-turn.
		bool ObserveAndCheckSustainedTurn(const float curFwd[3], const float prevFwd[3], float fwdDot) noexcept
		{
			if (!std::isfinite(fwdDot) || fwdDot < 0.342f || fwdDot > 0.9998f)
			{
				Reset();
				return false;
			}
			// 3D rotation axis = prevFwd x curFwd
			const float ax = prevFwd[1] * curFwd[2] - prevFwd[2] * curFwd[1];
			const float ay = prevFwd[2] * curFwd[0] - prevFwd[0] * curFwd[2];
			const float az = prevFwd[0] * curFwd[1] - prevFwd[1] * curFwd[0];
			const float lenSq = ax * ax + ay * ay + az * az;
			if (!std::isfinite(lenSq) || lenSq <= 1e-8f)
			{
				Reset();
				return false;
			}
			const float invLen = 1.0f / std::sqrt(lenSq);
			const float nx = ax * invLen;
			const float ny = ay * invLen;
			const float nz = az * invLen;

			bool sustained = false;
			if (hasPrevAxis)
			{
				const float axisDot = nx * prevAxis[0] + ny * prevAxis[1] + nz * prevAxis[2];
				if (axisDot > 0.65f)
				{
					++turnStreak;
					sustained = (turnStreak >= 1u);
				}
				else
				{
					turnStreak = 0;
				}
			}
			prevAxis[0] = nx;
			prevAxis[1] = ny;
			prevAxis[2] = nz;
			hasPrevAxis = true;
			return sustained;
		}
	};

	// Adaptive per-frame camera displacement tracker (scale-invariant across meter-scale and UE centimeter-scale engines).
	// Uses a 5-sample median of recent non-zero camera steps so 1-2 frame teleports never inflate the baseline,
	// while sustained movement (e.g. 5-50 cm/frame in Unreal Engine) raises thresholds automatically.
	struct CameraMotionScale
	{
		float recentSteps[5]{};
		uint32_t sampleCount = 0;
		uint32_t writeIndex = 0;
		float baselineStep = 0.0f;
		// Startup consistency buffer: detects sustained large-unit movement (e.g. 12+ cm/frame in UE)
		// that exceeds the initial 10-unit meter-scale teleport threshold, while still rejecting isolated jumps.
		float startupConsistentSteps[5]{};
		uint32_t startupConsistentCount = 0;

		static float Median5(const float (&vals)[5]) noexcept
		{
			float sorted[5];
			for (uint32_t i = 0; i < 5u; ++i)
				sorted[i] = vals[i];
			for (uint32_t i = 1; i < 5u; ++i)
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
			return sorted[2];
		}

		void Reset() noexcept
		{
			for (float& s : recentSteps)
				s = 0.0f;
			for (float& s : startupConsistentSteps)
				s = 0.0f;
			sampleCount = 0;
			writeIndex = 0;
			baselineStep = 0.0f;
			startupConsistentCount = 0;
		}

		// Record a non-zero normal-motion camera step distance and update the robust baseline.
		// Call AFTER evaluating cut/GhostBuster for the current frame.
		void Observe(float dist) noexcept
		{
			if (!std::isfinite(dist) || dist <= 1e-3f)
				return;

			startupConsistentCount = 0;
			recentSteps[writeIndex] = dist;
			writeIndex = (writeIndex + 1u) % 5u;
			if (sampleCount < 5u)
			{
				++sampleCount;
				// Wait until all 5 normal-motion samples are collected so a single early spike cannot poison the baseline.
				if (sampleCount < 5u)
					return;
			}

			const float median = Median5(recentSteps);

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

		// Unified per-frame observer: normal non-reset frames feed Observe(); during uncalibrated startup
		// (sampleCount < 5), 5 consecutive consistent reset-triggering steps (within 0.5x..2.0x of each other)
		// bootstrap the world-unit scale so cm-scale games (e.g. 12+ cm/frame) never deadlock in continuous resets,
		// while isolated teleports (e.g. 1000 followed by 1, 1, ...) are discarded cleanly.
		void ObserveCandidate(float dist, bool wasReset) noexcept
		{
			if (!std::isfinite(dist) || dist <= 1e-3f)
				return;

			if (!wasReset)
			{
				Observe(dist);
				return;
			}

			if (sampleCount >= 5u)
			{
				startupConsistentCount = 0;
				return;
			}

			if (startupConsistentCount > 0u)
			{
				const float prev = startupConsistentSteps[startupConsistentCount - 1u];
				if (dist < prev * 0.5f || dist > prev * 2.0f)
					startupConsistentCount = 0;
			}

			startupConsistentSteps[startupConsistentCount++] = dist;
			if (startupConsistentCount >= 5u)
			{
				for (uint32_t i = 0; i < 5u; ++i)
					recentSteps[i] = startupConsistentSteps[i];
				sampleCount = 5u;
				writeIndex = 0;
				baselineStep = Median5(recentSteps);
				startupConsistentCount = 0;
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
	// - Sharp single-frame camera rotation snap (> ~45 deg, cos(45 deg) ~= 0.707), unless sustainedTurn is active
	//   for a continuous player mouse turn (still resets on >70 deg instant cuts)
	// - Sudden FOV / zoom / ADS jump (> 0.25 rad ~= 14.3 deg)
	// - Missing camera motion in motion vectors (DLSSG.CameraMotionIncluded == 0) combined with camera movement
	inline bool ShouldGhostBusterReset(bool enabled, bool hasCamera, bool hasPrevCamera, float fwdDot,
		float curFovRad, float prevFovRad, bool hasPrevFov, bool cameraMotionIncluded, float distSq,
		float moveThresholdSq = 4.0f, bool sustainedTurn = false) noexcept
	{
		if (!enabled)
			return false;

		if (hasCamera && hasPrevCamera && fwdDot < (sustainedTurn ? 0.342f : 0.707f))
			return true;

		if (hasPrevFov && curFovRad > 0.01f && prevFovRad > 0.01f && std::abs(curFovRad - prevFovRad) > 0.25f)
			return true;

		// Require a substantial jump (> moveThreshold, default 2m / adaptive for cm engines) or sharp snap (>30 deg,
		// cos(30 deg) ~= 0.866) when CameraMotionIncluded == 0 so ordinary movement/panning never spams resets.
		if (!cameraMotionIncluded && hasCamera && hasPrevCamera &&
			(distSq > moveThresholdSq || (!sustainedTurn && fwdDot < 0.866f)))
			return true;

		return false;
	}
} // namespace XeFgEnabler
