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

	// GhostBuster anti-ghosting guard (DLSS Enabler [GhostBuster]):
	// Forces a clean XeFG history reset when rapid disocclusion would otherwise leave ghosting trails:
	// - Sharp single-frame camera rotation snap (> ~45 deg, cos(45 deg) ~= 0.707)
	// - Sudden FOV / zoom / ADS jump (> 0.25 rad ~= 14.3 deg)
	// - Missing camera motion in motion vectors (DLSSG.CameraMotionIncluded == 0) combined with camera movement
	inline bool ShouldGhostBusterReset(bool enabled, bool hasCamera, bool hasPrevCamera, float fwdDot,
		float curFovRad, float prevFovRad, bool hasPrevFov, bool cameraMotionIncluded, float distSq) noexcept
	{
		if (!enabled)
			return false;

		if (hasCamera && hasPrevCamera && fwdDot < 0.707f)
			return true;

		if (hasPrevFov && curFovRad > 0.01f && prevFovRad > 0.01f && std::abs(curFovRad - prevFovRad) > 0.25f)
			return true;

		// Require a substantial jump (>2m) or sharp snap (>30 deg, cos(30 deg) ~= 0.866) when CameraMotionIncluded == 0
		// so ordinary mouse panning never triggers per-frame history resets that disable interpolation.
		if (!cameraMotionIncluded && hasCamera && hasPrevCamera && (distSq > 4.0f || fwdDot < 0.866f))
			return true;

		return false;
	}
} // namespace XeFgEnabler
