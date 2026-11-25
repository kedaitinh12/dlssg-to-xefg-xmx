#pragma once

#include <cstdint>
#include <mutex>

// Synchronizes ownership of process-wide D3D12/Vulkan/CUDA bridge state across FrameGenerator::Context
// lifecycles (including deferred destruction after NVSDK_NGX_D3D12_ReleaseFeature -> CreateFeature transitions).
//
// Invariants:
// 1. A new context claims ownership ONLY AFTER XeFG::CreateContext() and XeFG::Init() both succeed.
//    A failed context stays unowned (generation == 0) and never resets an active context's bridge/cache.
// 2. Ownership transfer (Commit) and global teardown (ResetIfOwner / ResetBeforeReplacement) are strictly
//    serialized under the same mutex, preventing a deferred destructor from racing with a new Commit().
class ContextOwnershipCoordinator
{
public:
	// Claim ownership for a newly initialized context; returns its non-zero generation token.
	// Optional onCommitted runs under the ownership lock immediately after ownership is transferred.
	template <class Fn>
	uint64_t Commit(Fn&& onCommitted) noexcept
	{
		std::lock_guard lock(mutex_);
		const uint64_t gen = ++nextGeneration_;
		activeGeneration_ = gen;
		onCommitted(gen);
		return gen;
	}

	uint64_t Commit() noexcept
	{
		std::lock_guard lock(mutex_);
		activeGeneration_ = ++nextGeneration_;
		return activeGeneration_;
	}

	// Explicitly reset global state before replacing an active context (e.g. settings/format change).
	// Clears active ownership if `generation` is still the current owner so its deferred destructor won't double-reset.
	template <class ResetFn>
	bool ResetBeforeReplacement(uint64_t generation, ResetFn&& resetGlobal) noexcept
	{
		std::lock_guard lock(mutex_);
		if (generation != 0 && generation == activeGeneration_)
			activeGeneration_ = 0;
		resetGlobal();
		return true;
	}

	// Called from Context::~Context(). Runs `resetGlobal()` ONLY if this context is NOT retired-for-replacement
	// AND is still the active owner. Serialized with Commit() so a stale destructor can never reset a newer context.
	template <class ResetFn>
	bool ResetIfOwner(uint64_t generation, bool retiredForReplacement, ResetFn&& resetGlobal) noexcept
	{
		std::lock_guard lock(mutex_);
		if (retiredForReplacement || generation == 0 || generation != activeGeneration_)
			return false;
		activeGeneration_ = 0;
		resetGlobal();
		return true;
	}

	uint64_t ActiveGeneration() const noexcept
	{
		std::lock_guard lock(mutex_);
		return activeGeneration_;
	}

	// For unit tests only.
	void ResetForTests() noexcept
	{
		std::lock_guard lock(mutex_);
		nextGeneration_ = 0;
		activeGeneration_ = 0;
	}

private:
	mutable std::mutex mutex_;
	uint64_t nextGeneration_ = 0;
	uint64_t activeGeneration_ = 0;
};
