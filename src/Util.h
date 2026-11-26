#pragma once

#include <Windows.h>
#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// Keeps objects a recorded command list may still use alive until every submitting D3D12 command queue's private
// fence completes (or, when queue-fence tracking is unavailable, for `Delay` evaluations).
// - Multiple Evaluate() calls on the same open (unsubmitted) command list coalesce onto one recording FenceSignal;
//   once submitted, subsequent recordings reusing the same command list address allocate a fresh FenceSignal.
// - If a queue accepts command lists but Signal() fails, those signals are tracked against that exact queue so a
//   subsequent successful Signal() on the same queue (or shutdown retry) completes them cleanly.
// - Multi-signal entries stay alive until EVERY associated queue fence confirms completion (including on Collect(true)).
// - Lightweight entries (textures/heaps) drain completely once GPU-ready so backlog never grows unbounded.
// - Heavy entries (XeFG contexts / FrameGenerators) pace to 1 per Collect() call while ready backlog <= MaxHeavyBacklog
//   so teardown bursts never stall Streamline's 500ms pacer flush window, yet still stay strictly bounded.
namespace DeferredDestroy
{
	constexpr uint64_t Delay = 64;
	constexpr size_t MaxHeavyBacklog = 4;

	struct FenceSignal
	{
		uint64_t recordingId = 0;
		const void* cmdList = nullptr;
		std::atomic<bool> submitted{ false };
		std::mutex mutex;
		std::function<bool()> isComplete;

		bool IsGpuComplete() noexcept
		{
			if (!submitted.load(std::memory_order_acquire))
				return false;
			std::lock_guard lock(mutex);
			return isComplete ? isComplete() : true;
		}
	};

	struct Entry
	{
		uint64_t destroyAt = 0;
		bool heavy = false;
		std::vector<std::shared_ptr<FenceSignal>> signals;
		std::shared_ptr<void> object;
	};

	inline std::mutex mutex;
	// Leaked on purpose: destroying XeFG contexts from our static destructors at process exit would call into a
	// libxess_fg.dll that may already be torn down
	inline std::vector<Entry>& entries = *new std::vector<Entry>;
	inline std::vector<std::shared_ptr<FenceSignal>>& awaitingSubmit = *new std::vector<std::shared_ptr<FenceSignal>>;
	inline std::vector<std::pair<const void*, std::shared_ptr<FenceSignal>>>& awaitingQueueSignal =
		*new std::vector<std::pair<const void*, std::shared_ptr<FenceSignal>>>;
	inline uint64_t nextRecordingId = 0;
	inline std::atomic<uint64_t> evaluations{ 0 };
	inline std::atomic<size_t> pendingCount{ 0 };
	inline std::atomic<size_t> awaitingSubmitCount{ 0 };
	inline std::atomic<size_t> awaitingQueueSignalCount{ 0 };
	inline std::atomic<bool> queueFenceTrackingEnabled{ false };
	inline thread_local std::shared_ptr<FenceSignal> tActiveSignal;
	inline thread_local std::vector<std::shared_ptr<FenceSignal>> tActiveSignals;

	inline void SetQueueFenceTrackingEnabled(bool enabled) noexcept
	{
		queueFenceTrackingEnabled.store(enabled, std::memory_order_release);
	}

	inline size_t AwaitingSubmitCount() noexcept
	{
		return awaitingSubmitCount.load(std::memory_order_acquire);
	}

	inline size_t AwaitingQueueSignalCount() noexcept
	{
		return awaitingQueueSignalCount.load(std::memory_order_acquire);
	}

	inline bool HasAnyPendingSubmitOrQueueSignal() noexcept
	{
		return awaitingSubmitCount.load(std::memory_order_acquire) != 0 ||
			awaitingQueueSignalCount.load(std::memory_order_acquire) != 0;
	}

	inline void PruneUnreferencedAwaitingLocked() noexcept
	{
		awaitingSubmit.erase(
			std::remove_if(awaitingSubmit.begin(), awaitingSubmit.end(),
				[](const std::shared_ptr<FenceSignal>& sig) noexcept {
					return !sig || sig.use_count() == 1;
				}),
			awaitingSubmit.end());
		awaitingSubmitCount.store(awaitingSubmit.size(), std::memory_order_release);

		awaitingQueueSignal.erase(
			std::remove_if(awaitingQueueSignal.begin(), awaitingQueueSignal.end(),
				[](const std::pair<const void*, std::shared_ptr<FenceSignal>>& kv) noexcept {
					return !kv.second || kv.second.use_count() == 1;
				}),
			awaitingQueueSignal.end());
		awaitingQueueSignalCount.store(awaitingQueueSignal.size(), std::memory_order_release);
	}

	// Coalesces multiple Evaluate() recordings on the SAME still-open (unsubmitted) command list into one FenceSignal.
	// Once that command list is submitted (or accepted by a queue), any later recording on the same address creates a
	// new FenceSignal with a strictly greater recordingId.
	inline std::shared_ptr<FenceSignal> RegisterCommandList(const void* cmdList)
	{
		if (!cmdList || !queueFenceTrackingEnabled.load(std::memory_order_acquire))
			return nullptr;
		std::scoped_lock lock(mutex);
		for (auto it = awaitingSubmit.rbegin(); it != awaitingSubmit.rend(); ++it)
		{
			if (*it && (*it)->cmdList == cmdList)
				return *it;
		}
		auto sig = std::make_shared<FenceSignal>();
		sig->cmdList = cmdList;
		sig->recordingId = ++nextRecordingId;
		// Only prune abandoned recordings that no live Entry, FrameGenerator, or ScopedCommandList references.
		if (awaitingSubmit.size() >= 64)
			PruneUnreferencedAwaitingLocked();
		awaitingSubmit.push_back(sig);
		awaitingSubmitCount.store(awaitingSubmit.size(), std::memory_order_release);
		return sig;
	}

	inline bool HasAwaitingSubmitFor(const void* const* lists, UINT count) noexcept
	{
		if (!lists || count == 0 || awaitingSubmitCount.load(std::memory_order_acquire) == 0)
			return false;
		std::scoped_lock lock(mutex);
		for (const auto& sig : awaitingSubmit)
		{
			for (UINT i = 0; i < count; ++i)
			{
				if (sig->cmdList == lists[i])
					return true;
			}
		}
		return false;
	}

	inline bool HasAwaitingQueueSignalFor(const void* queue) noexcept
	{
		if (!queue || awaitingQueueSignalCount.load(std::memory_order_acquire) == 0)
			return false;
		std::scoped_lock lock(mutex);
		for (const auto& kv : awaitingQueueSignal)
		{
			if (kv.first == queue)
				return true;
		}
		return false;
	}

	inline std::vector<const void*> AwaitingQueueSignalQueues()
	{
		std::vector<const void*> queues;
		if (awaitingQueueSignalCount.load(std::memory_order_acquire) == 0)
			return queues;
		std::scoped_lock lock(mutex);
		for (const auto& kv : awaitingQueueSignal)
		{
			if (kv.first && std::find(queues.begin(), queues.end(), kv.first) == queues.end())
				queues.push_back(kv.first);
		}
		return queues;
	}

	// Records that `queue` accepted `lists[0..count-1]` via ExecuteCommandLists, but queue->Signal() failed.
	// Moves matching signals out of awaitingSubmit into awaitingQueueSignal so a later Signal() on the same queue
	// (or shutdown retry) can complete them cleanly, while subsequent recordings on the same cmdList get fresh signals.
	inline void MarkCommandListsAcceptedWithoutFence(const void* queue, const void* const* lists, UINT count) noexcept
	{
		if (!queue || !lists || count == 0)
			return;
		std::scoped_lock lock(mutex);
		for (UINT i = 0; i < count; ++i)
		{
			const void* target = lists[i];
			auto it = std::find_if(awaitingSubmit.begin(), awaitingSubmit.end(),
				[target](const std::shared_ptr<FenceSignal>& sig) noexcept {
					return sig && sig->cmdList == target;
				});
			if (it != awaitingSubmit.end())
			{
				awaitingQueueSignal.emplace_back(queue, std::move(*it));
				awaitingSubmit.erase(it);
			}
		}
		awaitingSubmitCount.store(awaitingSubmit.size(), std::memory_order_release);
		awaitingQueueSignalCount.store(awaitingQueueSignal.size(), std::memory_order_release);
	}

	// Attaches a later successful fence on `queue` to all prior recordings accepted by `queue` whose Signal() failed.
	inline void AttachFenceToQueueSignals(const void* queue, std::function<bool()> isComplete) noexcept
	{
		if (!queue || awaitingQueueSignalCount.load(std::memory_order_acquire) == 0)
			return;
		std::vector<std::shared_ptr<FenceSignal>> matched;
		{
			std::scoped_lock lock(mutex);
			auto split = std::partition(awaitingQueueSignal.begin(), awaitingQueueSignal.end(),
				[queue](const std::pair<const void*, std::shared_ptr<FenceSignal>>& kv) noexcept {
					return kv.first != queue;
				});
			for (auto it = split; it != awaitingQueueSignal.end(); ++it)
			{
				if (it->second)
					matched.push_back(std::move(it->second));
			}
			awaitingQueueSignal.erase(split, awaitingQueueSignal.end());
			awaitingQueueSignalCount.store(awaitingQueueSignal.size(), std::memory_order_release);
		}
		for (auto& sig : matched)
		{
			{
				std::lock_guard slock(sig->mutex);
				sig->isComplete = isComplete;
			}
			sig->submitted.store(true, std::memory_order_release);
		}
	}

	// Matches awaiting recordings for `lists[0..count-1]` (FIFO) and, when `queue != nullptr`, also completes any
	// earlier recordings accepted by the same `queue` whose Signal() previously failed.
	inline void MarkCommandListsSubmitted(const void* const* lists, UINT count,
		std::function<bool()> isComplete, const void* queue = nullptr) noexcept
	{
		std::vector<std::shared_ptr<FenceSignal>> matched;
		{
			std::scoped_lock lock(mutex);
			if (lists && count > 0)
			{
				for (UINT i = 0; i < count; ++i)
				{
					const void* target = lists[i];
					auto it = std::find_if(awaitingSubmit.begin(), awaitingSubmit.end(),
						[target](const std::shared_ptr<FenceSignal>& sig) noexcept {
							return sig && sig->cmdList == target;
						});
					if (it != awaitingSubmit.end())
					{
						matched.push_back(std::move(*it));
						awaitingSubmit.erase(it);
					}
				}
				awaitingSubmitCount.store(awaitingSubmit.size(), std::memory_order_release);
			}
			if (queue && !awaitingQueueSignal.empty())
			{
				auto split = std::partition(awaitingQueueSignal.begin(), awaitingQueueSignal.end(),
					[queue](const std::pair<const void*, std::shared_ptr<FenceSignal>>& kv) noexcept {
						return kv.first != queue;
					});
				for (auto it = split; it != awaitingQueueSignal.end(); ++it)
				{
					if (it->second)
						matched.push_back(std::move(it->second));
				}
				awaitingQueueSignal.erase(split, awaitingQueueSignal.end());
				awaitingQueueSignalCount.store(awaitingQueueSignal.size(), std::memory_order_release);
			}
		}
		for (auto& sig : matched)
		{
			{
				std::lock_guard slock(sig->mutex);
				sig->isComplete = isComplete;
			}
			sig->submitted.store(true, std::memory_order_release);
		}
	}

	struct ScopedCommandList
	{
		std::shared_ptr<FenceSignal> prev;
		std::shared_ptr<FenceSignal> current;

		explicit ScopedCommandList(const void* cmdList)
			: prev(std::move(tActiveSignal)), current(RegisterCommandList(cmdList))
		{
			tActiveSignal = current;
		}

		~ScopedCommandList()
		{
			tActiveSignal = std::move(prev);
		}

		ScopedCommandList(const ScopedCommandList&) = delete;
		ScopedCommandList& operator=(const ScopedCommandList&) = delete;
	};

	// Binds a full set of active GPU signals (across all queues that have used the current FrameGenerator) to every
	// DeferredDestroy::Add() call within this scope (including texture replacements in depth/motion/HUD helpers).
	struct ScopedSignals
	{
		std::vector<std::shared_ptr<FenceSignal>> prev;

		explicit ScopedSignals(std::vector<std::shared_ptr<FenceSignal>> signals)
			: prev(std::move(tActiveSignals))
		{
			tActiveSignals = std::move(signals);
		}

		~ScopedSignals()
		{
			tActiveSignals = std::move(prev);
		}

		ScopedSignals(const ScopedSignals&) = delete;
		ScopedSignals& operator=(const ScopedSignals&) = delete;
	};

	inline std::shared_ptr<FenceSignal> MakeCompletedFenceSignal(std::function<bool()> isComplete)
	{
		auto sig = std::make_shared<FenceSignal>();
		sig->isComplete = std::move(isComplete);
		sig->submitted.store(true, std::memory_order_release);
		return sig;
	}

	inline void Add(std::shared_ptr<void> object, bool heavy,
		std::vector<std::shared_ptr<FenceSignal>> signals)
	{
		if (!object)
			return;
		signals.erase(
			std::remove_if(signals.begin(), signals.end(),
				[](const std::shared_ptr<FenceSignal>& s) noexcept { return !s; }),
			signals.end());
		if (signals.empty())
		{
			if (!tActiveSignals.empty())
				signals = tActiveSignals;
			else if (tActiveSignal)
				signals.push_back(tActiveSignal);
		}
		std::scoped_lock lock(mutex);
		const uint64_t now = evaluations.load(std::memory_order_relaxed);
		entries.push_back({ now + Delay, heavy, std::move(signals), std::move(object) });
		pendingCount.store(entries.size(), std::memory_order_release);
	}

	inline void Add(std::shared_ptr<void> object, bool heavy = false, std::shared_ptr<FenceSignal> signal = nullptr)
	{
		std::vector<std::shared_ptr<FenceSignal>> signals;
		if (signal)
			signals.push_back(std::move(signal));
		Add(std::move(object), heavy, std::move(signals));
	}

	inline bool AreAllSignalsComplete(const Entry& e) noexcept
	{
		for (const auto& sig : e.signals)
		{
			if (!sig || !sig->IsGpuComplete())
				return false;
		}
		return true;
	}

	inline bool IsEntryReady(const Entry& e, uint64_t now) noexcept
	{
		if (!e.signals.empty())
			return AreAllSignalsComplete(e);
		return now >= e.destroyAt;
	}

	inline void Collect(bool all = false)
	{
		const uint64_t now = evaluations.fetch_add(1, std::memory_order_relaxed) + 1;
		if (!all && pendingCount.load(std::memory_order_acquire) == 0)
			return;

		std::vector<Entry> due;
		{
			std::scoped_lock lock(mutex);
			if (all)
			{
				// Even on shutdown (all == true), never destroy entries whose attached GPU fences are incomplete
				// or whose command lists have not yet been confirmed submitted.
				std::vector<Entry> keep;
				keep.reserve(entries.size());
				for (auto& e : entries)
				{
					if (e.signals.empty() || AreAllSignalsComplete(e))
						due.push_back(std::move(e));
					else
						keep.push_back(std::move(e));
				}
				entries.swap(keep);
				PruneUnreferencedAwaitingLocked();
			}
			else
			{
				std::vector<Entry> keep;
				std::vector<Entry> readyHeavy;
				keep.reserve(entries.size());
				for (auto& e : entries)
				{
					if (!IsEntryReady(e, now))
					{
						keep.push_back(std::move(e));
					}
					else if (e.heavy)
					{
						readyHeavy.push_back(std::move(e));
					}
					else
					{
						due.push_back(std::move(e));
					}
				}
				if (!readyHeavy.empty())
				{
					// Pace heavy teardowns (XeFG::Destroy) to 1 per Collect() while backlog <= MaxHeavyBacklog,
					// draining excess if backlog exceeds MaxHeavyBacklog so memory stays strictly bounded.
					const size_t heavyToDrain = (readyHeavy.size() > MaxHeavyBacklog)
						? (readyHeavy.size() - MaxHeavyBacklog + 1)
						: 1;
					for (size_t i = 0; i < heavyToDrain; ++i)
						due.push_back(std::move(readyHeavy[i]));
					for (size_t i = heavyToDrain; i < readyHeavy.size(); ++i)
						keep.push_back(std::move(readyHeavy[i]));
				}
				entries.swap(keep);
			}
			pendingCount.store(entries.size(), std::memory_order_release);
		}
		// Destroyed here, outside the lock
	}

	inline size_t PendingCount() noexcept
	{
		return pendingCount.load(std::memory_order_acquire);
	}

	inline void ResetForTests() noexcept
	{
		std::scoped_lock lock(mutex);
		entries.clear();
		awaitingSubmit.clear();
		awaitingQueueSignal.clear();
		nextRecordingId = 0;
		evaluations.store(0, std::memory_order_relaxed);
		pendingCount.store(0, std::memory_order_release);
		awaitingSubmitCount.store(0, std::memory_order_release);
		awaitingQueueSignalCount.store(0, std::memory_order_release);
		queueFenceTrackingEnabled.store(false, std::memory_order_release);
		tActiveSignal.reset();
		tActiveSignals.clear();
	}
} // namespace DeferredDestroy

inline bool SameFile(const wchar_t* left, const wchar_t* right) noexcept
{
	if (!left || !right || !*left || !*right)
		return false;
	const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
	HANDLE a = CreateFileW(left, FILE_READ_ATTRIBUTES, share, nullptr, OPEN_EXISTING, 0, nullptr);
	HANDLE b = CreateFileW(right, FILE_READ_ATTRIBUTES, share, nullptr, OPEN_EXISTING, 0, nullptr);
	BY_HANDLE_FILE_INFORMATION x{}, y{};
	const bool same = a != INVALID_HANDLE_VALUE && b != INVALID_HANDLE_VALUE &&
		GetFileInformationByHandle(a, &x) && GetFileInformationByHandle(b, &y) &&
		x.dwVolumeSerialNumber == y.dwVolumeSerialNumber &&
		x.nFileIndexHigh == y.nFileIndexHigh && x.nFileIndexLow == y.nFileIndexLow;
	if (a != INVALID_HANDLE_VALUE)
		CloseHandle(a);
	if (b != INVALID_HANDLE_VALUE)
		CloseHandle(b);
	return same;
}

inline std::filesystem::path ModulePath(HMODULE module)
{
	wchar_t path[MAX_PATH]{};
	GetModuleFileNameW(module, path, MAX_PATH);
	return path;
}

// Full path of this dll
inline std::filesystem::path ThisModulePath()
{
	HMODULE module = nullptr;
	GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		reinterpret_cast<LPCWSTR>(&ThisModulePath), &module);

	return ModulePath(module);
}

inline void Barrier(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
	D3D12_RESOURCE_STATES after)
{
	if (before == after)
		return;

	D3D12_RESOURCE_BARRIER barrier{};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Transition.pResource = resource;
	barrier.Transition.StateBefore = before;
	barrier.Transition.StateAfter = after;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	cmdList->ResourceBarrier(1, &barrier);
}

// Both resources are back in their states afterwards; transitions are batched into at most one ResourceBarrier call
// before and one after CopyResource to halve CPU/GPU barrier overhead across all models.
inline void Copy(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* destination, D3D12_RESOURCE_STATES destinationState,
	ID3D12Resource* source, D3D12_RESOURCE_STATES sourceState)
{
	if (!cmdList || !destination || !source || destination == source)
		return;

	D3D12_RESOURCE_BARRIER before[2]{};
	UINT beforeCount = 0;
	if (destinationState != D3D12_RESOURCE_STATE_COPY_DEST)
	{
		before[beforeCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		before[beforeCount].Transition.pResource = destination;
		before[beforeCount].Transition.StateBefore = destinationState;
		before[beforeCount].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
		before[beforeCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		++beforeCount;
	}
	if (sourceState != D3D12_RESOURCE_STATE_COPY_SOURCE)
	{
		before[beforeCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		before[beforeCount].Transition.pResource = source;
		before[beforeCount].Transition.StateBefore = sourceState;
		before[beforeCount].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
		before[beforeCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		++beforeCount;
	}
	if (beforeCount)
		cmdList->ResourceBarrier(beforeCount, before);

	cmdList->CopyResource(destination, source);

	D3D12_RESOURCE_BARRIER after[2]{};
	UINT afterCount = 0;
	if (destinationState != D3D12_RESOURCE_STATE_COPY_DEST)
	{
		after[afterCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		after[afterCount].Transition.pResource = destination;
		after[afterCount].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		after[afterCount].Transition.StateAfter = destinationState;
		after[afterCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		++afterCount;
	}
	if (sourceState != D3D12_RESOURCE_STATE_COPY_SOURCE)
	{
		after[afterCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		after[afterCount].Transition.pResource = source;
		after[afterCount].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
		after[afterCount].Transition.StateAfter = sourceState;
		after[afterCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		++afterCount;
	}
	if (afterCount)
		cmdList->ResourceBarrier(afterCount, after);
}

// Typed formats for views on typeless or depth resources. 8-bit sRGB maps to UNORM, typed UAVs can't be sRGB
inline DXGI_FORMAT TypedFormat(DXGI_FORMAT format)
{
	switch (format)
	{
	case DXGI_FORMAT_R8G8B8A8_TYPELESS:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
		return DXGI_FORMAT_R8G8B8A8_UNORM;
	case DXGI_FORMAT_B8G8R8A8_TYPELESS:
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
		return DXGI_FORMAT_B8G8R8A8_UNORM;
	case DXGI_FORMAT_R10G10B10A2_TYPELESS:
		return DXGI_FORMAT_R10G10B10A2_UNORM;
	case DXGI_FORMAT_R16G16B16A16_TYPELESS:
		return DXGI_FORMAT_R16G16B16A16_FLOAT;
	case DXGI_FORMAT_R16G16_TYPELESS:
		return DXGI_FORMAT_R16G16_FLOAT;
	case DXGI_FORMAT_R32G32_TYPELESS:
		return DXGI_FORMAT_R32G32_FLOAT;
	case DXGI_FORMAT_R32_TYPELESS:
	case DXGI_FORMAT_D32_FLOAT:
		return DXGI_FORMAT_R32_FLOAT;
	case DXGI_FORMAT_R16_TYPELESS:
	case DXGI_FORMAT_D16_UNORM:
		return DXGI_FORMAT_R16_UNORM;
	default:
		return format;
	}
}
