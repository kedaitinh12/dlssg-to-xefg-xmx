#pragma once

#include "Util.h"

#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

// Lightweight, chain-safe ID3D12CommandQueue::ExecuteCommandLists observer that attaches a per-queue private
// ID3D12Fence signal to DeferredDestroy entries recorded on a submitted command list.
// - Fast path when no entries await submission or queue-signal retry: atomic loads (~0 overhead).
// - Claims awaiting signals BEFORE forwarding ExecuteCommandLists so concurrent recordings reusing the same
//   command list address allocate a fresh FenceSignal and never inherit the older fence.
// - Serializes per-queue ExecuteCommandLists + Signal + submitSeq + retry queue retain/release under the same
//   per-queue submitMutex so:
//   (1) an earlier fence can never satisfy a newer failed-Signal submission on the same queue, and
//   (2) ReleaseRetryQueueIfEmpty can never drop a queue reference while another thread is recording a retry.
// - Binds each cached fence to the exact queue instance & owning ID3D12Device (via private-data token + GetDevice
//   check + stale-fence recovery) so a recreated queue/device reusing the same pointer address always gets a fresh fence.
// - Chain-safe with D3D12VulkanWmma's vtable hook regardless of installation order.
namespace D3D12QueueFence
{
	namespace detail
	{
		constexpr size_t kExecuteCommandListsSlot = 10;
		// {9A4E2F61-8C1B-4D7E-A5E3-2C71F08B4D92}
		inline constexpr GUID kQueueTokenGuid = {
			0x9a4e2f61, 0x8c1b, 0x4d7e, { 0xa5, 0xe3, 0x2c, 0x71, 0xf0, 0x8b, 0x4d, 0x92 }
		};

		using ExecuteCommandListsFn = void (STDMETHODCALLTYPE*)(
			ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);

		struct QueueHookEntry
		{
			void** vtable = nullptr;
			ExecuteCommandListsFn original = nullptr;
		};

		struct PerQueueFence
		{
			std::shared_ptr<std::mutex> submitMutex;
			Microsoft::WRL::ComPtr<ID3D12Fence> fence;
			ID3D12Device* ownerDevice = nullptr; // identity tag only; never dereferenced
			uint64_t queueToken = 0;
			bool hasPrivateToken = false;
			uint64_t nextValue = 0;
		};

		struct State
		{
			std::mutex hookMutex;
			std::array<QueueHookEntry, 8> hooks{};
			std::atomic<size_t> hookCount{ 0 };
			std::atomic<bool> installed{ false };

			std::mutex fenceMutex;
			uint64_t nextQueueToken = 0;
			std::unordered_map<ID3D12CommandQueue*, PerQueueFence> queueFences;
			// Retired fences from queues whose pointer address was later reused by a new queue/device.
			std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Fence>, uint64_t>> retiredFences;
			// Strong references kept ONLY while a queue has pending failed-Signal retry entries.
			std::unordered_map<ID3D12CommandQueue*, Microsoft::WRL::ComPtr<ID3D12CommandQueue>> retryQueues;
		};

		// Intentionally immortal to avoid static-destructor COM release races during DLL_PROCESS_DETACH.
		inline State& S() noexcept
		{
			static State* s = new State;
			return *s;
		}

		inline ExecuteCommandListsFn FindOriginal(ID3D12CommandQueue* queue) noexcept
		{
			void** vt = *reinterpret_cast<void***>(queue);
			State& s = S();
			const size_t n = s.hookCount.load(std::memory_order_acquire);
			for (size_t i = 0; i < n; ++i)
			{
				if (s.hooks[i].vtable == vt)
					return s.hooks[i].original;
			}
			return nullptr;
		}

		inline std::shared_ptr<std::mutex> GetQueueSubmitMutex(ID3D12CommandQueue* queue)
		{
			State& s = S();
			std::lock_guard lock(s.fenceMutex);
			auto& slot = s.queueFences[queue];
			if (!slot.submitMutex)
				slot.submitMutex = std::make_shared<std::mutex>();
			return slot.submitMutex;
		}

		// Must be called while holding this queue's submitMutex, AFTER MarkSignalsAcceptedWithoutFence().
		inline void RetainRetryQueueUnderSubmitLock(ID3D12CommandQueue* queue) noexcept
		{
			if (!queue)
				return;
			State& s = S();
			std::lock_guard lock(s.fenceMutex);
			s.retryQueues[queue] = queue;
		}

		// Must be called while holding this queue's submitMutex, AFTER CompleteClaimedSubmits()/AttachFenceToQueueSignals().
		// Returns the detached ComPtr so caller releases it OUTSIDE both fenceMutex and submitMutex.
		inline Microsoft::WRL::ComPtr<ID3D12CommandQueue> DetachRetryQueueIfEmptyUnderSubmitLock(
			ID3D12CommandQueue* queue) noexcept
		{
			Microsoft::WRL::ComPtr<ID3D12CommandQueue> released;
			if (!queue || DeferredDestroy::HasAwaitingQueueSignalFor(queue))
				return released;
			State& s = S();
			std::lock_guard lock(s.fenceMutex);
			auto it = s.retryQueues.find(queue);
			if (it != s.retryQueues.end())
			{
				released = std::move(it->second);
				s.retryQueues.erase(it);
			}
			return released;
		}

		inline void RetireSlotFenceLocked(State& s, PerQueueFence& slot) noexcept
		{
			if (slot.fence && slot.nextValue > 0)
			{
				const uint64_t done = slot.fence->GetCompletedValue();
				if (done != UINT64_MAX && done < slot.nextValue)
					s.retiredFences.emplace_back(slot.fence, slot.nextValue);
			}
			slot.fence.Reset();
			slot.ownerDevice = nullptr;
			slot.queueToken = 0;
			slot.hasPrivateToken = false;
			slot.nextValue = 0;

			// Prune completed retired fences so the list stays tiny.
			s.retiredFences.erase(
				std::remove_if(s.retiredFences.begin(), s.retiredFences.end(),
					[](const std::pair<Microsoft::WRL::ComPtr<ID3D12Fence>, uint64_t>& item) noexcept {
						if (!item.first)
							return true;
						const uint64_t d = item.first->GetCompletedValue();
						return d == UINT64_MAX || d >= item.second;
					}),
				s.retiredFences.end());
		}

		inline bool SignalQueuePrivateFence(ID3D12CommandQueue* queue,
			Microsoft::WRL::ComPtr<ID3D12Fence>& outFence, uint64_t& outValue) noexcept
		{
			if (!queue)
				return false;

			// Always query the live owning device from the queue so a recreated device/queue reusing this
			// pointer address never inherits an older device's fence.
			Microsoft::WRL::ComPtr<ID3D12Device> dev;
			if (FAILED(queue->GetDevice(IID_PPV_ARGS(&dev))) || !dev)
				return false;

			// Also check per-queue private-data token so a brand-new queue reusing the same address on the
			// same device gets a fresh fence bound to its own lifetime.
			uint64_t observedToken = 0;
			UINT tokenSize = sizeof(observedToken);
			const bool gotToken =
				SUCCEEDED(queue->GetPrivateData(kQueueTokenGuid, &tokenSize, &observedToken)) &&
				tokenSize == sizeof(observedToken);

			State& s = S();
			std::lock_guard lock(s.fenceMutex);
			auto& slot = s.queueFences[queue];

			const bool deviceChanged = slot.fence && slot.ownerDevice != dev.Get();
			const bool queueInstanceChanged = slot.fence && slot.hasPrivateToken &&
				(!gotToken || observedToken != slot.queueToken);
			if (deviceChanged || queueInstanceChanged)
				RetireSlotFenceLocked(s, slot);

			bool createdFresh = false;
			if (!slot.fence)
			{
				if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&slot.fence))) || !slot.fence)
					return false;
				slot.ownerDevice = dev.Get();
				slot.nextValue = 0;
				const uint64_t newToken = ++s.nextQueueToken;
				if (SUCCEEDED(queue->SetPrivateData(kQueueTokenGuid, sizeof(newToken), &newToken)))
				{
					slot.queueToken = newToken;
					slot.hasPrivateToken = true;
				}
				else
				{
					slot.queueToken = 0;
					slot.hasPrivateToken = false;
				}
				createdFresh = true;
			}

			uint64_t val = ++slot.nextValue;
			if (SUCCEEDED(queue->Signal(slot.fence.Get(), val)))
			{
				outFence = slot.fence;
				outValue = val;
				return true;
			}

			// If a cached fence failed to signal (e.g. reused device+queue address on a mock without private data),
			// recreate the fence on the queue's live device and retry once.
			if (!createdFresh)
			{
				RetireSlotFenceLocked(s, slot);
				if (SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&slot.fence))) && slot.fence)
				{
					slot.ownerDevice = dev.Get();
					slot.nextValue = 1;
					const uint64_t newToken = ++s.nextQueueToken;
					if (SUCCEEDED(queue->SetPrivateData(kQueueTokenGuid, sizeof(newToken), &newToken)))
					{
						slot.queueToken = newToken;
						slot.hasPrivateToken = true;
					}
					if (SUCCEEDED(queue->Signal(slot.fence.Get(), 1)))
					{
						outFence = slot.fence;
						outValue = 1;
						return true;
					}
				}
			}

			return false;
		}

		inline void STDMETHODCALLTYPE HookExecuteCommandLists(
			ID3D12CommandQueue* self, UINT numCommandLists, ID3D12CommandList* const* ppCommandLists)
		{
			ExecuteCommandListsFn orig = FindOriginal(self);
			if (!orig)
				return;

			if (!DeferredDestroy::HasAnyPendingSubmitOrQueueSignal())
			{
				orig(self, numCommandLists, ppCommandLists);
				return;
			}

			// Claim awaiting signals for these command lists BEFORE forwarding ExecuteCommandLists so any concurrent
			// recording reusing the same cmdList address starts a fresh FenceSignal instead of inheriting this fence.
			const void* const* rawLists = reinterpret_cast<const void* const*>(ppCommandLists);
			std::vector<std::shared_ptr<DeferredDestroy::FenceSignal>> claimed;
			if (numCommandLists != 0 && ppCommandLists != nullptr)
				claimed = DeferredDestroy::ClaimAwaitingSubmits(rawLists, numCommandLists);

			const bool hasQueueRetry = DeferredDestroy::HasAwaitingQueueSignalFor(self);
			if (claimed.empty() && !hasQueueRetry)
			{
				orig(self, numCommandLists, ppCommandLists);
				return;
			}

			auto submitMutex = GetQueueSubmitMutex(self);
			Microsoft::WRL::ComPtr<ID3D12CommandQueue> releasedRetryQueue;
			{
				std::lock_guard qlock(*submitMutex);
				orig(self, numCommandLists, ppCommandLists);
				const uint64_t submitSeq = DeferredDestroy::NextQueueSubmitSeq();
				Microsoft::WRL::ComPtr<ID3D12Fence> fence;
				uint64_t value = 0;
				if (SignalQueuePrivateFence(self, fence, value))
				{
					auto pred = [fence = std::move(fence), value]() noexcept -> bool {
						const uint64_t done = fence->GetCompletedValue();
						return done == UINT64_MAX || done >= value;
					};
					// Complete and detach retry queue reference under the same per-queue submitMutex so
					// RetainRetryQueue and DetachRetryQueueIfEmpty can never race on the same queue.
					DeferredDestroy::CompleteClaimedSubmits(std::move(claimed), std::move(pred), self, submitSeq);
					releasedRetryQueue = DetachRetryQueueIfEmptyUnderSubmitLock(self);
				}
				else if (!claimed.empty())
				{
					DeferredDestroy::MarkSignalsAcceptedWithoutFence(self, std::move(claimed), submitSeq);
					RetainRetryQueueUnderSubmitLock(self);
				}
			}
			// releasedRetryQueue ComPtr is destroyed here, outside both fenceMutex and submitMutex.
		}

		inline bool PatchQueueVtable(ID3D12CommandQueue* queue) noexcept
		{
			if (!queue)
				return false;
			void** vt = *reinterpret_cast<void***>(queue);
			if (!vt || !vt[kExecuteCommandListsSlot])
				return false;

			State& s = S();
			std::lock_guard lock(s.hookMutex);
			const size_t n = s.hookCount.load(std::memory_order_relaxed);
			for (size_t i = 0; i < n; ++i)
			{
				if (s.hooks[i].vtable == vt)
					return true;
			}
			if (n >= s.hooks.size())
				return false;

			void* orig = vt[kExecuteCommandListsSlot];
			if (orig == reinterpret_cast<void*>(&HookExecuteCommandLists))
				return true;

			// Publish original before patching so concurrent ExecuteCommandLists callers never observe null.
			s.hooks[n] = { vt, reinterpret_cast<ExecuteCommandListsFn>(orig) };
			s.hookCount.store(n + 1, std::memory_order_release);

			DWORD oldProtect = 0;
			if (!VirtualProtect(&vt[kExecuteCommandListsSlot], sizeof(void*), PAGE_EXECUTE_READWRITE, &oldProtect))
			{
				s.hookCount.store(n, std::memory_order_release);
				return false;
			}

			InterlockedExchangePointer(&vt[kExecuteCommandListsSlot], reinterpret_cast<void*>(&HookExecuteCommandLists));
			DWORD ignored = 0;
			VirtualProtect(&vt[kExecuteCommandListsSlot], sizeof(void*), oldProtect, &ignored);
			return true;
		}
	} // namespace detail

	// Installs the ExecuteCommandLists observer on DIRECT and COMPUTE queue vtables once per process.
	inline bool Install(ID3D12Device* device) noexcept
	{
		if (!device)
			return false;
		if (detail::S().installed.load(std::memory_order_acquire))
			return true;

		bool patchedAny = false;
		for (D3D12_COMMAND_LIST_TYPE type : { D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE })
		{
			D3D12_COMMAND_QUEUE_DESC desc{};
			desc.Type = type;
			Microsoft::WRL::ComPtr<ID3D12CommandQueue> tmpQueue;
			if (SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&tmpQueue))) && tmpQueue)
				patchedAny |= detail::PatchQueueVtable(tmpQueue.Get());
		}

		if (patchedAny)
		{
			detail::S().installed.store(true, std::memory_order_release);
			DeferredDestroy::SetQueueFenceTrackingEnabled(true);
		}
		return patchedAny;
	}

	// Best-effort retry for any retained queues whose Signal() previously failed, followed by a bounded wait for all
	// tracked per-queue fences during shutdown before DeferredDestroy::Collect(true).
	inline void WaitAllPending(DWORD timeoutMs = 1000) noexcept
	{
		detail::State& s = detail::S();

		std::vector<Microsoft::WRL::ComPtr<ID3D12CommandQueue>> retrySnapshot;
		{
			std::lock_guard lock(s.fenceMutex);
			retrySnapshot.reserve(s.retryQueues.size());
			for (const auto& kv : s.retryQueues)
			{
				if (kv.second)
					retrySnapshot.push_back(kv.second);
			}
		}

		for (auto& queue : retrySnapshot)
		{
			auto submitMutex = detail::GetQueueSubmitMutex(queue.Get());
			Microsoft::WRL::ComPtr<ID3D12CommandQueue> releasedRetryQueue;
			{
				std::lock_guard qlock(*submitMutex);
				const uint64_t seq = DeferredDestroy::NextQueueSubmitSeq();
				Microsoft::WRL::ComPtr<ID3D12Fence> fence;
				uint64_t value = 0;
				if (detail::SignalQueuePrivateFence(queue.Get(), fence, value))
				{
					DeferredDestroy::AttachFenceToQueueSignals(queue.Get(),
						[fence = std::move(fence), value]() noexcept -> bool {
							const uint64_t done = fence->GetCompletedValue();
							return done == UINT64_MAX || done >= value;
						}, seq);
					releasedRetryQueue = detail::DetachRetryQueueIfEmptyUnderSubmitLock(queue.Get());
				}
			}
		}

		std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Fence>, uint64_t>> snapshot;
		{
			std::lock_guard lock(s.fenceMutex);
			snapshot.reserve(s.queueFences.size() + s.retiredFences.size());
			for (const auto& kv : s.queueFences)
			{
				if (kv.second.fence && kv.second.nextValue > 0)
					snapshot.emplace_back(kv.second.fence, kv.second.nextValue);
			}
			for (const auto& item : s.retiredFences)
			{
				if (item.first && item.second > 0)
					snapshot.push_back(item);
			}
		}
		if (snapshot.empty())
			return;

		HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		const ULONGLONG deadline = GetTickCount64() + static_cast<ULONGLONG>(timeoutMs);
		for (const auto& item : snapshot)
		{
			const uint64_t done = item.first->GetCompletedValue();
			if (done == UINT64_MAX || done >= item.second)
				continue;
			const ULONGLONG now = GetTickCount64();
			if (now >= deadline)
				break;
			const DWORD remaining = static_cast<DWORD>(deadline - now);
			if (ev && SUCCEEDED(item.first->SetEventOnCompletion(item.second, ev)))
				WaitForSingleObject(ev, remaining);
		}
		if (ev)
			CloseHandle(ev);
	}
} // namespace D3D12QueueFence
