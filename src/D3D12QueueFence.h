#pragma once

#include "Util.h"

#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>

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
// - Serializes per-queue ExecuteCommandLists + Signal + submitSeq so an earlier fence can never satisfy a newer
//   submission on the same queue whose Signal() failed, and retains ComPtr<ID3D12CommandQueue> while retry work
//   remains pending so shutdown WaitAllPending() never dereferences a released queue.
// - Chain-safe with D3D12VulkanWmma's vtable hook regardless of installation order.
namespace D3D12QueueFence
{
	namespace detail
	{
		constexpr size_t kExecuteCommandListsSlot = 10;
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
			uint64_t nextValue = 0;
		};

		struct State
		{
			std::mutex hookMutex;
			std::array<QueueHookEntry, 8> hooks{};
			std::atomic<size_t> hookCount{ 0 };
			std::atomic<bool> installed{ false };

			std::mutex fenceMutex;
			std::unordered_map<ID3D12CommandQueue*, PerQueueFence> queueFences;
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

		inline void RetainRetryQueue(ID3D12CommandQueue* queue) noexcept
		{
			if (!queue)
				return;
			State& s = S();
			std::lock_guard lock(s.fenceMutex);
			s.retryQueues[queue] = queue;
		}

		inline void ReleaseRetryQueueIfEmpty(ID3D12CommandQueue* queue) noexcept
		{
			if (!queue || DeferredDestroy::HasAwaitingQueueSignalFor(queue))
				return;
			Microsoft::WRL::ComPtr<ID3D12CommandQueue> released;
			{
				State& s = S();
				std::lock_guard lock(s.fenceMutex);
				// Re-check under fenceMutex in case another thread just retained it.
				if (DeferredDestroy::HasAwaitingQueueSignalFor(queue))
					return;
				auto it = s.retryQueues.find(queue);
				if (it != s.retryQueues.end())
				{
					released = std::move(it->second);
					s.retryQueues.erase(it);
				}
			}
			// ComPtr released outside fenceMutex.
		}

		inline bool SignalQueuePrivateFence(ID3D12CommandQueue* queue,
			Microsoft::WRL::ComPtr<ID3D12Fence>& outFence, uint64_t& outValue) noexcept
		{
			if (!queue)
				return false;
			State& s = S();
			std::lock_guard lock(s.fenceMutex);
			auto& slot = s.queueFences[queue];
			if (!slot.fence)
			{
				Microsoft::WRL::ComPtr<ID3D12Device> dev;
				if (FAILED(queue->GetDevice(IID_PPV_ARGS(&dev))) || !dev)
					return false;
				if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&slot.fence))) || !slot.fence)
					return false;
			}
			const uint64_t val = ++slot.nextValue;
			if (FAILED(queue->Signal(slot.fence.Get(), val)))
				return false;
			outFence = slot.fence;
			outValue = val;
			return true;
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
			Microsoft::WRL::ComPtr<ID3D12Fence> fence;
			uint64_t value = 0;
			uint64_t submitSeq = 0;
			bool signaled = false;
			{
				std::lock_guard qlock(*submitMutex);
				orig(self, numCommandLists, ppCommandLists);
				submitSeq = DeferredDestroy::NextQueueSubmitSeq();
				signaled = SignalQueuePrivateFence(self, fence, value);
				if (!signaled && !claimed.empty())
				{
					// Retain queue lifetime and record failed-Signal submission while still holding submitMutex
					// so any subsequent submission on this queue observes it with a strictly greater submitSeq.
					RetainRetryQueue(self);
					DeferredDestroy::MarkSignalsAcceptedWithoutFence(self, std::move(claimed), submitSeq);
				}
			}

			if (signaled)
			{
				auto pred = [fence = std::move(fence), value]() noexcept -> bool {
					const uint64_t done = fence->GetCompletedValue();
					return done == UINT64_MAX || done >= value;
				};
				DeferredDestroy::CompleteClaimedSubmits(std::move(claimed), std::move(pred), self, submitSeq);
				ReleaseRetryQueueIfEmpty(self);
			}
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
			Microsoft::WRL::ComPtr<ID3D12Fence> fence;
			uint64_t value = 0;
			uint64_t seq = 0;
			bool signaled = false;
			{
				std::lock_guard qlock(*submitMutex);
				seq = DeferredDestroy::NextQueueSubmitSeq();
				signaled = detail::SignalQueuePrivateFence(queue.Get(), fence, value);
			}
			if (signaled)
			{
				DeferredDestroy::AttachFenceToQueueSignals(queue.Get(),
					[fence = std::move(fence), value]() noexcept -> bool {
						const uint64_t done = fence->GetCompletedValue();
						return done == UINT64_MAX || done >= value;
					}, seq);
				detail::ReleaseRetryQueueIfEmpty(queue.Get());
			}
		}

		std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Fence>, uint64_t>> snapshot;
		{
			std::lock_guard lock(s.fenceMutex);
			snapshot.reserve(s.queueFences.size());
			for (const auto& kv : s.queueFences)
			{
				if (kv.second.fence && kv.second.nextValue > 0)
					snapshot.emplace_back(kv.second.fence, kv.second.nextValue);
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
