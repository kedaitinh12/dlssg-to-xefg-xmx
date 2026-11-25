#pragma once

#include "Util.h"

#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

// Lightweight, chain-safe ID3D12CommandQueue::ExecuteCommandLists observer that attaches a per-queue private
// ID3D12Fence signal to DeferredDestroy entries recorded on a submitted command list.
// - Fast path when no entries await submission: single atomic load (~0 overhead).
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

			if (DeferredDestroy::AwaitingSubmitCount() == 0 || !numCommandLists || !ppCommandLists)
			{
				orig(self, numCommandLists, ppCommandLists);
				return;
			}

			orig(self, numCommandLists, ppCommandLists);

			if (!DeferredDestroy::HasAwaitingSubmitFor(reinterpret_cast<const void* const*>(ppCommandLists), numCommandLists))
				return;

			Microsoft::WRL::ComPtr<ID3D12Fence> fence;
			uint64_t value = 0;
			if (SignalQueuePrivateFence(self, fence, value))
			{
				DeferredDestroy::MarkCommandListsSubmitted(reinterpret_cast<const void* const*>(ppCommandLists), numCommandLists,
					[fence = std::move(fence), value]() noexcept -> bool {
						const uint64_t done = fence->GetCompletedValue();
						return done == UINT64_MAX || done >= value;
					});
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

	// Best-effort bounded wait for all tracked per-queue fences during shutdown before DeferredDestroy::Collect(true).
	inline void WaitAllPending(DWORD timeoutMs = 1000) noexcept
	{
		detail::State& s = detail::S();
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
