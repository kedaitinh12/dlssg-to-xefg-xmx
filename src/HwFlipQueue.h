#pragma once

#include <atomic>
#include <cstdint>

// Queries WDDM 3.0 Hardware Flip Queue (HwFlipQueueEnabled) for the active D3D12 adapter LUID once at startup,
// using dynamically loaded gdi32.dll D3DKMT entry points with self-contained structs (zero WDK header dependency).
// Learned from NSS-OFG (NFRU_Dx12.cpp): 0% per-frame cost, read-only, skipped safely under Wine/Proton.
namespace HwFlipQueue
{
	enum class State : int32_t
	{
		Unknown = -1,
		Disabled = 0,
		Enabled = 1,
	};

	inline std::atomic<int32_t> gState{ static_cast<int32_t>(State::Unknown) };

	inline State CachedState() noexcept
	{
		return static_cast<State>(gState.load(std::memory_order_acquire));
	}

	inline const char* StateName(State s) noexcept
	{
		switch (s)
		{
		case State::Enabled: return "enabled";
		case State::Disabled: return "disabled";
		default: return "unknown";
		}
	}

#ifdef _WIN32
	namespace detail
	{
		using NTSTATUS = long;
		using D3DKMT_HANDLE = uint32_t;
		constexpr uint32_t kKmtQaiTypeWddm30Caps = 77u; // KMTQAITYPE_WDDM_3_0_CAPS

		struct OpenAdapterFromLuid
		{
			LUID AdapterLuid;
			D3DKMT_HANDLE hAdapter;
		};

		struct QueryAdapterInfo
		{
			D3DKMT_HANDLE hAdapter;
			uint32_t Type;
			void* pPrivateDriverData;
			uint32_t PrivateDriverDataSize;
		};

		struct Wddm30Caps
		{
			uint32_t Value; // bit 2 = HwFlipQueueEnabled
		};

		struct CloseAdapter
		{
			D3DKMT_HANDLE hAdapter;
		};

		using OpenAdapterFromLuidFn = NTSTATUS(WINAPI*)(OpenAdapterFromLuid*);
		using QueryAdapterInfoFn = NTSTATUS(WINAPI*)(const QueryAdapterInfo*);
		using CloseAdapterFn = NTSTATUS(WINAPI*)(const CloseAdapter*);
	} // namespace detail

	inline State QueryOnce(ID3D12Device* device, bool skipUnderWine = false) noexcept
	{
		const State cached = CachedState();
		if (cached != State::Unknown)
			return cached;
		if (!device || skipUnderWine)
		{
			gState.store(static_cast<int32_t>(State::Disabled), std::memory_order_release);
			return State::Disabled;
		}

		State result = State::Disabled;
		HMODULE gdi = LoadLibraryExW(L"gdi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
		if (gdi)
		{
			auto openFn = reinterpret_cast<detail::OpenAdapterFromLuidFn>(
				GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid"));
			auto queryFn = reinterpret_cast<detail::QueryAdapterInfoFn>(
				GetProcAddress(gdi, "D3DKMTQueryAdapterInfo"));
			auto closeFn = reinterpret_cast<detail::CloseAdapterFn>(
				GetProcAddress(gdi, "D3DKMTCloseAdapter"));

			if (openFn && queryFn && closeFn)
			{
				detail::OpenAdapterFromLuid openArg{};
				openArg.AdapterLuid = device->GetAdapterLuid();
				if (openFn(&openArg) >= 0 && openArg.hAdapter != 0u)
				{
					detail::Wddm30Caps caps{};
					detail::QueryAdapterInfo queryArg{};
					queryArg.hAdapter = openArg.hAdapter;
					queryArg.Type = detail::kKmtQaiTypeWddm30Caps;
					queryArg.pPrivateDriverData = &caps;
					queryArg.PrivateDriverDataSize = sizeof(caps);
					if (queryFn(&queryArg) >= 0 && (caps.Value & (1u << 2)) != 0u)
						result = State::Enabled;

					detail::CloseAdapter closeArg{ openArg.hAdapter };
					closeFn(&closeArg);
				}
			}
			FreeLibrary(gdi);
		}

		gState.store(static_cast<int32_t>(result), std::memory_order_release);
		return result;
	}
#endif
} // namespace HwFlipQueue
