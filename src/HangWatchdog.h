// Opt-in 15-second Hang Watchdog (MFGAdaUnlock-RenoDx 1.12.0 inspiration).
// Active ONLY when Plugin::options.forceDred is enabled (0% overhead, zero atomics written, and zero threads when OFF).
#pragma once

#include "Log.h"
#include "Plugin.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace HangWatchdog
{
	namespace detail
	{
		enum class Lifecycle : uint8_t
		{
			Stopped = 0,
			Running = 1,
			Stopping = 2,
		};

		struct State
		{
			std::atomic<uint64_t> lastHeartbeatMs{ 0 };
			std::atomic<uint64_t> frameSeq{ 0 };
			std::atomic<const char*> lastStage{ "init" };
			std::atomic<bool> inFlight{ false };
			std::atomic<bool> reportedForStall{ false };
			std::atomic<Lifecycle> lifecycle{ Lifecycle::Stopped };

			std::mutex lifecycleMutex;
			std::condition_variable stopCv;
			bool stopRequested = false;
			std::thread worker;
		};

		inline State& S() noexcept
		{
			static State s;
			return s;
		}

		inline uint64_t NowMs() noexcept
		{
			return static_cast<uint64_t>(
				std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::steady_clock::now().time_since_epoch())
					.count());
		}

		inline bool EnsureStarted() noexcept
		{
			State& s = S();
			const Lifecycle cur = s.lifecycle.load(std::memory_order_acquire);
			if (cur == Lifecycle::Running)
				return true;
			if (cur == Lifecycle::Stopping)
				return false;

			std::lock_guard lock(s.lifecycleMutex);
			const Lifecycle lockedCur = s.lifecycle.load(std::memory_order_relaxed);
			if (lockedCur == Lifecycle::Running)
				return true;
			if (lockedCur == Lifecycle::Stopping)
				return false;

			s.stopRequested = false;
			try
			{
				s.worker = std::thread([]() noexcept {
					constexpr uint64_t kHangThresholdMs = 15'000ull;
					State& st = S();
					std::unique_lock lk(st.lifecycleMutex);
					while (!st.stopRequested)
					{
						if (st.stopCv.wait_for(lk, std::chrono::seconds(1), [&st]() { return st.stopRequested; }))
							break;
						if (!st.inFlight.load(std::memory_order_acquire))
							continue;
						const uint64_t last = st.lastHeartbeatMs.load(std::memory_order_acquire);
						if (last == 0)
							continue;
						const uint64_t now = NowMs();
						if (now > last && (now - last) >= kHangThresholdMs)
						{
							if (!st.reportedForStall.exchange(true, std::memory_order_acq_rel))
							{
								const char* stage = st.lastStage.load(std::memory_order_acquire);
								const uint64_t seq = st.frameSeq.load(std::memory_order_acquire);
								lk.unlock();
								Log::Warn("[HangWatchdog] No FG frame progress for {}s (stage='{}', frameSeq={})",
									(now - last) / 1000ull, stage ? stage : "unknown", seq);
								lk.lock();
							}
						}
					}
				});
				s.lifecycle.store(Lifecycle::Running, std::memory_order_release);
				return true;
			}
			catch (...)
			{
				s.lifecycle.store(Lifecycle::Stopped, std::memory_order_release);
				return false;
			}
		}
	} // namespace detail

	inline void Beat(const char* stage, bool advanceSeq = false) noexcept
	{
		if (!Plugin::options.forceDred.load(std::memory_order_relaxed))
			return;
		if (!detail::EnsureStarted())
			return;
		detail::State& s = detail::S();
		s.lastStage.store(stage ? stage : "unknown", std::memory_order_release);
		if (advanceSeq)
			s.frameSeq.fetch_add(1u, std::memory_order_relaxed);
		s.lastHeartbeatMs.store(detail::NowMs(), std::memory_order_release);
		s.inFlight.store(true, std::memory_order_release);
		s.reportedForStall.store(false, std::memory_order_release);
	}

	inline void EndFrame() noexcept
	{
		if (!Plugin::options.forceDred.load(std::memory_order_relaxed))
			return;
		detail::State& s = detail::S();
		s.inFlight.store(false, std::memory_order_release);
		s.reportedForStall.store(false, std::memory_order_release);
	}

	inline void Shutdown() noexcept
	{
		detail::State& s = detail::S();
		std::thread toJoin;
		{
			std::lock_guard lock(s.lifecycleMutex);
			if (s.lifecycle.load(std::memory_order_relaxed) != detail::Lifecycle::Running)
				return;
			s.lifecycle.store(detail::Lifecycle::Stopping, std::memory_order_release);
			s.stopRequested = true;
			s.inFlight.store(false, std::memory_order_release);
			toJoin = std::move(s.worker);
		}
		s.stopCv.notify_all();
		if (toJoin.joinable())
			toJoin.join();
		{
			std::lock_guard lock(s.lifecycleMutex);
			s.lifecycle.store(detail::Lifecycle::Stopped, std::memory_order_release);
		}
	}
} // namespace HangWatchdog
