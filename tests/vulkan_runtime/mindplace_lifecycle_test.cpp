// Unit tests for ContextOwnershipCoordinator and fence-backed / backlog-safe DeferredDestroy.
#include "ContextLifecycle.h"
#include "Util.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace
{
	int gChecks = 0;
	int gFailed = 0;

	void Check(bool cond, const char* msg)
	{
		++gChecks;
		if (!cond)
		{
			++gFailed;
			std::fprintf(stderr, "FAIL: %s\n", msg);
		}
	}

	struct TrackedObject
	{
		std::atomic<int>* destroyedCount = nullptr;
		explicit TrackedObject(std::atomic<int>* c) : destroyedCount(c) {}
		TrackedObject(const TrackedObject&) = delete;
		TrackedObject& operator=(const TrackedObject&) = delete;
		~TrackedObject()
		{
			if (destroyedCount)
				destroyedCount->fetch_add(1, std::memory_order_relaxed);
		}
	};

	void TestFailedContextInitDoesNotStealOwnership()
	{
		ContextOwnershipCoordinator owner;
		int resets = 0;

		// Context A commits after XeFG::CreateContext + XeFG::Init succeed.
		const uint64_t genA = owner.Commit();
		Check(genA == 1, "Context A receives generation 1");

		// Context B fails XeFG::Init -> never calls Commit(), stays at generation 0.
		const uint64_t genB = 0;
		const bool bReset = owner.ResetIfOwner(genB, false, [&]() noexcept { ++resets; });
		Check(!bReset && resets == 0, "Failed Context B must not reset active bridge");
		Check(owner.ActiveGeneration() == genA, "Context A retains ownership after Context B init failure");

		// Later, when Context A is destroyed, A still owns and resets cleanly once.
		const bool aReset = owner.ResetIfOwner(genA, false, [&]() noexcept { ++resets; });
		Check(aReset && resets == 1, "Context A resets once when destroyed as active owner");
	}

	void TestReleaseFeatureThenCreateFeatureDoesNotWipeNewBridge()
	{
		ContextOwnershipCoordinator owner;
		int resets = 0;

		const uint64_t genA = owner.Commit();
		// Simulate ReleaseFeature(A) followed by CreateFeature(B) + successful CreateContext(B).
		const uint64_t genB = owner.Commit();
		Check(genB == genA + 1, "Context B advances active generation");

		// Deferred ~Context() of A runs later with retiredForReplacement == false.
		const bool aReset = owner.ResetIfOwner(genA, false, [&]() noexcept { ++resets; });
		Check(!aReset && resets == 0, "Stale deferred Context A destructor must not reset Context B");
		Check(owner.ActiveGeneration() == genB, "Context B remains active owner");
	}

	void TestConcurrentCommitAndDestructorSerialized()
	{
		ContextOwnershipCoordinator owner;
		std::atomic<int> resetCalls{ 0 };
		std::atomic<int> activeOwnerAtReset{ 0 };

		for (int iter = 0; iter < 200; ++iter)
		{
			owner.ResetForTests();
			const uint64_t genA = owner.Commit();
			std::atomic<bool> go{ false };
			uint64_t genB = 0;

			std::atomic<bool> bCommitted{ false };
			std::thread tDestroy([&]() {
				while (!go.load(std::memory_order_acquire)) {}
				owner.ResetIfOwner(genA, false, [&]() noexcept {
					resetCalls.fetch_add(1, std::memory_order_relaxed);
					if (bCommitted.load(std::memory_order_acquire))
						activeOwnerAtReset.fetch_add(1, std::memory_order_relaxed);
				});
			});
			std::thread tCommit([&]() {
				while (!go.load(std::memory_order_acquire)) {}
				genB = owner.Commit([&](uint64_t) noexcept {
					bCommitted.store(true, std::memory_order_release);
				});
			});

			go.store(true, std::memory_order_release);
			tDestroy.join();
			tCommit.join();

			// Regardless of thread scheduling, Context B MUST end up as the active owner and never be reset.
			Check(owner.ActiveGeneration() == genB && genB > genA, "Concurrent Commit always wins and stays intact");
		}
		Check(activeOwnerAtReset.load() == 0, "ResetIfOwner never ran while another owner was active");
	}

	void TestDeferredDestroyWaitsForQueueFencePast64Evaluations()
	{
		DeferredDestroy::ResetForTests();
		DeferredDestroy::SetQueueFenceTrackingEnabled(true);

		std::atomic<int> destroyed{ 0 };
		std::atomic<uint64_t> gpuCompletedValue{ 0 };
		const void* fakeCmdList = reinterpret_cast<const void*>(uintptr_t(0x12340000));

		{
			DeferredDestroy::ScopedCommandList scope(fakeCmdList);
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/false);
		}

		// Queue submits cmdList with fence value 1, but GPU has not completed it yet.
		const void* lists[1] = { fakeCmdList };
		Check(DeferredDestroy::HasAwaitingSubmitFor(lists, 1), "Command list is tracked awaiting submission");
		DeferredDestroy::MarkCommandListsSubmitted(lists, 1, [&]() noexcept {
			return gpuCompletedValue.load(std::memory_order_acquire) >= 1;
		});
		Check(DeferredDestroy::AwaitingSubmitCount() == 0, "Awaiting submit cleared after submission");

		// Run 128 evaluations while GPU fence is still 0 (not complete).
		for (int i = 0; i < 128; ++i)
			DeferredDestroy::Collect(false);
		Check(destroyed.load() == 0, "Resource must NOT be freed after 128 evaluations while queue fence is incomplete");
		Check(DeferredDestroy::PendingCount() == 1, "Entry stays pending until queue fence completes");

		// Now GPU completes fence value 1 -> next Collect() frees it immediately.
		gpuCompletedValue.store(1, std::memory_order_release);
		DeferredDestroy::Collect(false);
		Check(destroyed.load() == 1, "Resource is freed immediately once queue fence signals completion");
		Check(DeferredDestroy::PendingCount() == 0, "Pending count drops to 0 after fence completion");
	}

	void TestDeferredDestroyNoUnboundedBacklogAtTwoAddsPerEval()
	{
		DeferredDestroy::ResetForTests();
		std::atomic<int> destroyed{ 0 };
		auto readySig = DeferredDestroy::MakeCompletedFenceSignal([]() noexcept { return true; });

		// Add 2 lightweight entries per evaluation for 256 evaluations (the exact stress case from review).
		for (int i = 0; i < 256; ++i)
		{
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/false, readySig);
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/false, readySig);
			DeferredDestroy::Collect(false);
		}
		Check(destroyed.load() == 512, "All 512 GPU-ready lightweight entries drained with zero backlog");
		Check(DeferredDestroy::PendingCount() == 0, "Pending count is 0 after 256 evals of 2 lightweight adds/eval");
	}

	void TestDeferredDestroyHeavyPacingAndBacklogCap()
	{
		DeferredDestroy::ResetForTests();
		std::atomic<int> destroyed{ 0 };
		auto readySig = DeferredDestroy::MakeCompletedFenceSignal([]() noexcept { return true; });

		// Enqueue 4 ready heavy entries: should drain paced at 1 per Collect().
		for (int i = 0; i < 4; ++i)
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/true, readySig);

		DeferredDestroy::Collect(false);
		Check(destroyed.load() == 1 && DeferredDestroy::PendingCount() == 3, "Heavy entries pace at 1/eval within cap");

		// Enqueue 6 more (total 9 ready heavy entries > MaxHeavyBacklog=4): excess drains so pending stays <= 3.
		for (int i = 0; i < 6; ++i)
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/true, readySig);

		DeferredDestroy::Collect(false);
		Check(DeferredDestroy::PendingCount() <= DeferredDestroy::MaxHeavyBacklog, "Heavy backlog stays strictly bounded");

		DeferredDestroy::Collect(true);
		Check(destroyed.load() == 10 && DeferredDestroy::PendingCount() == 0, "Collect(true) drains all remaining entries");
	}
} // namespace

int main()
{
	TestFailedContextInitDoesNotStealOwnership();
	TestReleaseFeatureThenCreateFeatureDoesNotWipeNewBridge();
	TestConcurrentCommitAndDestructorSerialized();
	TestDeferredDestroyWaitsForQueueFencePast64Evaluations();
	TestDeferredDestroyNoUnboundedBacklogAtTwoAddsPerEval();
	TestDeferredDestroyHeavyPacingAndBacklogCap();

	std::printf("mindplace_lifecycle_test: %d checks, %d failed\n", gChecks, gFailed);
	return gFailed == 0 ? 0 : 1;
}
