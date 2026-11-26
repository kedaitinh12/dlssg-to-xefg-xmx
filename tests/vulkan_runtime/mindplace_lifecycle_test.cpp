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

		const uint64_t genA = owner.Commit();
		Check(genA == 1, "Context A receives generation 1");

		const uint64_t genB = 0;
		const bool bReset = owner.ResetIfOwner(genB, false, [&]() noexcept { ++resets; });
		Check(!bReset && resets == 0, "Failed Context B must not reset active bridge");
		Check(owner.ActiveGeneration() == genA, "Context A retains ownership after Context B init failure");

		const bool aReset = owner.ResetIfOwner(genA, false, [&]() noexcept { ++resets; });
		Check(aReset && resets == 1, "Context A resets once when destroyed as active owner");
	}

	void TestStaleResetBeforeReplacementCannotResetNewOwner()
	{
		ContextOwnershipCoordinator owner;
		int resets = 0;

		const uint64_t genA = owner.Commit();
		const uint64_t genB = owner.Commit();
		Check(genB > genA && owner.ActiveGeneration() == genB, "Context B owns bridge after Commit()");

		const bool zeroReset = owner.ResetBeforeReplacement(0, [&]() noexcept { ++resets; });
		Check(!zeroReset && resets == 0, "ResetBeforeReplacement(0) returns false without resetting");

		const bool staleReset = owner.ResetBeforeReplacement(genA, [&]() noexcept { ++resets; });
		Check(!staleReset && resets == 0, "ResetBeforeReplacement(genA) returns false without resetting Context B");
		Check(owner.ActiveGeneration() == genB, "Context B remains active owner after stale ResetBeforeReplacement(genA)");

		const bool activeReset = owner.ResetBeforeReplacement(genB, [&]() noexcept { ++resets; });
		Check(activeReset && resets == 1, "ResetBeforeReplacement(genB) resets once when B is active owner");
		Check(owner.ActiveGeneration() == 0, "Active generation cleared after owner ResetBeforeReplacement");
	}

	void TestReleaseFeatureThenCreateFeatureDoesNotWipeNewBridge()
	{
		ContextOwnershipCoordinator owner;
		int resets = 0;

		const uint64_t genA = owner.Commit();
		const uint64_t genB = owner.Commit();
		Check(genB == genA + 1, "Context B advances active generation");

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

		const void* lists[1] = { fakeCmdList };
		Check(DeferredDestroy::HasAwaitingSubmitFor(lists, 1), "Command list is tracked awaiting submission");
		DeferredDestroy::MarkCommandListsSubmitted(lists, 1, [&]() noexcept {
			return gpuCompletedValue.load(std::memory_order_acquire) >= 1;
		});
		Check(DeferredDestroy::AwaitingSubmitCount() == 0, "Awaiting submit cleared after submission");

		for (int i = 0; i < 128; ++i)
			DeferredDestroy::Collect(false);
		Check(destroyed.load() == 0, "Resource must NOT be freed after 128 evaluations while queue fence is incomplete");
		Check(DeferredDestroy::PendingCount() == 1, "Entry stays pending until queue fence completes");

		gpuCompletedValue.store(1, std::memory_order_release);
		DeferredDestroy::Collect(false);
		Check(destroyed.load() == 1, "Resource is freed immediately once queue fence signals completion");
		Check(DeferredDestroy::PendingCount() == 0, "Pending count drops to 0 after fence completion");
	}

	void TestMultiQueueRetirementAndTextureReplacementWaitForAllQueues()
	{
		DeferredDestroy::ResetForTests();
		DeferredDestroy::SetQueueFenceTrackingEnabled(true);

		std::atomic<int> genDestroyed{ 0 };
		std::atomic<int> texDestroyed{ 0 };
		std::atomic<bool> directDone{ false };
		std::atomic<bool> computeDone{ false };
		const void* directCmd = reinterpret_cast<const void*>(uintptr_t(0x20000000));
		const void* computeCmd = reinterpret_cast<const void*>(uintptr_t(0x30000000));

		std::vector<std::shared_ptr<DeferredDestroy::FenceSignal>> activeSignals;
		{
			DeferredDestroy::ScopedCommandList s1(directCmd);
			activeSignals.push_back(s1.current);
		}
		DeferredDestroy::MarkCommandListsSubmitted(&directCmd, 1, [&]() noexcept {
			return directDone.load(std::memory_order_acquire);
		});

		{
			DeferredDestroy::ScopedCommandList s2(computeCmd);
			activeSignals.push_back(s2.current);
			DeferredDestroy::ScopedSignals scoped(activeSignals);
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&texDestroyed), /*heavy=*/false);
		}
		DeferredDestroy::MarkCommandListsSubmitted(&computeCmd, 1, [&]() noexcept {
			return computeDone.load(std::memory_order_acquire);
		});

		computeDone.store(true, std::memory_order_release);
		DeferredDestroy::Add(std::make_shared<TrackedObject>(&genDestroyed), /*heavy=*/true, activeSignals);

		for (int i = 0; i < 10; ++i)
			DeferredDestroy::Collect(false);
		Check(texDestroyed.load() == 0, "Replaced texture must NOT be freed while Direct queue is still reading it");
		Check(genDestroyed.load() == 0, "Generator must NOT be freed while Direct queue is still busy even if Compute finished");

		directDone.store(true, std::memory_order_release);
		DeferredDestroy::Collect(false);
		Check(texDestroyed.load() == 1 && genDestroyed.load() == 1 && DeferredDestroy::PendingCount() == 0,
			"Replaced texture and generator are freed once all involved queues complete");
	}

	void TestMultipleEvaluatesPerOpenCommandListCoalesceWithZeroLeak()
	{
		DeferredDestroy::ResetForTests();
		DeferredDestroy::SetQueueFenceTrackingEnabled(true);

		std::atomic<int> destroyed{ 0 };
		const void* cmdList = reinterpret_cast<const void*>(uintptr_t(0x40000000));

		for (int round = 0; round < 256; ++round)
		{
			std::shared_ptr<DeferredDestroy::FenceSignal> sig1;
			std::shared_ptr<DeferredDestroy::FenceSignal> sig2;
			{
				DeferredDestroy::ScopedCommandList s1(cmdList);
				sig1 = s1.current;
				DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/false);
			}
			{
				DeferredDestroy::ScopedCommandList s2(cmdList);
				sig2 = s2.current;
				DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/false);
			}
			if (round == 0)
			{
				Check(sig1 == sig2, "Two Evaluates on the same open command list coalesce onto one FenceSignal");
				Check(DeferredDestroy::AwaitingSubmitCount() == 1, "Coalesced open recording occupies 1 awaitingSubmit slot");
			}
			DeferredDestroy::MarkCommandListsSubmitted(&cmdList, 1, []() noexcept { return true; });
			DeferredDestroy::Collect(false);
		}
		Check(DeferredDestroy::AwaitingSubmitCount() == 0, "Zero orphaned signals left in awaitingSubmit after 256 rounds");
		Check(destroyed.load() == 512 && DeferredDestroy::PendingCount() == 0,
			"All 512 resources from 256 coalesced 2-eval batches freed cleanly");
	}

	void TestPreSubmitClaimPreventsConcurrentRecordingFromInheritingOldFence()
	{
		DeferredDestroy::ResetForTests();
		DeferredDestroy::SetQueueFenceTrackingEnabled(true);

		std::atomic<int> destroyed1{ 0 };
		std::atomic<int> destroyed2{ 0 };
		std::atomic<bool> fence1Done{ true };
		std::atomic<bool> fence2Done{ false };
		const void* reusedCmdList = reinterpret_cast<const void*>(uintptr_t(0x50000000));

		// Recording 1 on reusedCmdList.
		std::shared_ptr<DeferredDestroy::FenceSignal> sig1;
		{
			DeferredDestroy::ScopedCommandList s1(reusedCmdList);
			sig1 = s1.current;
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed1), /*heavy=*/false);
		}

		// HookExecuteCommandLists claims Recording 1 BEFORE forwarding orig(ExecuteCommandLists).
		auto claimed1 = DeferredDestroy::ClaimAwaitingSubmits(&reusedCmdList, 1);
		Check(claimed1.size() == 1 && DeferredDestroy::AwaitingSubmitCount() == 0,
			"Pre-submit claim removes Recording 1 before forwarding ExecuteCommandLists");

		// Concurrently (during/right after origExec), another thread reuses the exact same cmdList address for Recording 2.
		std::shared_ptr<DeferredDestroy::FenceSignal> sig2;
		{
			DeferredDestroy::ScopedCommandList s2(reusedCmdList);
			sig2 = s2.current;
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed2), /*heavy=*/false);
		}
		Check(sig1 != sig2 && sig2->recordingId > sig1->recordingId,
			"Concurrent recording on reused cmdList gets a brand-new FenceSignal");

		// Recording 1's fence completes -> ONLY Recording 1 is freed; Recording 2 MUST stay alive!
		DeferredDestroy::CompleteClaimedSubmits(std::move(claimed1), [&]() noexcept {
			return fence1Done.load(std::memory_order_acquire);
		});
		DeferredDestroy::Collect(false);
		Check(destroyed1.load() == 1, "Recording 1 freed when fence 1 completes");
		Check(destroyed2.load() == 0, "Concurrent Recording 2 MUST NOT inherit Recording 1's completed fence");

		// Recording 2 later submits and waits for fence 2.
		DeferredDestroy::MarkCommandListsSubmitted(&reusedCmdList, 1, [&]() noexcept {
			return fence2Done.load(std::memory_order_acquire);
		});
		DeferredDestroy::Collect(false);
		Check(destroyed2.load() == 0, "Recording 2 waits until fence 2 completes");

		fence2Done.store(true, std::memory_order_release);
		DeferredDestroy::Collect(false);
		Check(destroyed2.load() == 1 && DeferredDestroy::PendingCount() == 0,
			"Recording 2 freed once fence 2 completes");
	}

	void TestEarlierFenceCannotSatisfyLaterFailedSignalOnSameQueue()
	{
		DeferredDestroy::ResetForTests();
		DeferredDestroy::SetQueueFenceTrackingEnabled(true);

		std::atomic<int> destroyedA{ 0 };
		std::atomic<int> destroyedB{ 0 };
		const void* queue = reinterpret_cast<const void*>(uintptr_t(0xB0000001));
		const void* cmdA = reinterpret_cast<const void*>(uintptr_t(0x63000000));
		const void* cmdB = reinterpret_cast<const void*>(uintptr_t(0x64000000));

		// Thread A records and submits first (seqA = 1), Signal() succeeds, then pauses before CompleteClaimedSubmits.
		{
			DeferredDestroy::ScopedCommandList sA(cmdA);
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyedA), /*heavy=*/false);
		}
		auto claimedA = DeferredDestroy::ClaimAwaitingSubmits(&cmdA, 1);
		const uint64_t seqA = DeferredDestroy::NextQueueSubmitSeq();

		// Thread B submits AFTER Thread A on the SAME queue (seqB = 2), Signal() fails -> queued with seqB.
		{
			DeferredDestroy::ScopedCommandList sB(cmdB);
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyedB), /*heavy=*/false);
		}
		auto claimedB = DeferredDestroy::ClaimAwaitingSubmits(&cmdB, 1);
		const uint64_t seqB = DeferredDestroy::NextQueueSubmitSeq();
		Check(seqB > seqA, "Later submission on same queue receives higher submitSeq");
		DeferredDestroy::MarkSignalsAcceptedWithoutFence(queue, std::move(claimedB), seqB);

		// Thread A resumes and attaches its ALREADY-COMPLETE fence at seqA.
		DeferredDestroy::CompleteClaimedSubmits(std::move(claimedA), []() noexcept { return true; }, queue, seqA);
		DeferredDestroy::Collect(false);
		DeferredDestroy::Collect(true);
		Check(destroyedA.load() == 1, "Thread A freed by its own fence");
		Check(destroyedB.load() == 0 && DeferredDestroy::AwaitingQueueSignalCount() == 1,
			"Thread A's older fence (seqA) MUST NOT satisfy Thread B's newer failed-Signal submission (seqB)");

		// Later submission seqC > seqB on the SAME queue succeeds and completes -> Thread B is now freed cleanly.
		const uint64_t seqC = DeferredDestroy::NextQueueSubmitSeq();
		DeferredDestroy::AttachFenceToQueueSignals(queue, []() noexcept { return true; }, seqC);
		DeferredDestroy::Collect(false);
		Check(destroyedB.load() == 1 && DeferredDestroy::AwaitingQueueSignalCount() == 0 && DeferredDestroy::PendingCount() == 0,
			"Thread B is freed once a later fence (seqC > seqB) on the same queue completes");
	}

	void TestFailedQueueSignalRecoversOnLaterSubmissionOfSameQueue()
	{
		DeferredDestroy::ResetForTests();
		DeferredDestroy::SetQueueFenceTrackingEnabled(true);

		std::atomic<int> destroyed1{ 0 };
		std::atomic<int> destroyed2{ 0 };
		std::atomic<bool> queue1LaterFenceDone{ false };
		const void* queue1 = reinterpret_cast<const void*>(uintptr_t(0xA0000001));
		const void* queue2 = reinterpret_cast<const void*>(uintptr_t(0xA0000002));
		const void* cmd1 = reinterpret_cast<const void*>(uintptr_t(0x61000000));
		const void* cmd2 = reinterpret_cast<const void*>(uintptr_t(0x62000000));

		{
			DeferredDestroy::ScopedCommandList s1(cmd1);
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed1), /*heavy=*/false);
		}
		DeferredDestroy::MarkCommandListsAcceptedWithoutFence(queue1, &cmd1, 1);
		Check(DeferredDestroy::AwaitingSubmitCount() == 0, "Accepted cmd1 moved out of awaitingSubmit");
		Check(DeferredDestroy::AwaitingQueueSignalCount() == 1 && DeferredDestroy::HasAwaitingQueueSignalFor(queue1),
			"Accepted cmd1 tracked against queue1 awaiting fence retry");

		DeferredDestroy::AttachFenceToQueueSignals(queue2, []() noexcept { return true; });
		DeferredDestroy::Collect(false);
		DeferredDestroy::Collect(true);
		Check(destroyed1.load() == 0 && DeferredDestroy::AwaitingQueueSignalCount() == 1,
			"Failed Signal() entry stays safe across Collect(true) and unrelated queue2 fences");

		{
			DeferredDestroy::ScopedCommandList s2(cmd2);
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed2), /*heavy=*/false);
		}
		const uint64_t seq2 = DeferredDestroy::NextQueueSubmitSeq();
		DeferredDestroy::MarkCommandListsSubmitted(&cmd2, 1, [&]() noexcept {
			return queue1LaterFenceDone.load(std::memory_order_acquire);
		}, queue1, seq2);
		Check(DeferredDestroy::AwaitingQueueSignalCount() == 0, "queue1 retry list drained once later fence attached");

		DeferredDestroy::Collect(true);
		Check(destroyed1.load() == 0 && destroyed2.load() == 0, "Both submissions wait until queue1's later fence completes");

		queue1LaterFenceDone.store(true, std::memory_order_release);
		DeferredDestroy::Collect(true);
		Check(destroyed1.load() == 1 && destroyed2.load() == 1 && DeferredDestroy::PendingCount() == 0,
			"Earlier failed-Signal() submission is freed once a later fence on the same queue completes");
	}

	void TestUnconfirmedSubmitNeverBypassesFenceAt256()
	{
		DeferredDestroy::ResetForTests();
		DeferredDestroy::SetQueueFenceTrackingEnabled(true);

		std::atomic<int> destroyed{ 0 };
		const void* fakeCmdList = reinterpret_cast<const void*>(uintptr_t(0x60000000));

		{
			DeferredDestroy::ScopedCommandList scope(fakeCmdList);
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/false);
		}

		for (int i = 0; i < 512; ++i)
			DeferredDestroy::Collect(false);
		Check(destroyed.load() == 0, "Unconfirmed submit must NEVER be destroyed at Collect #256 or #512");
		Check(DeferredDestroy::PendingCount() == 1, "Unconfirmed entry remains pending");

		DeferredDestroy::MarkCommandListsSubmitted(&fakeCmdList, 1, []() noexcept { return true; });
		DeferredDestroy::Collect(false);
		Check(destroyed.load() == 1 && DeferredDestroy::PendingCount() == 0,
			"Entry is freed once submission and fence completion are confirmed");
	}

	void TestShutdownCollectAllRespectsIncompleteAndUnsubmittedFences()
	{
		DeferredDestroy::ResetForTests();
		DeferredDestroy::SetQueueFenceTrackingEnabled(true);

		std::atomic<int> destroyedReady{ 0 };
		std::atomic<int> destroyedIncomplete{ 0 };
		std::atomic<int> destroyedUnsubmitted{ 0 };
		std::atomic<bool> incompleteFenceDone{ false };
		const void* unsubmittedCmd = reinterpret_cast<const void*>(uintptr_t(0x70000000));

		auto readySig = DeferredDestroy::MakeCompletedFenceSignal([]() noexcept { return true; });
		auto incompleteSig = DeferredDestroy::MakeCompletedFenceSignal([&]() noexcept {
			return incompleteFenceDone.load(std::memory_order_acquire);
		});

		DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyedReady), /*heavy=*/true, readySig);
		DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyedIncomplete), /*heavy=*/true, incompleteSig);
		{
			DeferredDestroy::ScopedCommandList scope(unsubmittedCmd);
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyedUnsubmitted), /*heavy=*/false);
		}

		DeferredDestroy::Collect(true);
		Check(destroyedReady.load() == 1, "Collect(true) frees GPU-complete entries");
		Check(destroyedIncomplete.load() == 0, "Collect(true) must NOT destroy entry with incomplete fence");
		Check(destroyedUnsubmitted.load() == 0, "Collect(true) must NOT destroy entry with unsubmitted command list");
		Check(DeferredDestroy::PendingCount() == 2, "Both in-flight entries remain pending after Collect(true)");
		Check(DeferredDestroy::AwaitingSubmitCount() == 1, "Unsubmitted signal stays tracked after Collect(true)");

		incompleteFenceDone.store(true, std::memory_order_release);
		DeferredDestroy::MarkCommandListsSubmitted(&unsubmittedCmd, 1, []() noexcept { return true; });
		DeferredDestroy::Collect(true);
		Check(destroyedIncomplete.load() == 1 && destroyedUnsubmitted.load() == 1 && DeferredDestroy::PendingCount() == 0,
			"Collect(true) frees remaining entries once their fences complete");
	}

	void TestDeferredDestroyNoUnboundedBacklogAtTwoAddsPerEval()
	{
		DeferredDestroy::ResetForTests();
		std::atomic<int> destroyed{ 0 };
		auto readySig = DeferredDestroy::MakeCompletedFenceSignal([]() noexcept { return true; });

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

		for (int i = 0; i < 4; ++i)
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/true, readySig);

		DeferredDestroy::Collect(false);
		Check(destroyed.load() == 1 && DeferredDestroy::PendingCount() == 3, "Heavy entries pace at 1/eval within cap");

		for (int i = 0; i < 6; ++i)
			DeferredDestroy::Add(std::make_shared<TrackedObject>(&destroyed), /*heavy=*/true, readySig);

		DeferredDestroy::Collect(false);
		Check(DeferredDestroy::PendingCount() <= DeferredDestroy::MaxHeavyBacklog, "Heavy backlog stays strictly bounded");

		DeferredDestroy::Collect(true);
		Check(destroyed.load() == 10 && DeferredDestroy::PendingCount() == 0, "Collect(true) drains all remaining ready entries");
	}
} // namespace

int main()
{
	TestFailedContextInitDoesNotStealOwnership();
	TestStaleResetBeforeReplacementCannotResetNewOwner();
	TestReleaseFeatureThenCreateFeatureDoesNotWipeNewBridge();
	TestConcurrentCommitAndDestructorSerialized();
	TestDeferredDestroyWaitsForQueueFencePast64Evaluations();
	TestMultiQueueRetirementAndTextureReplacementWaitForAllQueues();
	TestMultipleEvaluatesPerOpenCommandListCoalesceWithZeroLeak();
	TestPreSubmitClaimPreventsConcurrentRecordingFromInheritingOldFence();
	TestEarlierFenceCannotSatisfyLaterFailedSignalOnSameQueue();
	TestFailedQueueSignalRecoversOnLaterSubmissionOfSameQueue();
	TestUnconfirmedSubmitNeverBypassesFenceAt256();
	TestShutdownCollectAllRespectsIncompleteAndUnsubmittedFences();
	TestDeferredDestroyNoUnboundedBacklogAtTwoAddsPerEval();
	TestDeferredDestroyHeavyPacingAndBacklogCap();

	std::printf("mindplace_lifecycle_test: %d checks, %d failed\n", gChecks, gFailed);
	return gFailed == 0 ? 0 : 1;
}
