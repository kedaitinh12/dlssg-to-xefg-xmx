#include "Plugin.h"
#include "HudTierPolicy.h"
#include "AmdFlipMeterReg.h"
#include "WineCompat.h"
#include "Log.h"
#include "Util.h"
#include "VulkanPrograms.h"
#include "VulkanProvider.h"
#include "VulkanRuntime.h"
#include "XeFG.h"
#include "loader/EarlyLoaderCore.h"
#include "optiscaler_plugin.h"
#include "vulkan/VulkanPolicy.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <mutex>
#include <thread>
#include <string>
#include <string_view>
#include <system_error>

namespace
{
	const OptiPluginApi* api = nullptr;
	OptiPlugin* self = nullptr;

	std::mutex amdFlipMeterMutex;
	AmdFlipMeterReg::ScanResult amdFlipMeterState;
	bool amdFlipMeterScanned = false;

	void RefreshAmdFlipMeterState(bool tryAutoApply) noexcept
	{
		try
		{
			if (WineCompat::IsWineOrProton())
			{
				std::scoped_lock lock(amdFlipMeterMutex);
				if (!amdFlipMeterScanned)
				{
					Log::Info("AMD RDNA 4 FlipMeter: skipped under Wine/Proton (Windows Adrenalin registry only)");
					amdFlipMeterScanned = true;
				}
				return;
			}

			auto scan = AmdFlipMeterReg::ScanRdna4Adapters();
			if (tryAutoApply && !scan.rdna4Adapters.empty() && !scan.allConfigured)
			{
				if (AmdFlipMeterReg::TryDirectApply(scan, false))
				{
					scan = AmdFlipMeterReg::ScanRdna4Adapters();
					scan.appliedThisSession = true;
					scan.rebootRecommended = true;
					Log::Info("AMD RDNA 4 FlipMeter: auto-applied Gmx_DriverFramePacingMaxGenFrames=5 (6x) to {} RDNA 4 adapter(s); restart Windows to activate",
						scan.rdna4Adapters.size());
				}
				else
				{
					for (const auto& a : scan.rdna4Adapters)
					{
						if (!a.valuePresent || a.currentValue < AmdFlipMeterReg::kTargetMaxGenFrames)
						{
							Log::Info("AMD RDNA 4 FlipMeter: detected {} at registry subkey {} (Gmx_DriverFramePacingMaxGenFrames={}); click 'Apply 6x FlipMeter (Admin)' in OptiScaler menu or run tools/ApplyAmdRdna4FlipMeter.ps1",
								a.driverDesc, a.subkeyUtf8, a.valuePresent ? std::to_string(a.currentValue) : std::string("default(1/2x)"));
						}
					}
				}
			}
			else if (tryAutoApply && !scan.rdna4Adapters.empty() && scan.allConfigured)
			{
				for (const auto& a : scan.rdna4Adapters)
				{
					Log::Info("AMD RDNA 4 FlipMeter: {} at subkey {} already configured (Gmx_DriverFramePacingMaxGenFrames={})",
						a.driverDesc, a.subkeyUtf8, a.currentValue);
				}
			}

			std::scoped_lock lock(amdFlipMeterMutex);
			const bool prevReboot = amdFlipMeterState.rebootRecommended;
			const bool prevApplied = amdFlipMeterState.appliedThisSession;
			amdFlipMeterState = std::move(scan);
			if (!tryAutoApply)
			{
				amdFlipMeterState.rebootRecommended = prevReboot || amdFlipMeterState.rebootRecommended;
				amdFlipMeterState.appliedThisSession = prevApplied || amdFlipMeterState.appliedThisSession;
			}
			amdFlipMeterScanned = true;
		}
		catch (...)
		{
		}
	}

	std::mutex mutex;
	Plugin::Status status;
	ULONGLONG reportedAt = 0;

	// No real frame for this long and frame generation is off
	constexpr ULONGLONG IdleAfterMs = 1000;

	void Green(const char* text) { api->TextColored(0.45f, 0.85f, 0.45f, 1.0f, text); }
	void Yellow(const char* text) { api->TextColored(1.0f, 0.80f, 0.30f, 1.0f, text); }
	void Red(const char* text) { api->TextColored(1.0f, 0.45f, 0.4f, 1.0f, text); }
	void HelpMarker(const char* text)
	{
		api->SameLine();
		api->TextDisabled("(?)");
		api->ItemTooltip(text);
	}

	struct ScopedDisable
	{
		bool active = false;
		explicit ScopedDisable(bool disabled) noexcept : active(disabled && OPTI_API_HAS(api, Disabled))
		{
			if (active)
				api->Disabled(true);
		}
		~ScopedDisable() noexcept
		{
			if (active)
				api->Disabled(false);
		}
	};

	bool initialVkWmmaBridge = true;
	bool initialVkWmmaCaptured = false;

	enum class NoticeKind
	{
		None,
		LiveSaved,
		LiveTemp,
		RestartGame,
		RestartPc,
	};

	NoticeKind lastNoticeKind = NoticeKind::None;
	std::string lastNoticeFeature;
	bool unsavedSettings = false;

	void NotifySettingChanged(NoticeKind kind, const char* featureName)
	{
		lastNoticeKind = kind;
		lastNoticeFeature = featureName ? featureName : "";
		if (kind == NoticeKind::LiveSaved || kind == NoticeKind::RestartGame)
			unsavedSettings = true;
	}

	const char* NetworkName(Plugin::Network network)
	{
		switch (network)
		{
		case Plugin::Network::XmxNative:
			return "XMX";
		case Plugin::Network::XmxPort:
			return "XMX Port";
		case Plugin::Network::Cuda:
			return "CUDA";
		case Plugin::Network::Wmma:
			return "WMMA";
		case Plugin::Network::Fallback:
			return "Fallback (DP4a)";
		default:
			return "Unknown";
		}
	}

	void DrawStatus()
	{
		Plugin::Status current;
		bool active;
		{
			std::scoped_lock lock(mutex);
			current = status;
			active = reportedAt != 0 && GetTickCount64() - reportedAt < IdleAfterMs;
		}

		const auto state = XeFG::State();
		auto logTooltip = [] { api->ItemTooltip(ThisModulePath().replace_extension(L".log").string().c_str()); };

		api->Section("Status");

		if (state == XeFG::LoadState::Failed)
		{
			Red("No compatible libxess_fg.dll");
			logTooltip();
		}
		else if (active && current.failed && current.fp16)
		{
			Red("16-bit HDR isn't supported");
			api->Text("XeFG only supports SDR and HDR10, switch the game to one of them");
		}
		else if (active && current.failed)
		{
			Red("XeFG couldn't start (duplicating real frames)");
			api->Text("No interpolation active; FPS overlay counts duplicated real frames. Check log for details");
			logTooltip();
		}
		else if (active)
		{
			if (current.vramGuarded && current.requestedFrames > current.generatedFrames)
			{
				Yellow(std::format("Generating {}x (VRAM guard, requested {}x)",
					current.generatedFrames + 1, current.requestedFrames + 1).c_str());
				if (current.vramBudgetMiB > 0)
				{
					Yellow(std::format("VRAM near limit ({} / {} MiB) \xC2\xB7 MFG clamped to 2x",
						current.vramUsageMiB, current.vramBudgetMiB).c_str());
				}
				else
				{
					Yellow("VRAM near limit \xC2\xB7 MFG clamped to 2x");
				}
				api->ItemTooltip(
					"Local VRAM usage reached the Adaptive VRAM guard threshold (budget - 200 MiB).\n"
					"Multi-Frame Generation is temporarily clamped to 2x to prevent VRAM swap stutter.\n"
					"Turn off 'Adaptive VRAM guard' below or reduce VRAM usage to restore the full multiplier.");
			}
			else
			{
				Green(std::format("Generating {}x", current.generatedFrames + 1).c_str());
			}
			const char* hudLabel = current.hudTier > 0
				? HudTier::ActiveTierShortName(static_cast<HudTier::ActiveTier>(current.hudTier))
				: (current.hudless ? "HUD T1: Native" : "UI interpolated");
			auto hudLine = std::format("{}x{} \xC2\xB7 {}", current.width, current.height, hudLabel);
			if (current.toneMapActiveMode == 1)
				hudLine += " \xC2\xB7 Tone-Map: 1D LUT";
			else if (current.toneMapActiveMode == 2)
				hudLine += " \xC2\xB7 Tone-Map: 3D Bilateral";
			api->Text(hudLine.c_str());
			if (current.profilerActive)
			{
				api->Text(std::format("GPU: {:.2f} ms total (Interp {:.2f} ms \xC2\xB7 HUD {:.2f} ms)",
					current.totalGpuMs, current.interpGpuMs, current.hudGpuMs).c_str());
			}
		}
		else
		{
			api->TextDisabled("Idle");
		}

		if (state != XeFG::LoadState::Loaded)
			return;

		const auto& library = XeFG::Library();
		auto line = std::format("XeFG {}.{}.{}", library.major, library.minor, library.patch);

		if (active && !current.failed)
			line += std::format(" \xC2\xB7 {}", NetworkName(current.network));

		api->Text(line.c_str());
		api->ItemTooltip(std::format("{}\n{}", library.path.string(),
			library.knownBuild ? "Known build" : "Functions found by scanning").c_str());
	}

	const char* HooksText(const VulkanRuntime::Status& host)
	{
		if (!host.enabled)
			return "off";

		switch (host.hooks)
		{
		case VulkanRuntime::Hooks::Waiting:
			return "waiting for vulkan-1.dll";
		case VulkanRuntime::Hooks::Installed:
			return "installed";
		case VulkanRuntime::Hooks::Failed:
			return "failed";
		default:
			return "not started";
		}
	}

	// What the tensor-core route (src/vulkan/VulkanTensor.h) is doing for the ported model's int8 layers: on an NVIDIA RTX GPU
	// they run as CUDA kernels in the game's command buffers. Nothing is drawn until a Vulkan game gave the runtime a
	// device, most games are DX12 ones and those have a tensor-core route of their own
	void DrawVulkanTensor()
	{
		using State = VulkanRuntime::TensorState;

		const auto tensor = VulkanRuntime::GetTensorStatus();

		if (tensor.state == State::NotAttempted && (tensor.reason.empty() || tensor.reason == "not_attempted"))
			return;

		const auto why = VulkanPolicy::DescribeTensor(tensor.reason);

		if (tensor.state == State::Active)
		{
			const auto architecture = std::format("sm_{}{}", tensor.major, tensor.minor);

			if (tensor.replacing)
			{
				// Green once a layer has run there. Selected and installed is all that can be said before that: no frame
				// was generated yet, or no program of this XeFG build is one of the route's layers (its table is of
				// libxess_fg.dll 1.3.1, XeSS SDK 3.0.2, whose 13 Intel programs 1.3.4.95 has byte for byte too,
				// tools/CheckXeFgBuild.py) and the SPIR-V programs run everything
				if (tensor.launches)
					Green(std::format("Tensor cores: on \xC2\xB7 {}", architecture).c_str());
				else
					api->Text(std::format("Tensor cores: idle \xC2\xB7 {}", architecture).c_str());

				api->ItemTooltip(std::format("The ported model's int8 layers run as CUDA kernels on this GPU's tensor cores, in the\n"
					"game's own command buffers.\n"
					"Layers launched on the tensor cores: {}\n"
					"Layers left to their SPIR-V program: {}\n"
					"A layer the route doesn't take (a shape or an image view it doesn't know) keeps its SPIR-V program in the same\n"
					"frame, the log says which. While both counts are 0 no layer has gone through the route yet: no frame was\n"
					"generated, or none of this XeFG build's programs is in the route's layer table (the same in libxess_fg.dll\n"
					"1.3.1 and 1.3.4.95; another build may not have them).\n"
					"XEFG_TENSOR_CORES=off in the environment turns the route off.",
					tensor.launches, tensor.fallbacks).c_str());
			}
			else
			{
				api->Text(std::format("Tensor cores: ready \xC2\xB7 {}", architecture).c_str());
				api->ItemTooltip("Installed on this GPU, but the layers run as SPIR-V programs now: the XMX Port or Fallback (DP4a)\n"
								 "network path is selected. The CUDA path (and XMX (native), which runs as CUDA on Vulkan) puts them on the\n"
								 "tensor cores.");
			}

			return;
		}

		if (tensor.state == State::NotAttempted)
		{
			// The route reports nothing for a GPU it was never an option for
			api->TextDisabled("Tensor cores: NVIDIA RTX GPUs only");
			api->ItemTooltip(why.c_str());
			return;
		}

		// The driver crashing under the route is no matter of this GPU not having it, the log has the details
		const bool faulted = tensor.reason.ends_with("_fault");

		if (faulted)
			Red("Tensor cores: the driver faulted");
		else
			api->TextDisabled(tensor.state == State::Disabled ? "Tensor cores: off" : "Tensor cores: unavailable");

		api->ItemTooltip(std::format("{}\nThe layers run as SPIR-V programs instead.", why).c_str());
	}

	// What the VK_KHR_cooperative_matrix route (src/vulkan/VulkanCoopMat.h) is doing on AMD RDNA3/RDNA4 (WMMA) and Intel Arc (XMX)
	void DrawVulkanCoopMat()
	{
		using State = VulkanRuntime::TensorState;

		const auto coop = VulkanRuntime::GetCoopMatrixStatus();

		if (coop.state == State::NotAttempted && (coop.reason.empty() || coop.reason == "not_attempted" || coop.reason == "no_extensions"))
			return;

		const auto why = VulkanPolicy::DescribeCoopMatrix(coop.reason);

		if (coop.state == State::Active)
		{
			const auto shape = std::format("{}x{}x{}", coop.tileM, coop.tileN, coop.tileK);

			if (coop.replacing)
			{
				if (coop.launches)
					Green(std::format("Coop matrix (WMMA/XMX): on \xC2\xB7 {}", shape).c_str());
				else
					api->Text(std::format("Coop matrix (WMMA/XMX): idle \xC2\xB7 {}", shape).c_str());

				api->ItemTooltip(std::format("The ported model's int8 layers run on VK_KHR_cooperative_matrix hardware units\n"
					"(AMD RDNA3/RDNA4 WMMA or Intel Arc/Battlemage XMX) in the game's command buffers.\n"
					"Layers launched on cooperative matrix: {}\n"
					"Layers left to their DP4a SPIR-V program: {}\n"
					"XEFG_COOP_MATRIX=off in the environment turns the route off.",
					coop.launches, coop.fallbacks).c_str());
			}
			else
			{
				api->Text(std::format("Coop matrix (WMMA/XMX): ready \xC2\xB7 {}", shape).c_str());
				api->ItemTooltip("Installed on this GPU, ready for the WMMA, XMX Port and XMX (native) paths.");
			}

			return;
		}

		api->TextDisabled(coop.state == State::Disabled ? "Coop matrix: off" : "Coop matrix: unavailable");
		api->ItemTooltip(std::format("{}\nThe layers run as DP4a SPIR-V programs instead.", why).c_str());
	}

	// Vulkan games run on the runtime built into this dll: MFG-Unlocker's D3D12-on-Vulkan layer and the models of
	// xefg_mfg_programs\vulkan (src/VulkanRuntime.h)
	void DrawVulkanRuntime()
	{
		const auto info = VulkanProvider::Query();
		const auto host = VulkanRuntime::GetStatus();
		const bool ready = info.state == VulkanProvider::State::Ready;
		const bool initialized = VulkanProvider::initialized.load();
		const bool running = VulkanProvider::liveFeatures.load() != 0;

		if (!ready && initialized)
		{
			// A Vulkan game asked for frame generation and it can't have it
			Red("Unavailable");
			api->Text(info.reason.c_str());
		}
		else if (!ready)
		{
			// Most games aren't Vulkan games, what keeps this from running there isn't news
			api->TextDisabled("Idle");
			api->ItemTooltip(info.reason.c_str());
		}
		else if (running)
		{
			Green("Active");
		}
		else if (initialized)
		{
			Green("Ready");
		}
		else
		{
			api->TextDisabled("Idle");
			api->ItemTooltip("Waiting for a Vulkan game to start frame generation. DX12 games don't use this.");
		}

		if (ready)
		{
			auto line = std::format("Up to {}x", info.maxFrames + 1);

			if (const auto model = VulkanProvider::ModelName(); !model.empty())
				line += std::format(" \xC2\xB7 {} model", model);

			api->Text(line.c_str());
			api->ItemTooltip("port / WMMA: the ported MFG model, the WMMA, XMX Port, XMX (native) and CUDA network paths\n"
							 "DP4a: XeFG's own model, the Fallback (DP4a) network path\n"
							 "XeFG's own XMX network isn't used on Vulkan, XMX (native) and WMMA run the port. The WMMA path puts the port's\n"
							 "int8 layers on VK_KHR_cooperative_matrix (AMD RDNA 4 / RDNA 3 WMMA), CUDA puts them on NVIDIA tensor cores.");
		}

		DrawVulkanTensor();
		DrawVulkanCoopMat();

		auto hooks = std::format("Hooks: {}", HooksText(host));

		if (host.devices)
			hooks += std::format(" \xC2\xB7 {} of {} devices prepared", host.prepared, host.devices);

		api->Text(hooks.c_str());

		std::string tip = "The hooks add what XeFG's programs need (Vulkan 1.3 features, VK_KHR_push_descriptor) to the game's\n"
			"VkDevice when it creates it. They have to be in before that: with this dll as an .asi in OptiScaler's\n"
			"plugins folder (see the README), or a game that creates its device after OptiScaler loaded this dll.";

		if (!host.detail.empty())
			tip += "\n" + host.detail;

		if (!host.refusal.empty())
			tip += "\nLast device turned down: " + VulkanPolicy::Describe(host.refusal);

		api->ItemTooltip(tip.c_str());

		bool hooksOn = host.enabled;

		const bool vkHooksDisabled = Plugin::activeApi.load(std::memory_order_acquire) == Plugin::ActiveApi::D3D12;
		{
			ScopedDisable dis(vkHooksDisabled);
			if (api->Checkbox("Vulkan hooks", &hooksOn))
			{
				VulkanRuntime::SetHooksEnabled(hooksOn);
				api->SetBool(self, "VulkanHooks", hooksOn);
				NotifySettingChanged(NoticeKind::RestartGame, "Vulkan hooks");
			}
		}
		HelpMarker("Enables or disables Vulkan frame generation hooks (Vulkan games only).\n"
				   "Off: leaves the game's Vulkan device untouched (no Vulkan FG).\n"
				   "Requires clicking 'Save Settings' and restarting the game when changing.");
	}

	// A build without the Vulkan headers has no runtime of its own: Vulkan games run through MFG-Unlocker's XeMFG provider
	// (MFGUnlocker.asi), the Vulkan models are its own. The choice between them is the asi's, not the network path
	void DrawVulkanProvider()
	{
		const auto info = VulkanProvider::Query();

		if (info.state == VulkanProvider::State::NotLoaded)
		{
			api->TextDisabled("MFGUnlocker.asi isn't loaded");
			api->ItemTooltip("This build of the dll has no Vulkan runtime, Vulkan games need MFG-Unlocker's MFGUnlocker.asi, it\n"
							 "runs XeFG on Vulkan. DX12 games don't need it.");
			return;
		}

		if (info.state == VulkanProvider::State::Blocked)
		{
			Red("Unavailable");
			api->ItemTooltip(info.module.c_str());
			api->Text(info.reason.c_str());
			return;
		}

		const bool running = VulkanProvider::liveFeatures.load() != 0;

		if (running)
			Green("Active");
		else if (VulkanProvider::initialized.load())
			Green("Ready");
		else
			api->TextDisabled("Idle");

		auto line = std::format("Up to {}x", info.maxFrames + 1);

		if (const auto model = VulkanProvider::ModelName(); !model.empty())
			line += std::format(" \xC2\xB7 {} model", model);

		api->Text(line.c_str());
		api->ItemTooltip(std::format("{}\n"
			"port: the ported MFG model, used unless XeFGModel says otherwise\n"
			"2X: XeFG's own 2x model, XeFGModel = default\n"
			"CUDA-port: the port with its int8 layers on tensor cores, XeFGTensorCores = off turns it off\n"
			"XeFGModel is in [XeFG] of OptiScaler.ini, restart the game after changing it.",
			info.module).c_str());
	}

	void DrawVulkan()
	{
		api->Section("Vulkan");

		if (VulkanRuntime::Built())
			DrawVulkanRuntime();
		else
			DrawVulkanProvider();
	}

	// The Vulkan models in xefg_mfg_programs\vulkan, lazily loaded when vulkan-1.dll or the WMMA bridge is used
	void DrawVulkanModels()
	{
		const auto* set = VulkanPrograms::Loaded();

		if (!set)
		{
			api->TextDisabled("Vulkan models: deferred (loads on demand)");
			api->ItemTooltip(
				"Vulkan models are only read and verified when vulkan-1.dll or the D3D12-Vulkan WMMA bridge is used.\n"
				"Pure DX12 games skip loading them to avoid start-up I/O overhead.");
			return;
		}

		if (!set->present)
		{
			api->TextDisabled("Vulkan models: not installed");
			api->ItemTooltip(std::format("{}\n"
				"The models XeFG runs on Vulkan. tools/ExtractXeFgVulkanPrograms.py makes this folder from MFGUnlocker.asi.\n"
				"DX12 games don't need it.",
				VulkanPrograms::PathText(set->directory)).c_str());
			return;
		}

		if (!set->Usable())
		{
			std::string problems;

			for (const auto& problem : set->problems)
				problems += "\n" + problem;

			Red("Vulkan models: can't be used");
			api->ItemTooltip(std::format("{}{}", VulkanPrograms::PathText(set->directory), problems).c_str());
			return;
		}

		api->Text(std::format("Vulkan models: {} programs", set->programs.size()).c_str());
		api->ItemTooltip(std::format("{}\n{}\n"
			"Every program was checked against its SHA-256 when loaded.{}",
			VulkanPrograms::Summary(*set), VulkanPrograms::PathText(set->directory),
			VulkanRuntime::Built() ? " Vulkan games run them on this dll's own D3D12-on-Vulkan layer."
			                       : " Running them takes the D3D12 layer of MFGUnlocker.asi (this build has none of its own).")
				.c_str());
	}

	void DrawSettings()
	{
		api->Section("Settings");

		const auto curApi = Plugin::activeApi.load(std::memory_order_acquire);
		const bool isD3D12 = (curApi == Plugin::ActiveApi::D3D12);
		const bool isVulkan = (curApi == Plugin::ActiveApi::Vulkan);

		// Immediate per-click notifications (OptiScaler-style):
		// - Live settings: confirm immediate change & remind to click 'Save Settings' for next launch
		// - Game-restart settings: remind to click 'Save Settings' and restart the game (both ON and OFF)
		// - PC-restart settings: remind to restart Windows
		const bool gameRestartNeeded = initialVkWmmaCaptured &&
			(Plugin::options.d3d12VulkanWmmaBridge.load(std::memory_order_relaxed) != initialVkWmmaBridge);
		const bool showRestartGameNotice = gameRestartNeeded || lastNoticeKind == NoticeKind::RestartGame;
		bool pcRestartNeeded = false;
		{
			std::scoped_lock lock(amdFlipMeterMutex);
			pcRestartNeeded = amdFlipMeterState.rebootRecommended;
		}

		if (lastNoticeKind == NoticeKind::LiveSaved && !lastNoticeFeature.empty())
		{
			Green(std::format("{}: applied immediately! Click 'Save Settings' to keep for next launch.", lastNoticeFeature).c_str());
		}
		else if (lastNoticeKind == NoticeKind::LiveTemp && !lastNoticeFeature.empty())
		{
			Green(std::format("{}: applied immediately (temporary test mode, not saved).", lastNoticeFeature).c_str());
		}

		if (showRestartGameNotice)
		{
			const std::string& restartLabel = (lastNoticeKind == NoticeKind::RestartGame && !lastNoticeFeature.empty())
				? lastNoticeFeature
				: std::string("Hardware WMMA acceleration (D3D12)");
			Yellow(std::format("{}: click 'Save Settings' and restart the game to apply.", restartLabel).c_str());
		}
		if (pcRestartNeeded)
			Yellow("AMD RDNA 4 Flip Meter: updated — restart your PC (Windows) to activate.");

		static constexpr const char* PathNames[] = { "XMX Port", "CUDA", "Fallback (DP4a)", "XMX (native)", "WMMA" };
		const bool cudaDisabled = Plugin::cudaSupport.load() == Plugin::CudaSupport::Unsupported;
		const bool xmxDisabled = Plugin::xmxSupport.load() != Plugin::XmxSupport::Supported;
		const bool wmmaDisabled = Plugin::wmmaSupport.load() == Plugin::WmmaSupport::Unsupported;

		int32_t path = (int32_t)Plugin::options.path.load();

		// A saved choice the GPU doesn't support runs something else; auto-switch to the fallback model and notify.
		const bool xmxReady = !xmxDisabled;
		const bool cudaReady = Plugin::cudaSupport.load() == Plugin::CudaSupport::Supported;
		const bool wmmaReady = Plugin::wmmaSupport.load() == Plugin::WmmaSupport::Supported;
		const auto resolvedPath = Plugin::ResolveEffectivePath((Plugin::Path)path,
			Plugin::gpuVendorId.load(std::memory_order_relaxed), xmxReady, cudaReady, wmmaReady, true);
		if ((int32_t)resolvedPath != path)
		{
			Plugin::RecordPathFallback((Plugin::Path)path, resolvedPath);
			path = (int32_t)resolvedPath;
			Plugin::options.path = resolvedPath;
			api->SetInt(self, "NetworkPath", path);
		}

		const bool disabled[] = { false, cudaDisabled, false, xmxDisabled, wmmaDisabled };

		if (api->Combo("Network path", &path, PathNames, 5, disabled))
		{
			Plugin::ClearPathFallback();
			Plugin::options.path = (Plugin::Path)path;
			api->SetInt(self, "NetworkPath", path);
			NotifySettingChanged(NoticeKind::LiveSaved, "Network path");
		}
		{
			const int32_t fbFrom = Plugin::fallbackFromPath.load(std::memory_order_relaxed);
			const int32_t fbTo = Plugin::fallbackToPath.load(std::memory_order_relaxed);
			if (fbFrom >= 0 && fbFrom < 5 && fbTo >= 0 && fbTo < 5 && fbFrom != fbTo)
			{
				Yellow(std::format("Model '{}' is not supported on this GPU; automatically switched to '{}'.",
					PathNames[fbFrom], PathNames[fbTo]).c_str());
			}
		}
		HelpMarker("Picks which AI engine on your graphics card creates extra frames.\n"
				   "- WMMA: Fastest on newer AMD RX 7000 / 9000 cards.\n"
				   "- CUDA: Fastest on NVIDIA RTX cards.\n"
				   "- XMX (native): Best on Intel Arc cards.\n"
				   "- XMX Port / Fallback: Compatible backups for any card.");

		bool hudlessUi = Plugin::options.hudlessUi;

		if (api->Checkbox("HUDless UI", &hudlessUi))
		{
			Plugin::options.hudlessUi = hudlessUi;
			api->SetBool(self, "HudlessUi", hudlessUi);
			NotifySettingChanged(NoticeKind::LiveSaved, "HUDless UI");
		}
		HelpMarker("Keeps on-screen menus, health bars, and crosshairs sharp instead of blurry.\n"
				   "Turn OFF only if FPS feels stuck or the HUD looks glitched.");

		static const char* const HudTierNames[] = {
			"Auto (3-Tier)", "Tier 1 (Native HUDless)", "Tier 2 (HUD-Diff Composite)", "Tier 3 (Hybrid: Tone-Match + Static)", "Off (Pure XeFG / Max FPS)"
		};
		int32_t hudTierMode = std::clamp(Plugin::options.hudTierMode.load(), 0, 4);
		if (api->Combo("HUD protection tier", &hudTierMode, HudTierNames, 5, nullptr))
		{
			Plugin::options.hudTierMode = hudTierMode;
			api->SetInt(self, "HudTierMode", hudTierMode);
			NotifySettingChanged(NoticeKind::LiveSaved, "HUD protection tier");
		}
		HelpMarker("Controls how the mod keeps your crosshair and game HUD clean:\n"
				   "- Auto (3-Tier): Recommended. Uses fast Tier 1 on normal games (zero\n"
				   "  extra cost) and auto-switches to Tier 3 only if HUD data breaks.\n"
				   "- Tier 1: Uses only the game's built-in HUD data.\n"
				   "- Tier 2: Separates HUD manually (helps if Tier 1 flickers).\n"
				   "- Tier 3 (Hybrid): Combines Tone-Map Matching (fixes HDR/dynamic UI)\n"
				   "  with Static Mask (protects fixed crosshairs & text) on DX12 & Vulkan.\n"
				   "- Off (Pure XeFG / Max FPS): Disables all extra HUD compute passes\n"
				   "  for maximum DX12/Vulkan FPS and lowest frametime overhead.");

		bool hudDebug = Plugin::options.hudDebugMagenta;
		if (api->Checkbox("HUD mask debug (magenta)", &hudDebug))
		{
			Plugin::options.hudDebugMagenta = hudDebug;
			NotifySettingChanged(NoticeKind::LiveTemp, "HUD mask debug");
		}
		HelpMarker("Test mode: highlights detected HUD areas in bright pink so you can\n"
				   "check if Tier 2 or Tier 3 is protecting your crosshair and UI accurately.");

		bool hideHudless = Plugin::options.debugHideHudless;
		if (api->Checkbox("Hide HUD-less (diagnostic)", &hideHudless))
		{
			Plugin::options.debugHideHudless = hideHudless;
			NotifySettingChanged(NoticeKind::LiveTemp, "Hide HUD-less");
		}
		HelpMarker("Diagnostic toggle (not saved): temporarily hides the game's HUD-less texture\n"
				   "from Frame Generation to isolate pre-FG judder vs HUD-less compositing issues.");

		bool cameraCut = Plugin::options.cameraCutDetection;

		if (api->Checkbox("Camera cut detection", &cameraCut))
		{
			Plugin::options.cameraCutDetection = cameraCut;
			api->SetBool(self, "CameraCutDetection", cameraCut);
			NotifySettingChanged(NoticeKind::LiveSaved, "Camera cut detection");
		}
		HelpMarker("Stops brief image smearing or warping when a cutscene switches\n"
				   "camera angles or when your character teleports.");

		bool vramGuard = Plugin::options.vramGuard;

		if (api->Checkbox("Adaptive VRAM guard", &vramGuard))
		{
			Plugin::options.vramGuard = vramGuard;
			api->SetBool(self, "VramGuard", vramGuard);
			NotifySettingChanged(NoticeKind::LiveSaved, "Adaptive VRAM guard");
		}
		HelpMarker("Watches your graphics card memory. When memory is almost full,\n"
				   "it temporarily lowers 3x-6x down to 2x to prevent heavy stuttering.");

		bool autoVrr = Plugin::options.autoVrrPacing;

		{
			ScopedDisable dis(isD3D12);
			if (api->Checkbox("Auto VRR pacing (Hz - 3)", &autoVrr))
			{
				Plugin::options.autoVrrPacing = autoVrr;
				api->SetBool(self, "AutoVrrPacing", autoVrr);
				NotifySettingChanged(NoticeKind::LiveSaved, "Auto VRR pacing");
			}
		}
		HelpMarker("Keeps Vulkan frame pacing just under your monitor refresh rate (Hz - 3)\n"
				   "so FreeSync / G-Sync stays smooth without V-Sync input lag.\n"
				   "Avoid combining with RTSS / NVCP / AMD Chill FPS caps to prevent double-limiting stutter.");

		bool vkAmdAntiLag = Plugin::options.vkAmdAntiLag;

		{
			ScopedDisable dis(isD3D12);
			if (api->Checkbox("AMD Anti-Lag 2 (Vulkan, experimental)", &vkAmdAntiLag))
			{
				Plugin::options.vkAmdAntiLag = vkAmdAntiLag;
				api->SetBool(self, "VkAmdAntiLag", vkAmdAntiLag);
				NotifySettingChanged(NoticeKind::RestartGame, "AMD Anti-Lag 2 (Vulkan)");
			}
		}
		HelpMarker("Opt-in (OFF by default; requires game restart): enables VK_AMD_anti_lag\n"
				   "on AMD GPUs in Vulkan games to reduce input latency during Frame Generation.\n"
				   "WARNING: Keep OFF if the game or OptiScaler already enables Anti-Lag 2 or\n"
				   "Reflex, as running two low-latency pacers together can cause stutter or hangs.");

		bool vkWmmaBridge = Plugin::options.d3d12VulkanWmmaBridge;

		{
			ScopedDisable dis(isVulkan);
			if (api->Checkbox("Hardware WMMA acceleration (D3D12)", &vkWmmaBridge))
			{
				Plugin::options.d3d12VulkanWmmaBridge = vkWmmaBridge;
				api->SetBool(self, "D3D12VulkanWmmaBridge", vkWmmaBridge);
				NotifySettingChanged(NoticeKind::RestartGame, "Hardware WMMA acceleration (D3D12)");
			}
		}
		HelpMarker("Lets DirectX 12 games use faster hardware AI acceleration (WMMA) on AMD cards.\n"
				   "Keep ON for best speed; turn OFF if a specific game freezes or crashes.\n"
				   "Requires clicking 'Save Settings' and restarting the game.");
		if (showRestartGameNotice)
		{
			api->SameLine();
			Yellow("(Save Settings & restart game)");
		}

		bool uiCooldown = Plugin::options.uiTransitionCooldown;

		if (api->Checkbox("UI / HUDless transition guard (10 frames)", &uiCooldown))
		{
			Plugin::options.uiTransitionCooldown = uiCooldown;
			api->SetBool(self, "UiTransitionCooldown", uiCooldown);
			NotifySettingChanged(NoticeKind::LiveSaved, "UI transition guard");
		}
		HelpMarker("Opt-in (OFF by default for max smoothness): briefly pauses extra frames\n"
				   "when opening or closing in-game menus if the screen flashes or glitches.");

		bool ghostBuster = Plugin::options.ghostBuster;

		if (api->Checkbox("GhostBuster anti-ghosting", &ghostBuster))
		{
			Plugin::options.ghostBuster = ghostBuster;
			api->SetBool(self, "GhostBuster", ghostBuster);
			NotifySettingChanged(NoticeKind::LiveSaved, "GhostBuster anti-ghosting");
		}
		HelpMarker("Opt-in (OFF by default for max smoothness): resets history on rapid\n"
				   "camera snaps (>45 deg/frame, or >30 deg if camera MV is missing) and\n"
				   "sudden scope/ADS FOV jumps (>14 deg), skipping 1 FG beat each time.\n"
				   "Avoid in fast-paced shooters/esports or high-speed action games with\n"
				   "frequent quick-scoping or fast mouse flicks (can cause brief 1-frame\n"
				   "hitches). Best reserved for single-player games with heavy ghosting.");

		bool spinSleep = Plugin::options.pacingSpinlockSleep;

		{
			ScopedDisable dis(isD3D12);
			if (api->Checkbox("Pacing spinlock sleep (lower CPU)", &spinSleep))
			{
				Plugin::options.pacingSpinlockSleep = spinSleep;
				api->SetBool(self, "PacingSpinlockSleep", spinSleep);
				NotifySettingChanged(NoticeKind::LiveSaved, "Pacing spinlock sleep");
			}
		}
		HelpMarker("Lets your processor rest briefly between frames on Vulkan games to reduce\n"
				   "CPU usage and heat. Turn OFF if frame timing feels less steady.");

		static const char* const ToneMapNames[] = { "Auto (1D -> 3D Bilateral)", "1D LUT (Fastest)", "3D Bilateral Grid (Local)" };
		int32_t tmMode = std::clamp(Plugin::options.toneMapMatchMode.load(), 0, 2);
		if (api->Combo("Tone-Map Matcher", &tmMode, ToneMapNames, 3, nullptr))
		{
			Plugin::options.toneMapMatchMode = tmMode;
			api->SetInt(self, "ToneMapMatchMode", tmMode);
			NotifySettingChanged(NoticeKind::LiveSaved, "Tone-Map Matcher");
		}
		HelpMarker("Matches HUD-less colors to HDR/ReShade/RenoDX backbuffers when Tier 4 activates.\n"
				   "Auto starts with the fast 1D LUT and promotes asynchronously to 3D Bilateral Grid\n"
				   "(amortized every 4 frames) if spatial vignette/local exposure mismatch persists.");

		bool debugLines = Plugin::options.debugLines;

		if (api->Checkbox("Debug lines", &debugLines))
		{
			Plugin::options.debugLines = debugLines;
			NotifySettingChanged(NoticeKind::LiveTemp, "Debug lines");
		}
		HelpMarker("Test mode: draws Intel's built-in indicator lines on generated frames\n"
				   "so you can visually confirm frame generation is running.");

		bool loggingOn = Plugin::options.logging;
		if (api->Checkbox("Enable logging", &loggingOn))
		{
			Plugin::options.logging = loggingOn;
			Log::SetEnabled(loggingOn);
			api->SetBool(self, "Logging", loggingOn);
			NotifySettingChanged(NoticeKind::LiveSaved, "Enable logging");
		}
		HelpMarker("Writes startup diagnostics, HUD tier switches, and crash/TDR records to\n"
				   "dlssg_to_xefg_intel_is_better.log. Turn OFF for zero disk I/O.");

		bool crashDump = Plugin::options.crashMiniDump;
		if (api->Checkbox("Crash minidump (.dmp)", &crashDump))
		{
			Plugin::options.crashMiniDump = crashDump;
			Log::SetCrashMiniDumpEnabled(crashDump);
			api->SetBool(self, "CrashMiniDump", crashDump);
			NotifySettingChanged(NoticeKind::LiveSaved, "Crash minidump");
		}
		HelpMarker("Off by default. Attempts to write a compact .dmp file next to the mod\n"
				   "on unhandled crash using System32 dbghelp.dll (or set XEFG_CRASH_MINIDUMP=1).");

		bool forceDred = Plugin::options.forceDred;
		{
			ScopedDisable dis(isVulkan);
			if (api->Checkbox("Force DRED crash diagnostics (DX12)", &forceDred))
			{
				Plugin::options.forceDred = forceDred;
				api->SetBool(self, "ForceDRED", forceDred);
				NotifySettingChanged(NoticeKind::RestartGame, "Force DRED crash diagnostics");
			}
		}
		HelpMarker("DirectX 12 only (opt-in; 0% overhead when OFF): enables D3D12 DRED\n"
				   "AutoBreadcrumbs + GPU Page Fault tracking so TDR crashes log the exact\n"
				   "unfinished command list and faulting resource. Adds slight GPU overhead\n"
				   "while ON; click 'Save Settings' and restart the game to apply.");

		bool gpuProf = Plugin::options.gpuProfiler;
		if (api->Checkbox("Per-stage GPU profiler", &gpuProf))
		{
			Plugin::options.gpuProfiler = gpuProf;
			NotifySettingChanged(NoticeKind::LiveTemp, "Per-stage GPU profiler");
		}
		HelpMarker("Measures real GPU time per stage (Total / Interp / HUD) in Status above.\n"
				   "Zero overhead when OFF (allocates no query heap or readback buffer).");

		bool splitCmp = Plugin::options.hudSplitCompare;
		if (api->Checkbox("Split-screen compare", &splitCmp))
		{
			Plugin::options.hudSplitCompare = splitCmp;
			NotifySettingChanged(NoticeKind::LiveTemp, "Split-screen compare");
		}
		HelpMarker("Debug split view: left = original real frame, right = generated/HUD-protected output.\n"
				   "Zero overhead when OFF.");
		if (splitCmp)
		{
			float splitPos = Plugin::options.hudSplitPos.load();
			if (api->SliderFloat("Split position", &splitPos, 0.05f, 0.95f, "%.2f"))
			{
				Plugin::options.hudSplitPos = splitPos;
				NotifySettingChanged(NoticeKind::LiveTemp, "Split position");
			}
			HelpMarker("Moves the vertical split divider between the original frame (left)\n"
					   "and the generated/HUD-protected output (right). Applies immediately.");
		}

		// AMD RDNA 4+ Flip Meter (6x) multi-GPU registry section (shown only when an AMD RDNA 4+ GPU is detected)
		{
			AmdFlipMeterReg::ScanResult snap;
			bool scanned = false;
			{
				std::scoped_lock lock(amdFlipMeterMutex);
				snap = amdFlipMeterState;
				scanned = amdFlipMeterScanned;
			}
			if (!scanned)
			{
				RefreshAmdFlipMeterState(false);
				std::scoped_lock lock(amdFlipMeterMutex);
				snap = amdFlipMeterState;
			}
			if (!snap.rdna4Adapters.empty())
			{
				api->Section("AMD RDNA 4 Flip Meter (6x)");
				for (const auto& a : snap.rdna4Adapters)
				{
					const bool ok = a.valuePresent && a.currentValue >= AmdFlipMeterReg::kTargetMaxGenFrames;
					const auto line = std::format("{} [{}] \xC2\xB7 {}",
						a.driverDesc, a.subkeyUtf8,
						ok ? std::format("6x unlocked ({})", a.currentValue)
						   : (a.valuePresent ? std::format("value={} (below 5)", a.currentValue) : std::string("2x default")));
					if (ok)
						Green(line.c_str());
					else
						Yellow(line.c_str());
				}
				if (snap.rebootRecommended)
					Yellow("Restart your PC (Windows) so the AMD driver activates this change.");

				if (!snap.allConfigured)
				{
					{
						ScopedDisable dis(isVulkan);
						if (api->Button("Apply 6x FlipMeter (Admin)"))
						{
							if (AmdFlipMeterReg::LaunchElevatedApply(snap, false))
							{
								Sleep(250);
								RefreshAmdFlipMeterState(false);
								std::scoped_lock lock(amdFlipMeterMutex);
								amdFlipMeterState.appliedThisSession = true;
								amdFlipMeterState.rebootRecommended = true;
								NotifySettingChanged(NoticeKind::RestartPc, "AMD RDNA 4 Flip Meter");
							}
						}
					}
					HelpMarker("Unlocks smoother 3x-6x frame pacing in the AMD driver for your RX 9000 card.\n"
							   "Automatically selects your AMD RDNA 4 card and ignores integrated graphics.\n"
							   "Requires Admin permission and restarting Windows afterwards.");
				}
				else
				{
					{
						ScopedDisable dis(isVulkan);
						if (api->Button("Revert to 2x default (Admin)"))
						{
							if (AmdFlipMeterReg::LaunchElevatedApply(snap, true))
							{
								Sleep(250);
								RefreshAmdFlipMeterState(false);
								std::scoped_lock lock(amdFlipMeterMutex);
								amdFlipMeterState.rebootRecommended = true;
								NotifySettingChanged(NoticeKind::RestartPc, "AMD RDNA 4 Flip Meter");
							}
						}
					}
					HelpMarker("Restores the AMD driver's original 2x frame pacing limit on your RX 9000 card.\n"
							   "Requires Admin permission and restarting Windows afterwards.");
				}
				api->SameLine();
				if (api->Button("Recheck##AmdFlipMeter"))
					RefreshAmdFlipMeterState(false);
				HelpMarker("Checks Windows settings again to confirm whether 6x Flip Meter is active.");
			}
		}

		api->Section("Extended XeFG Overrides");

		static constexpr const char* TriStateNames[] = { "Auto (Game)", "Force Off", "Force On" };

		int32_t depthOv = Plugin::options.depthInvertedOverride.load();
		if (api->Combo("Depth inverted", &depthOv, TriStateNames, 3, nullptr))
		{
			Plugin::options.depthInvertedOverride = depthOv;
			api->SetInt(self, "DepthInvertedOverride", depthOv);
			NotifySettingChanged(NoticeKind::LiveSaved, "Depth inverted");
		}
		HelpMarker("Fixes broken background/foreground blurring if a game sends depth backwards.\n"
				   "Leave on 'Auto (Game)' unless generated frames look heavily distorted.");

		int32_t jitterOv = Plugin::options.jitteredMvOverride.load();
		if (api->Combo("Jittered motion vectors", &jitterOv, TriStateNames, 3, nullptr))
		{
			Plugin::options.jitteredMvOverride = jitterOv;
			api->SetInt(self, "JitteredMvOverride", jitterOv);
			NotifySettingChanged(NoticeKind::LiveSaved, "Jittered motion vectors");
		}
		HelpMarker("Fixes shaky or vibrating edges in games that send jittered motion data.\n"
				   "Leave on 'Auto (Game)' unless thin edges shimmer during motion.");

		static const char* const UiAlphaNames[] = { "Auto", "Straight", "Premultiplied" };
		int32_t uiAlpha = std::clamp(Plugin::options.uiAlphaMode.load(), 0, 2);
		{
			ScopedDisable dis(isD3D12);
			if (api->Combo("UI alpha mode", &uiAlpha, UiAlphaNames, 3, nullptr))
			{
				Plugin::options.uiAlphaMode = uiAlpha;
				api->SetInt(self, "UiAlphaMode", uiAlpha);
				NotifySettingChanged(NoticeKind::LiveSaved, "UI alpha mode");
			}
		}
		HelpMarker("Fixes dark borders or glowing halos around HUD icons in HDR Vulkan games.\n"
				   "Leave on 'Auto' unless HUD edges look dark or washed out.");

		bool ignoreReset = Plugin::options.ignoreGameReset;
		if (api->Checkbox("Ignore game Reset signals", &ignoreReset))
		{
			Plugin::options.ignoreGameReset = ignoreReset;
			api->SetBool(self, "IgnoreGameReset", ignoreReset);
			NotifySettingChanged(NoticeKind::LiveSaved, "Ignore game Reset signals");
		}
		HelpMarker("Fixes rare games where frame generation stutters or turns itself off\n"
				   "every frame because the game keeps sending false reset signals.");

		api->Spacing();
		api->Separator();
		if (api->Button("Save Settings"))
		{
			if (api->SaveSettings(self))
			{
				unsavedSettings = false;
				lastNoticeKind = NoticeKind::None;
			}
		}
		api->SameLine();
		if (api->Button("Reset Defaults"))
		{
			static const char* const kAllKeys[] = {
				"NetworkPath", "HudlessUi", "CameraCutDetection", "VramGuard", "AutoVrrPacing", "VkAmdAntiLag",
				"D3D12VulkanWmmaBridge", "UiTransitionCooldown", "GhostBuster", "PacingSpinlockSleep",
				"DepthInvertedOverride", "JitteredMvOverride", "UiAlphaMode", "HudTierMode",
				"ToneMapMatchMode", "IgnoreGameReset", "Logging", "CrashMiniDump", "ForceDRED"
			};
			for (const char* key : kAllKeys)
				api->ResetValue(self, key);

			Plugin::ClearPathFallback();
			Plugin::options.path = Plugin::PreferredPathForGpu(
				Plugin::gpuVendorId.load(std::memory_order_relaxed),
				Plugin::xmxSupport.load() == Plugin::XmxSupport::Supported,
				Plugin::cudaSupport.load() == Plugin::CudaSupport::Supported,
				Plugin::wmmaSupport.load() == Plugin::WmmaSupport::Supported,
				true);
			Plugin::options.hudlessUi = true;
			Plugin::ApplyBrokenHudlessQuirkOnce();
			Plugin::options.cameraCutDetection = true;
			Plugin::options.vramGuard = true;
			Plugin::options.autoVrrPacing = false;
			Plugin::options.vkAmdAntiLag = false;
			Plugin::options.d3d12VulkanWmmaBridge = true;
			Plugin::options.uiTransitionCooldown = false;
			Plugin::options.ghostBuster = false;
			Plugin::options.pacingSpinlockSleep = false;
			Plugin::options.depthInvertedOverride = 0;
			Plugin::options.jitteredMvOverride = 0;
			Plugin::options.uiAlphaMode = 0;
			Plugin::options.hudTierMode = 0;
			Plugin::options.toneMapMatchMode = 0;
			Plugin::options.ignoreGameReset = false;
			Plugin::options.logging = true;
			Plugin::options.forceDred = false;
			Plugin::options.crashMiniDump = false;
			Log::SetEnabled(true);
			Log::SetCrashMiniDumpEnabled(false);
			Plugin::options.hudDebugMagenta = false;
			Plugin::options.debugHideHudless = false;
			Plugin::options.hudSplitCompare = false;
			Plugin::options.hudSplitPos = 0.5f;
			Plugin::options.gpuProfiler = false;
			Plugin::options.debugLines = false;
			api->SaveSettings(self);
			unsavedSettings = false;
			lastNoticeKind = NoticeKind::None;
		}
		HelpMarker("Resets all XeMFG XMX plugin keys in OptiScaler.ini to defaults without touching other tabs.");
		if (unsavedSettings)
		{
			api->SameLine();
			Yellow("(Unsaved)");
		}
	}

	void DrawTab(void*)
	{
		DrawStatus();
		DrawVulkan();
		DrawVulkanModels();
		DrawSettings();
	}

	// A copy of this dll renamed to .asi, in OptiScaler's plugins folder. See InitializeASI
	bool IsEarlyCopy()
	{
		return _wcsicmp(ThisModulePath().extension().c_str(), L".asi") == 0;
	}

	// Lazily pre-read and validate xefg_mfg_programs\vulkan on a background thread only when
	// vulkan-1.dll is already loaded in the process. Pure DX12 games skip the disk I/O and SHA-256
	// verification completely at start-up; Vulkan games that load vulkan-1.dll dynamically trigger
	// VulkanPrograms::PreloadAsync() from VulkanHost::InstallFromLoad(), and the D3D12-Vulkan WMMA
	// bridge loads on demand via VulkanPrograms::Installed().
	void PreloadVulkanModelsIfLoaded() noexcept
	{
		if (GetModuleHandleW(L"vulkan-1.dll") != nullptr)
			VulkanPrograms::PreloadAsync();
	}

	// Legacy fallback when a user manually copies the main DLL as .asi into plugins\.
	// Uses EarlyLoaderCore so OptiScaler.ini FGNvngxReplacement and PE export verification apply here too.
	std::filesystem::path FindMainDll()
	{
		try
		{
			const auto self = ThisModulePath();
			const auto optiRoot = self.parent_path().parent_path();
			const auto ini = EarlyLoader::ParseOptiScalerIni(optiRoot / L"OptiScaler.ini");
			const auto selected = EarlyLoader::SelectMainDll(optiRoot, ini, self.stem().string());
			return selected.path;
		}
		catch (...)
		{
		}

		return {};
	}

	std::once_flag asiStartOnce;

	void StartFromAsiImpl()
	{
		const auto self = ThisModulePath();
		Log::Info("InitializeASI, {}", self.string());

		// The dll itself, started by OptiScaler as a plugin or by its early loader
		if (!IsEarlyCopy())
		{
			VulkanRuntime::Start("InitializeASI");
			PreloadVulkanModelsIfLoaded();
			return;
		}

		// The copy has no settings, no models and no menu of its own, those are the main dll's. It only gets that one
		// loaded now, so that its Vulkan hooks are in before the game creates its instance and device
		const auto main = FindMainDll();

		if (main.empty())
		{
			Log::Warn("This copy of the dll is a plugin, but there is no matching main dll in custom\\<name>\\dlssg next to the plugins "
				"folder to start");
			return;
		}

		HMODULE module = LoadLibraryW(main.c_str());

		if (!module)
		{
			Log::Error("Can't load {} ({})", main.string(), GetLastError());
			return;
		}

		using BootstrapFn = void (*)(const char*);
		if (const auto bootstrap = reinterpret_cast<BootstrapFn>(GetProcAddress(module, EarlyLoader::kBootstrapExportName)))
		{
			Log::Info("Starting {} via {}", main.string(), EarlyLoader::kBootstrapExportName);
			bootstrap("InitializeASI (early copy)");
			return;
		}

		const auto start = reinterpret_cast<void (*)()>(GetProcAddress(module, "InitializeASI"));

		if (!start)
		{
			Log::Error("{} has no InitializeASI, it isn't a build this copy can start", main.string());
			return;
		}

		Log::Info("Starting {}", main.string());
		start();
	}

	void StartFromAsi()
	{
		std::call_once(asiStartOnce, StartFromAsiImpl);
	}
} // namespace

Plugin::Path Plugin::PreferredPathForGpu(uint32_t vendorId, bool xmxReady, bool cudaReady, bool wmmaReady, bool portReady) noexcept
{
	switch (vendorId)
	{
	case 0x1002u: // AMD: WMMA -> XMX Port -> Fallback (DP4a)
		if (wmmaReady && portReady)
			return Path::Wmma;
		return portReady ? Path::XmxPort : Path::Fallback;
	case 0x10DEu: // NVIDIA: CUDA -> XMX Port -> Fallback (DP4a)
		if (cudaReady && portReady)
			return Path::Cuda;
		return portReady ? Path::XmxPort : Path::Fallback;
	case 0x8086u: // Intel: XMX (native) -> XMX Port -> Fallback (DP4a)
	default:      // Unknown GPU: XMX (native) -> XMX Port -> Fallback (DP4a)
		if (xmxReady)
			return Path::XmxNative;
		return portReady ? Path::XmxPort : Path::Fallback;
	}
}

Plugin::Path Plugin::ResolveEffectivePath(Path requested, uint32_t vendorId, bool xmxReady, bool cudaReady, bool wmmaReady, bool portReady) noexcept
{
	switch (requested)
	{
	case Path::Fallback:
		return Path::Fallback;
	case Path::XmxPort:
		return portReady ? Path::XmxPort : Path::Fallback;
	case Path::Wmma:
		if (wmmaReady && portReady)
			return Path::Wmma;
		return portReady ? Path::XmxPort : Path::Fallback;
	case Path::Cuda:
		if (cudaReady && portReady)
			return Path::Cuda;
		return portReady ? Path::XmxPort : Path::Fallback;
	case Path::XmxNative:
		if (xmxReady && (vendorId == 0x8086u || vendorId == 0u))
			return Path::XmxNative;
		// Safety net: any GPU that ends up with XMX (native) without native XMX support falls back via its vendor chain.
		return PreferredPathForGpu(vendorId, false, cudaReady, wmmaReady, portReady);
	default:
		return PreferredPathForGpu(vendorId, xmxReady, cudaReady, wmmaReady, portReady);
	}
}

const char* Plugin::PathName(Path path)
{
	switch (path)
	{
	case Path::XmxPort:
		return "XMX Port";
	case Path::Cuda:
		return "CUDA";
	case Path::Fallback:
		return "Fallback (DP4a)";
	case Path::XmxNative:
		return "XMX (native)";
	case Path::Wmma:
		return "WMMA";
	default:
		return "Unknown";
	}
}

void Plugin::Report(const Status& reported)
{
	std::scoped_lock lock(mutex);
	status = reported;
	reportedAt = GetTickCount64();
}

static std::string DetectBrokenHudlessExeName() noexcept
{
	try
	{
		wchar_t path[MAX_PATH]{};
		const DWORD len = GetModuleFileNameW(nullptr, path, MAX_PATH);
		if (len == 0 || len >= MAX_PATH)
			return {};

		std::filesystem::path exePath(path);
		std::wstring name = exePath.filename().wstring();
		for (wchar_t& ch : name)
		{
			if (ch >= L'A' && ch <= L'Z')
				ch = static_cast<wchar_t>(ch - L'A' + L'a');
		}

		static constexpr std::wstring_view kBrokenHudlessExes[] = {
			L"spider-man2.exe",
			L"tlou-i.exe",
			L"tlou-i-l.exe",
		};

		for (const auto known : kBrokenHudlessExes)
		{
			if (name == known)
				return exePath.filename().string();
		}
	}
	catch (...)
	{
	}

	return {};
}

bool Plugin::IsKnownBrokenHudlessGame() noexcept
{
	static const bool known = !DetectBrokenHudlessExeName().empty();
	return known;
}

void Plugin::ApplyBrokenHudlessQuirkOnce() noexcept
{
	static std::once_flag once;
	std::call_once(once, []
	{
		const std::string exeName = DetectBrokenHudlessExeName();
		if (!exeName.empty())
		{
			Plugin::options.hudlessUi.store(false, std::memory_order_relaxed);
			Log::Info("Known broken HUD-less buffer in {}: disabling HUDless UI by default to prevent XeFG UI-mask overwrite (toggle HUDless UI in menu to override)",
				exeName);
		}
	});
}

// OptiScaler's ASI plugin entry point: with LoadAsiPlugins=true it loads every .asi in its plugins folder at start-up and
// calls this right away, long before a game creates its Vulkan instance and device. The dll OptiScaler uses is loaded
// later, when the game asks for DLSS Frame Generation, which for a Vulkan game is after the device exists and too late
// for the Vulkan hooks. So for Vulkan games this dll is also copied to OptiScaler's plugins folder renamed to .asi: that
// copy loads the real one from custom\<name>\dlssg here, and the real one installs the hooks.
// Dedicated handshake entry point called by dlssg_to_xefg_loader.asi before the game creates its Vulkan device.
// Exporting this unique symbol also lets the loader identify this mod's DLL via its PE export table without running DllMain.
extern "C" __declspec(dllexport) void XeFgEarlyBootstrap(const char* caller)
{
	try
	{
		const char* tag = (caller && *caller) ? caller : "XeFgEarlyBootstrap";
		Log::Info("XeFgEarlyBootstrap ({}), {}", tag, ThisModulePath().string());
		VulkanRuntime::Start(tag);
		PreloadVulkanModelsIfLoaded();
	}
	catch (const std::exception& error)
	{
		Log::Error("XeFgEarlyBootstrap: {}", error.what());
	}
	catch (...)
	{
		Log::Error("XeFgEarlyBootstrap failed");
	}
}

extern "C" __declspec(dllexport) void InitializeASI()
{
	try
	{
		StartFromAsi();
	}
	catch (const std::exception& error)
	{
		Log::Error("InitializeASI: {}", error.what());
	}
	catch (...)
	{
		Log::Error("InitializeASI failed");
	}
}

extern "C" __declspec(dllexport) bool OptiPluginInit(const OptiPluginApi* optiApi, OptiPlugin* plugin)
{
	// The early copy has no menu or settings of its own; those belong to the main dll.
	// OptiScaler v10+ may invoke OptiPluginInit without calling InitializeASI when OptiPluginInit is exported,
	// so trigger StartFromAsi() here as well (idempotent via std::call_once).
	if (IsEarlyCopy())
	{
		try
		{
			StartFromAsi();
		}
		catch (const std::exception& error)
		{
			Log::Error("OptiPluginInit (early copy): {}", error.what());
		}
		catch (...)
		{
			Log::Error("OptiPluginInit (early copy) failed");
		}

		return false;
	}

	api = optiApi;
	self = plugin;

	Log::Info("OptiScaler plugin API version {}", api->version);

	const int32_t path = api->GetInt(self, "NetworkPath", (int32_t)Plugin::options.path.load());
	if (path >= (int32_t)Plugin::Path::XmxPort && path <= (int32_t)Plugin::Path::Wmma)
		Plugin::options.path = (Plugin::Path)path;

	Plugin::options.hudlessUi = api->GetBool(self, "HudlessUi", Plugin::options.hudlessUi);
	Plugin::ApplyBrokenHudlessQuirkOnce();
	Plugin::options.cameraCutDetection = api->GetBool(self, "CameraCutDetection", Plugin::options.cameraCutDetection);
	Plugin::options.vramGuard = api->GetBool(self, "VramGuard", Plugin::options.vramGuard);
	Plugin::options.autoVrrPacing = api->GetBool(self, "AutoVrrPacing", Plugin::options.autoVrrPacing);
	Plugin::options.vkAmdAntiLag = api->GetBool(self, "VkAmdAntiLag", Plugin::options.vkAmdAntiLag);
	Plugin::options.d3d12VulkanWmmaBridge = api->GetBool(self, "D3D12VulkanWmmaBridge", Plugin::options.d3d12VulkanWmmaBridge);
	initialVkWmmaBridge = Plugin::options.d3d12VulkanWmmaBridge.load(std::memory_order_relaxed);
	initialVkWmmaCaptured = true;
	Plugin::options.uiTransitionCooldown = api->GetBool(self, "UiTransitionCooldown", Plugin::options.uiTransitionCooldown);
	Plugin::options.ghostBuster = api->GetBool(self, "GhostBuster", Plugin::options.ghostBuster);
	Plugin::options.pacingSpinlockSleep = api->GetBool(self, "PacingSpinlockSleep", Plugin::options.pacingSpinlockSleep);
	Plugin::options.depthInvertedOverride = std::clamp(api->GetInt(self, "DepthInvertedOverride", 0), 0, 2);
	Plugin::options.jitteredMvOverride = std::clamp(api->GetInt(self, "JitteredMvOverride", 0), 0, 2);
	Plugin::options.uiAlphaMode = std::clamp(api->GetInt(self, "UiAlphaMode", 0), 0, 2);
	Plugin::options.hudTierMode = std::clamp(api->GetInt(self, "HudTierMode", 0), 0, 4);
	Plugin::options.toneMapMatchMode = std::clamp(api->GetInt(self, "ToneMapMatchMode", 0), 0, 2);
	Plugin::options.ignoreGameReset = api->GetBool(self, "IgnoreGameReset", Plugin::options.ignoreGameReset);
	Plugin::options.logging = api->GetBool(self, "Logging", Plugin::options.logging);
	Log::SetEnabled(Plugin::options.logging);
	Plugin::options.crashMiniDump = Log::CrashMiniDumpRequestedByEnvironment() ||
		api->GetBool(self, "CrashMiniDump", false);
	Log::SetCrashMiniDumpEnabled(Plugin::options.crashMiniDump);
	Plugin::options.forceDred = api->GetBool(self, "ForceDRED", Plugin::options.forceDred);
	{
		wchar_t envBuf[16]{};
		const DWORD envLen = GetEnvironmentVariableW(L"XEFG_FORCE_DRED", envBuf, static_cast<DWORD>(std::size(envBuf)));
		if (envLen > 0 && envLen < std::size(envBuf) &&
			(_wcsicmp(envBuf, L"1") == 0 || _wcsicmp(envBuf, L"true") == 0 || _wcsicmp(envBuf, L"on") == 0 || _wcsicmp(envBuf, L"yes") == 0))
		{
			Plugin::options.forceDred = true;
		}
	}
	if (Plugin::options.forceDred)
		Log::EnableDred();
	RefreshAmdFlipMeterState(true);

	// Off in OptiScaler.ini, the Vulkan hooks aren't installed. Said before they are, and a copy started by
	// InitializeASI that has installed them already only passes the game's calls on from here
	if (!api->GetBool(self, "VulkanHooks", true))
	{
		Log::Info("VulkanHooks is off in OptiScaler.ini, the game's Vulkan device is left alone");
		VulkanRuntime::SetHooksEnabled(false);
	}

	// Lazy-load Vulkan models: only kick off background verification if vulkan-1.dll is already in the process.
	// Dynamic vulkan-1.dll loads trigger VulkanPrograms::PreloadAsync() from VulkanHost, and the D3D12-Vulkan WMMA
	// bridge loads on demand via VulkanPrograms::Installed().
	PreloadVulkanModelsIfLoaded();

	// Late for a game that already created its Vulkan device, which the log says when it asks for frame generation. Not
	// from DllMain: that is why this is here and not there
	try
	{
		VulkanRuntime::Start("OptiPluginInit");
	}
	catch (const std::exception& error)
	{
		Log::Warn("Vulkan hooks: {}", error.what());
	}

	return api->RegisterTab(self, "XeMFG XMX", "XeFG replacing DLSS Frame Generation", DrawTab, nullptr);
}
