#include "Plugin.h"
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
#include <system_error>

namespace
{
	const OptiPluginApi* api = nullptr;
	OptiPlugin* self = nullptr;

	std::mutex mutex;
	Plugin::Status status;
	ULONGLONG reportedAt = 0;

	// No real frame for this long and frame generation is off
	constexpr ULONGLONG IdleAfterMs = 1000;

	void Green(const char* text) { api->TextColored(0.45f, 0.85f, 0.45f, 1.0f, text); }
	void Red(const char* text) { api->TextColored(1.0f, 0.45f, 0.4f, 1.0f, text); }

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
			Red("XeFG couldn't start");
			logTooltip();
		}
		else if (active)
		{
			Green(std::format("Generating {}x", current.generatedFrames + 1).c_str());
			api->Text(std::format("{}x{} \xC2\xB7 {}", current.width, current.height,
				current.hudless ? "HUDless UI" : "UI interpolated").c_str());
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

		if (api->Checkbox("Vulkan hooks", &hooksOn))
		{
			VulkanRuntime::SetHooksEnabled(hooksOn);
			api->SetBool(self, "VulkanHooks", hooksOn);
		}

		api->ItemTooltip("Off: the game's Vulkan device is left alone and Vulkan games get no frame generation. Hooks that\n"
						 "aren't installed yet come with the next start of the game. XEFG_VULKAN_HOOKS=0 in the environment\n"
						 "does the same. Saved as VulkanHooks in OptiScaler.ini.");
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

	// The Vulkan models in xefg_mfg_programs\vulkan, read and checked when the plugin starts
	void DrawVulkanModels()
	{
		const auto* set = VulkanPrograms::Loaded();

		if (!set)
			return;

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
			"Every program was checked against its SHA-256 at start-up.{}",
			VulkanPrograms::Summary(*set), VulkanPrograms::PathText(set->directory),
			VulkanRuntime::Built() ? " Vulkan games run them on this dll's own D3D12-on-Vulkan layer."
			                       : " Running them takes the D3D12 layer of MFGUnlocker.asi (this build has none of its own).")
				.c_str());
	}

	void DrawSettings()
	{
		api->Section("Settings");

		static constexpr const char* PathNames[] = { "XMX Port", "CUDA", "Fallback (DP4a)", "XMX (native)", "WMMA" };
		const bool cudaDisabled = Plugin::cudaSupport.load() == Plugin::CudaSupport::Unsupported;
		const bool xmxDisabled = Plugin::xmxSupport.load() != Plugin::XmxSupport::Supported;
		const bool wmmaDisabled = Plugin::wmmaSupport.load() == Plugin::WmmaSupport::Unsupported;

		int32_t path = (int32_t)Plugin::options.path.load();

		// A saved choice the GPU doesn't support runs something else, show what is actually running
		if (path == (int32_t)Plugin::Path::XmxNative && xmxDisabled)
			path = !cudaDisabled ? (int32_t)Plugin::Path::Cuda : (int32_t)Plugin::Path::XmxPort;

		if (path == (int32_t)Plugin::Path::Cuda && cudaDisabled)
			path = (int32_t)Plugin::Path::XmxPort;

		if (path == (int32_t)Plugin::Path::Wmma && wmmaDisabled)
			path = (int32_t)Plugin::Path::XmxPort;

		const bool disabled[] = { false, cudaDisabled, false, xmxDisabled, wmmaDisabled };

		if (api->Combo("Network path", &path, PathNames, 5, disabled))
		{
			Plugin::options.path = (Plugin::Path)path;
			api->SetInt(self, "NetworkPath", path);
		}

		api->ItemTooltip("XMX (native): XeFG's own XMX network, the default on Intel GPUs with XMX\n"
						 "XMX Port: ported XMX model on DXIL / SPIR-V (DP4a INT8)\n"
						 "CUDA: ported XMX model on NVIDIA tensor cores, RTX GPUs only\n"
						 "WMMA: ported XMX model accelerated with WMMA INT8 (AMD RDNA 4 / RDNA 3 WaveMatrix & Vulkan Cooperative Matrix)\n"
						 "Fallback (DP4a): XeFG's default model\n"
						 "Vulkan games run the ported model on VK_KHR_cooperative_matrix (AMD WMMA / Intel XMX) for WMMA, on NVIDIA tensor\n"
						 "cores for CUDA, or as DP4a SPIR-V programs for XMX Port. Fallback (DP4a) runs XeFG's own model (see Vulkan)");

		bool hudlessUi = Plugin::options.hudlessUi;

		if (api->Checkbox("HUDless UI", &hudlessUi))
		{
			Plugin::options.hudlessUi = hudlessUi;
			api->SetBool(self, "HudlessUi", hudlessUi);
		}

		api->ItemTooltip("Uses the game's HUDless image to draw the UI into generated frames, instead of\n"
						 "interpolating the UI with the scene. It doesn't highlight the HUD.\n"
						 "Turn off if the game's HUDless image is broken.");

		bool cameraCut = Plugin::options.cameraCutDetection;

		if (api->Checkbox("Camera cut detection", &cameraCut))
		{
			Plugin::options.cameraCutDetection = cameraCut;
			api->SetBool(self, "CameraCutDetection", cameraCut);
		}

		api->ItemTooltip("Automatically resets XeFG frame history on sudden camera teleports or cutscene cuts\n"
						 "to prevent jelly/smearing artifacts. Saved as CameraCutDetection in OptiScaler.ini.");

		bool vramGuard = Plugin::options.vramGuard;

		if (api->Checkbox("Adaptive VRAM guard", &vramGuard))
		{
			Plugin::options.vramGuard = vramGuard;
			api->SetBool(self, "VramGuard", vramGuard);
		}

		api->ItemTooltip("Monitors local GPU VRAM via IDXGIAdapter3 and temporarily clamps 3x-6x MFG down to 2x\n"
						 "when within 200 MiB of budget to prevent system-RAM swap stutters. Saved as VramGuard.");

		bool autoVrr = Plugin::options.autoVrrPacing;

		if (api->Checkbox("Auto VRR pacing (Hz - 3)", &autoVrr))
		{
			Plugin::options.autoVrrPacing = autoVrr;
			api->SetBool(self, "AutoVrrPacing", autoVrr);
		}

		api->ItemTooltip("Limits the Vulkan present pacer cadence to (Monitor Hz - 3) to stay inside the VRR\n"
						 "(G-Sync / FreeSync) window and avoid V-Sync latency spikes. Saved as AutoVrrPacing.");

		bool vkWmmaBridge = Plugin::options.d3d12VulkanWmmaBridge;

		if (api->Checkbox("D3D12->Vulkan WMMA bridge (experimental)", &vkWmmaBridge))
		{
			Plugin::options.d3d12VulkanWmmaBridge = vkWmmaBridge;
			api->SetBool(self, "D3D12VulkanWmmaBridge", vkWmmaBridge);
		}

		api->ItemTooltip("Routes D3D12 WMMA batches through a private Vulkan VK_KHR_cooperative_matrix device\n"
						 "(takes effect on next game launch; set XEFG_D3D12_VULKAN_WMMA=0 to force off).\n"
						 "On by default; disable to fall back to D3D12 cs_6_6 WMMA overrides without duplicate VRAM\n"
						 "or OptiScaler Vulkan_wDx12 hook interference. Saved as D3D12VulkanWmmaBridge.");

		bool uiCooldown = Plugin::options.uiTransitionCooldown;

		if (api->Checkbox("UI / HUDless transition guard (10 frames)", &uiCooldown))
		{
			Plugin::options.uiTransitionCooldown = uiCooldown;
			api->SetBool(self, "UiTransitionCooldown", uiCooldown);
		}

		api->ItemTooltip("Skips interpolation for 10 frames whenever HUD-less or UI composition state toggles "
						 "(from DLSS Enabler XeFG) to prevent glitched menu/overlay transition frames. Saved as UiTransitionCooldown.");

		bool ghostBuster = Plugin::options.ghostBuster;

		if (api->Checkbox("GhostBuster anti-ghosting", &ghostBuster))
		{
			Plugin::options.ghostBuster = ghostBuster;
			api->SetBool(self, "GhostBuster", ghostBuster);
		}

		api->ItemTooltip("Tightens disocclusion history resets on sharp camera snaps (>45 deg), sudden FOV/ADS jumps, "
						 "and missing camera motion vectors (from DLSS Enabler GhostBuster). Saved as GhostBuster.");

		bool spinSleep = Plugin::options.pacingSpinlockSleep;

		if (api->Checkbox("Pacing spinlock sleep (lower CPU)", &spinSleep))
		{
			Plugin::options.pacingSpinlockSleep = spinSleep;
			api->SetBool(self, "PacingSpinlockSleep", spinSleep);
		}

		api->ItemTooltip("Shrinks the Vulkan pacer spin tail and yields the thread while waiting, reducing CPU usage "
						 "(from DLSS Enabler). Saved as PacingSpinlockSleep.");

		bool debugLines = Plugin::options.debugLines;

		if (api->Checkbox("Debug lines", &debugLines))
			Plugin::options.debugLines = debugLines;

		api->ItemTooltip("XeFG draws debug lines into the frames it generated. Not saved.");

		api->Section("Extended XeFG Overrides");

		static constexpr const char* TriStateNames[] = { "Auto (Game)", "Force Off", "Force On" };

		int32_t depthOv = Plugin::options.depthInvertedOverride.load();
		if (api->Combo("Depth inverted", &depthOv, TriStateNames, 3, nullptr))
		{
			Plugin::options.depthInvertedOverride = depthOv;
			api->SetInt(self, "DepthInvertedOverride", depthOv);
		}
		api->ItemTooltip("Overrides XeFG DepthInverted flag when a game reports inverted depth incorrectly. "
						 "Saved as DepthInvertedOverride.");

		int32_t jitterOv = Plugin::options.jitteredMvOverride.load();
		if (api->Combo("Jittered motion vectors", &jitterOv, TriStateNames, 3, nullptr))
		{
			Plugin::options.jitteredMvOverride = jitterOv;
			api->SetInt(self, "JitteredMvOverride", jitterOv);
		}
		api->ItemTooltip("Overrides XeFG JitteredMV flag when a game misreports motion-vector jitter. "
						 "Saved as JitteredMvOverride.");

		bool ignoreReset = Plugin::options.ignoreGameReset;
		if (api->Checkbox("Ignore game Reset signals", &ignoreReset))
		{
			Plugin::options.ignoreGameReset = ignoreReset;
			api->SetBool(self, "IgnoreGameReset", ignoreReset);
		}
		api->ItemTooltip("Ignores DLSSG.Reset=1 from games that spam reset every frame. Internal safety resets "
						 "(subrect change, camera cut, HUD-less toggle) still run. Saved as IgnoreGameReset.");
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

	// Pre-read and validate xefg_mfg_programs\vulkan on a background thread during early bootstrap so
	// OptiPluginInit doesn't stall the main thread for disk I/O when the game initializes.
	std::once_flag preloadModelsOnce;

	void StartAsyncModelPreload()
	{
		std::call_once(preloadModelsOnce,
			[]
			{
				try
				{
					std::thread(
						[]
						{
							try
							{
								VulkanPrograms::Installed();
							}
							catch (const std::exception& error)
							{
								Log::Warn("Vulkan models (early preload): {}", error.what());
							}
							catch (...)
							{
							}
						}).detach();
				}
				catch (...)
				{
				}
			});
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
			StartAsyncModelPreload();
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
		StartAsyncModelPreload();
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
	Plugin::options.cameraCutDetection = api->GetBool(self, "CameraCutDetection", Plugin::options.cameraCutDetection);
	Plugin::options.vramGuard = api->GetBool(self, "VramGuard", Plugin::options.vramGuard);
	Plugin::options.autoVrrPacing = api->GetBool(self, "AutoVrrPacing", Plugin::options.autoVrrPacing);
	Plugin::options.d3d12VulkanWmmaBridge = api->GetBool(self, "D3D12VulkanWmmaBridge", Plugin::options.d3d12VulkanWmmaBridge);
	Plugin::options.uiTransitionCooldown = api->GetBool(self, "UiTransitionCooldown", Plugin::options.uiTransitionCooldown);
	Plugin::options.ghostBuster = api->GetBool(self, "GhostBuster", Plugin::options.ghostBuster);
	Plugin::options.pacingSpinlockSleep = api->GetBool(self, "PacingSpinlockSleep", Plugin::options.pacingSpinlockSleep);
	Plugin::options.depthInvertedOverride = std::clamp(api->GetInt(self, "DepthInvertedOverride", 0), 0, 2);
	Plugin::options.jitteredMvOverride = std::clamp(api->GetInt(self, "JitteredMvOverride", 0), 0, 2);
	Plugin::options.ignoreGameReset = api->GetBool(self, "IgnoreGameReset", Plugin::options.ignoreGameReset);

	// Off in OptiScaler.ini, the Vulkan hooks aren't installed. Said before they are, and a copy started by
	// InitializeASI that has installed them already only passes the game's calls on from here
	if (!api->GetBool(self, "VulkanHooks", true))
	{
		Log::Info("VulkanHooks is off in OptiScaler.ini, the game's Vulkan device is left alone");
		VulkanRuntime::SetHooksEnabled(false);
	}

	// Reads and checks the Vulkan models next to the dll now, so neither the menu nor a frame pays for it
	try
	{
		VulkanPrograms::Installed();
	}
	catch (const std::exception& error)
	{
		Log::Warn("Vulkan models: {}", error.what());
	}

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
