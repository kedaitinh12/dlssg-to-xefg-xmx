#include "VulkanHost.h"

#include "../Log.h"
#include "VulkanCoopMat.h"
#include "VulkanPolicy.h"
#include "VulkanFeatureChain.h"
#include "VulkanTensor.h"

#include <MinHook.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

// See VulkanHost.h. The hooks are MinHook inline detours on vulkan-1.dll's own vkCreateInstance and vkCreateDevice, so
// they catch the game however it got those functions: linked, GetProcAddress, vkGetInstanceProcAddr, a cached pointer.
// An overlay or a layer that hooks the same functions chains with them, MinHook follows the jump an earlier hook left.

namespace
{
	constexpr size_t MaxInstances = 16;
	constexpr size_t MaxDevices = 64;
	constexpr size_t MaxSwapchains = 64;
	constexpr wchar_t LoaderName[] = L"vulkan-1.dll";
	constexpr DWORD HighResTimerFlag = 0x00000002u; // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION

	struct State
	{
		// The maps and the strings, held briefly and never while calling out
		std::mutex mutex;
		std::map<VkInstance, uint32_t> instances; // the API version the instance was created with
		std::map<VkDevice, VulkanHost::DeviceRecord> devices;
		std::map<VkSwapchainKHR, VkPresentModeKHR> swapchains;
		std::map<VkPhysicalDevice, std::string> refused;
		std::string detail;
		std::string lastRefusal;
		VulkanRuntime::Hooks hooks = VulkanRuntime::Hooks::Off;

		// Start and the load hook installing the hooks. loaderHooked is also read by the load hook without the lock
		std::mutex install;
		bool started = false;
		std::atomic_bool loaderHooked = false;

		std::atomic_bool enabled = true;
		std::atomic_uint32_t instanceCalls = 0;
		std::atomic_uint32_t deviceCalls = 0;
		std::atomic_uint32_t prepared = 0;

		PFN_vkGetInstanceProcAddr loaderGipa = nullptr;
		PFN_vkGetDeviceProcAddr loaderGdpa = nullptr;
		PFN_vkCreateInstance createInstance = nullptr;
		PFN_vkCreateDevice createDevice = nullptr;
		PFN_vkCreateSwapchainKHR createSwapchain = nullptr;
		PFN_vkDestroySwapchainKHR destroySwapchain = nullptr;
		PFN_vkQueuePresentKHR queuePresent = nullptr;
		void* rawCreateInstance = nullptr;
		void* rawCreateDevice = nullptr;
		void* rawGdpa = nullptr;
		void* rawCreateSwapchain = nullptr;
		void* rawDestroySwapchain = nullptr;
		void* rawQueuePresent = nullptr;
		std::atomic<PFN_vkCreateInstance> gipaCreateInstance = nullptr;
		std::atomic<PFN_vkCreateDevice> gipaCreateDevice = nullptr;
		std::atomic<PFN_vkGetDeviceProcAddr> gipaGdpa = nullptr;
		std::atomic<PFN_vkCreateSwapchainKHR> gdpaCreateSwapchain = nullptr;
		std::atomic<PFN_vkDestroySwapchainKHR> gdpaDestroySwapchain = nullptr;
		std::atomic<PFN_vkQueuePresentKHR> gdpaQueuePresent = nullptr;
		bool gipaHooked = false;
		bool presentHooked = false;
		HMODULE(WINAPI* loadLibraryExW)(LPCWSTR, HANDLE, DWORD) = nullptr;

		// Dedicated lock for low-overhead present pacing
		std::mutex pacerMutex;
		VulkanPolicy::PresentPacer pacer;
		std::atomic_bool hwPacingActive = false;
		std::atomic_bool pacingLogged = false;
	};

	// Leaked on purpose: a game's thread can call the hooks while the process exits, after the static destructors
	State& S()
	{
		static State* state = new State;
		return *state;
	}

	template <class T> T* FindStruct(const void* next, VkStructureType type)
	{
		for (unsigned links = 0; next && links < 64; ++links)
		{
			auto* base = static_cast<const VkBaseInStructure*>(next);

			if (base->sType == type)
				return reinterpret_cast<T*>(const_cast<VkBaseInStructure*>(base));

			next = base->pNext;
		}

		return nullptr;
	}

	void SetHookState(VulkanRuntime::Hooks hooks, std::string detail)
	{
		State& s = S();
		std::lock_guard lock(s.mutex);
		s.hooks = hooks;
		s.detail = std::move(detail);
	}

	enum class PacingMode { Off, Auto, Force };

	PacingMode GetPacingMode() noexcept
	{
		static const PacingMode mode = [] {
			wchar_t value[16]{};
			const DWORD len = GetEnvironmentVariableW(L"XEFG_VK_PACING", value, 16);
			if (len && len < 16)
			{
				if (value[0] == L'0' || _wcsicmp(value, L"off") == 0 || _wcsicmp(value, L"false") == 0)
					return PacingMode::Off;
				if (_wcsicmp(value, L"force") == 0 || _wcsicmp(value, L"always") == 0)
					return PacingMode::Force;
			}
			return PacingMode::Auto;
		}();
		return mode;
	}

	bool PacingActive(const State& s) noexcept
	{
		const PacingMode mode = GetPacingMode();
		if (mode == PacingMode::Off)
			return false;
		if (mode == PacingMode::Force)
			return true;
		return !s.hwPacingActive.load(std::memory_order_relaxed);
	}

	uint64_t NowNs() noexcept
	{
		static const uint64_t freq = [] {
			LARGE_INTEGER f{};
			QueryPerformanceFrequency(&f);
			return f.QuadPart > 0 ? static_cast<uint64_t>(f.QuadPart) : 10'000'000ull;
		}();
		LARGE_INTEGER counter{};
		QueryPerformanceCounter(&counter);
		const uint64_t ticks = counter.QuadPart > 0 ? static_cast<uint64_t>(counter.QuadPart) : 0ull;
		const uint64_t sec = ticks / freq;
		const uint64_t rem = ticks % freq;
		return sec * 1'000'000'000ull + (rem * 1'000'000'000ull) / freq;
	}

	// Sub-millisecond wait using a per-thread Windows 10/11 high-resolution waitable timer (avoiding winmm.dll
	// which OptiScaler frequently proxies) plus a short QPC spin-wait tail.
	void WaitPreciseNs(uint64_t waitNs) noexcept
	{
		if (waitNs == 0)
			return;

		const uint64_t deadline = NowNs() + waitNs;
		constexpr uint64_t SpinTailNs = 1'200'000ull; // 1.2 ms QPC spin tail

		using CreateWaitableTimerExWFn = HANDLE(WINAPI*)(void*, LPCWSTR, DWORD, DWORD);
		static const auto createTimerEx = [] {
			const HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
			return k32 ? reinterpret_cast<CreateWaitableTimerExWFn>(GetProcAddress(k32, "CreateWaitableTimerExW")) : nullptr;
		}();

		struct ThreadTimer
		{
			HANDLE handle = nullptr;
			~ThreadTimer() { if (handle) CloseHandle(handle); }
		};
		static thread_local ThreadTimer timer;
		static thread_local bool timerAttempted = false;
		if (!timerAttempted)
		{
			timerAttempted = true;
			if (createTimerEx)
				timer.handle = createTimerEx(nullptr, nullptr, HighResTimerFlag, 0x001F0003u /* TIMER_ALL_ACCESS */);
		}

		uint64_t now = NowNs();
		if (now + SpinTailNs < deadline)
		{
			const uint64_t sleepNs = (deadline - now) - SpinTailNs;
			if (timer.handle)
			{
				LARGE_INTEGER due{};
				due.QuadPart = -static_cast<int64_t>(sleepNs / 100ull); // relative 100 ns intervals
				if (due.QuadPart < 0 && SetWaitableTimer(timer.handle, &due, 0, nullptr, nullptr, 0))
					WaitForSingleObject(timer.handle, 50u);
			}
			else if (sleepNs >= 2'000'000ull)
			{
				Sleep(static_cast<DWORD>(sleepNs / 1'000'000ull));
			}
		}

		while (NowNs() < deadline)
			YieldProcessor();
	}

	// ----------------------------------------------------------------------------------------------- instance

	struct InstanceEdit
	{
		VkInstanceCreateInfo info{};
		VkApplicationInfo application{};
	};

	uint32_t RequestedVersion(const VkInstanceCreateInfo* info)
	{
		return info->pApplicationInfo && info->pApplicationInfo->apiVersion ? info->pApplicationInfo->apiVersion
		                                                                  : VK_API_VERSION_1_0;
	}

	// XeFG's programs need Vulkan 1.3. A game that asks for less gets 1.3 when the loader has it: the physical device
	// decides what the device can use, the instance only has to allow it
	const VkInstanceCreateInfo* PrepareInstance(const VkInstanceCreateInfo* info, InstanceEdit& edit) noexcept
	{
		State& s = S();

		if (!info || !s.loaderGipa)
			return info;

		try
		{
			const uint32_t requested = RequestedVersion(info);

			if (requested >= xvk::kRequiredApiVersion)
				return info;

			auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
				s.loaderGipa(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
			uint32_t loaderVersion = VK_API_VERSION_1_0;

			if (!enumerate || enumerate(&loaderVersion) != VK_SUCCESS || loaderVersion < xvk::kRequiredApiVersion)
			{
				Log::Warn("Vulkan loader is older than 1.3, the game's instance stays at {}.{}",
					VK_API_VERSION_MAJOR(requested), VK_API_VERSION_MINOR(requested));
				return info;
			}

			edit.info = *info;

			if (info->pApplicationInfo)
				edit.application = *info->pApplicationInfo;
			else
				edit.application = { VK_STRUCTURE_TYPE_APPLICATION_INFO };

			edit.application.apiVersion = xvk::kRequiredApiVersion;
			edit.info.pApplicationInfo = &edit.application;

			Log::Info("Vulkan instance: the game asks for {}.{}, created as 1.3", VK_API_VERSION_MAJOR(requested),
				VK_API_VERSION_MINOR(requested));

			return &edit.info;
		}
		catch (...)
		{
			return info;
		}
	}

	void InstanceCreated(VkInstance instance, const VkInstanceCreateInfo* used) noexcept
	{
		try
		{
			State& s = S();
			std::lock_guard lock(s.mutex);

			if (s.instances.size() >= MaxInstances)
				s.instances.clear();

			s.instances[instance] = RequestedVersion(used);
		}
		catch (...)
		{
		}
	}

	struct ReentryGuard
	{
		bool& flag;
		explicit ReentryGuard(bool& f) : flag(f) { flag = true; }
		~ReentryGuard() { flag = false; }
	};

	VkResult CreateInstanceImpl(PFN_vkCreateInstance next, const VkInstanceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkInstance* instance)
	{
		if (!next)
			return VK_ERROR_INITIALIZATION_FAILED;

		static thread_local bool inCreateInstance = false;
		if (inCreateInstance)
			return next(info, allocator, instance);

		ReentryGuard guard(inCreateInstance);

		State& s = S();
		s.instanceCalls++;

		InstanceEdit edit;
		const VkInstanceCreateInfo* used = s.enabled ? PrepareInstance(info, edit) : info;

		VkResult result = next(used, allocator, instance);

		if (result != VK_SUCCESS && used != info)
		{
			Log::Warn("The loader refused the instance at 1.3 ({}), created as the game asked", static_cast<int>(result));
			used = info;
			result = next(info, allocator, instance);
		}

		if (result == VK_SUCCESS && instance && *instance)
			InstanceCreated(*instance, used);

		return result;
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateInstance(const VkInstanceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkInstance* instance)
	{
		return CreateInstanceImpl(S().createInstance, info, allocator, instance);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateInstanceViaGipa(const VkInstanceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkInstance* instance)
	{
		State& s = S();
		PFN_vkCreateInstance next = s.gipaCreateInstance.load(std::memory_order_acquire);
		return CreateInstanceImpl(next ? next : s.createInstance, info, allocator, instance);
	}

	// ------------------------------------------------------------------------------------------------ device

	struct DeviceEdit
	{
		VkDeviceCreateInfo info{};
		std::vector<const char*> extensions;
		VkPhysicalDeviceFeatures features10{};
		VkPhysicalDeviceTimelineSemaphoreFeatures timeline{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
		VkPhysicalDeviceSynchronization2Features sync2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES };
		VkPhysicalDeviceMaintenance4Features maintenance4{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES };
		VkPhysicalDeviceShaderIntegerDotProductFeatures dot{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES
		};
		VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
		};

		VulkanHost::FeatureChain chain;
		VulkanHost::DeviceRecord record;

		// How many of the last entries of `extensions` are the tensor-core route's NVX or cooperative-matrix extensions
		// this edit added (none the game asked for itself): the part of the edit XeFG can do without
		size_t tensorAdded = 0;

		// The device the driver takes without the NVX / cooperative-matrix extensions: XeFG's features and push descriptors
		// stay, the hardware matrix route doesn't exist and the SPIR-V programs run every layer
		void DropTensorExtensions()
		{
			extensions.resize(extensions.size() - std::min(tensorAdded, extensions.size()));
			tensorAdded = 0;
			record.tensorExtensions = false;
			record.coopMatrixExtensions = false;
			if (info.pNext == &coop)
				info.pNext = coop.pNext;
			info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
			info.ppEnabledExtensionNames = extensions.data();
		}

		void Raise(VkBool32& field)
		{
			// An already enabled field may belong to the borrowed tail: do not write it.
			if (field != VK_TRUE) field = VK_TRUE;
		}

	};

	std::unique_ptr<DeviceEdit> Refuse(VkPhysicalDevice physical, const char* code) noexcept
	{
		try
		{
			State& s = S();
			{
				std::lock_guard lock(s.mutex);
				s.lastRefusal = code;
				s.refused[physical] = code;
			}

			Log::Warn("Vulkan device {} left alone: {}", static_cast<const void*>(physical), VulkanPolicy::Describe(code));
		}
		catch (...)
		{
		}

		return nullptr;
	}

	// What XeFG's programs need from a device, and the game's device edited to have it. Null (the game's request goes
	// through as it is) when the device can't have it
	std::unique_ptr<DeviceEdit> PrepareDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info) noexcept
	{
		State& s = S();

		if (!physical || !info || !s.loaderGipa)
			return nullptr;

		try
		{
			auto edit = std::make_unique<DeviceEdit>();

			// The instance that enumerated this physical device and the API version it was created with
			std::vector<std::pair<VkInstance, uint32_t>> known;
			{
				std::lock_guard lock(s.mutex);
				known.assign(s.instances.begin(), s.instances.end());
			}

			uint32_t instanceVersion = 0;

			for (const auto& [instance, apiVersion] : known)
			{
				auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
					s.loaderGipa(instance, "vkEnumeratePhysicalDevices"));
				uint32_t count = 0;

				if (!enumerate || enumerate(instance, &count, nullptr) != VK_SUCCESS || !count || count > 64)
					continue;

				std::vector<VkPhysicalDevice> list(count);

				if (enumerate(instance, &count, list.data()) != VK_SUCCESS ||
					std::find(list.begin(), list.begin() + std::min<size_t>(count, list.size()), physical) == list.end())
					continue;

				edit->record.instance = instance;
				instanceVersion = apiVersion;
				break;
			}

			if (!edit->record.instance)
				return Refuse(physical, "instance_unknown");

			if (instanceVersion < xvk::kRequiredApiVersion)
				return Refuse(physical, "instance_api_version");

			const auto gipa = s.loaderGipa;
			const auto instance = edit->record.instance;

			auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gipa(instance, "vkGetPhysicalDeviceProperties2"));
			auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(gipa(instance, "vkGetPhysicalDeviceFeatures2"));
			auto familiesFn = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
				gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
			auto extensionsFn = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
				gipa(instance, "vkEnumerateDeviceExtensionProperties"));

			if (!properties2 || !features2 || !familiesFn || !extensionsFn)
				return Refuse(physical, "instance_functions");

			VkPhysicalDeviceVulkan11Properties v11Properties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES };
			VkPhysicalDeviceProperties2 properties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &v11Properties };
			properties2(physical, &properties);

			edit->record.vendor = properties.properties.vendorID;

			if (properties.properties.apiVersion < xvk::kRequiredApiVersion)
				return Refuse(physical, "device_api_version");

			// XeFG's inverted depth motion splat reduces with wave min/max: subgroup arithmetic in compute shaders
			constexpr VkSubgroupFeatureFlags SubgroupOps = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;

			if (!(v11Properties.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
				(v11Properties.subgroupSupportedOperations & SubgroupOps) != SubgroupOps)
				return Refuse(physical, "subgroup_operations");

			// What the layer and the program set use
			VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopSupported{
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
			};
			VkPhysicalDeviceVulkan13Features v13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &coopSupported };
			VkPhysicalDeviceVulkan12Features v12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13 };
			VkPhysicalDeviceFeatures2 supported{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12 };
			features2(physical, &supported);

			if (!v12.timelineSemaphore || !v13.synchronization2 || !v13.maintenance4 || !v13.shaderIntegerDotProduct ||
				!supported.features.shaderStorageImageExtendedFormats)
				return Refuse(physical, "device_features");

			uint32_t count = 0;
			extensionsFn(physical, nullptr, &count, nullptr);
			std::vector<VkExtensionProperties> available(count);
			extensionsFn(physical, nullptr, &count, available.data());

			for (const auto& extension : available)
			{
				if (!strcmp(extension.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
					edit->record.coopMatrixSpec = extension.specVersion;
			}

			// Intel GPUs are refused on Vulkan unless cooperative-matrix acceleration is available and allowed
			if (edit->record.vendor == 0x8086 &&
				(!edit->record.coopMatrixSpec || !coopSupported.cooperativeMatrix || !VulkanCoopMat::AllowIntelAdapter()))
				return Refuse(physical, "intel_adapter");

			auto offered = [&](const char* name)
				{
					return std::any_of(available.begin(), available.end(),
						[&](const VkExtensionProperties& extension) { return !strcmp(extension.extensionName, name); });
				};

			auto enabled = [&](const char* name)
				{
					for (uint32_t i = 0; i < info->enabledExtensionCount; ++i)
					{
						if (info->ppEnabledExtensionNames[i] && !strcmp(info->ppEnabledExtensionNames[i], name))
							return true;
					}

					return false;
				};

			if (!offered(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
				return Refuse(physical, "push_descriptor");

			// XeFG's dispatches are recorded into the game's own command buffers, so the family is the game's graphics one
			familiesFn(physical, &count, nullptr);
			std::vector<VkQueueFamilyProperties> families(count);
			familiesFn(physical, &count, families.data());

			bool family = false;

			for (uint32_t i = 0; i < info->queueCreateInfoCount && !family; ++i)
			{
				const auto& queue = info->pQueueCreateInfos[i];

				if (queue.queueFamilyIndex < families.size() &&
					(families[queue.queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT))
				{
					edit->record.family = queue.queueFamilyIndex;
					family = true;
				}
			}

			if (!family)
				return Refuse(physical, "no_graphics_family");

			// Only push descriptors (plus optional tensor/coopmat extensions) are added to the extensions, nothing is taken away
			if (info->enabledExtensionCount)
				edit->extensions.assign(info->ppEnabledExtensionNames, info->ppEnabledExtensionNames + info->enabledExtensionCount);

			if (!enabled(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
				edit->extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

			// The tensor-core route launches CUDA kernels in the game's command buffers (VK_NVX_binary_import) and gets the
			// handles of the images they read and write (VK_NVX_image_view_handle).
			if (edit->record.vendor == 0x10DE)
			{
				for (const auto& extension : available)
				{
					if (!strcmp(extension.extensionName, VK_NVX_BINARY_IMPORT_EXTENSION_NAME))
						edit->record.binaryImportSpec = extension.specVersion;

					if (!strcmp(extension.extensionName, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME))
						edit->record.imageViewHandleSpec = extension.specVersion;
				}

				if (edit->record.binaryImportSpec && edit->record.imageViewHandleSpec && VulkanTensor::Wanted())
				{
					for (const char* name : { VK_NVX_BINARY_IMPORT_EXTENSION_NAME, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME })
					{
						if (!enabled(name))
						{
							edit->extensions.push_back(name);
							edit->tensorAdded++;
						}
					}

					edit->record.tensorExtensions = true;
				}
			}

			// VK_KHR_cooperative_matrix for AMD RDNA3/RDNA4 (WMMA) and Intel Arc (XMX), or NVIDIA fallback when NVX is off
			const bool wantCoopMat = edit->record.coopMatrixSpec && coopSupported.cooperativeMatrix &&
				VulkanCoopMat::Wanted() && !edit->record.tensorExtensions;

			if (wantCoopMat)
			{
				if (!enabled(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
				{
					edit->extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
					edit->tensorAdded++;
				}

				edit->record.coopMatrixExtensions = true;
			}

			// Detect native NVIDIA hardware pacing extensions (e.g. VK_NV_present_metering / VK_NV_low_latency2)
			// so the CPU present pacer automatically yields to hardware flip metering on NVIDIA GPUs.
			edit->record.hwPacing =
				(offered("VK_NV_present_metering") && enabled("VK_NV_present_metering")) ||
				(offered("VK_NV_low_latency2") && enabled("VK_NV_low_latency2"));

			edit->info = *info;
			edit->info.enabledExtensionCount = static_cast<uint32_t>(edit->extensions.size());
			edit->info.ppEnabledExtensionNames = edit->extensions.data();

			// Features: raise bits in an owned copy of the game's chain (never in its const storage); the individual structures it doesn't pass are added in front of the chain. A request that
			// holds a Vulkan 1.2 or 1.3 features structure must not also hold the single feature structures it covers
			if (!edit->chain.Copy(info->pNext))
				return Refuse(physical, "feature_chain_unsupported");
			const void* chain = edit->chain.Head();

			if (auto* f2 = FindStruct<VkPhysicalDeviceFeatures2>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2))
				edit->Raise(f2->features.shaderStorageImageExtendedFormats);
			else
			{
				if (info->pEnabledFeatures)
					edit->features10 = *info->pEnabledFeatures;

				edit->features10.shaderStorageImageExtendedFormats = VK_TRUE;
				edit->info.pEnabledFeatures = &edit->features10;
			}

			void* head = const_cast<void*>(chain);

			auto prepend = [&head](auto& structure)
				{
					structure.pNext = head;
					head = &structure;
				};

			if (auto* f12 = FindStruct<VkPhysicalDeviceVulkan12Features>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES))
				edit->Raise(f12->timelineSemaphore);
			else if (auto* t = FindStruct<VkPhysicalDeviceTimelineSemaphoreFeatures>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES))
				edit->Raise(t->timelineSemaphore);
			else
			{
				edit->timeline.timelineSemaphore = VK_TRUE;
				prepend(edit->timeline);
			}

			if (auto* f13 = FindStruct<VkPhysicalDeviceVulkan13Features>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES))
			{
				edit->Raise(f13->synchronization2);
				edit->Raise(f13->maintenance4);
				edit->Raise(f13->shaderIntegerDotProduct);
			}
			else
			{
				if (auto* sync = FindStruct<VkPhysicalDeviceSynchronization2Features>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES))
					edit->Raise(sync->synchronization2);
				else
				{
					edit->sync2.synchronization2 = VK_TRUE;
					prepend(edit->sync2);
				}

				if (auto* m = FindStruct<VkPhysicalDeviceMaintenance4Features>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES))
					edit->Raise(m->maintenance4);
				else
				{
					edit->maintenance4.maintenance4 = VK_TRUE;
					prepend(edit->maintenance4);
				}

				if (auto* d = FindStruct<VkPhysicalDeviceShaderIntegerDotProductFeatures>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES))
					edit->Raise(d->shaderIntegerDotProduct);
				else
				{
					edit->dot.shaderIntegerDotProduct = VK_TRUE;
					prepend(edit->dot);
				}
			}

			if (edit->record.coopMatrixExtensions &&
				!FindStruct<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR))
			{
				edit->coop.cooperativeMatrix = VK_TRUE;
				prepend(edit->coop);
			}

			edit->info.pNext = head;
			edit->record.physical = physical;
			edit->record.gipa = gipa;

			return edit;
		}
		catch (...)
		{
			return Refuse(physical, "prepare_exception");
		}
	}

	void DeviceCreated(const DeviceEdit& edit, VkDevice device) noexcept
	{
		try
		{
			State& s = S();
			{
				std::lock_guard lock(s.mutex);

				if (s.devices.size() >= MaxDevices)
					s.devices.clear();

				s.devices[device] = edit.record;
				s.refused.erase(edit.record.physical);
			}

			if (edit.record.hwPacing)
				s.hwPacingActive.store(true, std::memory_order_relaxed);

			s.prepared++;
			Log::Info("Vulkan device {} prepared for XeFG: graphics family {}, vendor {:#06x}{}{}{}",
				static_cast<const void*>(device), edit.record.family, edit.record.vendor,
				edit.record.tensorExtensions ? ", tensor-core extensions enabled"
				: edit.record.vendor == 0x10DE ? (edit.record.binaryImportSpec && edit.record.imageViewHandleSpec
					? ", tensor-core extensions not enabled (XEFG_TENSOR_CORES is off or the driver refused them)"
					: ", the driver doesn't offer the tensor-core extensions")
				: "",
				edit.record.coopMatrixExtensions ? ", VK_KHR_cooperative_matrix enabled" : "",
				edit.record.hwPacing ? ", hardware pacing detected (CPU present pacer auto-bypassed)" : "");
		}
		catch (...)
		{
		}
	}

	VkResult CreateDeviceImpl(PFN_vkCreateDevice next, VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkDevice* device)
	{
		if (!next)
			return VK_ERROR_INITIALIZATION_FAILED;

		static thread_local bool inCreateDevice = false;
		if (inCreateDevice)
			return next(physical, info, allocator, device);

		ReentryGuard guard(inCreateDevice);

		State& s = S();
		s.deviceCalls++;

		std::unique_ptr<DeviceEdit> edit;

		if (s.enabled)
			edit = PrepareDevice(physical, info);

		VkResult result = next(physical, edit ? &edit->info : info, allocator, device);

		if (result != VK_SUCCESS && edit && edit->tensorAdded)
		{
			// The NVX / cooperative-matrix extensions are the part of the edit XeFG can do without: keep the rest and let
			// the SPIR-V programs run every layer
			Log::Warn("The driver refused the device with the tensor/cooperative-matrix extensions ({}), trying again without them",
				static_cast<int>(result));
			edit->DropTensorExtensions();
			result = next(physical, &edit->info, allocator, device);
		}

		if (result != VK_SUCCESS && edit)
		{
			// The driver took the game's request but not the edited one: the game gets the device it asked for, without
			// the bits that were raised in its own structures
			Log::Warn("The driver refused the device with XeFG's features ({}), created as the game asked",
				static_cast<int>(result));
			edit.reset();
			Refuse(physical, "device_features");
			result = next(physical, info, allocator, device);
		}

		if (result == VK_SUCCESS && edit && device && *device)
			DeviceCreated(*edit, *device);

		return result;
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkDevice* device)
	{
		return CreateDeviceImpl(S().createDevice, physical, info, allocator, device);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateDeviceViaGipa(VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
		const VkAllocationCallbacks* allocator, VkDevice* device)
	{
		State& s = S();
		PFN_vkCreateDevice next = s.gipaCreateDevice.load(std::memory_order_acquire);
		return CreateDeviceImpl(next ? next : s.createDevice, physical, info, allocator, device);
	}

	// OptiScaler hooks vulkan-1.dll before ASI plugins run and returns its own hkvkCreateDevice from
	// vkGetInstanceProcAddr. Wrapping vkGetInstanceProcAddr catches games that resolve vkCreateDevice via GIPA
	// and call OptiScaler's hook directly (whose saved trampoline would otherwise bypass our loader prologue hook).
	// --------------------------------------------------------------------------------------- swapchain / present pacing

	bool IsFifoMode(VkPresentModeKHR mode) noexcept
	{
		return mode == VK_PRESENT_MODE_FIFO_KHR || mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR;
	}

	VkResult CreateSwapchainImpl(PFN_vkCreateSwapchainKHR next, VkDevice device,
		const VkSwapchainCreateInfoKHR* info, const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain)
	{
		if (!next)
			return VK_ERROR_INITIALIZATION_FAILED;

		static thread_local bool inCreateSwapchain = false;
		if (inCreateSwapchain)
			return next(device, info, allocator, swapchain);

		ReentryGuard guard(inCreateSwapchain);
		const VkResult result = next(device, info, allocator, swapchain);

		if (result == VK_SUCCESS && info && swapchain && *swapchain)
		{
			State& s = S();
			std::lock_guard lock(s.mutex);
			if (s.swapchains.size() >= MaxSwapchains)
				s.swapchains.clear();
			s.swapchains[*swapchain] = info->presentMode;
		}

		return result;
	}

	void DestroySwapchainImpl(PFN_vkDestroySwapchainKHR next, VkDevice device, VkSwapchainKHR swapchain,
		const VkAllocationCallbacks* allocator)
	{
		if (!next)
			return;

		static thread_local bool inDestroySwapchain = false;
		if (inDestroySwapchain)
		{
			next(device, swapchain, allocator);
			return;
		}

		ReentryGuard guard(inDestroySwapchain);
		if (swapchain)
		{
			State& s = S();
			std::lock_guard lock(s.mutex);
			s.swapchains.erase(swapchain);
		}
		next(device, swapchain, allocator);
	}

	VkResult QueuePresentImpl(PFN_vkQueuePresentKHR next, VkQueue queue, const VkPresentInfoKHR* info)
	{
		if (!next)
			return VK_ERROR_INITIALIZATION_FAILED;

		static thread_local bool inQueuePresent = false;
		if (inQueuePresent)
			return next(queue, info);

		ReentryGuard guard(inQueuePresent);

		State& s = S();
		if (PacingActive(s))
		{
			bool vsyncFifo = false;
			if (info && info->swapchainCount > 0 && info->pSwapchains)
			{
				std::lock_guard lock(s.mutex);
				const auto it = s.swapchains.find(info->pSwapchains[0]);
				if (it != s.swapchains.end())
					vsyncFifo = IsFifoMode(it->second);
			}

			uint64_t waitNs = 0;
			{
				std::lock_guard lock(s.pacerMutex);
				waitNs = s.pacer.ComputePresentWaitNs(NowNs(), vsyncFifo);
			}

			if (waitNs > 0)
			{
				if (!s.pacingLogged.exchange(true, std::memory_order_relaxed))
					Log::Info("Vulkan present pacer active (V-Sync FIFO off, smoothing multi-frame burst presents)");
				WaitPreciseNs(waitNs);
			}
		}

		return next(queue, info);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info,
		const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain)
	{
		return CreateSwapchainImpl(S().createSwapchain, device, info, allocator, swapchain);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookCreateSwapchainViaGdpa(VkDevice device, const VkSwapchainCreateInfoKHR* info,
		const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain)
	{
		State& s = S();
		PFN_vkCreateSwapchainKHR next = s.gdpaCreateSwapchain.load(std::memory_order_acquire);
		return CreateSwapchainImpl(next ? next : s.createSwapchain, device, info, allocator, swapchain);
	}

	VKAPI_ATTR void VKAPI_CALL HookDestroySwapchain(VkDevice device, VkSwapchainKHR swapchain,
		const VkAllocationCallbacks* allocator)
	{
		DestroySwapchainImpl(S().destroySwapchain, device, swapchain, allocator);
	}

	VKAPI_ATTR void VKAPI_CALL HookDestroySwapchainViaGdpa(VkDevice device, VkSwapchainKHR swapchain,
		const VkAllocationCallbacks* allocator)
	{
		State& s = S();
		PFN_vkDestroySwapchainKHR next = s.gdpaDestroySwapchain.load(std::memory_order_acquire);
		DestroySwapchainImpl(next ? next : s.destroySwapchain, device, swapchain, allocator);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookQueuePresent(VkQueue queue, const VkPresentInfoKHR* info)
	{
		return QueuePresentImpl(S().queuePresent, queue, info);
	}

	VKAPI_ATTR VkResult VKAPI_CALL HookQueuePresentViaGdpa(VkQueue queue, const VkPresentInfoKHR* info)
	{
		State& s = S();
		PFN_vkQueuePresentKHR next = s.gdpaQueuePresent.load(std::memory_order_acquire);
		return QueuePresentImpl(next ? next : s.queuePresent, queue, info);
	}

	PFN_vkVoidFunction WrapDeviceProc(const char* pName, PFN_vkVoidFunction fn)
	{
		if (!fn || !pName)
			return fn;

		State& s = S();
		void* addr = reinterpret_cast<void*>(fn);

		if (std::strcmp(pName, "vkQueuePresentKHR") == 0)
		{
			if (addr != s.rawQueuePresent &&
				addr != reinterpret_cast<void*>(&HookQueuePresent) &&
				addr != reinterpret_cast<void*>(&HookQueuePresentViaGdpa))
			{
				s.gdpaQueuePresent.store(reinterpret_cast<PFN_vkQueuePresentKHR>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookQueuePresentViaGdpa);
			}
		}
		else if (std::strcmp(pName, "vkCreateSwapchainKHR") == 0)
		{
			if (addr != s.rawCreateSwapchain &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchain) &&
				addr != reinterpret_cast<void*>(&HookCreateSwapchainViaGdpa))
			{
				s.gdpaCreateSwapchain.store(reinterpret_cast<PFN_vkCreateSwapchainKHR>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookCreateSwapchainViaGdpa);
			}
		}
		else if (std::strcmp(pName, "vkDestroySwapchainKHR") == 0)
		{
			if (addr != s.rawDestroySwapchain &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchain) &&
				addr != reinterpret_cast<void*>(&HookDestroySwapchainViaGdpa))
			{
				s.gdpaDestroySwapchain.store(reinterpret_cast<PFN_vkDestroySwapchainKHR>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookDestroySwapchainViaGdpa);
			}
		}

		return fn;
	}

	VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookGetDeviceProcAddr(VkDevice device, const char* pName)
	{
		State& s = S();
		PFN_vkVoidFunction fn = s.loaderGdpa ? s.loaderGdpa(device, pName) : nullptr;
		return WrapDeviceProc(pName, fn);
	}

	VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookGetDeviceProcAddrViaGipa(VkDevice device, const char* pName)
	{
		State& s = S();
		PFN_vkGetDeviceProcAddr next = s.gipaGdpa.load(std::memory_order_acquire);
		PFN_vkVoidFunction fn = next ? next(device, pName) : (s.loaderGdpa ? s.loaderGdpa(device, pName) : nullptr);
		return WrapDeviceProc(pName, fn);
	}

	// OptiScaler hooks vulkan-1.dll before ASI plugins run and returns its own hkvkCreateDevice from
	// vkGetInstanceProcAddr. Wrapping vkGetInstanceProcAddr catches games that resolve vkCreateDevice via GIPA
	// and call OptiScaler's hook directly (whose saved trampoline would otherwise bypass our loader prologue hook).
	VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HookGetInstanceProcAddr(VkInstance instance, const char* pName)
	{
		State& s = S();
		PFN_vkVoidFunction fn = s.loaderGipa ? s.loaderGipa(instance, pName) : nullptr;

		if (!fn || !pName)
			return fn;

		if (std::strcmp(pName, "vkCreateDevice") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawCreateDevice &&
				addr != reinterpret_cast<void*>(&HookCreateDevice) &&
				addr != reinterpret_cast<void*>(&HookCreateDeviceViaGipa))
			{
				s.gipaCreateDevice.store(reinterpret_cast<PFN_vkCreateDevice>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookCreateDeviceViaGipa);
			}
		}
		else if (std::strcmp(pName, "vkCreateInstance") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawCreateInstance &&
				addr != reinterpret_cast<void*>(&HookCreateInstance) &&
				addr != reinterpret_cast<void*>(&HookCreateInstanceViaGipa))
			{
				s.gipaCreateInstance.store(reinterpret_cast<PFN_vkCreateInstance>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookCreateInstanceViaGipa);
			}
		}
		else if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0)
		{
			void* addr = reinterpret_cast<void*>(fn);
			if (addr != s.rawGdpa &&
				addr != reinterpret_cast<void*>(&HookGetDeviceProcAddr) &&
				addr != reinterpret_cast<void*>(&HookGetDeviceProcAddrViaGipa))
			{
				s.gipaGdpa.store(reinterpret_cast<PFN_vkGetDeviceProcAddr>(fn), std::memory_order_release);
				return reinterpret_cast<PFN_vkVoidFunction>(&HookGetDeviceProcAddrViaGipa);
			}
		}
		else
		{
			return WrapDeviceProc(pName, fn);
		}

		return fn;
	}

	// ------------------------------------------------------------------------------------------------ install

	// With the install lock held
	bool InstallLoader(HMODULE loader, std::string& why)
	{
		State& s = S();

		auto create = GetProcAddress(loader, "vkCreateInstance");
		auto createDevice = GetProcAddress(loader, "vkCreateDevice");
		auto gipa = GetProcAddress(loader, "vkGetInstanceProcAddr");

		if (!create || !createDevice || !gipa)
		{
			why = "vulkan-1.dll has no vkCreateInstance, vkCreateDevice or vkGetInstanceProcAddr";
			return false;
		}

		if (const auto status = MH_Initialize(); status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED)
		{
			why = std::string("MinHook can't start: ") + MH_StatusToString(status);
			return false;
		}

		s.rawCreateInstance = reinterpret_cast<void*>(create);
		s.rawCreateDevice = reinterpret_cast<void*>(createDevice);
		s.loaderGipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(gipa);

		auto fail = [&why](const char* what, MH_STATUS status)
			{
				why = std::string(what) + ": " + MH_StatusToString(status);
				return false;
			};

		if (const auto status = MH_CreateHook(reinterpret_cast<void*>(create), reinterpret_cast<void*>(&HookCreateInstance),
				reinterpret_cast<void**>(&s.createInstance)); status != MH_OK)
			return fail("Can't hook vkCreateInstance", status);

		if (const auto status = MH_CreateHook(reinterpret_cast<void*>(createDevice), reinterpret_cast<void*>(&HookCreateDevice),
				reinterpret_cast<void**>(&s.createDevice)); status != MH_OK)
		{
			MH_RemoveHook(reinterpret_cast<void*>(create));
			return fail("Can't hook vkCreateDevice", status);
		}

		if (const auto status = MH_EnableHook(reinterpret_cast<void*>(create)); status != MH_OK)
		{
			MH_RemoveHook(reinterpret_cast<void*>(create));
			MH_RemoveHook(reinterpret_cast<void*>(createDevice));
			return fail("Can't enable the vkCreateInstance hook", status);
		}

		if (const auto status = MH_EnableHook(reinterpret_cast<void*>(createDevice)); status != MH_OK)
		{
			MH_DisableHook(reinterpret_cast<void*>(create));
			MH_RemoveHook(reinterpret_cast<void*>(create));
			MH_RemoveHook(reinterpret_cast<void*>(createDevice));
			return fail("Can't enable the vkCreateDevice hook", status);
		}

		PFN_vkGetInstanceProcAddr gipaTrampoline = nullptr;
		if (MH_CreateHook(reinterpret_cast<void*>(gipa), reinterpret_cast<void*>(&HookGetInstanceProcAddr),
				reinterpret_cast<void**>(&gipaTrampoline)) == MH_OK)
		{
			s.loaderGipa = gipaTrampoline;
			if (MH_EnableHook(reinterpret_cast<void*>(gipa)) == MH_OK)
			{
				s.gipaHooked = true;
			}
			else
			{
				s.loaderGipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(gipa);
				MH_RemoveHook(reinterpret_cast<void*>(gipa));
				Log::Warn("Vulkan hooks: couldn't enable vkGetInstanceProcAddr hook, using loader detours only");
			}
		}
		else
		{
			Log::Warn("Vulkan hooks: couldn't hook vkGetInstanceProcAddr, using loader detours only");
		}

		// Optional present/swapchain hooks for recorded-mode frame pacing (non-fatal if unavailable)
		auto tryHook = [](HMODULE mod, const char* name, void* detour, void** rawOut, auto& trampolineOut) -> bool {
			FARPROC proc = GetProcAddress(mod, name);
			if (!proc)
				return false;
			*rawOut = reinterpret_cast<void*>(proc);
			std::remove_reference_t<decltype(trampolineOut)> tramp = nullptr;
			if (MH_CreateHook(reinterpret_cast<void*>(proc), detour, reinterpret_cast<void**>(&tramp)) != MH_OK)
				return false;
			trampolineOut = tramp;
			if (MH_EnableHook(reinterpret_cast<void*>(proc)) != MH_OK)
			{
				trampolineOut = nullptr;
				MH_RemoveHook(reinterpret_cast<void*>(proc));
				return false;
			}
			return true;
		};

		tryHook(loader, "vkGetDeviceProcAddr", reinterpret_cast<void*>(&HookGetDeviceProcAddr), &s.rawGdpa, s.loaderGdpa);
		tryHook(loader, "vkCreateSwapchainKHR", reinterpret_cast<void*>(&HookCreateSwapchain), &s.rawCreateSwapchain, s.createSwapchain);
		tryHook(loader, "vkDestroySwapchainKHR", reinterpret_cast<void*>(&HookDestroySwapchain), &s.rawDestroySwapchain, s.destroySwapchain);
		s.presentHooked = tryHook(loader, "vkQueuePresentKHR", reinterpret_cast<void*>(&HookQueuePresent), &s.rawQueuePresent, s.queuePresent);

		s.loaderHooked = true;
		return true;
	}

	void InstallFromLoad(HMODULE loader) noexcept
	{
		try
		{
			State& s = S();
			std::lock_guard lock(s.install);

			if (s.loaderHooked)
				return;

			std::string why;

			if (InstallLoader(loader, why))
			{
				SetHookState(VulkanRuntime::Hooks::Installed, {});
				Log::Info("Vulkan hooks installed on vkCreateInstance and vkCreateDevice, vulkan-1.dll was just loaded");
			}
			else
			{
				SetHookState(VulkanRuntime::Hooks::Failed, why);
				Log::Error("Vulkan hooks: {}", why);
			}
		}
		catch (...)
		{
		}
	}

	bool IsLoaderName(const wchar_t* name)
	{
		const wchar_t* leaf = name;

		for (const wchar_t* p = name; *p; ++p)
		{
			if (*p == L'\\' || *p == L'/')
				leaf = p + 1;
		}

		return _wcsicmp(leaf, L"vulkan-1.dll") == 0 || _wcsicmp(leaf, L"vulkan-1") == 0;
	}

	HMODULE WINAPI HookLoadLibraryExW(LPCWSTR name, HANDLE file, DWORD flags)
	{
		State& s = S();
		const HMODULE module = s.loadLibraryExW(name, file, flags);

		constexpr DWORD NotCode = LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE;
		static thread_local bool inside = false;

		if (module && name && !(flags & NotCode) && !inside && !s.loaderHooked && IsLoaderName(name))
		{
			inside = true;
			InstallFromLoad(module);
			inside = false;
		}

		return module;
	}

	// With the install lock held: vulkan-1.dll isn't loaded, catch its loading. This hook is never taken out again, not
	// even once vulkan-1.dll is hooked: another hook (OptiScaler's own LoadLibrary hooks, an overlay's) may have been put
	// over this one since, and restoring the function's first bytes would cut it out. It only passes calls on from then on
	bool HookLoading(std::string& why)
	{
		State& s = S();

		if (const auto status = MH_Initialize(); status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED)
		{
			why = std::string("MinHook can't start: ") + MH_StatusToString(status);
			return false;
		}

		LPVOID target = nullptr;

		if (const auto status = MH_CreateHookApiEx(L"kernelbase", "LoadLibraryExW",
				reinterpret_cast<LPVOID>(&HookLoadLibraryExW), reinterpret_cast<LPVOID*>(&s.loadLibraryExW), &target);
			status != MH_OK)
		{
			why = std::string("Can't hook LoadLibraryExW: ") + MH_StatusToString(status);
			return false;
		}

		if (const auto status = MH_EnableHook(target); status != MH_OK)
		{
			MH_RemoveHook(target);
			s.loadLibraryExW = nullptr;
			why = std::string("Can't enable the LoadLibraryExW hook: ") + MH_StatusToString(status);
			return false;
		}

		return true;
	}
} // namespace

void VulkanHost::Start(const char* caller)
{
	State& s = S();
	std::lock_guard lock(s.install);

	if (s.started)
		return;

	s.started = true;

	if (const HMODULE loader = GetModuleHandleW(LoaderName))
	{
		std::string why;

		if (InstallLoader(loader, why))
		{
			SetHookState(VulkanRuntime::Hooks::Installed, {});
			Log::Info("Vulkan hooks installed on vkCreateInstance and vkCreateDevice ({}). A game that already has its "
				"Vulkan device isn't covered, see the VulkanHooks notes in the readme", caller);
		}
		else
		{
			SetHookState(VulkanRuntime::Hooks::Failed, why);
			Log::Error("Vulkan hooks: {}", why);
		}

		return;
	}

	std::string why;

	if (!HookLoading(why))
	{
		SetHookState(VulkanRuntime::Hooks::Failed, why);
		Log::Error("Vulkan hooks: {}", why);
		return;
	}

	SetHookState(VulkanRuntime::Hooks::Waiting, {});
	Log::Info("vulkan-1.dll isn't loaded yet ({}), the Vulkan hooks go in when it is", caller);

	// It may have been loaded between the check and the hook
	if (const HMODULE loader = GetModuleHandleW(LoaderName); loader && !s.loaderHooked)
	{
		std::string late;

		if (InstallLoader(loader, late))
		{
			SetHookState(VulkanRuntime::Hooks::Installed, {});
			Log::Info("Vulkan hooks installed on vkCreateInstance and vkCreateDevice");
		}
		else
		{
			SetHookState(VulkanRuntime::Hooks::Failed, late);
			Log::Error("Vulkan hooks: {}", late);
		}
	}
}

void VulkanHost::SetEnabled(bool enabled)
{
	State& s = S();

	if (s.enabled.exchange(enabled) != enabled)
		Log::Info("Vulkan hooks {}", enabled ? "on" : "off, they only pass the game's calls on");
}

bool VulkanHost::Find(VkDevice device, DeviceRecord& out)
{
	State& s = S();
	std::lock_guard lock(s.mutex);

	const auto found = s.devices.find(device);

	if (found == s.devices.end())
		return false;

	out = found->second;
	return true;
}

std::string VulkanHost::RefusalFor(VkPhysicalDevice physical)
{
	State& s = S();
	std::lock_guard lock(s.mutex);

	const auto found = s.refused.find(physical);
	return found == s.refused.end() ? std::string() : found->second;
}

void VulkanHost::NotifyBatchStart(uint32_t generatedCount) noexcept
{
	State& s = S();
	if (!PacingActive(s))
		return;
	std::lock_guard lock(s.pacerMutex);
	s.pacer.OnBatchStart(NowNs(), generatedCount);
}

void VulkanHost::NotifyPaused() noexcept
{
	State& s = S();
	std::lock_guard lock(s.pacerMutex);
	s.pacer.OnPaused();
}

VulkanRuntime::Status VulkanHost::Snapshot()
{
	State& s = S();
	VulkanRuntime::Status status;

	{
		std::lock_guard lock(s.mutex);
		status.hooks = s.hooks;
		status.detail = s.detail;
		status.refusal = s.lastRefusal;
	}

	status.enabled = s.enabled;
	status.instances = s.instanceCalls;
	status.devices = s.deviceCalls;
	status.prepared = s.prepared;

	return status;
}
