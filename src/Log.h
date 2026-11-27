#pragma once

#include <cstdint>
#include <format>
#include <string_view>

struct ID3D12Device;
struct ID3D12GraphicsCommandList;

// Appends to dlssg_to_xefg_intel_is_better.log next to the dll and to OutputDebugString
namespace Log
{
	void Write(std::string_view level, std::string_view message);
	void SetEnabled(bool enabled) noexcept;
	bool IsEnabled() noexcept;
	// Opt-in lightweight crash minidump (.dmp via System32 dbghelp.dll on unhandled exception).
	void SetCrashMiniDumpEnabled(bool enabled) noexcept;
	bool CrashMiniDumpRequestedByEnvironment() noexcept;
	const char* DeviceRemovedName(uint32_t hresult) noexcept;
	// Opt-in DX12 DRED (Device Removed Extended Data): enables AutoBreadcrumbs + PageFault before D3D12CreateDevice.
	bool EnableDred() noexcept;
	bool IsDredEnabled() noexcept;
	// Zero-overhead RAII DRED stage marker: emits BeginEvent/EndEvent ONLY when DRED is enabled.
	struct ScopedDredEvent
	{
		ScopedDredEvent(ID3D12GraphicsCommandList* cmdList, const wchar_t* label) noexcept;
		~ScopedDredEvent() noexcept;
		ScopedDredEvent(const ScopedDredEvent&) = delete;
		ScopedDredEvent& operator=(const ScopedDredEvent&) = delete;

	private:
		ID3D12GraphicsCommandList* cmdList_ = nullptr;
	};
	// One-shot DRED dump on DX12 device loss (TDR): logs unfinished command lists and GPU page-fault allocations.
	void DumpDred(ID3D12Device* device) noexcept;
	// RAII D3D12 device-removed watchdog: arms an ID3D12Fence(UINT64_MAX) + threadpool wait so GPU TDRs are logged
	// and DRED-dumped immediately even if the game exits or stalls before calling XeFG::Execute again.
	struct DredWatchdog
	{
		DredWatchdog() noexcept = default;
		explicit DredWatchdog(ID3D12Device* device) noexcept;
		~DredWatchdog() noexcept;
		DredWatchdog(const DredWatchdog&) = delete;
		DredWatchdog& operator=(const DredWatchdog&) = delete;
		DredWatchdog(DredWatchdog&& other) noexcept;
		DredWatchdog& operator=(DredWatchdog&& other) noexcept;

		void Arm(ID3D12Device* device) noexcept;
		void Disarm() noexcept;
		bool IsArmed() const noexcept { return state_ != nullptr; }

	private:
		void* state_ = nullptr;
	};
	// One-shot startup scan after first CreateFeature: logs competing FG mods/ReShade FG add-ons without blocking.
	void WarnForeignFgMods(bool isVulkan, bool usingExternalVulkanProvider) noexcept;

	template <class... Args> void Debug(std::format_string<Args...> format, Args&&... args)
	{
		if (IsEnabled())
			Write("debug", std::format(format, std::forward<Args>(args)...));
	}

	template <class... Args> void Info(std::format_string<Args...> format, Args&&... args)
	{
		if (IsEnabled())
			Write("info", std::format(format, std::forward<Args>(args)...));
	}

	template <class... Args> void Warn(std::format_string<Args...> format, Args&&... args)
	{
		if (IsEnabled())
			Write("warn", std::format(format, std::forward<Args>(args)...));
	}

	template <class... Args> void Error(std::format_string<Args...> format, Args&&... args)
	{
		if (IsEnabled())
			Write("error", std::format(format, std::forward<Args>(args)...));
	}
} // namespace Log
