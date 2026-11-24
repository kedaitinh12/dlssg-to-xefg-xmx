#pragma once

#include <cstdint>
#include <format>
#include <string_view>

struct ID3D12Device;

// Appends to dlssg_to_xefg_intel_is_better.log next to the dll and to OutputDebugString
namespace Log
{
	void Write(std::string_view level, std::string_view message);
	void SetEnabled(bool enabled) noexcept;
	bool IsEnabled() noexcept;
	const char* DeviceRemovedName(uint32_t hresult) noexcept;
	// Opt-in DX12 DRED (Device Removed Extended Data): enables AutoBreadcrumbs + PageFault before D3D12CreateDevice.
	bool EnableDred() noexcept;
	// One-shot DRED dump on DX12 device loss (TDR): logs unfinished command lists and GPU page-fault allocations.
	void DumpDred(ID3D12Device* device) noexcept;
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
