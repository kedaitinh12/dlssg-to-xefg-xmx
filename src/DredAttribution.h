#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

// Pure DRED breadcrumb attribution helper (cross-platform so unit tests run in Linux CI).
// Classifies whether an unfinished D3D12 command list hung before, inside, or after a
// dlssg-to-xefg BeginEvent/EndEvent region.
namespace DredAttribution
{
	inline constexpr std::wstring_view kStagePrefixW = L"dlssg-to-xefg:";
	inline constexpr std::string_view kStagePrefixA = "dlssg-to-xefg:";

	enum class Location : uint8_t
	{
		NoXefg = 0,
		BeforeXefg,
		InsideXefg,
		AfterXefg,
	};

	struct ContextEntry
	{
		uint32_t breadcrumbIndex = 0;
		const wchar_t* contextStringW = nullptr;
		const char* contextStringA = nullptr;
	};

	struct Result
	{
		Location location = Location::NoXefg;
		bool hasXefgSpan = false;
		uint32_t xefgBegin = 0;
		uint32_t xefgEnd = 0;
		const wchar_t* activeStageW = nullptr;
		const char* activeStageA = nullptr;
		uint32_t activeStageIndex = 0;
	};

	inline const char* LocationName(Location loc) noexcept
	{
		switch (loc)
		{
		case Location::NoXefg:     return "no_xefg";
		case Location::BeforeXefg: return "before_xefg";
		case Location::InsideXefg: return "inside_xefg";
		case Location::AfterXefg:  return "after_xefg";
		}
		return "no_xefg";
	}

	inline bool IsXefgMarker(const ContextEntry& entry) noexcept
	{
		if (entry.contextStringW && entry.contextStringW[0] != L'\0')
			return std::wstring_view(entry.contextStringW).starts_with(kStagePrefixW);
		if (entry.contextStringA && entry.contextStringA[0] != '\0')
			return std::string_view(entry.contextStringA).starts_with(kStagePrefixA);
		return false;
	}

	inline bool HasContextString(const ContextEntry& entry) noexcept
	{
		return (entry.contextStringW && entry.contextStringW[0] != L'\0') ||
			(entry.contextStringA && entry.contextStringA[0] != L'\0');
	}

	// Classifies the hang position `lastCompleted` against recorded DRED breadcrumb contexts.
	// When D3D12 records BeginEvent(L"dlssg-to-xefg: ...") and later EndEvent(), DRED attaches
	// the stage string to the BeginEvent breadcrumb index. If a subsequent non-XeFG context exists
	// after the last XeFG marker, `xefgEnd` is bounded by the op before that next context; otherwise
	// `xefgEnd` extends to `breadcrumbCount - 1` (or `xefgBegin` if `breadcrumbCount == 0`).
	inline Result Classify(uint32_t lastCompleted, uint32_t breadcrumbCount,
		const ContextEntry* contexts, size_t count) noexcept
	{
		Result out{};
		if (!contexts || count == 0)
			return out;

		uint32_t minXefg = UINT32_MAX;
		uint32_t maxXefg = 0;
		bool foundXefg = false;

		for (size_t i = 0; i < count; ++i)
		{
			const auto& ctx = contexts[i];
			if (!IsXefgMarker(ctx))
				continue;

			foundXefg = true;
			if (ctx.breadcrumbIndex < minXefg)
				minXefg = ctx.breadcrumbIndex;
			if (ctx.breadcrumbIndex > maxXefg)
				maxXefg = ctx.breadcrumbIndex;

			if (ctx.breadcrumbIndex <= lastCompleted &&
				((!out.activeStageW && !out.activeStageA) || ctx.breadcrumbIndex >= out.activeStageIndex))
			{
				out.activeStageW = ctx.contextStringW;
				out.activeStageA = ctx.contextStringA;
				out.activeStageIndex = ctx.breadcrumbIndex;
			}
		}

		if (!foundXefg)
			return out;

		// Bound the end of the XeFG span by the earliest non-XeFG context after maxXefg, if any.
		uint32_t nextNonXefg = UINT32_MAX;
		for (size_t i = 0; i < count; ++i)
		{
			const auto& ctx = contexts[i];
			if (!HasContextString(ctx) || IsXefgMarker(ctx))
				continue;
			if (ctx.breadcrumbIndex > maxXefg && ctx.breadcrumbIndex < nextNonXefg)
				nextNonXefg = ctx.breadcrumbIndex;
		}

		out.hasXefgSpan = true;
		out.xefgBegin = minXefg;
		if (nextNonXefg != UINT32_MAX)
			out.xefgEnd = nextNonXefg - 1u;
		else if (breadcrumbCount > 0 && breadcrumbCount - 1u >= maxXefg)
			out.xefgEnd = breadcrumbCount - 1u;
		else
			out.xefgEnd = maxXefg;

		if (lastCompleted < out.xefgBegin)
			out.location = Location::BeforeXefg;
		else if (lastCompleted <= out.xefgEnd)
			out.location = Location::InsideXefg;
		else
			out.location = Location::AfterXefg;

		return out;
	}
} // namespace DredAttribution
