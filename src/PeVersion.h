#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace PeVersion
{
	struct FileVersion
	{
		uint16_t major = 0;
		uint16_t minor = 0;
		uint16_t build = 0;
		uint16_t revision = 0;

		bool operator==(const FileVersion&) const noexcept = default;
	};

	inline uint32_t ReadLe32(const uint8_t* p) noexcept
	{
		return static_cast<uint32_t>(p[0]) |
			(static_cast<uint32_t>(p[1]) << 8) |
			(static_cast<uint32_t>(p[2]) << 16) |
			(static_cast<uint32_t>(p[3]) << 24);
	}

	// Parses a raw PE RT_VERSION (VS_VERSION_INFO) resource buffer to extract VS_FIXEDFILEINFO file version.
	// Uses only in-memory resource bytes (zero disk I/O, zero version.lib dependency).
	inline bool ParseVersionResource(const uint8_t* data, size_t size, FileVersion& out) noexcept
	{
		constexpr uint32_t kVsFfiSignature = 0xFEEF04BDu;
		constexpr size_t kVsFfiSize = 52u; // sizeof(VS_FIXEDFILEINFO) = 13 * DWORD

		if (!data || size < kVsFfiSize)
			return false;

		// VS_FIXEDFILEINFO is 4-byte aligned right after the VS_VERSION_INFO UTF-16 header (typically offset 40).
		const size_t scanLimit = size - kVsFfiSize;
		for (size_t offset = 0; offset <= scanLimit; offset += 4u)
		{
			if (ReadLe32(data + offset) != kVsFfiSignature)
				continue;

			const uint32_t strucVersion = ReadLe32(data + offset + 4u);
			if ((strucVersion >> 16) == 0u)
				continue;

			const uint32_t fileVerMs = ReadLe32(data + offset + 8u);
			const uint32_t fileVerLs = ReadLe32(data + offset + 12u);

			out.major = static_cast<uint16_t>((fileVerMs >> 16) & 0xFFFFu);
			out.minor = static_cast<uint16_t>(fileVerMs & 0xFFFFu);
			out.build = static_cast<uint16_t>((fileVerLs >> 16) & 0xFFFFu);
			out.revision = static_cast<uint16_t>(fileVerLs & 0xFFFFu);
			return true;
		}

		return false;
	}

	inline std::string Format(const FileVersion& v)
	{
		return std::to_string(v.major) + "." + std::to_string(v.minor) + "." +
			std::to_string(v.build) + "." + std::to_string(v.revision);
	}

#ifdef _WIN32
	// Queries VS_FIXEDFILEINFO FileVersion directly from an already-loaded HMODULE's RT_VERSION resource.
	inline bool QueryModuleFileVersion(HMODULE module, FileVersion& out) noexcept
	{
		if (!module)
			return false;

		// Resource ID 1 = VS_VERSION_INFO, type 16 = RT_VERSION
		HRSRC res = FindResourceW(module, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(16));
		if (!res)
			return false;

		const DWORD resSize = SizeofResource(module, res);
		if (resSize < 52u)
			return false;

		HGLOBAL loaded = LoadResource(module, res);
		if (!loaded)
			return false;

		const void* bytes = LockResource(loaded);
		if (!bytes)
			return false;

		return ParseVersionResource(static_cast<const uint8_t*>(bytes), static_cast<size_t>(resSize), out);
	}

	inline std::string QueryModuleFileVersionString(HMODULE module)
	{
		FileVersion v{};
		if (!QueryModuleFileVersion(module, v))
			return "unknown";
		return Format(v);
	}
#endif
} // namespace PeVersion
