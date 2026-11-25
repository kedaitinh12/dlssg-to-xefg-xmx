#include "PeVersion.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
	void WriteLe32(std::vector<uint8_t>& buf, size_t offset, uint32_t value)
	{
		buf[offset + 0] = static_cast<uint8_t>(value & 0xFFu);
		buf[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
		buf[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
		buf[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
	}

	std::vector<uint8_t> MakeVersionResource(size_t ffiOffset, uint32_t strucVer,
		uint16_t maj, uint16_t min, uint16_t bld, uint16_t rev)
	{
		std::vector<uint8_t> buf(ffiOffset + 52u, 0u);
		WriteLe32(buf, ffiOffset + 0u, 0xFEEF04BDu);
		WriteLe32(buf, ffiOffset + 4u, strucVer);
		WriteLe32(buf, ffiOffset + 8u, (static_cast<uint32_t>(maj) << 16) | min);
		WriteLe32(buf, ffiOffset + 12u, (static_cast<uint32_t>(bld) << 16) | rev);
		return buf;
	}
} // namespace

int main()
{
	// 1. Standard VS_VERSION_INFO layout (offset 40, e.g. sl.dlss_g.dll 2.14.16.0)
	{
		auto buf = MakeVersionResource(40u, 0x00010000u, 2, 14, 16, 0);
		PeVersion::FileVersion v{};
		assert(PeVersion::ParseVersionResource(buf.data(), buf.size(), v));
		assert(v.major == 2 && v.minor == 14 && v.build == 16 && v.revision == 0);
		assert(PeVersion::Format(v) == "2.14.16.0");
	}

	// 2. libxess_fg.dll style version at offset 0 (e.g. 3.0.2.95)
	{
		auto buf = MakeVersionResource(0u, 0x00010000u, 3, 0, 2, 95);
		PeVersion::FileVersion v{};
		assert(PeVersion::ParseVersionResource(buf.data(), buf.size(), v));
		assert(PeVersion::Format(v) == "3.0.2.95");
	}

	// 3. Unaligned false signature must be ignored
	{
		auto buf = MakeVersionResource(41u, 0x00010000u, 9, 9, 9, 9);
		PeVersion::FileVersion v{};
		assert(!PeVersion::ParseVersionResource(buf.data(), buf.size(), v));
	}

	// 4. Truncated VS_FIXEDFILEINFO (< 52 bytes remaining) must fail safely
	{
		auto buf = MakeVersionResource(40u, 0x00010000u, 1, 2, 3, 4);
		buf.pop_back();
		PeVersion::FileVersion v{};
		assert(!PeVersion::ParseVersionResource(buf.data(), buf.size(), v));
	}

	// 5. Corrupt dwStrucVersion == 0 must be skipped
	{
		auto buf = MakeVersionResource(40u, 0x00000000u, 1, 2, 3, 4);
		PeVersion::FileVersion v{};
		assert(!PeVersion::ParseVersionResource(buf.data(), buf.size(), v));
	}

	// 6. Null / empty / missing signature
	{
		PeVersion::FileVersion v{};
		assert(!PeVersion::ParseVersionResource(nullptr, 100u, v));
		std::vector<uint8_t> empty(64u, 0u);
		assert(!PeVersion::ParseVersionResource(empty.data(), empty.size(), v));
	}

	std::cout << "pe_version_test: 6/6 passed\n";
	return 0;
}
