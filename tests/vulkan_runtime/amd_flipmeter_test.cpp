// Unit tests for AMD RDNA 4+ multi-GPU PCI ID & DriverDesc matching (src/AmdFlipMeterReg.h).
#include "AmdFlipMeterReg.h"

#include <cstdio>

namespace
{
	int gChecks = 0;
	int gFailures = 0;

	void Check(bool cond, const char* expr, int line)
	{
		++gChecks;
		if (!cond)
		{
			++gFailures;
			std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
		}
	}
} // namespace

#define CHECK(expr) Check((expr), #expr, __LINE__)

int main()
{
	using namespace AmdFlipMeterReg;

	uint32_t ven = 0, dev = 0;

	// 1. AMD Radeon RX 9070 XT (Navi 48, 0x7550) => RDNA 4 match
	CHECK(ParsePciVendorDevice(L"pci\\ven_1002&dev_7550&subsys_12345678&rev_c0", ven, dev));
	CHECK(ven == 0x1002u && dev == 0x7550u);
	CHECK(IsRdna4OrNewerAdapter(ven, dev, L"AMD Radeon RX 9070 XT"));

	// 2. AMD Radeon RX 9060 XT (Navi 44, 0x7590) => RDNA 4 match
	CHECK(ParsePciVendorDevice(L"PCI\\VEN_1002&DEV_7590&REV_C0", ven, dev));
	CHECK(ven == 0x1002u && dev == 0x7590u);
	CHECK(IsRdna4OrNewerAdapter(ven, dev, L"AMD Radeon RX 9060 XT"));

	// 3. AMD Ryzen iGPU (e.g. 780M / 890M / Radeon(TM) Graphics) => MUST NOT match
	CHECK(ParsePciVendorDevice(L"PCI\\VEN_1002&DEV_164E", ven, dev));
	CHECK(!IsRdna4OrNewerAdapter(ven, dev, L"AMD Radeon(TM) Graphics"));
	CHECK(ParsePciVendorDevice(L"PCI\\VEN_1002&DEV_15BF", ven, dev));
	CHECK(!IsRdna4OrNewerAdapter(ven, dev, L"AMD Radeon 780M Graphics"));
	CHECK(ParsePciVendorDevice(L"PCI\\VEN_1002&DEV_150E", ven, dev));
	CHECK(!IsRdna4OrNewerAdapter(ven, dev, L"AMD Radeon 890M Graphics"));

	// 4. Older AMD RDNA 3 discrete GPU (RX 7900 XTX, 0x744C) => MUST NOT match
	CHECK(ParsePciVendorDevice(L"PCI\\VEN_1002&DEV_744C", ven, dev));
	CHECK(!IsRdna4OrNewerAdapter(ven, dev, L"AMD Radeon RX 7900 XTX"));

	// 5. Intel Arc / UHD iGPU & NVIDIA RTX => MUST NOT match even with tricky names
	CHECK(ParsePciVendorDevice(L"PCI\\VEN_8086&DEV_7550", ven, dev));
	CHECK(!IsRdna4OrNewerAdapter(ven, dev, L"Intel(R) UHD Graphics"));
	CHECK(ParsePciVendorDevice(L"PCI\\VEN_10DE&DEV_2684", ven, dev));
	CHECK(!IsRdna4OrNewerAdapter(ven, dev, L"NVIDIA GeForce RTX 4090"));

	// 6. Future AMD RDNA 5 / Radeon AI PRO R9xxx SKU => match via vendor 0x1002 + description
	CHECK(ParsePciVendorDevice(L"PCI\\VEN_1002&DEV_7610", ven, dev));
	CHECK(IsRdna4OrNewerAdapter(ven, dev, L"AMD Radeon AI PRO R9700"));

	// 7. Invalid / malformed MatchingDeviceId => fails cleanly
	CHECK(!ParsePciVendorDevice(L"ROOT\\BASICDISPLAY", ven, dev));

	std::printf("amd_flipmeter_test: %d checks, %d failed\n", gChecks, gFailures);
	return gFailures == 0 ? 0 : 1;
}
