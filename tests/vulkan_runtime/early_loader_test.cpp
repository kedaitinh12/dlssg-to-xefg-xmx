#include "loader/EarlyLoaderCore.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace
{
	void WriteU16LE(std::vector<uint8_t>& buf, size_t off, uint16_t val)
	{
		buf[off] = static_cast<uint8_t>(val & 0xFF);
		buf[off + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
	}

	void WriteU32LE(std::vector<uint8_t>& buf, size_t off, uint32_t val)
	{
		buf[off] = static_cast<uint8_t>(val & 0xFF);
		buf[off + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
		buf[off + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
		buf[off + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
	}

	// Builds a minimal, valid PE32+ (x64) image with an export table containing the given symbol names.
	void WriteSyntheticPe64(const fs::path& path, const std::vector<std::string>& exports)
	{
		fs::create_directories(path.parent_path());

		constexpr uint32_t kPeOffset = 0x80;
		constexpr uint16_t kOptionalSize = 0xF0;
		constexpr uint32_t kSectionVirtualAddress = 0x1000;
		constexpr uint32_t kSectionRawPointer = 0x200;

		std::vector<uint8_t> edata;
		if (!exports.empty())
		{
			const uint32_t dirSize = 40;
			const uint32_t namesArrayOff = dirSize;
			const uint32_t stringsOff = namesArrayOff + static_cast<uint32_t>(exports.size()) * 4;

			std::vector<uint32_t> nameRvas;
			std::vector<uint8_t> stringBlob;
			for (const auto& name : exports)
			{
				nameRvas.push_back(kSectionVirtualAddress + stringsOff + static_cast<uint32_t>(stringBlob.size()));
				stringBlob.insert(stringBlob.end(), name.begin(), name.end());
				stringBlob.push_back(0);
			}

			edata.resize(stringsOff + stringBlob.size(), 0);
			WriteU32LE(edata, 24, static_cast<uint32_t>(exports.size())); // NumberOfNames
			WriteU32LE(edata, 32, kSectionVirtualAddress + namesArrayOff); // AddressOfNames
			for (size_t i = 0; i < nameRvas.size(); ++i)
				WriteU32LE(edata, namesArrayOff + i * 4, nameRvas[i]);
			std::memcpy(edata.data() + stringsOff, stringBlob.data(), stringBlob.size());
		}

		const size_t totalSize = kSectionRawPointer + std::max<size_t>(edata.size(), 0x100);
		std::vector<uint8_t> image(totalSize, 0);

		// DOS header
		image[0] = 'M';
		image[1] = 'Z';
		WriteU32LE(image, 0x3C, kPeOffset);

		// PE signature + COFF header
		image[kPeOffset] = 'P';
		image[kPeOffset + 1] = 'E';
		WriteU16LE(image, kPeOffset + 6, 1);             // NumberOfSections = 1
		WriteU16LE(image, kPeOffset + 20, kOptionalSize); // SizeOfOptionalHeader

		// PE32+ Optional Header
		const size_t opt = kPeOffset + 24;
		WriteU16LE(image, opt, 0x20B); // PE32+ magic
		if (!exports.empty())
		{
			WriteU32LE(image, opt + 112, kSectionVirtualAddress);
			WriteU32LE(image, opt + 116, static_cast<uint32_t>(edata.size()));
		}

		// Section header (.edata)
		const size_t sec = opt + kOptionalSize;
		std::memcpy(image.data() + sec, ".edata", 6);
		WriteU32LE(image, sec + 8, static_cast<uint32_t>(edata.size()));
		WriteU32LE(image, sec + 12, kSectionVirtualAddress);
		WriteU32LE(image, sec + 16, static_cast<uint32_t>(edata.size()));
		WriteU32LE(image, sec + 20, kSectionRawPointer);

		if (!edata.empty())
			std::memcpy(image.data() + kSectionRawPointer, edata.data(), edata.size());

		std::ofstream out(path, std::ios::binary | std::ios::trunc);
		out.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size()));
		assert(out.good());
	}

	void WriteText(const fs::path& path, const std::string& content)
	{
		fs::create_directories(path.parent_path());
		std::ofstream out(path, std::ios::binary | std::ios::trunc);
		out << content;
		assert(out.good());
	}

	void TestIniParsing(const fs::path& root)
	{
		const fs::path iniPath = root / "OptiScaler.ini";
		WriteText(iniPath,
			"\xEF\xBB\xBF; header comment\r\n"
			"[FrameGen]\r\n"
			"FGNvngxReplacement = \"MyCustomXeFG\" ; inline comment\r\n"
			"[XeMFG XMX]\r\n"
			"VulkanHooks = false # disabled\r\n");

		const auto cfg = EarlyLoader::ParseOptiScalerIni(iniPath);
		assert(cfg.found);
		assert(cfg.fgNvngxReplacement == "MyCustomXeFG");
		assert(cfg.hasVulkanHooks);
		assert(!cfg.vulkanHooks);
	}

	void TestPeExportInspection(const fs::path& root)
	{
		const fs::path good = root / "pe" / "good.dll";
		WriteSyntheticPe64(good, { "InitializeASI", "OptiPluginInit", "XeFgEarlyBootstrap" });
		assert(EarlyLoader::PeFileHasExport(good, EarlyLoader::kBootstrapExportName));
		assert(EarlyLoader::PeFileHasExport(good, "InitializeASI"));
		assert(!EarlyLoader::PeFileHasExport(good, "NonExistentExport"));

		const fs::path foreign = root / "pe" / "foreign.dll";
		WriteSyntheticPe64(foreign, { "NVSDK_NGX_VULKAN_Init", "DllMain" });
		assert(!EarlyLoader::PeFileHasExport(foreign, EarlyLoader::kBootstrapExportName));

		const fs::path truncated = root / "pe" / "bad.dll";
		WriteText(truncated, "MZ\x90\x00not_a_real_pe");
		assert(!EarlyLoader::PeFileHasExport(truncated, EarlyLoader::kBootstrapExportName));
	}

	void TestRenamedFolderAndIniPriority(const fs::path& root)
	{
		const fs::path opti = root / "case_renamed";
		const fs::path foreignDll = opti / "custom" / "NukemFSR3" / "dlssg" / "dlssg_to_fsr3_amd_is_better.dll";
		const fs::path oldBackupDll = opti / "custom" / "XeMFG XMX_old" / "dlssg" / "dlssg_to_xefg_intel_is_better.dll";
		const fs::path renamedDll = opti / "custom" / "MyRenamedXeFG" / "dlssg" / "custom_name_xefg.dll";

		WriteSyntheticPe64(foreignDll, { "NVSDK_NGX_D3D12_Init" });
		WriteSyntheticPe64(oldBackupDll, { "InitializeASI", "OptiPluginInit", "XeFgEarlyBootstrap" });
		WriteSyntheticPe64(renamedDll, { "InitializeASI", "OptiPluginInit", "XeFgEarlyBootstrap" });

		// Make renamedDll strictly newer than oldBackupDll
		const auto baseTime = fs::last_write_time(oldBackupDll);
		fs::last_write_time(renamedDll, baseTime + std::chrono::seconds(10));

		// Case 1: OptiScaler.ini explicitly selects XeMFG XMX_old -> must pick oldBackupDll
		EarlyLoader::OptiScalerIniConfig iniSpecific;
		iniSpecific.found = true;
		iniSpecific.fgNvngxReplacement = "XeMFG XMX_old";
		const auto sel1 = EarlyLoader::SelectMainDll(opti, iniSpecific);
		assert(sel1.path == oldBackupDll);
		assert(!sel1.legacyMatch);

		// Case 2: OptiScaler.ini is "auto" (first-time launch) -> must still early-load newest valid XeFG DLL
		// even though both folder and DLL filename were renamed, and must ignore foreignDll.
		EarlyLoader::OptiScalerIniConfig iniAuto;
		iniAuto.found = true;
		iniAuto.fgNvngxReplacement = "auto";
		const auto sel2 = EarlyLoader::SelectMainDll(opti, iniAuto);
		assert(sel2.path == renamedDll);
		assert(sel2.totalMatches == 2);
		assert(!sel2.legacyMatch);

		// Case 3: OptiScaler.ini points to another mod ("NukemFSR3") -> does NOT load foreignDll;
		// falls back to our mod so first-time switch on Vulkan still has early hooks ready.
		EarlyLoader::OptiScalerIniConfig iniOther;
		iniOther.found = true;
		iniOther.fgNvngxReplacement = "NukemFSR3";
		const auto sel3 = EarlyLoader::SelectMainDll(opti, iniOther);
		assert(sel3.path == renamedDll);
	}

	void TestLegacyFallback(const fs::path& root)
	{
		const fs::path opti = root / "case_legacy";
		const fs::path legacyDll = opti / "custom" / "XeMFG XMX" / "dlssg" / "dlssg_to_xefg_intel_is_better.dll";
		WriteSyntheticPe64(legacyDll, { "InitializeASI", "OptiPluginInit" });

		EarlyLoader::OptiScalerIniConfig ini;
		const auto sel = EarlyLoader::SelectMainDll(opti, ini);
		assert(sel.path == legacyDll);
		assert(sel.legacyMatch);
	}
	void TestHelperProcesses()
	{
		using EarlyLoader::IsHelperProcessExe;
		for (const wchar_t* exe : { L"CrashClientReporter.exe", L"crashclientreporter.EXE", L"CrashReportClient.exe",
			     L"UnrealCEFSubProcess.exe", L"EpicWebHelper.exe", L"UnityCrashHandler64.exe", L"BsSndRpt64.exe",
			     L"C:\\Program Files\\Neverness To Everness\\Client\\WindowsNoEditor\\HT\\Binaries\\win64\\CrashClientReporter.exe",
			     L"D:/Games/x/Engine/Binaries/Win64/CRASHREPORTCLIENT.exe" })
			assert(IsHelperProcessExe(exe));

		for (const wchar_t* exe : { L"", L"HTGame.exe", L"b1-Win64-Shipping.exe", L"CrashClientReporter.exe.bak",
			     L"MyCrashClientReporter.exe", L"CrashClientReporter", L"C:\\CrashClientReporter.exe\\HTGame.exe" })
			assert(!IsHelperProcessExe(exe));
	}

	void TestSessionMarker()
	{
		using EarlyLoader::ParseSessionMarker;
		auto m = ParseSessionMarker("pid=2764 exe=HTGame.exe build=Nov 27 2026 22:29:17");
		assert(m.pid == 2764 && m.exe == "HTGame.exe");
		assert(m.line == "pid=2764 exe=HTGame.exe build=Nov 27 2026 22:29:17");

		m = ParseSessionMarker("pid=6192 exe=CrashClientReporter.exe build=x\r\n");
		assert(m.pid == 6192 && m.exe == "CrashClientReporter.exe" && m.line.back() == 'x');

		m = ParseSessionMarker("pid=12 exe=Game.exe\r");
		assert(m.pid == 12 && m.exe == "Game.exe");

		// No marker, garbage, or a pid that is not a whole decimal number: nobody owns it
		for (const char* bad : { "", "garbage", "pid=", "pid=-5 exe=a.exe", "pid=12a exe=a.exe", "pid=99999999999 exe=a.exe",
			     "xpid=12 exe=a.exe", "pid = 12" })
			assert(ParseSessionMarker(bad).pid == 0);

		assert(ParseSessionMarker("exe=a.exe pid=77").pid == 77);
		assert(ParseSessionMarker("pid=4294967295 exe=a.exe").pid == 4294967295u);

		using EarlyLoader::SameExeName;
		assert(SameExeName("HTGame.exe", "C:\\Games\\NTE\\HT\\Binaries\\win64\\htgame.exe"));
		assert(SameExeName("HTGame.exe", "HTGame.exe"));
		assert(!SameExeName("HTGame.exe", "C:\\Windows\\System32\\svchost.exe")); // reused pid
		assert(!SameExeName("", "HTGame.exe"));
		assert(!SameExeName("HTGame.exe", ""));
	}

	void TestSessionDecision()
	{
		using EarlyLoader::DecideSession;
		constexpr uint32_t self = 6192;

		// No marker: a fresh session takes the marker and starts the log afresh (or appends after the early loader)
		auto d = DecideSession(0, false, self, false);
		assert(!d.reportPreviousUnclean && d.takeMarker && d.truncateLog);
		d = DecideSession(0, false, self, true);
		assert(!d.reportPreviousUnclean && d.takeMarker && !d.truncateLog);

		// The game (pid 2764) still runs and owns the marker: a crash reporter neither reports it, takes it nor
		// truncates the log; it appends
		d = DecideSession(2764, true, self, false);
		assert(!d.reportPreviousUnclean && !d.takeMarker && !d.truncateLog);
		d = DecideSession(2764, true, self, true);
		assert(!d.reportPreviousUnclean && !d.takeMarker && !d.truncateLog);

		// The marker's process is gone (or its pid now runs another program): that session ended uncleanly
		d = DecideSession(2764, false, self, false);
		assert(d.reportPreviousUnclean && d.takeMarker && d.truncateLog);
		d = DecideSession(2764, false, self, true);
		assert(d.reportPreviousUnclean && d.takeMarker && !d.truncateLog);

		// Our own pid in the marker (the early loader's copy, or a reused pid of ours): not unclean
		d = DecideSession(self, true, self, true);
		assert(!d.reportPreviousUnclean && d.takeMarker && !d.truncateLog);
		d = DecideSession(self, false, self, false);
		assert(!d.reportPreviousUnclean && d.takeMarker && d.truncateLog);

		using EarlyLoader::ShouldDeleteMarkerAtExit;
		assert(ShouldDeleteMarkerAtExit(self, self));
		assert(!ShouldDeleteMarkerAtExit(2764, self)); // the game took it over: leave it
		assert(!ShouldDeleteMarkerAtExit(0, self));    // gone or unreadable
	}
} // namespace

int main()
{
	const fs::path tempRoot = fs::temp_directory_path() / "xefg_early_loader_test";
	std::error_code ec;
	fs::remove_all(tempRoot, ec);
	fs::create_directories(tempRoot);

	TestIniParsing(tempRoot);
	TestPeExportInspection(tempRoot);
	TestRenamedFolderAndIniPriority(tempRoot);
	TestLegacyFallback(tempRoot);
	TestHelperProcesses();
	TestSessionMarker();
	TestSessionDecision();

	fs::remove_all(tempRoot, ec);
	std::cout << "early loader: ini parsing, PE export check, renamed folder/dll, priority, legacy fallback, helper processes, .session marker: PASS\n";
	return 0;
}
