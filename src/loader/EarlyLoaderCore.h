#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace EarlyLoader
{
	// Unique export declared by this mod's main DLL (src/Plugin.cpp).
	// Checking for this export directly in the PE export table on disk guarantees the early .asi loader
	// never loads or runs DllMain on unrelated DLSSG mods inside OptiScaler's custom\ directory, even
	// when the user renames both the custom\<folder> and the .dll file.
	inline constexpr const char* kBootstrapExportName = "XeFgEarlyBootstrap";
	inline constexpr const char* kLegacyAsiExportName = "InitializeASI";
	inline constexpr const char* kLegacyPluginExportName = "OptiPluginInit";

	struct OptiScalerIniConfig
	{
		bool found = false;
		std::string fgNvngxReplacement;
		bool hasVulkanHooks = false;
		bool vulkanHooks = true;
		bool hasForceDred = false;
		bool forceDred = false;
	};

	struct CandidateSelection
	{
		std::filesystem::path path;
		std::string reason;
		size_t totalMatches = 0;
		bool legacyMatch = false;
	};

	namespace detail
	{
		inline std::string TrimAscii(std::string_view value)
		{
			size_t begin = 0;
			while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])))
				++begin;

			size_t end = value.size();
			while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
				--end;

			return std::string(value.substr(begin, end - begin));
		}

		inline std::string ToLowerAscii(std::string_view value)
		{
			std::string out;
			out.reserve(value.size());
			for (char ch : value)
				out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
			return out;
		}

		inline bool EqualsIgnoreCaseAscii(std::string_view a, std::string_view b)
		{
			if (a.size() != b.size())
				return false;
			for (size_t i = 0; i < a.size(); ++i)
			{
				if (std::tolower(static_cast<unsigned char>(a[i])) !=
					std::tolower(static_cast<unsigned char>(b[i])))
					return false;
			}
			return true;
		}

		inline bool ParseIniBool(std::string_view raw, bool& out)
		{
			const std::string lower = ToLowerAscii(TrimAscii(raw));
			if (lower == "true" || lower == "1" || lower == "yes" || lower == "on")
			{
				out = true;
				return true;
			}
			if (lower == "false" || lower == "0" || lower == "no" || lower == "off")
			{
				out = false;
				return true;
			}
			return false;
		}

		inline uint16_t ReadU16LE(const std::vector<uint8_t>& buf, size_t offset)
		{
			return static_cast<uint16_t>(buf[offset]) |
				(static_cast<uint16_t>(buf[offset + 1]) << 8);
		}

		inline uint32_t ReadU32LE(const std::vector<uint8_t>& buf, size_t offset)
		{
			return static_cast<uint32_t>(buf[offset]) |
				(static_cast<uint32_t>(buf[offset + 1]) << 8) |
				(static_cast<uint32_t>(buf[offset + 2]) << 16) |
				(static_cast<uint32_t>(buf[offset + 3]) << 24);
		}

		inline bool IsDllFile(const std::filesystem::path& path)
		{
			return EqualsIgnoreCaseAscii(path.extension().string(), ".dll");
		}
	} // namespace detail

	// Parses OptiScaler.ini for FGNvngxReplacement (under [FrameGen]) and VulkanHooks.
	inline OptiScalerIniConfig ParseOptiScalerIni(const std::filesystem::path& iniPath)
	{
		OptiScalerIniConfig config;
		std::ifstream stream(iniPath, std::ios::binary);
		if (!stream)
			return config;

		config.found = true;
		std::string section;
		std::string line;
		bool firstLine = true;

		while (std::getline(stream, line))
		{
			if (firstLine)
			{
				firstLine = false;
				if (line.size() >= 3 &&
					static_cast<unsigned char>(line[0]) == 0xEF &&
					static_cast<unsigned char>(line[1]) == 0xBB &&
					static_cast<unsigned char>(line[2]) == 0xBF)
				{
					line.erase(0, 3);
				}
			}

			if (!line.empty() && line.back() == '\r')
				line.pop_back();

			// Strip inline comments starting with ';' or '#'
			const size_t commentPos = line.find_first_of(";#");
			if (commentPos != std::string::npos)
				line.erase(commentPos);

			std::string trimmed = detail::TrimAscii(line);
			if (trimmed.empty())
				continue;

			if (trimmed.front() == '[' && trimmed.back() == ']')
			{
				section = detail::ToLowerAscii(detail::TrimAscii(
					std::string_view(trimmed).substr(1, trimmed.size() - 2)));
				continue;
			}

			const size_t eq = trimmed.find('=');
			if (eq == std::string::npos)
				continue;

			const std::string key = detail::ToLowerAscii(
				detail::TrimAscii(std::string_view(trimmed).substr(0, eq)));
			std::string value = detail::TrimAscii(std::string_view(trimmed).substr(eq + 1));

			// Strip optional surrounding quotes
			if (value.size() >= 2 &&
				((value.front() == '"' && value.back() == '"') ||
				 (value.front() == '\'' && value.back() == '\'')))
			{
				value = value.substr(1, value.size() - 2);
			}

			if (key == "fgnvngxreplacement" && (section.empty() || section == "framegen"))
			{
				config.fgNvngxReplacement = value;
			}
			else if (key == "vulkanhooks")
			{
				bool parsed = true;
				if (detail::ParseIniBool(value, parsed))
				{
					config.hasVulkanHooks = true;
					config.vulkanHooks = parsed;
				}
			}
			else if (key == "forcedred")
			{
				bool parsed = false;
				if (detail::ParseIniBool(value, parsed))
				{
					config.hasForceDred = true;
					config.forceDred = parsed;
				}
			}
		}

		return config;
	}

	// Inspects a PE32 / PE32+ file on disk for an exported ASCII symbol name without calling LoadLibrary
	// or executing DllMain. Returns false on any I/O error or malformed PE header.
	inline bool PeFileHasExport(const std::filesystem::path& filePath, std::string_view exportName)
	{
		if (exportName.empty())
			return false;

		std::error_code ec;
		const auto fileSize = std::filesystem::file_size(filePath, ec);
		if (ec || fileSize < 0x40 || fileSize > 256ull * 1024ull * 1024ull)
			return false;

		std::ifstream file(filePath, std::ios::binary);
		if (!file)
			return false;

		std::vector<uint8_t> data(static_cast<size_t>(fileSize));
		if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size())))
			return false;

		if (data[0] != 'M' || data[1] != 'Z')
			return false;

		const size_t peOffset = detail::ReadU32LE(data, 0x3C);
		if (peOffset > data.size() || data.size() - peOffset < 24)
			return false;

		if (data[peOffset] != 'P' || data[peOffset + 1] != 'E' ||
			data[peOffset + 2] != 0 || data[peOffset + 3] != 0)
			return false;

		const uint16_t sectionCount = detail::ReadU16LE(data, peOffset + 6);
		const uint16_t optionalSize = detail::ReadU16LE(data, peOffset + 20);
		const size_t optionalOffset = peOffset + 24;
		if (optionalSize < 2 || optionalOffset > data.size() || data.size() - optionalOffset < optionalSize)
			return false;

		const uint16_t magic = detail::ReadU16LE(data, optionalOffset);
		size_t exportDirRelOffset = 0;
		if (magic == 0x20B) // PE32+ (x64)
			exportDirRelOffset = 112;
		else if (magic == 0x10B) // PE32
			exportDirRelOffset = 96;
		else
			return false;

		if (optionalSize < exportDirRelOffset + 8)
			return false;

		const uint32_t exportRva = detail::ReadU32LE(data, optionalOffset + exportDirRelOffset);
		if (exportRva == 0)
			return false;

		const size_t sectionTableOffset = optionalOffset + optionalSize;
		const size_t sectionTableBytes = static_cast<size_t>(sectionCount) * 40;
		if (sectionTableOffset > data.size() || data.size() - sectionTableOffset < sectionTableBytes)
			return false;

		struct Section
		{
			uint32_t virtualAddress;
			uint32_t span;
			uint32_t rawPointer;
			uint32_t rawSize;
		};

		std::vector<Section> sections;
		sections.reserve(sectionCount);
		for (uint16_t i = 0; i < sectionCount; ++i)
		{
			const size_t entry = sectionTableOffset + static_cast<size_t>(i) * 40;
			const uint32_t virtualSize = detail::ReadU32LE(data, entry + 8);
			const uint32_t virtualAddress = detail::ReadU32LE(data, entry + 12);
			const uint32_t rawSize = detail::ReadU32LE(data, entry + 16);
			const uint32_t rawPointer = detail::ReadU32LE(data, entry + 20);
			const uint32_t span = std::max(virtualSize, rawSize);
			sections.push_back({ virtualAddress, span, rawPointer, rawSize });
		}

		auto rvaToOffset = [&](uint32_t rva, size_t& outOffset) -> bool
		{
			for (const Section& sec : sections)
			{
				if (rva >= sec.virtualAddress && rva - sec.virtualAddress < sec.span)
				{
					const uint64_t delta = static_cast<uint64_t>(rva - sec.virtualAddress);
					if (delta >= sec.rawSize)
						return false;
					const uint64_t fileOff = static_cast<uint64_t>(sec.rawPointer) + delta;
					if (fileOff >= data.size())
						return false;
					outOffset = static_cast<size_t>(fileOff);
					return true;
				}
			}
			return false;
		};

		size_t exportTableOffset = 0;
		if (!rvaToOffset(exportRva, exportTableOffset) || data.size() - exportTableOffset < 40)
			return false;

		const uint32_t numberOfNames = detail::ReadU32LE(data, exportTableOffset + 24);
		const uint32_t addressOfNamesRva = detail::ReadU32LE(data, exportTableOffset + 32);
		if (numberOfNames == 0 || numberOfNames > 65536)
			return false;

		size_t namesOffset = 0;
		if (!rvaToOffset(addressOfNamesRva, namesOffset))
			return false;
		if (data.size() - namesOffset < static_cast<size_t>(numberOfNames) * 4)
			return false;

		for (uint32_t i = 0; i < numberOfNames; ++i)
		{
			const uint32_t nameRva = detail::ReadU32LE(data, namesOffset + static_cast<size_t>(i) * 4);
			size_t nameOffset = 0;
			if (!rvaToOffset(nameRva, nameOffset))
				continue;

			const size_t maxLen = data.size() - nameOffset;
			const void* nul = std::memchr(data.data() + nameOffset, 0, maxLen);
			if (!nul)
				continue;

			const size_t len = static_cast<const uint8_t*>(nul) - (data.data() + nameOffset);
			if (len == exportName.size() &&
				std::memcmp(data.data() + nameOffset, exportName.data(), len) == 0)
			{
				return true;
			}
		}

		return false;
	}

	// Selects the main DLL to early-load from <optiRoot>/custom/*/dlssg/*.dll.
	// Priority 1: OptiScaler.ini [FrameGen] FGNvngxReplacement (when non-empty and not "auto").
	// Priority 2: Scan all custom/*/dlssg/*.dll for kBootstrapExportName (supports renamed folders & DLLs,
	//             never blocks first-time installation; breaks ties by newest last_write_time).
	// Priority 3: Legacy fallback for older builds of dlssg_to_xefg_intel_is_better.dll without XeFgEarlyBootstrap.
	inline CandidateSelection SelectMainDll(
		const std::filesystem::path& optiRoot,
		const OptiScalerIniConfig& ini,
		std::string_view preferredStem = "dlssg_to_xefg_intel_is_better")
	{
		CandidateSelection result;
		const std::filesystem::path customDir = optiRoot / "custom";
		std::error_code ec;
		if (!std::filesystem::is_directory(customDir, ec))
			return result;

		const std::string preferredFile = std::string(preferredStem) + ".dll";

		auto findInDlssgFolder = [&](const std::filesystem::path& dlssgDir, bool legacy) -> std::vector<std::filesystem::path>
		{
			std::vector<std::filesystem::path> matches;
			std::error_code dirEc;
			if (!std::filesystem::is_directory(dlssgDir, dirEc))
				return matches;

			for (const auto& entry : std::filesystem::directory_iterator(dlssgDir, dirEc))
			{
				std::error_code fileEc;
				if (!entry.is_regular_file(fileEc))
					continue;
				const auto& candidate = entry.path();
				if (!detail::IsDllFile(candidate))
					continue;

				if (!legacy)
				{
					if (PeFileHasExport(candidate, kBootstrapExportName))
						matches.push_back(candidate);
				}
				else
				{
					if (detail::EqualsIgnoreCaseAscii(candidate.filename().string(), preferredFile) &&
						PeFileHasExport(candidate, kLegacyAsiExportName) &&
						PeFileHasExport(candidate, kLegacyPluginExportName))
					{
						matches.push_back(candidate);
					}
				}
			}

			// Prefer exact preferredFile first, then newest last_write_time
			std::sort(matches.begin(), matches.end(),
				[&](const std::filesystem::path& a, const std::filesystem::path& b)
				{
					const bool aPref = detail::EqualsIgnoreCaseAscii(a.filename().string(), preferredFile);
					const bool bPref = detail::EqualsIgnoreCaseAscii(b.filename().string(), preferredFile);
					if (aPref != bPref)
						return aPref;
					std::error_code taEc, tbEc;
					const auto ta = std::filesystem::last_write_time(a, taEc);
					const auto tb = std::filesystem::last_write_time(b, tbEc);
					if (!taEc && !tbEc && ta != tb)
						return ta > tb;
					return a.string() < b.string();
				});

			return matches;
		};

		// Priority 1: Explicit FGNvngxReplacement from OptiScaler.ini
		if (!ini.fgNvngxReplacement.empty() &&
			!detail::EqualsIgnoreCaseAscii(ini.fgNvngxReplacement, "auto"))
		{
			const std::filesystem::path configuredDlssg = customDir / ini.fgNvngxReplacement / "dlssg";
			const auto configuredMatches = findInDlssgFolder(configuredDlssg, false);
			if (!configuredMatches.empty())
			{
				result.path = configuredMatches.front();
				result.reason = "OptiScaler.ini FGNvngxReplacement=" + ini.fgNvngxReplacement;
				result.totalMatches = configuredMatches.size();
				result.legacyMatch = false;
				return result;
			}

			const auto configuredLegacy = findInDlssgFolder(configuredDlssg, true);
			if (!configuredLegacy.empty())
			{
				result.path = configuredLegacy.front();
				result.reason = "OptiScaler.ini FGNvngxReplacement=" + ini.fgNvngxReplacement + " (legacy)";
				result.totalMatches = configuredLegacy.size();
				result.legacyMatch = true;
				return result;
			}
		}

		// Priority 2: Scan all custom/*/dlssg/*.dll for kBootstrapExportName
		std::vector<std::filesystem::path> allMatches;
		for (const auto& entry : std::filesystem::directory_iterator(customDir, ec))
		{
			std::error_code subEc;
			if (!entry.is_directory(subEc))
				continue;
			auto found = findInDlssgFolder(entry.path() / "dlssg", false);
			allMatches.insert(allMatches.end(), found.begin(), found.end());
		}

		if (!allMatches.empty())
		{
			std::sort(allMatches.begin(), allMatches.end(),
				[&](const std::filesystem::path& a, const std::filesystem::path& b)
				{
					std::error_code taEc, tbEc;
					const auto ta = std::filesystem::last_write_time(a, taEc);
					const auto tb = std::filesystem::last_write_time(b, tbEc);
					if (!taEc && !tbEc && ta != tb)
						return ta > tb;
					const bool aDefault = detail::EqualsIgnoreCaseAscii(
						a.parent_path().parent_path().filename().string(), "XeMFG XMX");
					const bool bDefault = detail::EqualsIgnoreCaseAscii(
						b.parent_path().parent_path().filename().string(), "XeMFG XMX");
					if (aDefault != bDefault)
						return aDefault;
					return a.string() < b.string();
				});

			result.path = allMatches.front();
			result.totalMatches = allMatches.size();
			result.reason = (allMatches.size() == 1)
				? "unique XeFgEarlyBootstrap export match"
				: "newest XeFgEarlyBootstrap export match across custom folders";
			result.legacyMatch = false;
			return result;
		}

		// Priority 3: Legacy fallback for older main DLLs without XeFgEarlyBootstrap
		std::vector<std::filesystem::path> legacyMatches;
		ec.clear();
		for (const auto& entry : std::filesystem::directory_iterator(customDir, ec))
		{
			std::error_code subEc;
			if (!entry.is_directory(subEc))
				continue;
			auto found = findInDlssgFolder(entry.path() / "dlssg", true);
			legacyMatches.insert(legacyMatches.end(), found.begin(), found.end());
		}

		if (!legacyMatches.empty())
		{
			std::sort(legacyMatches.begin(), legacyMatches.end(),
				[&](const std::filesystem::path& a, const std::filesystem::path& b)
				{
					std::error_code taEc, tbEc;
					const auto ta = std::filesystem::last_write_time(a, taEc);
					const auto tb = std::filesystem::last_write_time(b, tbEc);
					if (!taEc && !tbEc && ta != tb)
						return ta > tb;
					return a.string() < b.string();
				});

			result.path = legacyMatches.front();
			result.totalMatches = legacyMatches.size();
			result.reason = "legacy InitializeASI fallback";
			result.legacyMatch = true;
			return result;
		}

		return result;
	}

	// ------------------------------------------------------------------ helper processes and the .session marker

	// Helper processes that live beside a game's exe and so load OptiScaler (and through it this mod) too. They must
	// not hand off to the main DLL, open the shared log or touch the game's .session marker.
	inline constexpr std::wstring_view kHelperProcessExes[] = {
		L"CrashClientReporter.exe",
		L"CrashReportClient.exe",
		L"CrashReportClientEditor.exe",
		L"UnrealCEFSubProcess.exe",
		L"EpicWebHelper.exe",
		L"UnityCrashHandler64.exe",
		L"UnityCrashHandler32.exe",
		L"BsSndRpt64.exe",
		L"BsSndRpt.exe",
	};

	namespace detail
	{
		inline wchar_t LowerAsciiW(wchar_t c) noexcept
		{
			return (c >= L'A' && c <= L'Z') ? static_cast<wchar_t>(c - L'A' + L'a') : c;
		}

		inline bool EqualsIgnoreCaseAsciiW(std::wstring_view a, std::wstring_view b) noexcept
		{
			if (a.size() != b.size())
				return false;
			for (size_t i = 0; i < a.size(); ++i)
			{
				if (LowerAsciiW(a[i]) != LowerAsciiW(b[i]))
					return false;
			}
			return true;
		}

		inline std::wstring_view FileNameOf(std::wstring_view path) noexcept
		{
			const size_t slash = path.find_last_of(L"\\/");
			return slash == std::wstring_view::npos ? path : path.substr(slash + 1);
		}

		inline std::string_view FileNameOf(std::string_view path) noexcept
		{
			const size_t slash = path.find_last_of("\\/");
			return slash == std::string_view::npos ? path : path.substr(slash + 1);
		}
	} // namespace detail

	// The exe's file name, or a full path to it, any case
	inline bool IsHelperProcessExe(std::wstring_view exePathOrName) noexcept
	{
		const std::wstring_view name = detail::FileNameOf(exePathOrName);
		for (const std::wstring_view helper : kHelperProcessExes)
		{
			if (detail::EqualsIgnoreCaseAsciiW(name, helper))
				return true;
		}
		return false;
	}

	// The marker's single line: "pid=<n> exe=<name> build=<date> <time>"
	struct SessionMarkerInfo
	{
		uint32_t pid = 0;
		std::string exe;
		std::string line; // as read, for the log
	};

	inline SessionMarkerInfo ParseSessionMarker(std::string_view line)
	{
		SessionMarkerInfo info;
		info.line.assign(line.data(), line.size());
		while (!info.line.empty() && (info.line.back() == '\r' || info.line.back() == '\n'))
			info.line.pop_back();
		const auto field = [&](std::string_view key) -> std::string_view {
			size_t at = 0;
			while ((at = line.find(key, at)) != std::string_view::npos)
			{
				if (at == 0 || line[at - 1] == ' ')
				{
					const size_t begin = at + key.size();
					const size_t end = line.find(' ', begin);
					return line.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin);
				}
				at += key.size();
			}
			return {};
		};

		const std::string_view pid = field("pid=");
		uint64_t value = 0;
		bool digits = !pid.empty() && pid.size() <= 10;
		for (const char c : pid)
		{
			if (c < '0' || c > '9')
			{
				digits = false;
				break;
			}
			value = value * 10u + static_cast<uint64_t>(c - '0');
		}
		if (digits && value <= 0xFFFFFFFFull)
			info.pid = static_cast<uint32_t>(value);

		const std::string_view exe = field("exe=");
		info.exe.assign(exe.data(), exe.size());
		while (!info.exe.empty() && (info.exe.back() == '\r' || info.exe.back() == '\n'))
			info.exe.pop_back();
		return info;
	}

	// Whether the marker's exe names the same program as a running process's image path (file names compared, any case)
	inline bool SameExeName(std::string_view markerExe, std::string_view processImage) noexcept
	{
		return !markerExe.empty() && !processImage.empty() &&
			detail::EqualsIgnoreCaseAscii(detail::FileNameOf(markerExe), detail::FileNameOf(processImage));
	}

	// What a process starting up does about the log and the .session marker it shares with every other process
	// loading this mod from the same folder
	struct SessionDecision
	{
		bool reportPreviousUnclean = false; // the marker's process is gone: it never reached its clean exit
		bool takeMarker = false;            // write our own pid into the marker (and delete it again at exit)
		bool truncateLog = false;           // start the log afresh instead of appending to it
	};

	// markerPid: from the marker, 0 when there is none or it can't be read. markerOwnerAlive: that pid is a running
	// process of the marker's exe (a reused pid running something else counts as gone). logAlreadyStarted: the early
	// loader already opened the log for this process, so the DLL appends to it.
	inline SessionDecision DecideSession(uint32_t markerPid, bool markerOwnerAlive, uint32_t selfPid,
		bool logAlreadyStarted) noexcept
	{
		SessionDecision d;
		if (markerPid != 0 && markerPid != selfPid && markerOwnerAlive)
		{
			// Another live process (the game) owns the log and the marker: leave both alone, add to the log
			return d;
		}

		d.reportPreviousUnclean = markerPid != 0 && markerPid != selfPid;
		d.takeMarker = true;
		d.truncateLog = !logAlreadyStarted;
		return d;
	}

	// At a clean exit the marker is deleted only if it is still ours
	inline bool ShouldDeleteMarkerAtExit(uint32_t markerPid, uint32_t selfPid) noexcept
	{
		return markerPid != 0 && markerPid == selfPid;
	}
} // namespace EarlyLoader
