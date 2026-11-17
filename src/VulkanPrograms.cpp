#include "VulkanPrograms.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <format>
#include <fstream>
#include <set>
#include <system_error>

// Everything above Installed() is plain C++. Installed() needs this dll's log, its program folder and its SHA-256, the
// test in tests/vulkan_programs provides stand-ins for them and defines VULKANPROGRAMS_HOSTED
#if defined(_WIN32) || defined(VULKANPROGRAMS_HOSTED)
#include "Log.h"
#include "MfgKernels.h"
#include "MfgPort/Sha256.h"
#endif

namespace
{
	using VulkanPrograms::Origin;
	using VulkanPrograms::Program;
	using VulkanPrograms::Set;
	using VulkanPrograms::Sha256Fn;

	// The real manifest is about 10 KB with 83 programs and the biggest module is 213 KB, these only stop a wrong
	// file from being read whole
	constexpr uint64_t MaxManifestBytes = 1 << 20;
	constexpr uint64_t MaxModuleBytes = 16 << 20;
	constexpr uint64_t MaxSourceBytes = 16 << 10;
	constexpr size_t MaxPrograms = 4096;
	constexpr uint32_t MaxPushBytes = 4096;
	constexpr uint32_t MaxBindings = 1024;
	constexpr size_t MaxProblems = 16;
	constexpr size_t ColumnCount = 10;
	// Binding numbers: SRVs from 0, UAVs from 64, CBVs from 128, samplers from 192 (xvk's binding bases)
	constexpr uint32_t MaxBindingNumber = 255;

	bool ReadWholeFile(const std::filesystem::path& path, uint64_t limit, std::vector<uint8_t>& data)
	{
		std::error_code error;
		const uint64_t size = std::filesystem::file_size(path, error);
		if (error || size > limit)
			return false;

		std::ifstream file(path, std::ios::binary);
		if (!file)
			return false;

		data.resize(static_cast<size_t>(size));

		return size == 0 || static_cast<bool>(file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size)));
	}

	std::string DigestHex(const std::array<uint8_t, 32>& digest)
	{
		static constexpr char digits[] = "0123456789abcdef";

		std::string text;
		text.reserve(64);

		for (uint8_t value : digest)
		{
			text += digits[value >> 4];
			text += digits[value & 15];
		}

		return text;
	}

	bool HashHex(Sha256Fn sha256, const void* data, size_t size, std::string& hex)
	{
		std::array<uint8_t, 32> digest{};

		if (!sha256 || !sha256(data, size, digest))
			return false;

		hex = DigestHex(digest);
		return true;
	}

	bool IsLowerHex64(std::string_view text)
	{
		return text.size() == 64 && std::ranges::all_of(text, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
	}

	bool ParseUint(std::string_view text, uint32_t& value)
	{
		if (text.empty() || text.size() > 10)
			return false;

		const char* end = text.data() + text.size();
		const auto result = std::from_chars(text.data(), end, value);

		return result.ec == std::errc{} && result.ptr == end;
	}

	// "8x8x1"
	bool IsLocalSize(std::string_view text)
	{
		int parts = 0;

		for (size_t at = 0;;)
		{
			const size_t end = text.find('x', at);
			const auto piece = text.substr(at, end == std::string_view::npos ? std::string_view::npos : end - at);

			uint32_t value = 0;
			if (!ParseUint(piece, value) || value == 0 || value > 1024)
				return false;

			parts++;

			if (end == std::string_view::npos)
				break;

			at = end + 1;
		}

		return parts == 3;
	}

	// "345116,345270"
	bool IsKeyList(std::string_view text)
	{
		bool digit = false;

		for (char c : text)
		{
			if (c == ',')
			{
				if (!digit)
					return false;

				digit = false;
			}
			else if (c >= '0' && c <= '9')
			{
				digit = true;
			}
			else
			{
				return false;
			}
		}

		return digit;
	}

	bool IsSpirv(const std::vector<uint8_t>& module)
	{
		// The magic number 0x07230203, little endian, and whole 32-bit words
		return module.size() >= 20 && module.size() % 4 == 0 && module[0] == 0x03 && module[1] == 0x02 &&
			module[2] == 0x23 && module[3] == 0x07;
	}

	bool ParseOrigin(std::string_view text, Origin& origin)
	{
		if (text == "port")
			origin = Origin::Port;
		else if (text == "port-extra")
			origin = Origin::PortExtra;
		else if (text == "intel")
			origin = Origin::Intel;
		else if (text == "unhashed")
			origin = Origin::Unhashed;
		else
			return false;

		return true;
	}

	// "0:2,1:0,64:0": binding:type, in binding order. A program with no descriptors has "-"
	bool ParseBindingRows(std::string_view text, std::vector<VulkanPrograms::Binding>& rows, std::string& error)
	{
		rows.clear();

		if (text == "-")
			return true;

		for (size_t at = 0;;)
		{
			const size_t end = text.find(',', at);
			const auto piece = text.substr(at, end == std::string_view::npos ? std::string_view::npos : end - at);
			const size_t colon = piece.find(':');

			uint32_t binding = 0;
			uint32_t type = 0;

			if (colon == std::string_view::npos || !ParseUint(piece.substr(0, colon), binding) ||
				!ParseUint(piece.substr(colon + 1), type))
			{
				error = "the binding types aren't a list of binding:type";
				return false;
			}

			if (binding > MaxBindingNumber)
			{
				error = "a binding number is out of range";
				return false;
			}

			if (type >= VulkanPrograms::BindingTypeCount)
			{
				error = "unknown binding type";
				return false;
			}

			if (!rows.empty() && binding <= rows.back().binding)
			{
				error = "the bindings aren't in increasing order";
				return false;
			}

			if (rows.size() == MaxBindings)
			{
				error = "too many bindings";
				return false;
			}

			rows.push_back({ static_cast<uint16_t>(binding), static_cast<VulkanPrograms::BindingType>(type) });

			if (end == std::string_view::npos)
				break;

			at = end + 1;
		}

		return true;
	}

	// One line of programs.tsv: id, sha256, bytes, pushBytes, bindings, localSize, sourceDxilSha256, origin, portKeys,
	// bindingTypes
	bool ParseRow(std::string_view line, Program& program, std::string& error)
	{
		std::array<std::string_view, ColumnCount> field;
		size_t count = 0;

		for (size_t at = 0;;)
		{
			if (count == field.size())
			{
				error = std::format("more than {} columns", ColumnCount);
				return false;
			}

			const size_t end = line.find('\t', at);
			field[count++] = line.substr(at, end == std::string_view::npos ? std::string_view::npos : end - at);

			if (end == std::string_view::npos)
				break;

			at = end + 1;
		}

		if (count != field.size())
		{
			error = std::format("{} columns instead of {}", count, ColumnCount);
			return false;
		}

		if (!ParseUint(field[0], program.id))
		{
			error = "the id isn't a number";
			return false;
		}

		if (!IsLowerHex64(field[1]))
		{
			error = "the SPIR-V SHA-256 isn't 64 lower case hex digits";
			return false;
		}

		program.sha256 = field[1];

		if (!ParseUint(field[2], program.bytes) || program.bytes == 0 || program.bytes % 4 != 0 || program.bytes > MaxModuleBytes)
		{
			error = "the size isn't a plausible SPIR-V size";
			return false;
		}

		if (!ParseUint(field[3], program.pushBytes) || program.pushBytes > MaxPushBytes)
		{
			error = "the push constant size isn't plausible";
			return false;
		}

		if (!ParseUint(field[4], program.bindings) || program.bindings > MaxBindings)
		{
			error = "the binding count isn't plausible";
			return false;
		}

		if (field[5] == "-")
		{
			program.localSize.clear();
		}
		else if (IsLocalSize(field[5]))
		{
			program.localSize = field[5];
		}
		else
		{
			error = "the local size isn't AxBxC";
			return false;
		}

		if (field[6] == "-")
		{
			program.dxilSha256.clear();
		}
		else if (IsLowerHex64(field[6]))
		{
			program.dxilSha256 = field[6];
		}
		else
		{
			error = "the DXIL SHA-256 isn't 64 lower case hex digits";
			return false;
		}

		if (!ParseOrigin(field[7], program.origin))
		{
			error = std::format("unknown origin '{}'", field[7].substr(0, 24));
			return false;
		}

		if (field[8] == "-")
		{
			program.portKeys.clear();
		}
		else if (IsKeyList(field[8]))
		{
			program.portKeys = field[8];
		}
		else
		{
			error = "the port keys aren't a list of numbers";
			return false;
		}

		if (!ParseBindingRows(field[9], program.bindingRows, error))
			return false;

		if (program.bindingRows.size() != program.bindings)
		{
			error = "the binding types don't list as many bindings as the binding count says";
			return false;
		}

		// A program translates a DXIL program unless it is the one listed without a source, and a port program says
		// which of the XMX Port programs it stands for
		if ((program.origin == Origin::Unhashed) != program.dxilSha256.empty())
		{
			error = "the DXIL source doesn't fit the origin";
			return false;
		}

		if (program.origin == Origin::Port && program.portKeys.empty())
		{
			error = "a port program without port keys";
			return false;
		}

		return true;
	}

	// SOURCE.txt tells which MFGUnlocker.asi the folder was made from. Optional, only for the log
	void ReadSource(const std::filesystem::path& path, Set& set)
	{
		std::vector<uint8_t> data;
		if (!ReadWholeFile(path, MaxSourceBytes, data))
			return;

		constexpr std::string_view buildKey = "MFG Unlocker build: ";
		constexpr std::string_view asiKey = "MFGUnlocker.asi sha256: ";

		const std::string_view text(reinterpret_cast<const char*>(data.data()), data.size());

		for (size_t at = 0; at < text.size();)
		{
			size_t end = text.find('\n', at);
			if (end == std::string_view::npos)
				end = text.size();

			std::string_view line = text.substr(at, end - at);
			at = end + 1;

			if (!line.empty() && line.back() == '\r')
				line.remove_suffix(1);

			if (line.starts_with(buildKey))
			{
				// Printable ASCII only, it ends up in the log and the menu
				for (char c : line.substr(buildKey.size(), 100))
				{
					if (c >= 0x20 && c < 0x7F)
						set.build += c;
				}
			}
			else if (line.starts_with(asiKey) && IsLowerHex64(line.substr(asiKey.size())))
			{
				set.asiSha256 = line.substr(asiKey.size());
			}
		}
	}
} // namespace

const VulkanPrograms::Program* VulkanPrograms::Set::FindById(uint32_t id) const
{
	const auto found = std::ranges::find(programs, id, &Program::id);
	return found == programs.end() ? nullptr : &*found;
}

const VulkanPrograms::Program* VulkanPrograms::Set::FindByDxil(std::string_view sha256) const
{
	const auto found = byDxil.find(sha256);
	return found == byDxil.end() ? nullptr : &programs[found->second];
}

VulkanPrograms::Set VulkanPrograms::Load(const std::filesystem::path& directory, Sha256Fn sha256)
{
	Set set;
	set.directory = directory;

	const auto problem = [&set](std::string text)
		{
			if (set.problems.size() < MaxProblems)
				set.problems.push_back(std::move(text));
			else if (set.problems.size() == MaxProblems)
				set.problems.push_back("and more");
		};

	try
	{
		std::error_code error;
		const auto manifestPath = directory / "programs.tsv";

		if (!std::filesystem::exists(manifestPath, error))
			return set;

		set.present = true;

		std::vector<uint8_t> manifest;
		if (!ReadWholeFile(manifestPath, MaxManifestBytes, manifest) || manifest.empty())
		{
			problem("programs.tsv can't be read");
			return set;
		}

		if (!HashHex(sha256, manifest.data(), manifest.size(), set.manifestSha256))
		{
			problem("the SHA-256 of programs.tsv can't be computed");
			return set;
		}

		ReadSource(directory / "SOURCE.txt", set);

		const std::string_view text(reinterpret_cast<const char*>(manifest.data()), manifest.size());
		std::set<uint32_t> ids;
		size_t lineNumber = 0;

		for (size_t at = 0; at < text.size();)
		{
			size_t end = text.find('\n', at);
			if (end == std::string_view::npos)
				end = text.size();

			std::string_view line = text.substr(at, end - at);
			at = end + 1;
			lineNumber++;

			if (!line.empty() && line.back() == '\r')
				line.remove_suffix(1);

			if (line.empty())
				continue;

			if (set.programs.size() == MaxPrograms)
			{
				problem(std::format("programs.tsv lists more than {} programs", MaxPrograms));
				break;
			}

			Program program;
			std::string reason;

			if (!ParseRow(line, program, reason))
			{
				problem(std::format("programs.tsv line {}: {}", lineNumber, reason));
				continue;
			}

			if (!ids.insert(program.id).second)
			{
				problem(std::format("programs.tsv lists id {} twice", program.id));
				continue;
			}

			program.file = directory / std::format("{}.spv", program.id);

			std::error_code sizeError;
			const uint64_t size = std::filesystem::file_size(program.file, sizeError);

			if (sizeError)
			{
				problem(std::format("{}.spv is missing", program.id));
			}
			else if (size != program.bytes)
			{
				problem(std::format("{}.spv is {} bytes, programs.tsv says {}", program.id, size, program.bytes));
			}
			else
			{
				std::vector<uint8_t> module;
				std::string hash;

				if (!ReadWholeFile(program.file, MaxModuleBytes, module) || module.size() != program.bytes)
					problem(std::format("{}.spv can't be read", program.id));
				else if (!IsSpirv(module))
					problem(std::format("{}.spv isn't a SPIR-V module", program.id));
				else if (!HashHex(sha256, module.data(), module.size(), hash))
					problem(std::format("the SHA-256 of {}.spv can't be computed", program.id));
				else if (hash != program.sha256)
					problem(std::format("{}.spv doesn't match its SHA-256", program.id));
			}

			set.perOrigin[static_cast<size_t>(program.origin)]++;
			set.totalBytes += program.bytes;

			if (!program.dxilSha256.empty() && !set.byDxil.emplace(program.dxilSha256, set.programs.size()).second)
				problem(std::format("programs.tsv: {}.spv translates a DXIL program another one already does", program.id));

			set.programs.push_back(std::move(program));
		}

		if (set.programs.empty() && set.problems.empty())
			problem("programs.tsv lists no programs");
	}
	catch (const std::exception& e)
	{
		problem(std::format("unexpected error: {}", e.what()));
	}

	return set;
}

std::vector<uint8_t> VulkanPrograms::ReadModule(const Program& program, Sha256Fn sha256)
{
	std::vector<uint8_t> module;
	std::string hash;

	if (!ReadWholeFile(program.file, MaxModuleBytes, module) || module.size() != program.bytes || !IsSpirv(module) ||
		!HashHex(sha256, module.data(), module.size(), hash) || hash != program.sha256)
		return {};

	return module;
}

std::string VulkanPrograms::Summary(const Set& set)
{
	if (!set.present)
		return "not installed";

	if (!set.Usable())
	{
		return std::format("{} problem{}, first: {}", set.problems.size(), set.problems.size() == 1 ? "" : "s",
			set.problems.empty() ? "no programs" : set.problems.front());
	}

	static constexpr const char* names[OriginCount] = { "port", "port-extra", "intel", "unhashed" };
	std::string origins;

	for (size_t i = 0; i < OriginCount; i++)
	{
		if (set.perOrigin[i])
			origins += std::format("{}{} {}", origins.empty() ? "" : ", ", names[i], set.perOrigin[i]);
	}

	auto line = std::format("{} programs ({}), {:.1f} MiB", set.programs.size(), origins,
		static_cast<double>(set.totalBytes) / (1024.0 * 1024.0));

	if (!set.build.empty())
		line += ", from MFG Unlocker " + set.build;

	return line;
}

std::string VulkanPrograms::PathText(const std::filesystem::path& path)
{
	const auto text = path.u8string();

	return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

std::filesystem::path VulkanPrograms::DirectoryFor(const std::filesystem::path& programsDirectory)
{
	auto root = programsDirectory;

	if (root.filename() == "programs")
		root = root.parent_path();

	return root / "vulkan";
}

#if defined(_WIN32) || defined(VULKANPROGRAMS_HOSTED)
namespace
{
	std::atomic<const VulkanPrograms::Set*> published = nullptr;
} // namespace

const VulkanPrograms::Set& VulkanPrograms::Installed()
{
	static const Set set = []
		{
			Set loaded = Load(DirectoryFor(MfgKernels::ProgramsDirectory()), &sha256_util::Compute);

			if (!loaded.present)
			{
				Log::Info("Vulkan models: none in {}. Vulkan games need them there, tools/ExtractXeFgVulkanPrograms.py makes the "
					"folder from an MFGUnlocker.asi (a build of this dll without the Vulkan runtime uses the models inside the "
					"asi instead)", PathText(loaded.directory));
			}
			else if (!loaded.Usable())
			{
				Log::Warn("Vulkan models in {} can't be used: {}", PathText(loaded.directory), Summary(loaded));

				for (const auto& problem : loaded.problems)
					Log::Warn("  {}", problem);
			}
			else
			{
				Log::Info("Vulkan models: {} in {}", Summary(loaded), PathText(loaded.directory));
			}

			return loaded;
		}();

	published.store(&set, std::memory_order_release);

	return set;
}

const VulkanPrograms::Set* VulkanPrograms::Loaded()
{
	return published.load(std::memory_order_acquire);
}
#endif
