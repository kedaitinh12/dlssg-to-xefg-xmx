// Runs the real src/VulkanPrograms.cpp, and the real src/MfgPort/Sha256.h over a stand-in for CNG, against program sets
// built in a temporary folder, and optionally against a folder ExtractXeFgVulkanPrograms.py made from an MFGUnlocker.asi.
//
//   programs_test            the synthetic sets and Installed(), the part CI runs with nothing else
//   programs_test --missing  Installed() with no xefg_mfg_programs\vulkan folder
//   programs_test --real DIR the folder the extractor made: 83 programs, every one verified, see RealSet(), and the
//                            input-copy shaders built into the dll against the ones in the folder
//
// It checks the reading and the checking. It doesn't run a program, nothing here does. Built with
// -DVULKANPROGRAMS_HOSTED so that VulkanPrograms.cpp includes Installed(), which uses the stand-ins below.

#include "Log.h"
#include "MfgKernels.h"
#include "MfgPort/Sha256.h"
#include "VulkanPrograms.h"
#include "vulkan/VulkanInputCopy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------------------------------
static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                                                                   \
	do                                                                                                                \
	{                                                                                                                 \
		checks++;                                                                                                     \
		if (!(cond))                                                                                                  \
		{                                                                                                             \
			failures++;                                                                                               \
			std::printf("FAIL line %d: %s\n", __LINE__, #cond);                                                       \
		}                                                                                                             \
	} while (0)

// ---------------------------------------------------------------------------------------------------------------
// The host side Installed() uses
static std::vector<std::string> logLines;
static fs::path programsDirectory;

void Log::Write(std::string_view level, std::string_view message)
{
	logLines.push_back(std::string(level) + ": " + std::string(message));
}

fs::path MfgKernels::ProgramsDirectory() { return programsDirectory; }

static bool Logged(const char* needle)
{
	for (const auto& line : logLines)
		if (line.find(needle) != std::string::npos)
			return true;
	return false;
}

// ---------------------------------------------------------------------------------------------------------------
static bool Hash(const void* data, size_t size, sha256_util::Digest& digest) { return sha256_util::Compute(data, size, digest); }

static std::string Hex(const std::vector<uint8_t>& bytes)
{
	sha256_util::Digest digest{};
	sha256_util::Compute(bytes.data(), bytes.size(), digest);
	return sha256_util::LowerHex(digest);
}

static bool IsHex(const std::string& text)
{
	if (text.size() != 64)
		return false;
	for (char c : text)
		if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
			return false;
	return true;
}

static std::string HexOf(const std::string& text)
{
	return Hex(std::vector<uint8_t>(text.begin(), text.end()));
}

static std::vector<uint8_t> ReadAll(const fs::path& path)
{
	std::ifstream file(path, std::ios::binary);
	return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

static void WriteAll(const fs::path& path, const std::string& text)
{
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	file << text;
}

static void WriteAll(const fs::path& path, const std::vector<uint8_t>& bytes)
{
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

// A SPIR-V looking module: the magic number, version 1.6, then words that depend on the seed
static std::vector<uint8_t> Module(uint32_t seed, size_t words)
{
	std::vector<uint8_t> bytes = { 0x03, 0x02, 0x23, 0x07, 0x00, 0x06, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00 };
	uint32_t x = seed * 2654435761u + 1;

	for (size_t i = 0; i < words; i++)
	{
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;

		for (int b = 0; b < 4; b++)
			bytes.push_back((uint8_t)(x >> (8 * b)));
	}

	return bytes;
}

static std::string Dxil(uint32_t n)
{
	char text[65];
	std::snprintf(text, sizeof(text), "%064x", n);
	return text;
}

// "binding:type" rows for a program with `count` descriptors: the first half from binding 0 (SRVs), the rest from binding 64
// (UAVs), the types cycling through the five there are
static std::string BindingText(uint32_t count)
{
	if (count == 0)
		return "-";

	const uint32_t half = (count + 1) / 2;
	std::string text;

	for (uint32_t i = 0; i < count; i++)
	{
		const uint32_t binding = i < half ? i : 64 + (i - half);
		text += (i ? "," : "") + std::to_string(binding) + ":" + std::to_string(i % 5);
	}

	return text;
}

struct Prog
{
	uint32_t id;
	std::vector<uint8_t> module;
	uint32_t push;
	uint32_t bindings;
	std::string local;
	std::string dxil;
	std::string origin;
	std::string keys;
	std::string types = {}; // the bindingTypes column, BindingText(bindings) when empty
};

static std::string RowText(const Prog& p)
{
	return std::to_string(p.id) + "\t" + Hex(p.module) + "\t" + std::to_string(p.module.size()) + "\t" +
		std::to_string(p.push) + "\t" + std::to_string(p.bindings) + "\t" + p.local + "\t" + p.dxil + "\t" + p.origin + "\t" +
		p.keys + "\t" + (p.types.empty() ? BindingText(p.bindings) : p.types);
}

// Two port programs (one with two keys), the port-extra one, two Intel ones and the one without a DXIL source
static std::vector<Prog> Baseline()
{
	return {
		{ 7400, Module(1, 40), 16, 2, "8x8x1", Dxil(0x1001), "port", "34511160110267935226759" },
		{ 7401, Module(2, 90), 204, 16, "8x16x1", Dxil(0x1002), "port", "3452703420262894380461,3453044563380101633987" },
		{ 7402, Module(3, 70), 48, 5, "16x16x1", Dxil(0x1003), "port-extra", "-" },
		{ 7403, Module(4, 30), 16, 3, "4x8x4", Dxil(0x1004), "intel", "-" },
		{ 7404, Module(4, 30), 16, 3, "4x8x4", Dxil(0x1005), "intel", "-" }, // the same module as 7403, as in the real set
		{ 7405, Module(6, 500), 16, 5, "-", "-", "unhashed", "-" },
	};
}

static fs::path MakeTemp()
{
	static int counter = 0;
	const fs::path path = fs::temp_directory_path() / ("vkprograms-" + std::to_string(getpid()) + "-" + std::to_string(counter++));
	fs::remove_all(path);
	fs::create_directories(path);
	return path;
}

static std::string Join(const std::vector<Prog>& programs, const std::string& eol)
{
	std::string text;
	for (const auto& p : programs)
		text += RowText(p) + eol;
	return text;
}

static void WriteSet(const fs::path& dir, const std::vector<Prog>& programs, const std::string& eol = "\r\n", bool source = true)
{
	fs::create_directories(dir);

	for (const auto& p : programs)
		WriteAll(dir / (std::to_string(p.id) + ".spv"), p.module);

	WriteAll(dir / "programs.tsv", Join(programs, eol));

	if (source)
	{
		WriteAll(dir / "SOURCE.txt",
			std::string("MFG Unlocker build: 1.7.1-rc.2 (file version 1.7.1.173)\r\n"
						"MFGUnlocker.asi sha256: ") + std::string(64, 'a') + "\r\nprograms: 6\r\n");
	}
}

static void ReplaceInManifest(const fs::path& dir, const std::string& from, const std::string& to)
{
	auto bytes = ReadAll(dir / "programs.tsv");
	std::string text(bytes.begin(), bytes.end());
	const auto at = text.find(from);
	if (at == std::string::npos)
	{
		std::printf("test bug: '%s' isn't in the manifest\n", from.c_str());
		std::exit(2);
	}
	text.replace(at, from.size(), to);
	WriteAll(dir / "programs.tsv", text);
}

static bool HasProblem(const VulkanPrograms::Set& set, const std::string& needle)
{
	for (const auto& problem : set.problems)
		if (problem.find(needle) != std::string::npos)
			return true;
	return false;
}

static VulkanPrograms::Set LoadDir(const fs::path& dir) { return VulkanPrograms::Load(dir, &Hash); }

// ---------------------------------------------------------------------------------------------------------------
static void CheckSha256()
{
	// FIPS 180-2 test vectors, so the stand-in for CNG is known to be right before anything is hashed with it
	CHECK(HexOf("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
	CHECK(HexOf("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
	CHECK(HexOf("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
		"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
	CHECK(HexOf(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
	// Lengths around the padding boundaries, from Python's hashlib
	CHECK(HexOf(std::string(55, 'a')) == "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
	CHECK(HexOf(std::string(56, 'a')) == "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
	CHECK(HexOf(std::string(63, 'a')) == "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34");
	CHECK(HexOf(std::string(64, 'a')) == "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
	CHECK(HexOf(std::string(65, 'a')) == "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0");
	CHECK(HexOf(std::string(119, 'a')) == "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb");
	CHECK(HexOf(std::string(120, 'a')) == "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c");
	CHECK(HexOf(std::string(128, 'a')) == "6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e");
}

static void CheckBaseline()
{
	for (const char* eol : { "\r\n", "\n" })
	{
		const fs::path dir = MakeTemp();
		const auto programs = Baseline();
		WriteSet(dir, programs, eol);

		const auto set = LoadDir(dir);
		CHECK(set.present);
		CHECK(set.problems.empty());
		CHECK(set.Usable());
		CHECK(set.programs.size() == 6);
		CHECK(set.perOrigin[(size_t)VulkanPrograms::Origin::Port] == 2);
		CHECK(set.perOrigin[(size_t)VulkanPrograms::Origin::PortExtra] == 1);
		CHECK(set.perOrigin[(size_t)VulkanPrograms::Origin::Intel] == 2);
		CHECK(set.perOrigin[(size_t)VulkanPrograms::Origin::Unhashed] == 1);

		uint64_t total = 0;
		for (const auto& p : programs)
			total += p.module.size();
		CHECK(set.totalBytes == total);

		CHECK(set.manifestSha256 == HexOf(Join(programs, eol)));
		CHECK(set.build == "1.7.1-rc.2 (file version 1.7.1.173)");
		CHECK(set.asiSha256 == std::string(64, 'a'));

		const auto* second = set.FindById(7401);
		CHECK(second && second->bindings == 16 && second->bytes == programs[1].module.size());
		CHECK(second && second->pushBytes == 204 && second->localSize == "8x16x1");
		CHECK(second && second->origin == VulkanPrograms::Origin::Port);
		CHECK(second && second->portKeys == "3452703420262894380461,3453044563380101633987");
		CHECK(set.FindById(1) == nullptr && set.FindById(7406) == nullptr);

		// Two programs share a SPIR-V module, each is found by its own DXIL
		const auto* a = set.FindByDxil(Dxil(0x1004));
		const auto* b = set.FindByDxil(Dxil(0x1005));
		CHECK(a && b && a->id == 7403 && b->id == 7404 && a->sha256 == b->sha256);
		CHECK(set.FindByDxil(Dxil(0x9999)) == nullptr);
		CHECK(set.FindByDxil("") == nullptr);
		const auto* unhashed = set.FindById(7405);
		CHECK(unhashed && unhashed->dxilSha256.empty() && unhashed->localSize.empty());

		const auto module = VulkanPrograms::ReadModule(*second, &Hash);
		CHECK(module == programs[1].module);

		// The binding rows: as many as the count says, in order, with their types
		using VulkanPrograms::Binding;
		using VulkanPrograms::BindingType;
		const auto* third = set.FindById(7403);
		CHECK(third && third->bindingRows.size() == 3 && third->bindingRows[0] == (Binding{ 0, BindingType::StorageBuffer }) &&
			third->bindingRows[1] == (Binding{ 1, BindingType::UniformBuffer }) &&
			third->bindingRows[2] == (Binding{ 64, BindingType::SampledImage }));
		CHECK(second && second->bindingRows.size() == 16 && second->bindingRows[0].binding == 0 &&
			second->bindingRows[8].binding == 64 && second->bindingRows[15].binding == 71 &&
			second->bindingRows[4].type == BindingType::Sampler);
		for (const auto& program : set.programs)
			CHECK(program.bindingRows.size() == program.bindings);

		const auto summary = VulkanPrograms::Summary(set);
		CHECK(summary.find("6 programs") != std::string::npos);
		CHECK(summary.find("port 2, port-extra 1, intel 2, unhashed 1") != std::string::npos);
		CHECK(summary.find("MFG Unlocker 1.7.1-rc.2") != std::string::npos);

		// A module that changes after the load is refused
		auto bytes = ReadAll(dir / "7401.spv");
		bytes[100] ^= 1;
		WriteAll(dir / "7401.spv", bytes);
		CHECK(VulkanPrograms::ReadModule(*second, &Hash).empty());

		fs::remove_all(dir);
	}

	// No SOURCE.txt, no trailing newline, blank lines between the rows
	{
		const fs::path dir = MakeTemp();
		const auto programs = Baseline();
		WriteSet(dir, programs, "\n", false);
		std::string text = Join(programs, "\n\n");
		text.pop_back();
		text.pop_back();
		WriteAll(dir / "programs.tsv", text);

		const auto set = LoadDir(dir);
		CHECK(set.Usable() && set.programs.size() == 6 && set.build.empty() && set.asiSha256.empty());
		CHECK(VulkanPrograms::Summary(set).find("from MFG Unlocker") == std::string::npos);
		fs::remove_all(dir);
	}

	// SOURCE.txt text goes to the log and the menu: control characters are dropped
	{
		const fs::path dir = MakeTemp();
		WriteSet(dir, Baseline());
		WriteAll(dir / "SOURCE.txt", std::string("MFG Unlocker build: 1.7\x01\x1b[31m.1\x7f\r\nMFGUnlocker.asi sha256: nothex\r\n"));
		const auto set = LoadDir(dir);
		CHECK(set.Usable() && set.build == "1.7[31m.1" && set.asiSha256.empty());
		fs::remove_all(dir);
	}
}

static void CheckMissingAndEmpty()
{
	{
		const fs::path dir = MakeTemp();
		const auto set = LoadDir(dir);
		CHECK(!set.present && set.problems.empty() && !set.Usable() && set.programs.empty());
		CHECK(VulkanPrograms::Summary(set) == "not installed");
		fs::remove_all(dir);
	}
	{
		const auto set = LoadDir(fs::path("/nonexistent/vkprograms/none"));
		CHECK(!set.present && set.problems.empty() && !set.Usable());
	}
	{
		// programs.tsv is a folder
		const fs::path dir = MakeTemp();
		fs::create_directories(dir / "programs.tsv");
		const auto set = LoadDir(dir);
		CHECK(set.present && !set.Usable() && !set.problems.empty());
		fs::remove_all(dir);
	}
	{
		const fs::path dir = MakeTemp();
		WriteAll(dir / "programs.tsv", std::string());
		const auto set = LoadDir(dir);
		CHECK(set.present && !set.Usable() && HasProblem(set, "can't be read"));
		fs::remove_all(dir);
	}
	{
		const fs::path dir = MakeTemp();
		WriteAll(dir / "programs.tsv", std::string("\r\n\r\n\n"));
		const auto set = LoadDir(dir);
		CHECK(set.present && !set.Usable() && HasProblem(set, "lists no programs"));
		CHECK(VulkanPrograms::Summary(set).find("1 problem, first: programs.tsv lists no programs") == 0);
		fs::remove_all(dir);
	}
	{
		// Without a SHA-256 there is nothing to check against: refused, not accepted
		const fs::path dir = MakeTemp();
		WriteSet(dir, Baseline());
		const auto set = VulkanPrograms::Load(dir, [](const void*, size_t, sha256_util::Digest&) { return false; });
		CHECK(set.present && !set.Usable() && HasProblem(set, "SHA-256"));
		const auto none = VulkanPrograms::Load(dir, nullptr);
		CHECK(none.present && !none.Usable());
		fs::remove_all(dir);
	}
}

static void CheckBrokenFiles()
{
	struct Case
	{
		const char* name;
		void (*damage)(const fs::path&);
		const char* problem;
	};

	static const Case cases[] = {
		{ "missing module", [](const fs::path& d) { fs::remove(d / "7401.spv"); }, "7401.spv is missing" },
		{ "truncated module", [](const fs::path& d) { fs::resize_file(d / "7401.spv", 100); }, "7401.spv is 100 bytes" },
		{ "grown module", [](const fs::path& d) { fs::resize_file(d / "7402.spv", fs::file_size(d / "7402.spv") + 4); }, "7402.spv is" },
		{ "flipped byte", [](const fs::path& d) { auto b = ReadAll(d / "7400.spv"); b[60] ^= 0x10; WriteAll(d / "7400.spv", b); },
			"7400.spv doesn't match its SHA-256" },
		{ "not SPIR-V", [](const fs::path& d)
			{
				// A module with another magic number and the manifest updated to its real hash and size
				auto bytes = Module(9, 40);
				bytes[0] = 0x04;
				WriteAll(d / "7400.spv", bytes);
				auto manifest = ReadAll(d / "programs.tsv");
				std::string text(manifest.begin(), manifest.end());
				const auto oldRow = RowText(Baseline()[0]);
				Prog p = Baseline()[0];
				p.module = bytes;
				text.replace(text.find(oldRow), oldRow.size(), RowText(p));
				WriteAll(d / "programs.tsv", text);
			}, "7400.spv isn't a SPIR-V module" },
		{ "module is a folder", [](const fs::path& d) { fs::remove(d / "7403.spv"); fs::create_directories(d / "7403.spv"); },
			"7403.spv" },
	};

	for (const auto& c : cases)
	{
		const fs::path dir = MakeTemp();
		WriteSet(dir, Baseline());
		c.damage(dir);

		const auto set = LoadDir(dir);
		const bool found = HasProblem(set, c.problem);

		if (!found)
		{
			std::printf("  case '%s' expected '%s', problems:\n", c.name, c.problem);
			for (const auto& p : set.problems)
				std::printf("    %s\n", p.c_str());
		}

		CHECK(set.present && !set.Usable() && found);
		// The rest of the manifest is still read, the menu can say what is wrong
		CHECK(set.programs.size() == 6);
		fs::remove_all(dir);
	}
}

// Replaces the first `from` in the baseline's manifest with `to`, and expects the set to be refused for `problem`
static void RowCase(const std::string& from, const std::string& to, const std::string& problem)
{
	const fs::path dir = MakeTemp();
	WriteSet(dir, Baseline());
	ReplaceInManifest(dir, from, to);

	const auto set = LoadDir(dir);
	const bool found = HasProblem(set, problem);

	if (!found)
	{
		std::printf("  expected '%s', problems:\n", problem.c_str());
		for (const auto& p : set.problems)
			std::printf("    %s\n", p.c_str());
	}

	CHECK(set.present && !set.Usable() && found);
	// A broken row costs that row only, the rest is still read
	CHECK(set.programs.size() >= 5);

	fs::remove_all(dir);
}

static void CheckBrokenRows()
{
	const auto sha0 = Hex(Baseline()[0].module);
	const std::string keys0 = "34511160110267935226759";

	RowCase("7400\t", "x7400\t", "line 1: the id isn't a number");
	RowCase(sha0, sha0.substr(1), "line 1: the SPIR-V SHA-256 isn't 64 lower case hex digits");
	RowCase("\t8x8x1\t", "\t8x8\t", "line 1: the local size isn't AxBxC");
	RowCase("\t8x8x1\t", "\t8x0x1\t", "line 1: the local size isn't AxBxC");
	RowCase("\t8x8x1\t", "\t8xax1\t", "line 1: the local size isn't AxBxC");
	RowCase("\t8x8x1\t", "\t8x8x1x1\t", "line 1: the local size isn't AxBxC");
	RowCase("\tport\t" + keys0, "\tweird\t" + keys0, "line 1: unknown origin 'weird'");
	RowCase("\t" + keys0, "\t3451,,116", "line 1: the port keys aren't a list of numbers");
	RowCase("\t" + keys0, "\t-" + keys0, "line 1: the port keys aren't a list of numbers");
	RowCase("\t" + keys0, "\t" + keys0 + ",", "line 1: the port keys aren't a list of numbers");
	RowCase("\tport\t" + keys0, "\tport", "line 1: 9 columns instead of 10");
	RowCase("64:1\r\n", "64:1\t-\r\n", "line 1: more than 10 columns");
	RowCase("\t0:0,64:1\r\n", "\r\n", "line 1: 9 columns instead of 10");
	// The binding types: syntax, range, order, and as many as the binding count says
	RowCase("64:1\r\n", "64:9\r\n", "line 1: unknown binding type");
	RowCase("0:0,64:1", "64:0,0:1", "line 1: the bindings aren't in increasing order");
	RowCase("0:0,64:1", "0:0,0:1", "line 1: the bindings aren't in increasing order");
	RowCase("64:1\r\n", "256:1\r\n", "line 1: a binding number is out of range");
	RowCase("64:1\r\n", "64\r\n", "line 1: the binding types aren't a list of binding:type");
	RowCase("64:1\r\n", "64:x\r\n", "line 1: the binding types aren't a list of binding:type");
	RowCase("64:1\r\n", "64:1,\r\n", "line 1: the binding types aren't a list of binding:type");
	RowCase("\t0:0,64:1\r\n", "\t\r\n", "line 1: the binding types aren't a list of binding:type");
	RowCase("64:1\r\n", "64:1,65:1\r\n", "line 1: the binding types don't list as many bindings");
	RowCase("0:0,64:1", "-", "line 1: the binding types don't list as many bindings");
	RowCase(Dxil(0x1002), "-", "line 2: the DXIL source doesn't fit the origin");
	RowCase(Dxil(0x1002), "ABCDEF", "line 2: the DXIL SHA-256 isn't 64 lower case hex digits");
	RowCase(Dxil(0x1002), "A" + Dxil(0x1002).substr(1), "line 2: the DXIL SHA-256 isn't 64 lower case hex digits");
	RowCase("\tintel\t-\t0:0,1:1,64:2\r\n7404", "\tunhashed\t-\t0:0,1:1,64:2\r\n7404", "line 4: the DXIL source doesn't fit the origin");

	// port-extra programs say nothing about port keys, but may
	{
		const fs::path dir = MakeTemp();
		WriteSet(dir, Baseline());
		ReplaceInManifest(dir, "\tport-extra\t-", "\tport-extra\t12");
		CHECK(LoadDir(dir).Usable());
		fs::remove_all(dir);
	}

	// A port program without keys, an Intel program without a DXIL source
	{
		const fs::path dir = MakeTemp();
		WriteSet(dir, Baseline());
		ReplaceInManifest(dir, "\tport\t34511160110267935226759", "\tport\t-");
		const auto set = LoadDir(dir);
		CHECK(!set.Usable() && HasProblem(set, "line 1: a port program without port keys"));
		fs::remove_all(dir);
	}
	{
		const fs::path dir = MakeTemp();
		WriteSet(dir, Baseline());
		ReplaceInManifest(dir, Dxil(0x1004), "-");
		const auto set = LoadDir(dir);
		CHECK(!set.Usable() && HasProblem(set, "line 4: the DXIL source doesn't fit the origin"));
		fs::remove_all(dir);
	}

	// Sizes: zero, not whole words, absurd
	for (const char* size : { "0", "122", "99999999999" })
	{
		const fs::path dir = MakeTemp();
		auto programs = Baseline();
		WriteSet(dir, programs);
		ReplaceInManifest(dir, "\t" + std::to_string(programs[0].module.size()) + "\t", std::string("\t") + size + "\t");
		const auto set = LoadDir(dir);
		CHECK(!set.Usable() && HasProblem(set, "line 1: the size isn't a plausible SPIR-V size"));
		fs::remove_all(dir);
	}

	// push constants and bindings out of range
	{
		const fs::path dir = MakeTemp();
		WriteSet(dir, Baseline());
		ReplaceInManifest(dir, "\t16\t2\t8x8x1", "\t5000\t2\t8x8x1");
		CHECK(HasProblem(LoadDir(dir), "line 1: the push constant size isn't plausible"));
		ReplaceInManifest(dir, "\t5000\t2\t8x8x1", "\t16\t2000\t8x8x1");
		CHECK(HasProblem(LoadDir(dir), "line 1: the binding count isn't plausible"));
		fs::remove_all(dir);
	}

	// The same id twice, the same DXIL program twice
	{
		const fs::path dir = MakeTemp();
		WriteSet(dir, Baseline());
		ReplaceInManifest(dir, "7403\t", "7402\t");
		const auto set = LoadDir(dir);
		CHECK(!set.Usable() && HasProblem(set, "lists id 7402 twice"));
		fs::remove_all(dir);
	}
	{
		const fs::path dir = MakeTemp();
		WriteSet(dir, Baseline());
		ReplaceInManifest(dir, Dxil(0x1005), Dxil(0x1004));
		const auto set = LoadDir(dir);
		CHECK(!set.Usable() && HasProblem(set, "translates a DXIL program another one already does"));
		const auto* first = set.FindByDxil(Dxil(0x1004));
		CHECK(first && first->id == 7403);
		fs::remove_all(dir);
	}

	// Many problems are capped, a long manifest is cut off
	{
		const fs::path dir = MakeTemp();
		fs::create_directories(dir);
		std::string text;
		for (int i = 0; i < 40; i++)
			text += "garbage\r\n";
		WriteAll(dir / "programs.tsv", text);
		const auto set = LoadDir(dir);
		CHECK(!set.Usable() && set.problems.size() == 17 && set.problems.back() == "and more");
		fs::remove_all(dir);
	}
	{
		const fs::path dir = MakeTemp();
		fs::create_directories(dir);
		const auto module = Module(5, 10);
		std::string text;
		for (uint32_t i = 1; i <= 4097; i++)
		{
			Prog p{ i, module, 16, 2, "8x8x1", Dxil(i), "intel", "-" };
			text += RowText(p) + "\r\n";
			WriteAll(dir / (std::to_string(i) + ".spv"), module);
		}
		WriteAll(dir / "programs.tsv", text);
		const auto set = LoadDir(dir);
		CHECK(!set.Usable() && set.programs.size() == 4096 && HasProblem(set, "more than 4096 programs"));
		fs::remove_all(dir);
	}
}

// The input-copy shaders built into the dll: SPIR-V, the SHA-256 they're pinned to, and the bytes a little endian host
// reads them as. `dir` is a folder the extractor made, whose input_copy_<n>.spv they have to equal
static void CheckInputCopy(const fs::path& dir = {})
{
	CHECK(VulkanInputCopy::Count == 3);

	std::set<std::string> distinct;

	for (size_t i = 0; i < VulkanInputCopy::Count; i++)
	{
		const auto* words = VulkanInputCopy::Modules[i];
		const size_t bytes = VulkanInputCopy::Bytes[i];

		CHECK(bytes > 1000 && bytes < 4000 && bytes % 4 == 0);
		CHECK(words[0] == 0x07230203u); // the SPIR-V magic number

		std::vector<uint8_t> module(bytes);
		std::memcpy(module.data(), words, bytes);

		const auto hash = Hex(module);
		CHECK(IsHex(VulkanInputCopy::Sha256[i]));
		CHECK(hash == VulkanInputCopy::Sha256[i]);
		distinct.insert(hash);

		if (!dir.empty())
		{
			const auto file = ReadAll(dir / ("input_copy_" + std::to_string(i) + ".spv"));
			CHECK(file == module);
		}
	}

	CHECK(distinct.size() == 3);
	// MFG-Unlocker's generator pins these three
	CHECK(std::string(VulkanInputCopy::Sha256[0]) == "f6f48f06d01134d9596b117e0c98b4a1a098325ab5b67268eedee8554955b7ca");
	CHECK(std::string(VulkanInputCopy::Sha256[1]) == "112a67e0df42b09edab40be77f505d58f5b2d3b357bda64a45f145fb247ce0d3");
	CHECK(std::string(VulkanInputCopy::Sha256[2]) == "2e34c018e32f1a5fd819e93e2e46cf31c454312c8d3f53ae22d491d2e1470fbc");
}

static void CheckPathText()
{
	// UTF-8 whatever the platform's own encoding, never an exception
	const fs::path viet(std::u8string(u8"/tmp/Tiếng Việt"));
	CHECK(VulkanPrograms::PathText(viet) == "/tmp/Ti" "\xE1\xBA\xBF" "ng Vi" "\xE1\xBB\x87" "t");
	CHECK(VulkanPrograms::PathText("plain/ascii.spv") == "plain/ascii.spv");
	CHECK(VulkanPrograms::PathText(fs::path()).empty());
}

static void CheckDirectoryFor()
{
	const auto where = [](const char* programs) { return VulkanPrograms::DirectoryFor(programs).generic_string(); };

	// xefg_mfg_programs next to the dll holds the DXIL programs.tsv, the Vulkan set is its vulkan folder
	CHECK(where("g/xefg_mfg_programs") == "g/xefg_mfg_programs/vulkan");

	// MFG-Unlocker's build input layout has the DXIL set in xefg_mfg_programs\programs, the Vulkan one stays next to it
	CHECK(where("g/xefg_mfg_programs/programs") == "g/xefg_mfg_programs/vulkan");

	// Only a folder called programs counts
	CHECK(where("g/xefg_mfg_programs/programs2") == "g/xefg_mfg_programs/programs2/vulkan");
	CHECK(where("g/programs/xefg_mfg_programs") == "g/programs/xefg_mfg_programs/vulkan");
	CHECK(where("programs") == "vulkan");
	CHECK(where("") == "vulkan");

	// A set next to a nested DXIL set loads
	const fs::path root = MakeTemp();
	WriteSet(root / "xefg_mfg_programs" / "vulkan", Baseline());
	fs::create_directories(root / "xefg_mfg_programs" / "programs");
	WriteAll(root / "xefg_mfg_programs" / "programs" / "programs.tsv", std::string("1\t2\r\n"));

	const auto set = LoadDir(VulkanPrograms::DirectoryFor(root / "xefg_mfg_programs" / "programs"));
	CHECK(set.Usable() && set.programs.size() == 6);

	fs::remove_all(root);
}

static void CheckInstalled()
{
	const fs::path root = MakeTemp();
	programsDirectory = root / "xefg_mfg_programs";
	WriteSet(programsDirectory / "vulkan", Baseline());

	CHECK(VulkanPrograms::Loaded() == nullptr);

	const auto& set = VulkanPrograms::Installed();
	CHECK(set.Usable() && set.programs.size() == 6);
	CHECK(set.directory == programsDirectory / "vulkan");
	CHECK(&VulkanPrograms::Installed() == &set);
	CHECK(VulkanPrograms::Loaded() == &set);
	CHECK(Logged("info: Vulkan models: 6 programs"));
	CHECK(Logged("xefg_mfg_programs/vulkan"));

	// Loaded once: files changed afterwards don't change the answer
	fs::remove_all(programsDirectory / "vulkan");
	CHECK(VulkanPrograms::Installed().Usable());

	fs::remove_all(root);
}

static int CheckInstalledMissing()
{
	const fs::path root = MakeTemp();
	programsDirectory = root / "xefg_mfg_programs";
	fs::create_directories(programsDirectory);
	// The DXIL set's own programs.tsv is one folder up and must not be mistaken for the Vulkan one
	WriteAll(programsDirectory / "programs.tsv", std::string("1\t2\r\n"));

	CHECK(VulkanPrograms::Loaded() == nullptr);
	const auto& set = VulkanPrograms::Installed();
	CHECK(!set.present && !set.Usable() && set.problems.empty());
	CHECK(VulkanPrograms::Loaded() == &set);
	CHECK(Logged("info: Vulkan models: none in"));
	CHECK(Logged("tools/ExtractXeFgVulkanPrograms.py"));

	fs::remove_all(root);

	std::printf("%d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}

// The folder tools/ExtractXeFgVulkanPrograms.py made from an MFGUnlocker.asi
static int RealSet(const fs::path& dir)
{
	const auto set = LoadDir(dir);

	for (const auto& problem : set.problems)
		std::printf("problem: %s\n", problem.c_str());

	CHECK(set.present && set.Usable());
	CHECK(set.programs.size() == 83);
	CHECK(set.perOrigin[(size_t)VulkanPrograms::Origin::Port] == 26);
	CHECK(set.perOrigin[(size_t)VulkanPrograms::Origin::PortExtra] == 1);
	CHECK(set.perOrigin[(size_t)VulkanPrograms::Origin::Intel] == 56);
	CHECK(set.perOrigin[(size_t)VulkanPrograms::Origin::Unhashed] == 0);
	CHECK(set.manifestSha256 == "b2e2822b3281c31fc1f71e048e29ce283fc4e83e7f1b65651489ab1d9da5b37e");
	CHECK(!set.build.empty() && IsHex(set.asiSha256));

	// 83 programs have a DXIL source, each one is found by it and no two share one
	size_t withDxil = 0;
	std::set<std::string> modules;
	for (const auto& program : set.programs)
	{
		modules.insert(program.sha256);

		if (program.dxilSha256.empty())
			continue;

		withDxil++;
		const auto* found = set.FindByDxil(program.dxilSha256);
		CHECK(found == &program);
		CHECK(!VulkanPrograms::ReadModule(program, &Hash).empty());
	}

	// Every program has its descriptors, and the first port program is the one the extractor's own table says
	for (const auto& program : set.programs)
		CHECK(program.bindingRows.size() == program.bindings);

	using VulkanPrograms::Binding;
	using VulkanPrograms::BindingType;
	const auto* first = set.FindById(7400);
	CHECK(first && first->bindingRows.size() == 3 && first->bindingRows[0] == (Binding{ 0, BindingType::SampledImage }) &&
		first->bindingRows[1] == (Binding{ 1, BindingType::StorageBuffer }) &&
		first->bindingRows[2] == (Binding{ 64, BindingType::StorageBuffer }));

	CHECK(first->dxilSha256 == "00b1098a80a5e662de426284e6bf7bdf82ed236eb7c8aef1e74f9ebed204019a");
    CHECK(set.FindByDxil("6c9bf4b59956c7fec3d764ffe6520e7aec3e6137553b9fca0ba65ecdd7642ce8")->id == 7434);
    CHECK(withDxil == 83);
	CHECK(set.byDxil.size() == 83);
	CHECK(modules.size() == 82);   // two ids share one module
	CHECK(set.totalBytes > 5000000 && set.totalBytes < 6000000);

	// The folder carries the input-copy shaders too, the ones built into the dll must be those
	CheckInputCopy(dir);

	std::printf("%s\n", VulkanPrograms::Summary(set).c_str());
	std::printf("%d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}

int main(int argc, char** argv)
{
	if (argc >= 2 && std::string(argv[1]) == "--missing")
		return CheckInstalledMissing();

	if (argc >= 3 && std::string(argv[1]) == "--real")
		return RealSet(argv[2]);

	CheckSha256();
	CheckBaseline();
	CheckMissingAndEmpty();
	CheckBrokenFiles();
	CheckBrokenRows();
	CheckInputCopy();
	CheckPathText();
	CheckDirectoryFor();
	CheckInstalled();

	std::printf("%d checks, %d failed\n", checks, failures);
	return failures ? 1 : 0;
}
