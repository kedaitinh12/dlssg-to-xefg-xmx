// The Vulkan models: the SPIR-V programs MFG-Unlocker's xvk layer (a D3D12 API on Vulkan) dispatches for XeFG on a
// VkDevice, in place of the DXIL programs XeFG hands to CreateComputePipelineState. They are matched by the SHA-256 of
// that DXIL, so every program here says which DXIL it translates (sourceDxilSha256).
//
// They live in xefg_mfg_programs\vulkan next to this dll, laid out by tools/ExtractXeFgVulkanPrograms.py from an
// MFGUnlocker.asi: programs.tsv, one <id>.spv per program and SOURCE.txt. Like the DXIL set they are Intel derived program
// data, so they're a build artifact and not in the repository.
//
// This file reads and verifies the set, it doesn't run it: src/vulkan/ does (MFG-Unlocker's D3D12-on-Vulkan layer, built
// into this dll). Nothing below needs Windows.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace VulkanPrograms
{
	// The origin column of programs.tsv
	enum class Origin : uint8_t
	{
		Port,      // a translation of a program in the XMX Port set (xefg_mfg_programs\programs.tsv)
		PortExtra, // a translation of one of MFG-Unlocker's own replacement programs
		Intel,     // a translation of one of Intel's XeFG programs, as XeFG creates it
		Unhashed,  // listed without a DXIL source
		Unknown,   // an origin this reader doesn't know, the set is refused
	};

	constexpr size_t OriginCount = 4;

	// What kind of descriptor a binding of a program is (the type column of bindingTypes, xvk's BindingRow)
	enum class BindingType : uint8_t
	{
		StorageBuffer,
		UniformBuffer,
		SampledImage,
		StorageImage,
		Sampler,
	};

	constexpr size_t BindingTypeCount = 5;

	// One descriptor of a program: the binding number (D3D12 SRVs from 0, UAVs from 64, CBVs from 128, samplers from 192)
	// and its type, in binding order
	struct Binding
	{
		uint16_t binding = 0;
		BindingType type = BindingType::StorageBuffer;

		bool operator==(const Binding&) const = default;
	};

	struct Program
	{
		uint32_t id = 0;
		std::string sha256;     // of the SPIR-V module, lower case hex
		uint32_t bytes = 0;
		uint32_t pushBytes = 0; // push constant size
		uint32_t bindings = 0;  // bindingRows.size()
		std::vector<Binding> bindingRows;
		std::string localSize;  // "8x8x1", empty when the manifest has none
		std::string dxilSha256; // of the DXIL program XeFG creates, empty when the manifest lists none
		Origin origin = Origin::Unknown;
		std::string portKeys;   // keys of the XMX Port programs it translates, empty when none
		std::filesystem::path file;
	};

	// SHA-256 of a buffer, false on failure. Passed in so the reader doesn't depend on a platform API
	using Sha256Fn = bool (*)(const void* data, size_t size, std::array<uint8_t, 32>& digest);

	struct Set
	{
		std::filesystem::path directory;
		std::vector<Program> programs;
		std::array<size_t, OriginCount> perOrigin{};
		uint64_t totalBytes = 0;
		std::string manifestSha256; // programs.tsv, lower case hex
		std::string build;          // "1.7.1-rc.2 (file version 1.7.1.173)" from SOURCE.txt, empty without it
		std::string asiSha256;      // of the MFGUnlocker.asi it came from, empty without SOURCE.txt
		bool present = false;       // the folder has a programs.tsv
		std::vector<std::string> problems; // why the set can't be used, empty when every program checked out

		// Every program is there and verified
		bool Usable() const { return present && problems.empty() && !programs.empty(); }

		const Program* FindById(uint32_t id) const;
		// The program that replaces the DXIL program with this SHA-256 (lower case hex), null when there's none
		const Program* FindByDxil(std::string_view sha256) const;

		// dxilSha256 -> index into programs. Two programs can share a SPIR-V module, no two translate the same DXIL
		std::map<std::string, size_t, std::less<>> byDxil;
	};

	// Reads directory\programs.tsv and checks every program: the file is there, has the size the manifest says, is a
	// SPIR-V module and hashes to the manifest's SHA-256. Never throws. A folder without programs.tsv gives
	// present == false and no problems, that's the common case of a build that doesn't carry the Vulkan models.
	Set Load(const std::filesystem::path& directory, Sha256Fn sha256);

	// The SPIR-V of a program, read again and checked against its hash. Empty when it no longer matches
	std::vector<uint8_t> ReadModule(const Program& program, Sha256Fn sha256);

	// One line for the log and the menu
	std::string Summary(const Set& set);

	// A path as UTF-8 text, for the log and the menu. path::string() converts to the ANSI code page: that fails for a
	// character the page doesn't have, and the menu draws UTF-8
	std::string PathText(const std::filesystem::path& path);

	// Where the set is looked for, given the folder the DXIL programs are read from: its xefg_mfg_programs\vulkan. When
	// that folder is xefg_mfg_programs\programs (the layout of MFG-Unlocker's build input) it is still the one next to it
	std::filesystem::path DirectoryFor(const std::filesystem::path& programsDirectory);

	// The set next to this dll (xefg_mfg_programs\vulkan), loaded once on the first call and logged. Loaded lazily on
	// demand (or preloaded in the background via PreloadAsync() once vulkan-1.dll is present) so pure DX12 games do not
	// pay the disk I/O and SHA-256 verification cost at start-up
	const Set& Installed();

	// Starts loading Installed() on a background thread at most once (no-op if already loaded). Call when vulkan-1.dll
	// is present or has just been loaded so models are ready before the game creates its Vulkan device
	void PreloadAsync() noexcept;

	// The same set once Installed() has loaded it, null before. For the menu, which must not be the one that loads it
	const Set* Loaded();
} // namespace VulkanPrograms
