# Vulkan models test

Runs the real `src/VulkanPrograms.cpp`, the reader of the Vulkan models in `xefg_mfg_programs\vulkan`, and the real
`src/MfgPort/Sha256.h` over a stand-in for CNG, on Linux (or anywhere with g++ 13 or newer). `stubs/` holds just enough of the
Win32 and CNG API for that, the test itself stands in for the log and for the folder `MfgKernels` reads the DXIL programs from.

It checks the reading and the checking: the SHA-256 against known vectors, a good synthetic set (every field, the lookup by
DXIL SHA-256, `ReadModule`), no folder, an empty folder, a missing or empty manifest, broken sets (a missing file, a wrong size,
a flipped byte, a module that isn't SPIR-V) and broken manifest rows (every column, duplicate ids, two programs for one DXIL
program, more programs than any real set has), paths with non-ASCII characters as UTF-8 text, the folder next to a nested
`xefg_mfg_programs\programs`, `Installed()` and `Loaded()`: loaded once, logged, not read again, and the three input-copy
shaders built into the dll (`src/vulkan/VulkanInputCopy.h`): SPIR-V, the SHA-256 each is pinned to, the three hashes MFG-Unlocker's
generator pins. It doesn't run a program, nothing here does, and it says nothing about a game.

```
g++ -std=c++20 -Wall -Wextra -fsanitize=address,undefined -fno-sanitize-recover=undefined -g -DVULKANPROGRAMS_HOSTED \
    -I stubs -I ../../src programs_test.cpp ../../src/VulkanPrograms.cpp -o programs_test
./programs_test            # 206 checks, 0 failed
./programs_test --missing  # Installed() without a vulkan folder, 5 checks
./programs_test --real DIR # DIR made by tools/ExtractXeFgVulkanPrograms.py, 280 checks
```

`--real` is for a folder the extractor made from a real `MFGUnlocker.asi`: 83 programs, every one verified, each of the 82 with
a DXIL source found by it, every program's descriptors, the manifest's SHA-256 and the origin counts as in the four 1.7.1 builds
(1.7.1-beta2fix.1, 1.7.1-beta.3, 1.7.1-rc.2 and 1.7.1 final, file version 1.7.1.177), and the three `input_copy_<n>.spv` of the
folder equal to the ones built into the dll. `build.yml` runs it when the repository has the models
(`third_party/xefg_mfg_programs/vulkan`) or an `MFGUnlocker.asi` (`third_party/MFGUnlocker.asi`).
