#include "vulkan/VulkanProgramLookup.h"
#include <cassert>
#include <map>
#include <cstdio>
#include <vector>
int main() {
    const std::string intel = "c3649191906bdd300462dbafe6add8f82b09e8e10c41e92784a209d99513224b";
    const std::string replacement = "6c9bf4b59956c7fec3d764ffe6520e7aec3e6137553b9fca0ba65ecdd7642ce8";
    std::map<std::string, size_t> index{{replacement, 34}};
    assert(xvk::ResolveProgramHash(index, intel) == replacement);
    assert(xvk::ResolveProgramHash(index, replacement) == replacement);
    index[intel] = 35;
    assert(xvk::ResolveProgramHash(index, intel) == intel);
    assert(xvk::ResolveProgramHash(index, replacement) == replacement);
    index.erase(replacement);
    assert(xvk::ResolveProgramHash(index, replacement) == intel);
    index.clear();
    assert(xvk::ResolveProgramHash(index, intel) == intel);
    index[replacement] = 34;
    assert(xvk::ResolveProgramHash(index, "unknown") == "unknown");
    // Dynamic runtime DXIL replacement aliases (e.g. dxc-compiled splat_motion_aware.dxil):
    const std::string regularIntel = "3f3f91d2ecd4345042f0519f7455776314be9e7730453a3fcb42c57f0acb49b5";
    const std::string dxcSplat = "5cc613d4b028f1cec8a380ca3067b8f44b4917902c0abc511668c8694a34ced1";
    const std::string dxcSplatInv = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const std::vector<std::pair<std::string, std::string>> dynamicAliases{
        {dxcSplat, regularIntel},
        {dxcSplatInv, intel},
    };
    // 1. Direct match via native Intel hash in index
    index.clear();
    index[regularIntel] = 12;
    assert(xvk::ResolveProgramHash(index, dxcSplat, dynamicAliases) == regularIntel);
    // 2. Chained match through kReplacements when payload ships only MFG-Unlocker's pinned inverted splat
    index.clear();
    index[replacement] = 34;
    assert(xvk::ResolveProgramHash(index, dxcSplatInv, dynamicAliases) == replacement);
    // 3. Exact hash still wins if a future payload ships a dedicated SPIR-V for the dxc-compiled replacement
    index[dxcSplat] = 99;
    index[regularIntel] = 12;
    assert(xvk::ResolveProgramHash(index, dxcSplat, dynamicAliases) == dxcSplat);
    // 4. Unknown hash still fails closed
    assert(xvk::ResolveProgramHash(index, "unknown", dynamicAliases) == "unknown");

    puts("Program identity: exact hash, both alias directions, dynamic replacement aliases, missing/unknown: PASS");
}
