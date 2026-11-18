#include "vulkan/VulkanProgramLookup.h"
#include <cassert>
#include <map>
#include <cstdio>
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
    puts("Program identity: exact hash, both alias directions, missing/unknown: PASS");
}
