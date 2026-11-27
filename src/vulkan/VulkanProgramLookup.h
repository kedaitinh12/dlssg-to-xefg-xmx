#pragma once
#include <string>
#include <utility>

namespace xvk {
// Only these pinned byte-equivalent Intel/replacement pairs may share a translation.
// Exact hash wins; some payloads contain only the replacement (inverted-depth splat).
constexpr std::pair<const char*, const char*> kReplacements[] = {
    {"43ccfe670409fd807b604b870674012958fbb4067305a449304ee91ab6154f3f", "3f3f91d2ecd4345042f0519f7455776314be9e7730453a3fcb42c57f0acb49b5"},
    {"6c9bf4b59956c7fec3d764ffe6520e7aec3e6137553b9fca0ba65ecdd7642ce8", "c3649191906bdd300462dbafe6add8f82b09e8e10c41e92784a209d99513224b"},
    {"9900c30febf95bfa81b4da1178e5773863116beba4d3f032897da4254b6d8926", "01c8beac27d3f3f82ecba8f3dc2d9d8b60d4c175a41df3c24a24e8643f5578f5"},
    {"e78063f97ad2c0913412f07589cd17d0f2a6ebeeb6edd70647a69cca6b6a02bf", "8291bbc36b1100b07d0788f35a3a9fadfbfcc2267ff0ccb7afb114d76f5d69fe"},
    {"7a29fbf311cd8dee5b8c1cfdadf31ee983092a512cf9c10845a079b457eb62a9", "a60f7488872a675d66afd2bec02b0b89e3cc38c1c4ed1d25d745a7b2fe677412"},
    {"ce847b58086ce7e9d25f7d8b90e989247cd4582226506d9552112f1ef4130643", "8c2efb9e955f39379a91887ed3d90412c4b7ae2c142a874e3da5f699de197f7c"},
    {"efa8da5ef5f90c62b789b0fed8e77e2ce043e558d164cc552ca7fb9127cedb00", "40e159414d5b839ba7e0e07cbf8ff4ef425ee026c216b2928ef3bb6dd0e20d40"}};

template<class Index>
std::string ResolveProgramHash(const Index& index, const std::string& requested)
{
    if (index.find(requested) != index.end()) return requested;
    for (const auto& [replacement, intel] : kReplacements) {
        if (requested == replacement && index.find(intel) != index.end()) return intel;
        if (requested == intel && index.find(replacement) != index.end()) return replacement;
    }
    return requested;
}

template<class Index, class Aliases>
std::string ResolveProgramHash(const Index& index, const std::string& requested, const Aliases& extraAliases)
{
    const std::string resolved = ResolveProgramHash(index, requested);
    if (index.find(resolved) != index.end()) return resolved;
    for (const auto& [replacement, intel] : extraAliases) {
        if (requested == replacement) {
            const std::string viaIntel = ResolveProgramHash(index, intel);
            if (index.find(viaIntel) != index.end()) return viaIntel;
        }
        if (requested == intel) {
            const std::string viaRep = ResolveProgramHash(index, replacement);
            if (index.find(viaRep) != index.end()) return viaRep;
        }
    }
    return requested;
}
} // namespace xvk
