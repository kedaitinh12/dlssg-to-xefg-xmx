// Root signature (RTS0) parsing and the embedded SPIR-V program set (generate_xvk_programs.py).
#include "xvk_internal.h"
#include "VulkanProgramLookup.h"
#include "../MfgPort/Sha256.h"
#include "../VulkanPrograms.h"
#include <map>
#include <set>

namespace xvk {

namespace {

template <class T>
bool Read(const uint8_t* data, size_t size, size_t offset, T& out)
{
    if (offset > size || size - offset < sizeof(T)) return false;
    memcpy(&out, data + offset, sizeof(T));
    return true;
}

bool FindPart(const uint8_t* data, size_t size, uint32_t fourcc, const uint8_t*& part, size_t& partSize)
{
    uint32_t magic = 0, count = 0;
    if (!Read(data, size, 0, magic) || magic != 0x43425844 /* DXBC */) return false;
    if (!Read(data, size, 28, count)) return false;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t offset = 0, code = 0, length = 0;
        if (!Read(data, size, 32 + 4 * i, offset) || !Read(data, size, offset, code) || !Read(data, size, offset + 4, length)) return false;
        if (code != fourcc) continue;
        if (offset + 8 > size || size - offset - 8 < length) return false;
        part = data + offset + 8;
        partSize = length;
        return true;
    }
    return false;
}

uint32_t ClassBase(D3D12_DESCRIPTOR_RANGE_TYPE type)
{
    switch (type) {
    case D3D12_DESCRIPTOR_RANGE_TYPE_SRV: return kSrvBase;
    case D3D12_DESCRIPTOR_RANGE_TYPE_UAV: return kUavBase;
    case D3D12_DESCRIPTOR_RANGE_TYPE_CBV: return kCbvBase;
    default: return kSamplerBase;
    }
}

bool Compatible(D3D12_DESCRIPTOR_RANGE_TYPE range, VkDescriptorType type)
{
    switch (range) {
    case D3D12_DESCRIPTOR_RANGE_TYPE_SRV: return type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case D3D12_DESCRIPTOR_RANGE_TYPE_UAV: return type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case D3D12_DESCRIPTOR_RANGE_TYPE_CBV: return type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    default: return type == VK_DESCRIPTOR_TYPE_SAMPLER;
    }
}

bool MatchesRootLayout(const Program& program, const RootLayout& root)
{
    std::map<uint32_t, VkDescriptorType> declared;
    for (const auto& b : program.bindings) declared[b.binding] = b.type;
    std::map<uint32_t, bool> covered;
    uint32_t pushBytes = 0;
    int constantsParam = -1;

    for (uint32_t p = 0; p < root.parameters.size(); ++p) {
        const auto& parameter = root.parameters[p];
        if (parameter.type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
            if (constantsParam >= 0 || parameter.num32 > 64) return false;
            constantsParam = int(p);
            pushBytes = parameter.num32 * 4;
            continue;
        }
        if (parameter.type != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) return false;
        for (const auto& range : parameter.ranges) {
            if (range.space) return false;
            for (uint32_t i = 0; i < range.count; ++i) {
                const uint32_t binding = ClassBase(range.type) + range.baseRegister + i;
                auto it = declared.find(binding);
                if (it == declared.end()) continue;
                if (!Compatible(range.type, it->second)) return false;
                covered[binding] = true;
            }
        }
    }
    if (program.pushConstantBytes > pushBytes || constantsParam >= 16) return false;
    for (const auto& [binding, type] : declared) {
        if (type == VK_DESCRIPTOR_TYPE_SAMPLER && !covered[binding]) {
            for (const auto& s : root.samplers) {
                if (kSamplerBase + s.desc.ShaderRegister == binding && !s.desc.RegisterSpace) {
                    covered[binding] = true;
                    break;
                }
            }
        }
        if (!covered[binding]) return false;
    }
    return true;
}

} // namespace

bool ParseRootSignature(const void* blob, size_t size, RootLayout& out, std::string& error)
{
    const auto* data = static_cast<const uint8_t*>(blob);
    const uint8_t* rts = data;
    size_t rtsSize = size;
    if (size >= 4 && memcmp(data, "DXBC", 4) == 0 && !FindPart(data, size, 0x30535452 /* RTS0 */, rts, rtsSize)) {
        error = "container has no RTS0 part";
        return false;
    }
    uint32_t version = 0, parameterCount = 0, parameterOffset = 0, samplerCount = 0, samplerOffset = 0, flags = 0;
    if (!Read(rts, rtsSize, 0, version) || !Read(rts, rtsSize, 4, parameterCount) || !Read(rts, rtsSize, 8, parameterOffset) ||
        !Read(rts, rtsSize, 12, samplerCount) || !Read(rts, rtsSize, 16, samplerOffset) || !Read(rts, rtsSize, 20, flags)) {
        error = "truncated RTS0 header";
        return false;
    }
    if (version < 1 || version > 3) { error = "unsupported RTS0 version " + std::to_string(version); return false; }
    out = {};
    for (uint32_t i = 0; i < parameterCount; ++i) {
        uint32_t type = 0, visibility = 0, payload = 0;
        if (!Read(rts, rtsSize, parameterOffset + 12 * i, type) || !Read(rts, rtsSize, parameterOffset + 12 * i + 4, visibility) ||
            !Read(rts, rtsSize, parameterOffset + 12 * i + 8, payload)) { error = "truncated parameter"; return false; }
        RootParameter parameter;
        parameter.type = D3D12_ROOT_PARAMETER_TYPE(type);
        if (type == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
            uint32_t rangeCount = 0, rangeOffset = 0;
            if (!Read(rts, rtsSize, payload, rangeCount) || !Read(rts, rtsSize, payload + 4, rangeOffset)) { error = "truncated table"; return false; }
            const uint32_t stride = version == 1 ? 20 : 24;
            uint32_t running = 0;
            for (uint32_t r = 0; r < rangeCount; ++r) {
                const size_t at = rangeOffset + size_t(stride) * r;
                uint32_t rangeType = 0, count = 0, base = 0, space = 0, offset = 0;
                if (!Read(rts, rtsSize, at, rangeType) || !Read(rts, rtsSize, at + 4, count) || !Read(rts, rtsSize, at + 8, base) ||
                    !Read(rts, rtsSize, at + 12, space) || !Read(rts, rtsSize, at + (version == 1 ? 16 : 20), offset)) {
                    error = "truncated range";
                    return false;
                }
                if (offset == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) offset = running;
                if (count == UINT_MAX) { error = "unbounded range"; return false; }
                parameter.ranges.push_back({D3D12_DESCRIPTOR_RANGE_TYPE(rangeType), base, count, space, offset});
                running = offset + count;
            }
        } else if (type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
            if (!Read(rts, rtsSize, payload, parameter.shaderRegister) || !Read(rts, rtsSize, payload + 4, parameter.space) ||
                !Read(rts, rtsSize, payload + 8, parameter.num32)) { error = "truncated constants"; return false; }
        } else {
            if (!Read(rts, rtsSize, payload, parameter.shaderRegister) || !Read(rts, rtsSize, payload + 4, parameter.space)) {
                error = "truncated root descriptor";
                return false;
            }
        }
        out.parameters.push_back(std::move(parameter));
    }
    const uint32_t samplerStride = version == 3 ? 56 : 52;
    for (uint32_t i = 0; i < samplerCount; ++i) {
        StaticSampler sampler{};
        if (samplerOffset + size_t(samplerStride) * (i + 1) > rtsSize) { error = "truncated static sampler"; return false; }
        memcpy(&sampler.desc, rts + samplerOffset + size_t(samplerStride) * i, sizeof(D3D12_STATIC_SAMPLER_DESC));
        out.samplers.push_back(sampler);
    }
    return true;
}

void ProgramSet::Load(HMODULE module)
{
    // dlssg-to-xefg: the program set is the one in xefg_mfg_programs\vulkan (src/VulkanPrograms.h), read and verified once.
    // index_ maps the SHA-256 of the DXIL XeFG creates to the row of that set
    module_ = module;
    const auto& set = VulkanPrograms::Installed();
    if (set.Usable())
        for (const auto& [dxil, row] : set.byDxil) index_.emplace(std::string(dxil), row);
    Log("program set: %zu programs from xefg_mfg_programs\\vulkan, manifest %.16s", index_.size(), set.manifestSha256.c_str());
}


const Program* ProgramSet::LoadRow(size_t rowIdx, const std::string& key)
{
    if (auto it = loaded_.find(key); it != loaded_.end()) return it->second.get();
    const auto& programs = VulkanPrograms::Installed().programs;
    if (rowIdx >= programs.size()) return nullptr;
    const auto& asset = programs[rowIdx];
    std::unique_ptr<Program> program;
    const auto bytes = VulkanPrograms::ReadModule(asset, &sha256_util::Compute);
    if (bytes.empty() || bytes.size() % 4) {
        Log("program set: program %u (%.16s) failed verification (%zu bytes)", asset.id, key.c_str(), bytes.size());
    } else {
        program = std::make_unique<Program>();
        program->sha256 = key;
        program->spirv.resize(bytes.size() / 4);
        memcpy(program->spirv.data(), bytes.data(), bytes.size());
        program->pushConstantBytes = asset.pushBytes;
        static constexpr VkDescriptorType types[] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                     VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER};
        for (const auto& b : asset.bindingRows)
            program->bindings.push_back({b.binding, types[static_cast<size_t>(b.type)]});
    }
    const Program* result = program.get();
    loaded_.emplace(key, std::move(program));
    return result;
}

const Program* ProgramSet::Find(const std::string& requested)
{
    const std::string sha256 = ResolveProgramHash(index_, requested);
    std::lock_guard lock(mutex_);
    auto row = index_.find(sha256);
    if (row == index_.end()) return nullptr;
    return LoadRow(row->second, sha256);
}

const Program* ProgramSet::FindForRoot(const std::string& requested, const RootLayout& root)
{
    const std::string sha256 = ResolveProgramHash(index_, requested);
    std::lock_guard lock(mutex_);
    auto row = index_.find(sha256);
    if (row == index_.end()) return nullptr;
    const Program* direct = LoadRow(row->second, sha256);
    if (direct && MatchesRootLayout(*direct, root))
        return direct;

    // Descriptor compatibility cannot establish shader identity. Never substitute
    // a neighbouring program: different computations often share a root layout.
    return nullptr;
}

namespace {
std::mutex gMissingMutex;
std::set<std::string> gMissing;
std::atomic<uint64_t> gMissingCount{0};
} // namespace

void NoteMissingProgram(const std::string& sha256, const void*, size_t bytes)
{
    ++gMissingCount;
    std::lock_guard lock(gMissingMutex);
    if (gMissing.insert(sha256).second)
        Log("missing SPIR-V translation for DXIL %s (%zu bytes) - failing CreateComputePipelineState", sha256.c_str(), bytes);
}

size_t EmbeddedProgramCount(HMODULE) noexcept { return VulkanPrograms::Installed().programs.size(); }
uint64_t MissingProgramCount() noexcept { return gMissingCount.load(); }

} // namespace xvk
