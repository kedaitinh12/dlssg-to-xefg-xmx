#!/usr/bin/env python3
"""Copies MFG-Unlocker's D3D12-on-Vulkan layer (xvk) and MinHook into this repository (standard library only).

    VendorXvk.py MFG-UNLOCKER-DIR            copy and patch, writes src/vulkan and third_party/minhook
    VendorXvk.py MFG-UNLOCKER-DIR --check    only compare: exit 1 when the files in the repository aren't what a copy
                                             of that source would give

MFG-UNLOCKER-DIR is the root of a MFG-Unlocker source tree (it has source/native/xess/vulkan). xvk is MIT licensed
(Copyright (c) 2026 Michael Robles), MinHook is BSD-2-Clause, both are GPLv3 compatible; their licences are copied next
to the files and named in NOTICE.md. What changes in xvk is small and listed below, every edit has to apply exactly once:
a new MFG-Unlocker that moved those lines makes this script stop, instead of vendoring something half patched.

The tensor-core route comes along: xvk_tensor.{h,cpp} (CUDA kernels launched through VK_NVX_binary_import in the layer's own
command buffer) and cuda_layers.h, the layer table both routes share, which goes to src/MfgPort/Cuda next to the D3D12
route's files. Only their includes change.

What isn't copied:
  - xvk_embedded.h and the embedded program set: the programs come from xefg_mfg_programs\\vulkan (src/VulkanPrograms.h).
  - vk_provider.cpp, vk_host.cpp and the rest: the provider and the host are this repository's own code, in
    src/vulkan/VulkanRuntime.cpp, VulkanHost.cpp and VulkanTensor.cpp, written against the mod's XeFG loader and NGX types.
  - xess/cuda/cuda_route.cpp: the D3D12 route of this repository is its own older port; the few helpers the Vulkan route
    shares with it (Report, AdapterCapability, the layer mask and the status) are in src/MfgPort/Cuda/cuda_route.cpp and
    cuda_vulkan.cpp, and tests/vulkan_runtime/check_layer_tables.py keeps the two layer tables the same.
"""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

XVK_DIR = os.path.join("source", "native", "xess", "vulkan")
CUDA_DIR = os.path.join("source", "native", "xess", "cuda")
MINHOOK_DIR = os.path.join("source", "native", "third_party", "minhook")

# Copied as they are
VERBATIM = (
    "xvk.h",
    "xvk_internal.h",
    "xvk_output_frame.h",
    "xvk_commandlist.cpp",
    "xvk_dxgi.cpp",
    "xvk_pipeline.cpp",
    "xvk_queue.cpp",
    "xvk_resource.cpp",
)

MINHOOK_FILES = (
    "LICENSE.txt",
    "include/MinHook.h",
    "src/buffer.c",
    "src/buffer.h",
    "src/hook.c",
    "src/trampoline.c",
    "src/trampoline.h",
    "src/hde/hde64.c",
    "src/hde/hde64.h",
    "src/hde/pstdint.h",
    "src/hde/table64.h",
)

# file -> [(old, new)], each old text occurs once. The bare names of this dll's files are reached with "../" (the
# project has no include directory for src)
PATCHES = {
    # The tensor-core route: its layer table is src/MfgPort/Cuda/cuda_layers.h, the one the D3D12 route's files are in
    "xvk_device.cpp": [
        ('#include "../cuda/cuda_layers.h"\n', '#include "../MfgPort/Cuda/cuda_layers.h"\n'),
    ],
    "xvk_tensor.cpp": [
        ('#include "../cuda/cuda_layers.h"\n', '#include "../MfgPort/Cuda/cuda_layers.h"\n'),
    ],
    "xvk_tensor.h": [
        ('#include "../cuda/xefg_cuda_params.h"\n', '#include "../MfgPort/Cuda/xefg_cuda_params.h"\n'),
    ],
    # The log is this dll's, not RTXMFG's
    "xvk_util.cpp": [
        ('#include "single_module.h"\n', '#include "../Log.h"\n'),
        ("""    wchar_t wide[1100];
    swprintf_s(wide, L"XVK #%llu %hs", line, text);
    single_module::Log(wide);
""", """    ::Log::Write("info", std::format("xvk #{} {}", line, text));
"""),
    ],
    # The programs are the ones read from xefg_mfg_programs\vulkan, not embedded in the dll
    "xvk_programs.cpp": [
        ('#include "embedded_xvk_programs.generated.h"\n#include "xvk_programs.packed.h"\n',
         '#include "../MfgPort/Sha256.h"\n#include "../VulkanPrograms.h"\n'),
        ("""    module_ = module;
    for (size_t i = 0; i < std::size(embedded::programs); ++i) index_.emplace(embedded::programs[i].dxilSha256, i);
    Log("program set: %zu embedded programs, manifest %.16s", index_.size(), embedded::manifestSha256);
""", """    // dlssg-to-xefg: the program set is the one in xefg_mfg_programs\\vulkan (src/VulkanPrograms.h), read and verified once.
    // index_ maps the SHA-256 of the DXIL XeFG creates to the row of that set
    module_ = module;
    const auto& set = VulkanPrograms::Installed();
    if (set.Usable())
        for (const auto& [dxil, row] : set.byDxil) index_.emplace(std::string(dxil), row);
    Log("program set: %zu programs from xefg_mfg_programs\\\\vulkan, manifest %.16s", index_.size(), set.manifestSha256.c_str());
"""),
        ("""    const embedded::ProgramAsset& asset = embedded::programs[row->second];
    std::unique_ptr<Program> program;
    const auto payload = embedded_payload::Find(module_, embedded_payload::packed::xvk_programs,
        embedded_payload::packed::xvk_programs_items, asset.resource);
    const auto* bytes = payload.data();
    const DWORD size = static_cast<DWORD>(payload.size());
    if (!bytes || size != asset.spirvBytes || size % 4 || Sha256(bytes, size) != asset.spirvSha256) {
        Log("program set: embedded program %.16s failed verification (resource %u, %lu bytes)", asset.dxilSha256, asset.resource, size);
    } else {
        program = std::make_unique<Program>();
        program->sha256 = asset.dxilSha256;
        program->spirv.resize(size / 4);
        memcpy(program->spirv.data(), bytes, size);
        program->pushConstantBytes = asset.pushConstantBytes;
        static constexpr VkDescriptorType types[] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                     VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER};
        for (uint32_t i = 0; i < asset.bindingCount; ++i) {
            const embedded::BindingRow& b = embedded::bindings[asset.firstBinding + i];
            program->bindings.push_back({b.binding, types[b.type]});
        }
    }
""", """    // dlssg-to-xefg: the SPIR-V is read from its file again and checked against the hash the manifest lists
    const auto& asset = VulkanPrograms::Installed().programs[row->second];
    std::unique_ptr<Program> program;
    const auto bytes = VulkanPrograms::ReadModule(asset, &sha256_util::Compute);
    if (bytes.empty() || bytes.size() % 4) {
        Log("program set: program %u (%.16s) failed verification (%zu bytes)", asset.id, asset.dxilSha256.c_str(), bytes.size());
    } else {
        program = std::make_unique<Program>();
        program->sha256 = asset.dxilSha256;
        program->spirv.resize(bytes.size() / 4);
        memcpy(program->spirv.data(), bytes.data(), bytes.size());
        program->pushConstantBytes = asset.pushBytes;
        static constexpr VkDescriptorType types[] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                     VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER};
        for (const auto& b : asset.bindingRows)
            program->bindings.push_back({b.binding, types[static_cast<size_t>(b.type)]});
    }
"""),
        ("size_t EmbeddedProgramCount(HMODULE) noexcept { return std::size(embedded::programs); }\n",
         "size_t EmbeddedProgramCount(HMODULE) noexcept { return VulkanPrograms::Installed().programs.size(); }\n"),
    ],
}

# The layer table of the tensor-core route, copied as it is: file of source/native/xess/cuda -> file in src/MfgPort/Cuda.
# Its two includes (cuda_route.h, xefg_cuda_params.h) are this repository's own files of the same names in that folder
CUDA_VERBATIM = (
    "cuda_layers.h",
)

# Everything else of src/vulkan/ that came from MFG-Unlocker besides the files above
LICENSE_SOURCE = "LICENSE"
LICENSE_TARGET = "LICENSE-MFG-Unlocker.txt"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    with open(path, "rb") as file:
        return file.read()


def build(source):
    """{relative path in the repository: bytes} for a MFG-Unlocker tree, and the provenance lines"""
    files = {}
    lines = []
    xvk = os.path.join(source, XVK_DIR)

    if not os.path.isdir(xvk):
        raise SystemExit("%s has no %s" % (source, XVK_DIR))

    for name in sorted(set(VERBATIM) | set(PATCHES)):
        original = read(os.path.join(xvk, name))
        try:
            text = original.decode("utf-8")
        except UnicodeDecodeError:
            raise SystemExit("%s isn't UTF-8 text" % name)

        patched = False
        for old, new in PATCHES.get(name, ()):
            if text.count(old) != 1:
                raise SystemExit("%s: the text to change occurs %d times instead of once, MFG-Unlocker moved it:\n%s" %
                                 (name, text.count(old), old))
            text = text.replace(old, new)
            patched = True

        data = text.encode("utf-8")
        files["src/vulkan/" + name] = data
        lines.append("%s  %s  %s" % (name, sha256(original), "patched, now " + sha256(data)[:16] if patched else "as is"))

    cuda = os.path.join(source, CUDA_DIR)

    for name in CUDA_VERBATIM:
        data = read(os.path.join(cuda, name))
        files["src/MfgPort/Cuda/" + name] = data
        lines.append("MfgPort/Cuda/%s  %s  as is" % (name, sha256(data)))

    files["src/vulkan/" + LICENSE_TARGET] = read(os.path.join(source, LICENSE_SOURCE))
    lines.append("%s  %s  as is" % (LICENSE_TARGET, sha256(files["src/vulkan/" + LICENSE_TARGET])))

    minhook = os.path.join(source, MINHOOK_DIR)
    for name in MINHOOK_FILES:
        data = read(os.path.join(minhook, *name.split("/")))
        files["third_party/minhook/" + name] = data
        lines.append("minhook/%s  %s  as is" % (name, sha256(data)))

    return files, lines


def header(source, lines):
    version = ""
    changelog = os.path.join(source, "CHANGELOG.md")
    if os.path.exists(changelog):
        with open(changelog, encoding="utf-8", errors="replace") as file:
            for line in file:
                if line.startswith("## "):
                    version = line.strip("# \r\n")
                    break

    return "\n".join([
        "The files of src/vulkan/, src/MfgPort/Cuda/cuda_layers.h and third_party/minhook/ that came from MFG-Unlocker",
        "(https://github.com/dashdogy/MFG-Unlocker), copied by tools/VendorXvk.py: xvk, its D3D12 API on Vulkan, with its",
        "tensor-core route and the layer table that route shares with the D3D12 one, MIT licensed (LICENSE-MFG-Unlocker.txt),",
        "and MinHook, BSD-2-Clause (third_party/minhook/LICENSE.txt).",
        "",
        "Source: %s" % (version or "MFG-Unlocker source tree"),
        "",
        "file  SHA-256 of MFG-Unlocker's file  what this repository did",
        "",
    ] + lines + [
        "",
        "VulkanRuntime.cpp, VulkanHost.cpp, VulkanTensor.cpp, VulkanPolicy.h and VulkanInputCopy.h are this repository's own",
        "(the last holds MFG-Unlocker's xess/vulkan/vk_input_copy.comp, compiled, see tools/MakeVulkanInputCopyHeader.py), and",
        "so are the helpers the tensor-core route shares with the D3D12 one (src/MfgPort/Cuda/cuda_route.cpp, cuda_vulkan.cpp).",
        "",
    ])


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("source", help="root of a MFG-Unlocker source tree")
    parser.add_argument("--check", action="store_true", help="compare only")
    args = parser.parse_args(argv[1:])

    files, lines = build(args.source)
    files["src/vulkan/VENDORED.txt"] = header(args.source, lines).encode("utf-8")

    different = 0
    for relative, data in sorted(files.items()):
        path = os.path.join(ROOT, *relative.split("/"))

        if args.check:
            if not os.path.exists(path) or read(path) != data:
                print("differs: %s" % relative)
                different += 1
            continue

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            file.write(data)

    if args.check:
        print("%d of %d files differ" % (different, len(files)))
        return 1 if different else 0

    print("wrote %d files" % len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
