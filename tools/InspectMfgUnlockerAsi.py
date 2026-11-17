#!/usr/bin/env python3
"""Checks an MFGUnlocker.asi against what this dll relies on (standard library only, Python 3.8+).

    python tools/InspectMfgUnlockerAsi.py MFGUnlocker.asi

The dll runs Vulkan games on its own runtime, which needs the Vulkan models (xefg_mfg_programs\\vulkan) that
tools/ExtractXeFgVulkanPrograms.py makes from an asi. A build of the dll without the Vulkan headers has no runtime and
forwards the Vulkan exports to the XeMFG provider table the asi publishes (src/VulkanProvider.h), which MFG-Unlocker
doesn't document, so a new asi build can change it without notice. This reads the asi itself, the same way the
extractor does, and reports:

  * the build (version resource, SHA-256) and whether it is one the extractor was run on
  * the Vulkan program set (SPIR-V) and whether it is the one the known builds carry: this is what the verdict is about
  * the exports and the provider table MfgXeMFGProviderGetApi(1) returns (its size, version and the 12 slots, checked
    against the layout VulkanProvider.h expects), which only a build of the dll without the Vulkan runtime uses
  * a few features the build mentions, for information only

Exit code 0 when the Vulkan models can be made from the asi, 1 when they can't, 2 when the file isn't readable. Provider
table findings are warnings, they don't change the exit code.
Nothing here runs the asi, it's all read from the file.
"""
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True

import ExtractXeFgVulkanPrograms as extractor  # noqa: E402

# The table layout VulkanProvider.h reads (offsets in bytes from the start of the table)
EXPECTED_VERSION = 1
EXPECTED_SLOTS = {
    0x08: "init", 0x10: "create", 0x18: "evaluate", 0x20: "release", 0x28: "shutdown", 0x30: "supported",
    0x38: "reason",
}
REQUIRED_SLOTS = (0x08, 0x10, 0x18, 0x20, 0x28, 0x30)   # init .. supported, reason is optional
MIN_SIZE = 0x38                                         # VulkanProvider::MinApiSize
D3D12_SLOTS = {0x40: "d3d12 init", 0x48: "d3d12 create", 0x50: "d3d12 evaluate", 0x58: "d3d12 release"}
CORE_LOADED_SLOT = 0x60                                 # MfgUnlockCoreLoaded
SLOT_COUNT = 12

REQUIRED_EXPORTS = ("MfgXeMFGProviderGetApi",)
OPTIONAL_EXPORTS = {
    "XeFGModelName": "the model name in this dll's log and menu",
    "MfgUnlockCoreLoaded": "slot 0x60 of the table",
    "MfgXeFGVkGetApi": "OptiScaler's XeFG-on-Vulkan output (not used by this dll)",
    "OptiPluginInit": "OptiScaler's plugin entry",
    "InitializeASI": "OptiScaler's plugin host entry",
}

# (label, bytes to look for): features mentioned by the build, informational
FEATURES = (
    ("CUDA tensor-core route on Vulkan (VK_NVX_binary_import)", "A", b"VK_NVX_binary_import"),
    ("XeFGTensorCoresVulkanBarriers setting", "W", "XeFGTensorCoresVulkanBarriers"),
    ("XeFGTensorCoresMask setting", "W", "XeFGTensorCoresMask"),
    ("XeFGModel setting", "W", "XeFGModel"),
    ("registers itself as OptiScaler's FG Nvngx provider (no shim file)", "W",
     "XeMFG registered as the FG Nvngx provider"),
    ("DLSSG.DepthSubrect / DLSSG.MVecsSubrect inputs", "A", b"DLSSG.DepthSubrect"),
)


class Report:
    def __init__(self):
        self.problems = []           # the Vulkan models can't be made from the asi: the dll's runtime would have nothing to run
        self.provider_problems = []  # a dll built without the Vulkan runtime can't use the asi as its provider
        self.warnings = []           # something else is off

    def ok(self, text):
        print("  ok     " + text)

    def info(self, text):
        print("         " + text)

    def warn(self, text):
        self.warnings.append(text)
        print("  warn   " + text)

    def bad(self, text):
        self.problems.append(text)
        print("  FAIL   " + text)

    def provider_bad(self, text):
        self.provider_problems.append(text)
        print("  warn   " + text)


def exports(img):
    """{name: rva} of the export table"""
    data = img.data
    lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    export_rva = struct.unpack_from("<I", data, lfanew + 24 + 112)[0]
    if not export_rva:
        return {}
    table = img.rva_to_offset(export_rva)
    name_count = struct.unpack_from("<I", data, table + 24)[0]
    functions = img.rva_to_offset(struct.unpack_from("<I", data, table + 28)[0])
    names = img.rva_to_offset(struct.unpack_from("<I", data, table + 32)[0])
    ordinals = img.rva_to_offset(struct.unpack_from("<I", data, table + 36)[0])
    out = {}
    for i in range(name_count):
        at = img.rva_to_offset(struct.unpack_from("<I", data, names + 4 * i)[0])
        name = data[at:data.index(b"\0", at)].decode("ascii", "replace")
        ordinal = struct.unpack_from("<H", data, ordinals + 2 * i)[0]
        out[name] = struct.unpack_from("<I", data, functions + 4 * ordinal)[0]
    return out


def table_address(img, function_rva):
    """VA of the table MfgXeMFGProviderGetApi hands out. Every build so far is
        xor eax,eax / lea rdx,[rip+table] / cmp ecx,1 / cmove rax,rdx / ret
    so the first `lea r64,[rip+disp32]` of the function is looked for rather than the exact bytes."""
    start = img.rva_to_offset(function_rva)
    code = img.data[start:start + 64]
    for i in range(len(code) - 7):
        if code[i] in (0x48, 0x4C) and code[i + 1] == 0x8D and code[i + 2] & 0xC7 == 0x05:
            disp = struct.unpack_from("<i", code, i + 3)[0]
            return img.base + function_rva + i + 7 + disp
        if code[i] == 0xC3:   # ret: nothing found before the function ended
            break
    return None


def check_table(img, ex, report):
    function_rva = ex["MfgXeMFGProviderGetApi"]
    address = table_address(img, function_rva)
    if address is None:
        report.provider_bad("MfgXeMFGProviderGetApi doesn't load a table address the way earlier builds do (no lea r64,[rip+disp32])")
        return

    try:
        offset = img.va_to_offset(address)
        size, version = struct.unpack_from("<II", img.data, offset)
        slots = struct.unpack_from("<%dQ" % SLOT_COUNT, img.data, offset + 8)
    except (extractor.ExtractError, struct.error):
        report.provider_bad("the provider table at VA 0x%X isn't inside the file" % address)
        return

    report.info("table at rva 0x%X, size 0x%X, version %d" % (address - img.base, size, version))

    if version != EXPECTED_VERSION:
        report.provider_bad("table version %d, VulkanProvider.h expects %d" % (version, EXPECTED_VERSION))
    else:
        report.ok("table version %d" % version)

    if size < MIN_SIZE:
        report.provider_bad("table size 0x%X is shorter than the 0x%X the Vulkan slots need" % (size, MIN_SIZE))
    elif size != 0x68:
        report.info("table size 0x%X, the builds checked so far have 0x68. Slots past 0x%X are not used here" % (size, MIN_SIZE + 8))
    else:
        report.ok("table size 0x68")

    def slot_rva(slot_offset):
        value = slots[(slot_offset - 8) // 8] if 8 <= slot_offset < 8 + 8 * SLOT_COUNT else 0
        return value - img.base if value else 0

    sections = img.sections

    def in_text(rva):
        # a function slot has to point into an executable section; the first section is .text in every build
        va, span, _, _ = sections[0]
        return va <= rva < va + span

    for slot_offset, name in EXPECTED_SLOTS.items():
        rva = slot_rva(slot_offset)
        if not rva:
            if slot_offset in REQUIRED_SLOTS:
                report.provider_bad("slot +0x%02X (%s) is empty" % (slot_offset, name))
            else:
                report.info("slot +0x%02X (%s) is empty, optional" % (slot_offset, name))
        elif not in_text(rva):
            report.provider_bad("slot +0x%02X (%s) points outside .text (rva 0x%X)" % (slot_offset, name, rva))
        else:
            report.ok("slot +0x%02X %-9s -> rva 0x%X" % (slot_offset, name, rva))

    for slot_offset, name in D3D12_SLOTS.items():
        rva = slot_rva(slot_offset)
        report.info("slot +0x%02X (%s) %s" % (slot_offset, name, "rva 0x%X" % rva if rva else "empty"))

    core = slot_rva(CORE_LOADED_SLOT)
    if "MfgUnlockCoreLoaded" in ex and core and core != ex["MfgUnlockCoreLoaded"]:
        report.info("slot +0x60 isn't the MfgUnlockCoreLoaded export, the layout may have moved")


def check_programs(data, report):
    try:
        program_set = extractor.read_programs(data)
    except extractor.ExtractError as error:
        report.bad("no Vulkan program set could be read (%s), xefg_mfg_programs\\vulkan can't be made from this build" % error)
        return

    rows = program_set["rows"]
    counts = extractor.origin_counts(rows)
    summary = ", ".join("%s %d" % (name, counts[name]) for name in sorted(counts))
    report.ok("%d SPIR-V programs (%s), all match the SHA-256 table in the image" % (len(rows), summary))

    manifest = extractor.sha256(program_set["manifest"])
    if manifest == extractor.KNOWN_MANIFEST_SHA256:
        report.ok("the program set is identical to the one in every known build (programs.tsv %s...)" % manifest[:12])
    else:
        report.info("the program set differs from the known builds' (programs.tsv %s), docs/VULKAN-MODELS.md and "
                    "KNOWN_MANIFEST_SHA256 describe the old one" % manifest)


def wide(text):
    return text.encode("utf-16le")


def check_features(data):
    for label, kind, needle in FEATURES:
        pattern = needle if kind == "A" else wide(needle)
        print("         %-3s %s" % ("yes" if pattern in data else "no", label))


def main(argv):
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(__doc__.strip().split("\n\n")[0], file=sys.stderr)
        print("usage: InspectMfgUnlockerAsi.py MFGUnlocker.asi", file=sys.stderr)
        return 2

    try:
        with open(argv[1], "rb") as f:
            data = f.read()
        img = extractor.Image(data)
    except (OSError, extractor.ExtractError) as error:
        print("error: %s: %s" % (argv[1], error), file=sys.stderr)
        return 2

    report = Report()
    digest = extractor.sha256(data)
    product, file_version, known = extractor.describe_build(data, img)

    print("%s" % argv[1])
    print("  build  %s (file version %s), %d bytes" % (product, file_version, len(data)))
    print("  sha256 %s" % digest)
    print("  %s" % ("one of the builds this was run on" if known else
                    "not one of the builds this was run on, the checks below decide"))

    print("\nexports")
    try:
        ex = exports(img)
    except (extractor.ExtractError, struct.error, ValueError) as error:
        print("error: the export table can't be read: %s" % error, file=sys.stderr)
        return 2

    for name in REQUIRED_EXPORTS:
        if name in ex:
            report.ok("%s" % name)
        else:
            report.provider_bad("%s is not exported" % name)
    for name, use in OPTIONAL_EXPORTS.items():
        report.info("%s %s (%s)" % ("exported " if name in ex else "missing  ", name, use))

    print("\nXeMFG provider table (what a dll built without the Vulkan runtime forwards to; this dll's own runtime doesn't use it)")
    if "MfgXeMFGProviderGetApi" in ex:
        check_table(img, ex, report)

    print("\nVulkan program set")
    check_programs(data, report)

    print("\nbuild features (information only)")
    check_features(data)

    print()
    if report.warnings:
        print("%d warning%s above" % (len(report.warnings), "" if len(report.warnings) == 1 else "s"))
    if report.problems:
        print("INCOMPATIBLE: the Vulkan models can't be made from this asi (%d problem%s)" % (
            len(report.problems), "" if len(report.problems) == 1 else "s"))
        return 1

    print("COMPATIBLE: the Vulkan models can be made from this asi (tools/ExtractXeFgVulkanPrograms.py)")
    if report.provider_problems:
        print("            a dll built without the Vulkan runtime can't use it as its provider (%d warning%s above)" % (
            len(report.provider_problems), "" if len(report.provider_problems) == 1 else "s"))
    else:
        print("            a dll built without the Vulkan runtime can use it as its provider (src/VulkanProvider.cpp)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
