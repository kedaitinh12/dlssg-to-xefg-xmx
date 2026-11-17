#!/usr/bin/env python3
"""Checks the tensor-core route's layer table, which the repository has twice.

The D3D12 route keeps its own table inline in src/MfgPort/Cuda/cuda_route.cpp. The Vulkan route (src/vulkan/xvk_tensor.cpp)
reads the header MFG-Unlocker shares between its two routes, src/MfgPort/Cuda/cuda_layers.h, which tools/VendorXvk.py
copies verbatim. Both launch the same 16 kernels of xefg_cuda_layers.fatbin for the same programs, so the two tables must say
the same thing: the same kernels in the same order, the same programs with the same SHA-256, the same shapes. A difference
means one route launches a kernel for a program the other leaves alone, or a kernel with another shape.

With --fatbin FILE it also checks the file itself: its SHA-256 is the pinned one (resource 7300 of MFG-Unlocker v1.7.0's
release dll, resource 3903 of the 1.7.1 MFGUnlocker.asi builds) and it holds all 16 kernels, the check xvk_tensor.cpp makes
before it imports the module. That needs no GPU.

Usage: python3 -B -I tests/vulkan_runtime/check_layer_tables.py [--fatbin FILE] [--no-pin]
       --no-pin accepts a fatbin of another SHA-256 (a build with CudaRoot makes its own kernels)
"""
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROUTE = ROOT / "src" / "MfgPort" / "Cuda" / "cuda_route.cpp"
HEADER = ROOT / "src" / "MfgPort" / "Cuda" / "cuda_layers.h"
PINNED_SHA256 = "2ffae52e7878ae7e374ff953b864204f079c3199dc904b2ce7ecf8b37ee130f7"
KERNEL_COUNT = 16
LAYER_COUNT = 17
FAMILY_CONSTANTS = {"Pool": "0", "Decoder": "3", "Flow": "kFlowConstants"}

problems = []


def problem(text):
    problems.append(text)


def squash(text):
    return re.sub(r"\s+", " ", text).strip()


def table(path):
    """The kernels, the flow constants and the layer rows of the file, as parsed text."""
    text = path.read_text(encoding="utf-8")
    # Comments don't count, and some of them hold braces
    text = re.sub(r"//[^\n]*", "", text)

    kernels = re.search(r"kKernels\[\]\s*=\s*\{(.*?)\};", text, re.S)
    flow = re.search(r"kFlowConstants\s*=\s*([^;]+);", text)
    layers = re.search(r"kLayers\[\]\s*=\s*\{(.*?)\n\};", text, re.S)

    if not (kernels and flow and layers):
        problem(f"{path.name}: the kernel list, kFlowConstants or the layer table wasn't found")
        return None

    rows = []
    # {{program, sha256, kernel[, own sha256]}, kernel index, family, intelW, intelH, groupsZ, tileW, tileH, gridZ,
    #  constants, bytes, ownBytes}
    for info, rest in re.findall(r"\{\{(.*?)\},\s*([^{}]*?)\}\s*,?", layers.group(1), re.S):
        strings = re.findall(r'"([^"]*)"', info)
        fields = [squash(field) for field in rest.split(",")]
        rows.append((strings, fields))

    return {
        "kernels": re.findall(r'"([^"]+)"', kernels.group(1)),
        "flow": squash(flow.group(1)),
        "rows": rows,
    }


def check_one(name, parsed):
    kernels = parsed["kernels"]
    rows = parsed["rows"]

    if len(kernels) != KERNEL_COUNT:
        problem(f"{name}: {len(kernels)} kernels, {KERNEL_COUNT} expected")

    if len(rows) != LAYER_COUNT:
        problem(f"{name}: {len(rows)} layers, {LAYER_COUNT} expected")

    seen = set()
    used = set()

    for number, (strings, fields) in enumerate(rows, 1):
        label = f"{name}: layer {number}"

        if len(strings) not in (3, 4):
            problem(f"{label}: {len(strings)} strings, a program, its SHA-256, a kernel and maybe an own SHA-256 expected")
            continue

        program, sha256, kernel = strings[:3]
        own = strings[3] if len(strings) == 4 else None

        for digest in [sha256] + ([own] if own else []):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                problem(f"{label} ({program}): {digest!r} isn't a lower case SHA-256")
            elif digest in seen:
                problem(f"{label} ({program}): the SHA-256 {digest[:12]}... is in the table twice")
            seen.add(digest)

        if len(fields) != 11:
            problem(f"{label} ({program}): {len(fields)} numbers, 11 expected")
            continue

        index, family, intel_w, intel_h, groups_z, tile_w, tile_h, grid_z, constants, size, own_size = fields
        family = family.replace("Family::", "")

        if not index.isdigit() or int(index) >= len(kernels) or kernels[int(index)] != kernel:
            problem(f"{label} ({program}): kernel {kernel} isn't kernel {index} of the list")
        else:
            used.add(int(index))

        if family not in FAMILY_CONSTANTS:
            problem(f"{label} ({program}): family {family!r}")
        elif constants != FAMILY_CONSTANTS[family]:
            problem(f"{label} ({program}): a {family} layer reads root constants {constants}, {FAMILY_CONSTANTS[family]} expected")

        for value in (intel_w, intel_h, groups_z, tile_w, tile_h, grid_z, size):
            if not value.isdigit() or int(value) == 0:
                problem(f"{label} ({program}): {value!r} isn't a positive number")

        if not own_size.isdigit() or bool(int(own_size)) != bool(own):
            problem(f"{label} ({program}): an own program has a size and only an own program does ({own_size})")

    for index in range(len(kernels)):
        if index not in used:
            problem(f"{name}: no layer launches kernel {kernels[index]}")


def check_fatbin(path, pin):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()

    if pin and digest != PINNED_SHA256:
        problem(f"{path}: SHA-256 {digest}, the pinned fatbin is {PINNED_SHA256}")

    # What TensorRoute::Install looks for: the kernel's name and its NUL
    parsed = table(HEADER)

    if parsed:
        for kernel in parsed["kernels"]:
            if kernel.encode() + b"\0" not in data:
                problem(f"{path}: the kernel {kernel} isn't in it")

    return digest, len(data)


def main(argv):
    fatbin = None
    pin = True

    args = list(argv)
    while args:
        arg = args.pop(0)

        if arg == "--fatbin" and args:
            fatbin = pathlib.Path(args.pop(0))
        elif arg == "--no-pin":
            pin = False
        else:
            print(__doc__)
            return 2

    route = table(ROUTE)
    header = table(HEADER)

    if route:
        check_one(ROUTE.name, route)

    if header:
        check_one(HEADER.name, header)

    if route and header:
        for key, what in (("kernels", "kernel lists"), ("flow", "kFlowConstants")):
            if route[key] != header[key]:
                problem(f"the {what} of cuda_route.cpp and cuda_layers.h differ: {route[key]} / {header[key]}")

        if len(route["rows"]) == len(header["rows"]):
            for number, (a, b) in enumerate(zip(route["rows"], header["rows"]), 1):
                if a != b:
                    problem(f"layer {number} differs between cuda_route.cpp and cuda_layers.h: {a} / {b}")
        else:
            problem(f"cuda_route.cpp has {len(route['rows'])} layers, cuda_layers.h {len(header['rows'])}")

    summary = f"{len(header['rows']) if header else 0} layers, {len(header['kernels']) if header else 0} kernels"

    if fatbin:
        if fatbin.is_file():
            digest, size = check_fatbin(fatbin, pin)
            summary += f", fatbin {size} bytes SHA-256 {digest[:12]}..."
        else:
            problem(f"{fatbin} isn't a file")

    if problems:
        for text in problems:
            print("FAIL " + text)
        return 1

    print(f"ok: the two layer tables agree ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
