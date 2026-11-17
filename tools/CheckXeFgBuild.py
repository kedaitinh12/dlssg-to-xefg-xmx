#!/usr/bin/env python3
"""Says whether a libxess_fg.dll is a build this repository knows and whether the programs it relies on are in it.

    python3 tools/CheckXeFgBuild.py libxess_fg.dll [--vulkan-programs DIR] [--no-layout] [--dump DIR] [--debug]
    python3 tools/CheckXeFgBuild.py --selftest

It reads the file only, no GPU and no Windows needed, and checks what depends on the XeFG build:

 1. Identity: the version, the PE time stamp and size, the SHA-256, and whether `src/XeFG.cpp` lists the build. The API
    addresses are also found by the scan `XeFG::Scan` makes for unknown builds and compared with the listed ones.
 2. The programs XeFG carries: each is a zstd stream packed with one of two dictionaries that the dll holds itself, and the
    DXBC container XeFG hands to `CreateComputePipelineState` is `DXBC` followed by the decoded bytes. The tool decodes
    them all and hashes the containers, the way the tensor-core route and the Vulkan runtime recognize a program.
 3. The tensor-core route's layer table (`src/MfgPort/Cuda/cuda_layers.h`): which of its Intel programs this build has.
 4. With `--vulkan-programs`, the Intel programs of the Vulkan models (`xefg_mfg_programs/vulkan/programs.tsv`).
 5. The layout family of MFG-Unlocker's port (`src/MfgPort/RuntimeLayout.h`, `Discover`) and the identity checks of
    `ShaderPort` and `ModelPort`, which decide whether the XMX Port, CUDA and XMX (native) paths can be installed.
    `--no-layout` skips it.

`--dump DIR` also writes the decoded programs (`<key>.dxbc`, the containers) there; they are Intel's, don't commit them.
`--debug` lets an unexpected error of reading a damaged file show its traceback. `--selftest` only checks that the zstd decoder
works here, on known answers (no dll needed).

Needs libzstd: the Python package `zstandard` (pip install zstandard) or libzstd for ctypes (libzstd.so.1, libzstd.dll).
Exit code 0 when everything this build has to offer is where the repository expects it, 1 when something isn't, 2 when the
file or the zstd library can't be used.
"""
import argparse
import ctypes
import ctypes.util
import datetime
import hashlib
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXEC, WRITE, READ = 0x20000000, 0x80000000, 0x40000000

# The four programs of the layer table that aren't Intel's: RTXMFG's own ported flow programs, which come with the
# program set this dll loads and aren't in any libxess_fg.dll (see the header of cuda_layers.h)
PORTED_FLOW_PROGRAMS = {"3459485", "3452733", "34517926", "3453044"}


def fail(message, code=2):
    print(message, file=sys.stderr)
    sys.exit(code)


class ProgramError(Exception):
    pass


class PE:
    """What the checks need of a PE file image: sections, the unwind table, the version resource."""

    def __init__(self, data):
        self.data = data

        if data[:2] != b"MZ" or len(data) < 0x400:
            fail("not a PE file")

        self.lfanew = struct.unpack_from("<I", data, 0x3C)[0]

        if data[self.lfanew:self.lfanew + 4] != b"PE\0\0":
            fail("not a PE file")

        machine, count, self.stamp = struct.unpack_from("<HHI", data, self.lfanew + 4)
        optional_size = struct.unpack_from("<H", data, self.lfanew + 20)[0]
        opt = self.lfanew + 24

        if machine != 0x8664 or struct.unpack_from("<H", data, opt)[0] != 0x20B:
            fail("not a 64 bit PE file")

        self.image_base = struct.unpack_from("<Q", data, opt + 24)[0]
        self.size_of_image = struct.unpack_from("<I", data, opt + 56)[0]
        self.checksum = struct.unpack_from("<I", data, opt + 64)[0]
        self.dirs = [struct.unpack_from("<II", data, opt + 112 + 8 * i) for i in range(16)]
        self.sections = []

        for i in range(count):
            name, vsize, va, raw_size, raw = struct.unpack_from("<8sIIII", data, opt + optional_size + 40 * i)
            flags = struct.unpack_from("<I", data, opt + optional_size + 40 * i + 36)[0]
            self.sections.append(dict(name=name.rstrip(b"\0").decode("latin1"), va=va, size=vsize, raw=raw,
                                      raw_size=raw_size, flags=flags))

        self._functions = None

    def read(self, rva, count):
        """The bytes the loaded image has at rva (zeros past a section's raw data), None outside every section."""
        for s in self.sections:
            if s["va"] <= rva < s["va"] + s["size"] and count <= s["va"] + s["size"] - rva:
                offset = rva - s["va"]
                raw = self.data[s["raw"] + offset:s["raw"] + offset + count] if offset < s["raw_size"] else b""
                return raw + b"\0" * (count - len(raw))

        return None

    def section(self, name):
        return next(s for s in self.sections if s["name"] == name)

    def in_range(self, rva, count, kind):
        """xess_layout::Image::Range: kind 0 any section, 1 executable, 2 writable data, 3 read only data."""
        for s in self.sections:
            if rva < s["va"] or rva - s["va"] >= s["size"] or count > s["size"] - (rva - s["va"]):
                continue

            f = s["flags"]
            return (True if kind == 0 else bool(f & EXEC) if kind == 1 else
                    (bool(f & WRITE) and not f & EXEC) if kind == 2 else (not f & (WRITE | EXEC) and bool(f & READ)))

        return False

    def functions(self):
        if self._functions is None:
            rva, size = self.dirs[3]
            raw = self.read(rva, size) if rva else b""
            self._functions = [struct.unpack_from("<III", raw, 12 * i) for i in range(len(raw) // 12)]
            self._starts = [f[0] for f in self._functions]

        return self._functions

    def function_start(self, rva):
        import bisect
        self.functions()
        i = bisect.bisect_right(self._starts, rva) - 1

        if i >= 0 and self._functions[i][0] <= rva < self._functions[i][1]:
            return self._functions[i][0]

        return None

    def version(self):
        """(FileVersion, ProductVersion) of the version resource, or (None, None)."""
        rva, size = self.dirs[2]

        if not rva:
            return None, None

        base = self.read(rva, min(size, 0x100000))
        # Resource directory walk: type 16 (RT_VERSION), any name, any language
        def entries(offset):
            named, ids = struct.unpack_from("<HH", base, offset + 12)
            return [struct.unpack_from("<II", base, offset + 16 + 8 * i) for i in range(named + ids)]

        for ident, to in entries(0):
            if ident != 16 or not to & 0x80000000:
                continue

            for _, to2 in entries(to & 0x7FFFFFFF):
                for _, to3 in entries(to2 & 0x7FFFFFFF):
                    data_rva, data_size = struct.unpack_from("<II", base, to3)
                    blob = self.read(data_rva, data_size) or b""
                    i = blob.find(struct.pack("<I", 0xFEEF04BD))

                    if i >= 0:
                        fv = struct.unpack_from("<IIII", blob, i + 8)
                        return ("%d.%d.%d.%d" % (fv[0] >> 16, fv[0] & 0xFFFF, fv[1] >> 16, fv[1] & 0xFFFF),
                                "%d.%d.%d.%d" % (fv[2] >> 16, fv[2] & 0xFFFF, fv[3] >> 16, fv[3] & 0xFFFF))

        return None, None


# ---- zstd -------------------------------------------------------------------------------------------------------------

# Known answers, which a decoder has to reproduce before it says anything about a build. The frame is a plain zstd frame, as
# the dictionaries are stored in the dll. The asset is packed with SELFTEST_DICTIONARY (raw content) and has no magic number,
# as XeFG's programs are: it only decodes with the dictionary, and only by a decoder that takes the frame format without it.
SELFTEST_DICTIONARY = b"XeFG packs its programs: a dictionary, then one zstd frame each. " * 3
SELFTEST_TEXT = b"one zstd frame each, then a dictionary: XeFG packs its programs."
SELFTEST_FRAME = bytes.fromhex(
    "28b52ffd2040d5010002840d11a0edd084e85ce15f35d26f210d89a81a35a19b727a0cdf71bd99517832b65c4856f75634eb5cef56c5036574bf4d19fdc49"
    "e5ff70200")
SELFTEST_ASSET = bytes.fromhex("20408d0000186f3a2e04006c80528403520d80debc02")


class Zstd:
    """A zstd frame decoder, with the python package or with libzstd. Whichever is used has reproduced the known answers."""

    def __init__(self):
        self.module = self.lib = None
        why = None

        try:
            import zstandard
        except ImportError:
            zstandard = None

        if zstandard:
            self.module = zstandard

            try:
                self.selftest()
                return
            except Exception as error:  # a version of the package that has another API: libzstd is the other way
                why = f"the zstandard package can't decode what XeFG's programs need ({type(error).__name__}: {error})"
                self.module = None

        self.load_library(why)

        try:
            self.selftest()
        except ValueError as error:
            fail(f"this libzstd doesn't decode what XeFG's programs need: {error}")

        if why:
            print(f"note: {why}; libzstd is used", file=sys.stderr)

    def load_library(self, why):
        names = ["libzstd.so.1", "libzstd.so", "libzstd.dylib", "libzstd.dll", "zstd.dll"]
        found = ctypes.util.find_library("zstd")

        if found:
            names.insert(0, found)

        for name in names:
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                pass

        if not self.lib:
            fail("this needs libzstd: pip install zstandard, or put libzstd.so.1 / libzstd.dll where Python finds it"
                 + (f" ({why})" if why else ""))

        lib, size = self.lib, ctypes.c_size_t
        lib.ZSTD_decompress.restype = size
        lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, size, ctypes.c_char_p, size]
        lib.ZSTD_isError.restype = ctypes.c_uint
        lib.ZSTD_isError.argtypes = [size]
        lib.ZSTD_getErrorName.restype = ctypes.c_char_p
        lib.ZSTD_getErrorName.argtypes = [size]
        lib.ZSTD_createDCtx.restype = ctypes.c_void_p
        lib.ZSTD_createDDict.restype = ctypes.c_void_p
        lib.ZSTD_createDDict.argtypes = [ctypes.c_char_p, size]
        lib.ZSTD_DCtx_setParameter.restype = size
        lib.ZSTD_DCtx_setParameter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.ZSTD_decompress_usingDDict.restype = size
        lib.ZSTD_decompress_usingDDict.argtypes = [ctypes.c_void_p, ctypes.c_void_p, size, ctypes.c_char_p, size,
                                                   ctypes.c_void_p]

    def selftest(self):
        if self.frame(SELFTEST_FRAME, len(SELFTEST_TEXT)) != SELFTEST_TEXT:
            raise ValueError("a plain frame decodes to something else")

        if self.asset_decoder(SELFTEST_DICTIONARY)(SELFTEST_ASSET, len(SELFTEST_TEXT)) != SELFTEST_TEXT:
            raise ValueError("a frame without the magic number, packed with a dictionary, decodes to something else")

    def frame(self, blob, size):
        """A frame with its magic, no dictionary."""
        if self.module:
            try:
                return self.module.ZstdDecompressor().decompress(blob, max_output_size=size)
            except (self.module.ZstdError, TypeError) as error:
                raise ValueError(str(error)) from None

        out = ctypes.create_string_buffer(size)
        n = self.lib.ZSTD_decompress(out, size, blob, len(blob))

        if self.lib.ZSTD_isError(n):
            raise ValueError(self.lib.ZSTD_getErrorName(n).decode())

        return out.raw[:n]

    def asset_decoder(self, dictionary):
        """A function(blob, size) for the frames XeFG's assets are: no magic number, packed with the dictionary."""
        if self.module:
            z = self.module
            decompressor = z.ZstdDecompressor(dict_data=z.ZstdCompressionDict(dictionary), format=z.FORMAT_ZSTD1_MAGICLESS)

            def decode(blob, size):
                # The stream interface, which doesn't read the frame header for a size first (it can't without the magic)
                try:
                    return decompressor.decompressobj().decompress(blob)
                except (z.ZstdError, TypeError) as error:
                    raise ValueError(str(error)) from None

            return decode

        lib = self.lib
        dctx = lib.ZSTD_createDCtx()
        ddict = lib.ZSTD_createDDict(dictionary, len(dictionary))
        # ZSTD_d_format (ZSTD_d_experimentalParam1) = ZSTD_f_zstd1_magicless
        error = lib.ZSTD_DCtx_setParameter(dctx, 1000, 1)

        if lib.ZSTD_isError(error):
            fail("this libzstd can't decode frames without the magic number: " + lib.ZSTD_getErrorName(error).decode())

        def decode(blob, size):
            out = ctypes.create_string_buffer(size)
            n = lib.ZSTD_decompress_usingDDict(dctx, out, size, blob, len(blob), ddict)

            if lib.ZSTD_isError(n):
                raise ValueError(lib.ZSTD_getErrorName(n).decode())

            return out.raw[:n]

        return decode


# ---- the dll's own tables -----------------------------------------------------------------------------------------------

# Each program XeFG carries has a static initializer that stores {compressed size, decoded size, address of the
# compressed bytes, address of the dictionary's frame, the dictionary's size, its frame's size} on the stack and registers
# them under a key, a decimal string that also is the key the model graphs use
REGISTRATION = re.compile(
    rb"\xC7\x44\x24\x28(....)\xC7\x44\x24\x2C(....)\x48\x8D\x05(....)\x48\x89\x44\x24\x30\x48\x8D\x05(....)\x48\x89\x44\x24\x38"
    rb"\xC7\x44\x24\x40(....)\xC7\x44\x24\x44(....)", re.S)


def registrations(pe):
    text = pe.section(".text")
    code = pe.read(text["va"], text["size"])
    found = []

    for m in REGISTRATION.finditer(code):
        va = text["va"] + m.start()
        packed, size = struct.unpack("<II", m.group(1) + m.group(2))
        data_rva = va + (m.start(3) - m.start()) + 4 + struct.unpack("<i", m.group(3))[0]
        dict_rva = va + (m.start(4) - m.start()) + 4 + struct.unpack("<i", m.group(4))[0]
        dict_size, dict_packed = struct.unpack("<II", m.group(5) + m.group(6))
        tail = code[m.end():m.end() + 400]
        lengths = [struct.unpack_from("<I", tail, x.end())[0] for x in re.finditer(rb"\x48\xC7\x44\x24\x58", tail)]
        name = re.search(rb"\x0F\x10\x05(....)", tail)
        key = ""

        if name and len(lengths) >= 2:
            name_rva = va + (m.end() - m.start()) + name.start(1) + 4 + struct.unpack("<i", name.group(1))[0]
            key = (pe.read(name_rva, lengths[1]) or b"").decode("latin1")

        found.append(dict(key=key, packed=packed, size=size, data=data_rva, dictionary=dict_rva, dict_size=dict_size,
                          dict_packed=dict_packed))

    return found


def containers(pe, zstd):
    """{key: container bytes} of the DXBC programs the dll carries, and how many assets of any kind it has."""
    regs = registrations(pe)
    dictionaries = {}
    result = {}

    for r in regs:
        if r["dictionary"] not in dictionaries:
            try:
                dictionary = zstd.frame(pe.read(r["dictionary"], r["dict_packed"]), r["dict_size"])
            except (ValueError, TypeError) as error:
                raise ProgramError(f"the dictionary at {r['dictionary']:#x} doesn't decode ({error}): "
                                   "the file is damaged, or a build that packs its programs differently")

            dictionaries[r["dictionary"]] = zstd.asset_decoder(dictionary)

        try:
            asset = dictionaries[r["dictionary"]](pe.read(r["data"], r["packed"]), r["size"])
        except ValueError:
            continue

        # What XeFG makes of an asset is the DXBC container without its 4 byte magic. Its size field says so. The assets
        # of the other dictionary (kernels for Intel's own GPUs) don't have that shape
        container = b"DXBC" + asset

        if len(asset) > 28 and struct.unpack_from("<I", container, 24)[0] == len(container):
            result[r["key"]] = container

    return result, len(regs)


# ---- XeFG.cpp -------------------------------------------------------------------------------------------------------------

def table(text, pattern, what):
    """The first group of pattern in text; the tool has to be updated with the repository when the table isn't found."""
    found = re.search(pattern, text, re.S)

    if not found:
        fail(f"{what}: the table isn't found where this tool looks for it, the tool has to follow a change of the source")

    return found.group(1)


def known_builds(repo):
    text = (repo / "src" / "XeFG.cpp").read_text(encoding="utf-8")
    block = table(text, r"knownBuilds\[\]\s*=\s*\{(.*?)\n\t\};", "src/XeFG.cpp")
    builds = []

    for comment, row in re.findall(r"//\s*([^\n]*)\n\s*\{([^{}]*?)&Layout", block):
        n = [int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", row)]
        builds.append(dict(name=comment.strip(), stamp=n[0], size=n[1],
                           api=dict(zip(("createContext", "init", "execute", "destroy", "setLoggingCallback",
                                         "settingsInstance"), n[2:8]))))

    return builds


def scan_api(pe):
    """XeFG::Scan: the hidden C API found by the strings and the code around it."""
    text = pe.section(".text")
    code = pe.read(text["va"], text["size"])
    tva = text["va"]
    data_sections = [s for s in pe.sections if s["flags"] & READ and not s["flags"] & EXEC and s["flags"] & 0x40]
    strings = ["phContext is null.", "pCommandList is null.", "pInitParams is null.",
               "Unsupported loggingLevel ({}) value.", "Inside libxess_fg.dll"]
    wanted = {}

    for text_string in strings:
        needle = text_string.encode() + b"\0"

        for s in data_sections:
            blob = pe.read(s["va"], s["size"])
            at = blob.find(needle)

            while at >= 0:
                wanted.setdefault(s["va"] + at, text_string)
                at = blob.find(needle, at + 1)

    # A disp32 that ends its instruction and lands on one of the strings, at any alignment
    references = {t: [] for t in wanted}

    for shift in range(4):
        end = shift + (len(code) - shift) // 4 * 4
        numbers = memoryview(code[shift:end]).cast("i")

        for i, d in enumerate(numbers):
            target = tva + shift + 4 * i + 4 + d

            if target in references:
                references[target].append(tva + shift + 4 * i)

    functions = {s: [] for s in strings}

    for target, text_string in wanted.items():
        for position in references[target]:
            f = pe.function_start(position)

            if f is not None and f not in functions[text_string]:
                functions[text_string].append(f)

    def pattern(spec):
        rx = b"".join(b"." if t.startswith("?") else re.escape(bytes([int(t, 16)])) for t in spec.split())
        return [tva + m.start() for m in re.finditer(rx, code, re.S)]

    found = {}
    create = functions[strings[0]]

    if len(create) != 1:
        return found

    anchor = create[0]
    near = lambda fs: [f for f in fs if abs(f - anchor) < 0x10000]
    found["createContext"] = anchor
    execute = near(functions[strings[1]])
    init = near(functions[strings[2]])
    logging = near(functions[strings[3]])

    if len(execute) == 1:
        found["execute"] = execute[0]

    if init:
        found["init"] = max(init)

    if logging:
        found["setLoggingCallback"] = min(logging, key=lambda f: abs(f - anchor))

    destroy = pattern("48 8B 41 18 48 8B CA FF 15 ?? ?? ?? ?? 48 8B CB E8 ?? ?? ?? ?? 48 8B C8 48 85 C0 74 ?? "
                      "48 8B 10 48 8B 02 BA 01 00 00 00 FF 15 ?? ?? ?? ?? BA 10 00 00 00 48 8B CB E8")

    if len(destroy) == 1 and pe.function_start(destroy[0]) is not None:
        found["destroy"] = pe.function_start(destroy[0])

    constructors = functions[strings[4]]

    if len(constructors) == 1:
        targets = [constructors[0]]

        for p in pattern("E9 ?? ?? ?? ??"):
            if p + 5 + struct.unpack_from("<i", code, p - tva + 1)[0] == constructors[0]:
                targets.append(p)

        for p in pattern("48 8D 1D ?? ?? ?? ?? 48 8B CB E8"):
            if p + 15 + struct.unpack_from("<i", code, p - tva + 11)[0] in targets:
                found["settingsInstance"] = pe.function_start(p)
                break

    return found


# ---- the tables of this repository --------------------------------------------------------------------------------------

def layer_table(repo):
    """[(program, sha256, kernel, own sha256 or None)] of cuda_layers.h."""
    text = re.sub(r"//[^\n]*", "", (repo / "src" / "MfgPort" / "Cuda" / "cuda_layers.h").read_text(encoding="utf-8"))
    body = table(text, r"kLayers\[\]\s*=\s*\{(.*?)\n\};", "src/MfgPort/Cuda/cuda_layers.h")
    rows = []

    for info, _ in re.findall(r"\{\{(.*?)\},\s*([^{}]*?)\}\s*,?", body, re.S):
        strings = re.findall(r'"([^"]*)"', info)
        rows.append((strings[0], strings[1], strings[2], strings[3] if len(strings) == 4 else None))

    return rows


def vulkan_intel_programs(directory):
    rows = []

    for line in (Path(directory) / "programs.tsv").read_text(encoding="utf-8").splitlines():
        cells = line.split("\t")

        if len(cells) >= 8 and cells[7] == "intel":
            rows.append((cells[0], cells[6]))

    return rows


# ---- layout family (RuntimeLayout.h Discover) -------------------------------------------------------------------------

class LayoutFail(Exception):
    pass


def layout_discover(pe, repo):
    """Python port of xess_layout::Discover on the file image. Returns ({family: error or None}, addresses, shader port)."""
    gen = (repo / "src" / "MfgPort" / "RuntimeLayout.generated.h").read_text(encoding="utf-8")

    def number(x):
        return int(x, 16) if x.lower().startswith("0x") else int(x)

    def arrays(prefix):
        witnesses = re.search(r"Witness\s+" + prefix + r"_witnesses\[\]=\{(.*?)\};", gen, re.S).group(1)
        result = []

        for m in re.finditer(r"\{(\w+),sizeof\(\w+\),(\w+),(\d+),(true|false)\}", witnesses):
            data = bytes(int(x) for x in re.search(r"unsigned char\s+" + m.group(1) + r"\[\]=\{([^}]*)\}", gen).group(1).split(","))
            edges = re.search(r"Edge\s+" + m.group(2) + r"\[\]=\{(.*?)\};", gen, re.S).group(1)
            parsed = [tuple(number(v) if v else 0 for v in t.split(",")) for t in re.findall(r"\{([^{}]*)\}", edges)]
            result.append((data, parsed[:int(m.group(3))], m.group(4) == "true"))

        bindings = re.search(r"Binding\s+" + prefix + r"_bindings\[\]=\{(.*?)\};", gen, re.S).group(1)
        return result, [tuple(number(v) for v in t.split(",")) for t in re.findall(r"\{([^{}]*)\}", bindings)]

    def target(start, edge):
        offset, end, local, kind, _ = edge
        raw = pe.read(start + offset, 4)

        if raw is None:
            raise LayoutFail("layout_edge_read")

        t = start + end + struct.unpack("<i", raw)[0]

        if t <= 0 or t >= pe.size_of_image or not pe.in_range(t, 1, kind):
            raise LayoutFail("layout_edge_owner")

        if local >= 0 and t != start + local:
            raise LayoutFail("layout_local_branch")

        return t

    def find(witness):
        data, edges, unwind = witness
        mask = bytearray([1]) * len(data)

        for e in edges:
            mask[e[0]:e[0] + 4] = b"\0\0\0\0"

        found = []

        def match(start):
            raw = pe.read(start, len(data))

            if raw is None or any(mask[i] and raw[i] != data[i] for i in range(len(data))):
                return

            try:
                for e in edges:
                    target(start, e)
            except LayoutFail:
                return

            found.append(start)

        if unwind:
            for begin, end, _ in pe.functions():
                if end - begin == len(data):
                    match(begin)
        else:
            for s in pe.sections:
                if not s["flags"] & EXEC or s["size"] < len(data):
                    continue

                code = pe.read(s["va"], s["size"])

                for offset in range(0, s["size"] - len(data) + 1, 16):
                    if code[offset] != data[0]:
                        continue

                    i = 1

                    while i < len(data) and (not mask[i] or code[offset + i] == data[i]):
                        i += 1

                    if i == len(data):
                        match(s["va"] + offset)

        return found

    families = table(gen, r"inline constexpr const Family\* fg\[\]=\{([^}]*)\}", "src/MfgPort/RuntimeLayout.generated.h")
    results = {}
    chosen = None

    for name in re.findall(r"&(\w+)", families):
        step = "start"

        try:
            witnesses, bindings = arrays(name)
            found = []

            for i, w in enumerate(witnesses):
                step = f"witness {i}"
                hits = find(w)

                if len(hits) != 1:
                    raise LayoutFail(f"witness {i} matched {len(hits)} times")

                found.append(hits[0])

            addresses = {}

            for i, (key, witness, offset, end) in enumerate(bindings):
                step = f"binding {i}"
                addresses[key] = target(found[witness], (offset, end, -1, 0, 0)) if end else found[witness] + offset

            for i, w in enumerate(witnesses):
                step = f"witness {i} edges"

                for e in w[1]:
                    if e[4] and target(found[i], e) != addresses[e[4]]:
                        raise LayoutFail("layout_edge_mismatch")

            step = "getter"
            getter = struct.unpack("<Q", pe.read(addresses[0x157C3A0] + 0x80, 8))[0]

            if getter < pe.image_base or getter - pe.image_base > 0xFFFFFFFF:
                raise LayoutFail("layout_getter_pointer")

            thunk = getter - pe.image_base

            if pe.read(thunk, 1)[0] != 0xE9 or target(thunk, (1, 5, -1, 1, 0)) != addresses[0x211AE0]:
                raise LayoutFail("layout_getter_target")

            step = "decoder"
            decoder = addresses[0x11230]

            if pe.read(decoder, 1)[0] != 0xE9 or target(decoder, (1, 5, -1, 1, 0)) != addresses[0x19FDA0]:
                raise LayoutFail("layout_decoder_target")

            results[name] = None
            chosen = chosen or (name, addresses)
        except (LayoutFail, KeyError, TypeError, ValueError, IndexError, AttributeError, struct.error) as error:
            results[name] = f"{step}: {error}"

    return results, chosen


def shader_port_checks(pe, addresses, repo):
    """What ShaderPort's constructor tests before it patches anything, as far as the file image can say."""
    site = addresses[0x1AD217]
    call = struct.unpack("<i", pe.read(site + 1, 4))[0]
    decode = bytes([0x48, 0x89, 0x5C, 0x24, 0x20, 0x55, 0x56, 0x57, 0x48, 0x81, 0xEC, 0xB0, 0, 0, 0])
    resize = bytes([0x40, 0x53, 0x48, 0x83, 0xEC, 0x20, 0x4C, 0x8B, 0x49, 0x08])
    donor = re.search(r'name\(donor \+ 0x30\) != "([0-9a-f]{32})"',
                      (repo / "src" / "MfgPort" / "ModelPort.h").read_text(encoding="utf-8"))
    text = pe.data

    return [
        ("the call site is a call", pe.read(site, 1)[0] == 0xE8),
        ("it calls XeFG's decoder thunk", site + 5 + call == addresses[0x11230]),
        ("the decoder starts as expected", pe.read(addresses[0x19FDA0], len(decode)) == decode),
        ("the vector resize starts as expected", pe.read(addresses[0x1A09C0], len(resize)) == resize),
        ("the site is in code", pe.in_range(site, 5, 1)),
        ("the model catalog is in data", pe.in_range(addresses[0x15BE920], 1, 2)),
        ("the donor model's name is in the dll", bool(donor) and donor.group(1).encode() in text),
    ]


# ---- report ---------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Checks a libxess_fg.dll against what this repository knows.")
    parser.add_argument("dll", nargs="?")
    parser.add_argument("--repo", default=str(ROOT), help="the repository (default: the one this script is in)")
    parser.add_argument("--vulkan-programs", help="the xefg_mfg_programs\\vulkan folder (programs.tsv) to check as well")
    parser.add_argument("--no-layout", action="store_true", help="skip the layout family check of the MFG port")
    parser.add_argument("--dump", help="write the decoded programs (containers) to this folder")
    parser.add_argument("--debug", action="store_true", help="a failing check of the file shows its Python traceback")
    parser.add_argument("--selftest", action="store_true", help="only check that the zstd decoder works here (no dll needed)")
    args = parser.parse_args()

    if args.selftest:
        zstd = Zstd()
        print("zstd decoder: ok (" + ("the zstandard package" if zstd.module else "libzstd") + ")")
        return 0

    if not args.dll:
        parser.error("a libxess_fg.dll is needed (or --selftest)")

    repo = Path(args.repo)
    path = Path(args.dll)

    if not path.is_file():
        fail(f"{path} isn't a file")

    data = path.read_bytes()

    try:
        pe = PE(data)
        pe.section(".text")
    except (struct.error, StopIteration, IndexError):
        fail("damaged PE file (headers or sections can't be read)")

    problems = []

    def check(ok, text):
        print(("  ok    " if ok else "  FAIL  ") + text)

        if not ok:
            problems.append(text)

        return ok

    file_version, product_version = pe.version()
    stamp = datetime.datetime.fromtimestamp(pe.stamp, datetime.timezone.utc).strftime("%Y-%m-%d")
    print(f"{path.name}: {len(data)} bytes, SHA-256 {hashlib.sha256(data).hexdigest()}")
    print(f"version {file_version or '?'} (product {product_version or '?'}), time stamp {pe.stamp:#010x} ({stamp}), "
          f"size of image {pe.size_of_image:#010x}, checksum {pe.checksum:#x}")

    # 1. identity and API
    print("\n1. XeFG API (src/XeFG.cpp)")
    builds = known_builds(repo)
    known = next((b for b in builds if b["stamp"] == pe.stamp and b["size"] == pe.size_of_image), None)
    scanned = scan_api(pe)

    if known:
        check(True, f"a known build: {known['name']}")

        for name, rva in known["api"].items():
            check(scanned.get(name) == rva,
                  f"{name} listed at {rva:#x}, the scan finds {'%#x' % scanned[name] if name in scanned else 'nothing'}")
    else:
        print("  not a build that src/XeFG.cpp lists; XeFG::Scan would look for the API, and finds:")

        for name in ("createContext", "init", "execute", "destroy", "setLoggingCallback", "settingsInstance"):
            check(name in scanned, f"{name}: {'%#x' % scanned[name] if name in scanned else 'not found'}")

        print("  (a build that isn't listed is accepted only when its version is 1.3.x and the settings layout matches)")

    # 2. programs
    print("\n2. programs XeFG carries")
    zstd = Zstd()

    try:
        programs, assets = containers(pe, zstd)
    except ProgramError as error:
        check(False, str(error))
        programs, assets = {}, -1

    by_hash = {hashlib.sha256(c).hexdigest(): k for k, c in programs.items()}

    if assets >= 0:
        print(f"  {assets} assets, {len(programs)} DXBC programs ({len(by_hash)} different)")

        if not assets:
            check(False, "no program registry found: this build lays its assets out differently")

    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)

        for key, container in programs.items():
            (out / f"{key}.dxbc").write_bytes(container)

        print(f"  decoded programs written to {out}")

    # 3. tensor-core route layer table
    print("\n3. tensor-core route layer table (src/MfgPort/Cuda/cuda_layers.h)")
    rows = layer_table(repo)
    intel = [r for r in rows if r[0] not in PORTED_FLOW_PROGRAMS]
    present = [r for r in intel if r[1] in by_hash]

    for program, sha, kernel, own in intel:
        if sha not in by_hash:
            check(False, f"layer program {program} ({sha[:12]}) isn't in this build")

    check(len(present) == len(intel),
          f"{len(present)} of the table's {len(intel)} Intel programs are in this build, byte for byte "
          f"({len(rows) - len(intel)} more are the ported flow programs of this repository's own set)")

    # 4. Vulkan models
    if args.vulkan_programs:
        print("\n4. Vulkan models (programs.tsv)")
        wanted = vulkan_intel_programs(args.vulkan_programs)
        missing = [(i, sha) for i, sha in wanted if sha not in by_hash]

        for ident, sha in missing:
            check(False, f"Vulkan program {ident} ({sha[:12]}) isn't in this build")

        check(not missing, f"{len(wanted) - len(missing)} of the {len(wanted)} Intel programs of the Vulkan models are in this build")

    # 5. layout family
    if not args.no_layout:
        print("\n5. MFG port layout (src/MfgPort/RuntimeLayout.h)")
        results, chosen = layout_discover(pe, repo)

        for family, error in results.items():
            print(f"  {family}: {'found' if error is None else 'rejected, ' + error}")

        if check(chosen is not None, "a family of the generated table matches this build"):
            family, addresses = chosen

            for text, ok in shader_port_checks(pe, addresses, repo):
                check(ok, f"ShaderPort and ModelPort: {text}")

    print()

    if problems:
        print(f"{len(problems)} problem(s). The route or path that depends on the first of them won't work with this build.")
        return 1

    print("This build is what the repository expects, everywhere the file can say so.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (struct.error, ValueError, IndexError, AttributeError, KeyError, OSError) as error:
        if "--debug" in sys.argv:
            raise

        fail(f"can't check this file: {type(error).__name__}: {error} (--debug shows where)")
