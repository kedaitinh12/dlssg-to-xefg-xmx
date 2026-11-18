#!/usr/bin/env python3
"""Extracts the XeMFG-on-Vulkan SPIR-V program set from MFGUnlocker.asi (standard library only, Python 3.8+).

MFG-Unlocker embeds the SPIR-V translations of every compute program XeSS Frame Generation creates on Vulkan as LZMA
compressed RCDATA resources, and describes them with two tables in the image's read-only data:

  * 12 byte records   { u32 id, u32 offset, u32 size }          where the SPIR-V module is at offset in the pack
  * 40 byte records   { char* sourceDxilSha256, u32 id, u32 pad, char* spirvSha256, u32 bytes,
                        u32 pushConstantBytes, u32 bindingStart, u32 bindingCount }
  * 4 byte records    { u16 binding, u16 type }               the descriptors of every program, bindingStart on, where
                        type is 0 storage buffer, 1 uniform buffer, 2 sampled image, 3 storage image, 4 sampler

The tables are located by scanning, not by fixed addresses, so any MFGUnlocker.asi with the same layout works, a build
this was never run on included. Every module is checked against the SHA-256 the image itself lists, against its own
SPIR-V structure (entry point, binding count) and against the DXIL set the image embeds, so a layout change fails loudly
instead of writing wrong data. The descriptor types of each program are read from its SPIR-V and then looked up in the
image's own binding table, which has to hold exactly those rows. --only-known refuses a build that isn't in KNOWN_BUILDS.

The host's three input-copy compute shaders (MFG-Unlocker's own vk_input_copy.comp, which resamples the game's depth and
motion vectors into XeFG's images) are compiled into the image as plain arrays. They are found by the SHA-256 their
generator (xess/vulkan/generate_vk_input_shaders.py) pins for each variant.

Output (--output DIR), the layout the dll loads from xefg_mfg_programs\\vulkan next to itself:
    DIR/programs.tsv        one row per program, see MANIFEST_COLUMNS
    DIR/<id>.spv            the SPIR-V modules
    DIR/input_copy_<n>.spv  the three input-copy shaders, n = 0 (r32f depth), 1 (rg16f) and 2 (rg32f motion vectors)
    DIR/SOURCE.txt          which MFGUnlocker.asi it came from (version, SHA-256) and the SHA-256 of programs.tsv
    DIR/LICENSE             MFG-Unlocker's MIT license
    DIR/XEFG-PROGRAMS-NOTICE.txt   MFG-Unlocker's note on the Intel derived program data (resource 104 of the image)

The Intel derived programs stay subject to Intel's license terms, so the output is a build artifact, not a source file.
"""
import argparse
import hashlib
import lzma
import os
import re
import struct
import sys

# The MFGUnlocker.asi builds this was run on, SHA-256 -> (ProductVersion, FileVersion). All of them embed the same
# program set, see docs/VULKAN-MODELS.md
KNOWN_BUILDS = {
    "8748f4498ad00b8cb85e8336d5ebf902247cddbf64aa9fe00dc78b719d3e88e5": ("1.7.1-beta2fix.1", "1.7.1.156"),
    "095a0321488f2905af849a02f6446bcb43dd22d41b3455d937287f3fa7e5505f": ("1.7.1-beta.3", "1.7.1.165"),
    "171ac18d4d632d12dee22d5caf2085f301bacfdb9a462561d93dbcbb61d6a4f8": ("1.7.1-rc.2", "1.7.1.173"),
    "5a389e6cad23b2b607e1dc224e9e86ee32d6ef46f0557b6c6f8137feadf0aa74": ("1.7.1", "1.7.1.177"),
}

# programs.tsv of that program set
KNOWN_MANIFEST_SHA256 = "18ce39ef3e69c064ba46ee70738b2e9cbf13e5fb000780e7061efc0543724079"

MANIFEST_COLUMNS = ("id", "sha256", "bytes", "pushBytes", "bindings", "localSize", "sourceDxilSha256", "origin", "portKeys",
                    "bindingTypes")

# MFG-Unlocker's xess/vulkan/generate_vk_input_shaders.py pins the SPIR-V of each variant of vk_input_copy.comp
INPUT_COPY_SHA256 = (
    "f6f48f06d01134d9596b117e0c98b4a1a098325ab5b67268eedee8554955b7ca",  # variant 0, r32f (depth)
    "112a67e0df42b09edab40be77f505d58f5b2d3b357bda64a45f145fb247ce0d3",  # variant 1, rg16f (motion vectors)
    "2e34c018e32f1a5fd819e93e2e46cf31c454312c8d3f53ae22d491d2e1470fbc",  # variant 2, rg32f (motion vectors)
)

# The descriptor types of a binding (xvk's BindingRow), the values programs.tsv carries
BINDING_TYPES = ("storage buffer", "uniform buffer", "sampled image", "storage image", "sampler")

SPIRV_MAGIC = b"\x03\x02\x23\x07"

LICENSE_TEXT = """MIT License

Copyright (c) 2026 Michael Robles

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class ExtractError(Exception):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------------------------------------------------
# PE image
# --------------------------------------------------------------------------------------------------------------------
class Image:
    def __init__(self, data):
        self.data = data
        if data[:2] != b"MZ":
            raise ExtractError("not a PE image")
        lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        if data[lfanew:lfanew + 4] != b"PE\0\0":
            raise ExtractError("not a PE image")
        nsec, = struct.unpack_from("<H", data, lfanew + 6)
        optsize, = struct.unpack_from("<H", data, lfanew + 20)
        opt = lfanew + 24
        if struct.unpack_from("<H", data, opt)[0] != 0x20B:
            raise ExtractError("not a PE32+ image")
        self.base, = struct.unpack_from("<Q", data, opt + 24)
        self.resource_rva, = struct.unpack_from("<I", data, opt + 112 + 2 * 8)
        self.sections = []
        for i in range(nsec):
            at = lfanew + 24 + optsize + i * 40
            vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", data, at + 8)
            self.sections.append((va, max(vsize, rawsize), rawptr, rawsize))

    def rva_to_offset(self, rva):
        for va, span, rawptr, rawsize in self.sections:
            if va <= rva < va + span:
                off = rva - va
                if off >= rawsize:
                    break
                return rawptr + off
        raise ExtractError("RVA 0x%x is not backed by the file" % rva)

    def va_to_offset(self, va):
        return self.rva_to_offset(va - self.base)

    def rcdata(self):
        """{resource id: bytes} of every RT_RCDATA resource"""
        return self.resources(10)

    def version_strings(self):
        """{name: value} of the first string table of the image's RT_VERSION resource, {} when it has none"""
        try:
            blobs = self.resources(16)
        except ExtractError:
            return {}
        if not blobs:
            return {}
        text = blobs[min(blobs)].decode("utf-16le", "ignore")
        found = {}
        for name in ("FileVersion", "ProductVersion", "ProductName", "FileDescription", "OriginalFilename"):
            match = re.search(name + r"\x00+([^\x00]{1,100})", text)
            if match:
                found[name] = match.group(1)
        return found

    def resources(self, type_id):
        """{resource id: bytes} of every resource of one type (10 RT_RCDATA, 16 RT_VERSION)"""
        if not self.resource_rva:
            raise ExtractError("the image has no resources")
        d = self.data
        root = self.rva_to_offset(self.resource_rva)
        out = {}

        def entries(off):
            named, ids = struct.unpack_from("<HH", d, off + 12)
            for i in range(named + ids):
                name, target = struct.unpack_from("<II", d, off + 16 + i * 8)
                yield name, target

        for type_name, type_target in entries(root):
            if type_name != type_id or not type_target & 0x80000000:
                continue
            for res_id, id_target in entries(root + (type_target & 0x7FFFFFFF)):
                if res_id & 0x80000000 or not id_target & 0x80000000:
                    continue
                lang_dir = root + (id_target & 0x7FFFFFFF)
                for _, lang_target in entries(lang_dir):
                    if lang_target & 0x80000000:
                        continue
                    rva, size, _, _ = struct.unpack_from("<IIII", d, root + lang_target)
                    start = self.rva_to_offset(rva)
                    out[res_id] = d[start:start + size]
                    break
        return out


def lzma_alone(blob):
    """The resources are .lzma ('LZMA alone') streams with an unknown size and an end marker"""
    try:
        return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(blob)
    except lzma.LZMAError:
        return None


# --------------------------------------------------------------------------------------------------------------------
# SPIR-V
# --------------------------------------------------------------------------------------------------------------------
def spirv_walk(blob):
    if len(blob) % 4 or len(blob) < 20:
        raise ExtractError("SPIR-V size is not a whole number of words")
    words = struct.unpack("<%dI" % (len(blob) // 4), blob)
    if words[0] != 0x07230203:
        raise ExtractError("bad SPIR-V magic")
    i = 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 0xFFFF
        if count == 0 or i + count > len(words):
            raise ExtractError("SPIR-V instruction runs past the module")
        yield op, words[i + 1:i + count]
        i += count


def spirv_string(words):
    raw = b"".join(struct.pack("<I", w) for w in words)
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


def spirv_summary(blob):
    """(entry point names, local size, descriptor binding count) of a compute module"""
    entries, local, names, decorations, variables = [], None, {}, {}, {}
    for op, a in spirv_walk(blob):
        if op == 15:  # OpEntryPoint
            entries.append((a[0], spirv_string(a[2:])))
        elif op == 16 and a[1] == 17:  # OpExecutionMode LocalSize
            local = tuple(a[2:5])
        elif op == 71:  # OpDecorate
            decorations.setdefault(a[0], {})[a[1]] = a[2:]
        elif op == 59:  # OpVariable
            variables[a[1]] = a[2]
    bindings = sum(1 for v in variables if 33 in decorations.get(v, {}) and 34 in decorations.get(v, {}))
    return entries, local, bindings


def spirv_bindings(blob):
    """[(binding, type)] of every descriptor of a module, in binding order. type is the index into BINDING_TYPES: a
    struct in the StorageBuffer class (or BufferBlock in Uniform) is a storage buffer, Block in Uniform a uniform buffer,
    an image is sampled (1) or storage (2), a sampler is a sampler. Arrays are looked through. Anything else is an error"""
    types, decorations, variables = {}, {}, []
    for op, a in spirv_walk(blob):
        if op == 71:  # OpDecorate
            decorations.setdefault(a[0], {})[a[1]] = a[2:]
        elif op in (25, 26, 27, 28, 29, 30, 32):  # OpTypeImage, Sampler, SampledImage, Array, RuntimeArray, Struct, Pointer
            types[a[0]] = (op, a[1:])
        elif op == 59:  # OpVariable: result type, result id, storage class
            variables.append((a[1], a[0], a[2]))

    found = {}
    for variable, type_id, storage in variables:
        decoration = decorations.get(variable, {})
        if 33 not in decoration or 34 not in decoration:
            continue
        binding = decoration[33][0]
        if decoration[34][0] != 0:
            raise ExtractError("binding %d is not in descriptor set 0" % binding)
        if type_id not in types or types[type_id][0] != 32:
            raise ExtractError("binding %d is not a pointer variable" % binding)
        target = types[type_id][1][1]
        while target in types and types[target][0] in (28, 29):
            target = types[target][1][0]
        if target not in types:
            raise ExtractError("binding %d has a type this reader does not know" % binding)
        op, operands = types[target]
        if op == 26:
            kind = 4
        elif op == 25:
            kind = {1: 2, 2: 3}.get(operands[5])
        elif op == 30:
            block = decorations.get(target, {})
            kind = 0 if storage == 12 else 1 if storage == 2 and 2 in block else 0 if storage == 2 and 3 in block else None
        else:
            kind = None
        if kind is None:
            raise ExtractError("binding %d has a descriptor type this reader does not know" % binding)
        if binding in found:
            raise ExtractError("binding %d is declared twice" % binding)
        found[binding] = kind
    return sorted(found.items())


def find_input_copy(data):
    """{variant: SPIR-V module} of the input-copy shaders, found as the SPIR-V modules in the image whose SHA-256 the
    generator pins. They are plain arrays in the image: a module is cut at an OpFunctionEnd and hashed"""
    pinned = {digest: variant for variant, digest in enumerate(INPUT_COPY_SHA256)}
    found = {}
    pos = data.find(SPIRV_MAGIC)
    while pos != -1:
        if pos % 4 == 0:
            i, limit = pos + 20, min(len(data), pos + 16384)
            while i + 4 <= limit:
                word = struct.unpack_from("<I", data, i)[0]
                count, op = word >> 16, word & 0xFFFF
                if count == 0 or i + 4 * count > limit:
                    break
                i += 4 * count
                if op == 56:  # OpFunctionEnd
                    module = data[pos:i]
                    if sha256(module) in pinned:
                        found[pinned[sha256(module)]] = module
        pos = data.find(SPIRV_MAGIC, pos + 4)
    return found


def split_modules(pack):
    offsets = []
    p = pack.find(SPIRV_MAGIC)
    while p != -1:
        offsets.append(p)
        p = pack.find(SPIRV_MAGIC, p + 4)
    return offsets


# --------------------------------------------------------------------------------------------------------------------
# DXIL set (to say which Vulkan programs come from the project's own ported programs)
# --------------------------------------------------------------------------------------------------------------------
def dxbc_blobs(pack):
    out, off = [], 0
    while off + 32 <= len(pack) and pack[off:off + 4] == b"DXBC":
        size = struct.unpack_from("<I", pack, off + 24)[0]
        if size < 32 or off + size > len(pack):
            break
        out.append(pack[off:off + size])
        off += size
    return out, pack[off:]


def read_program_manifest(tail):
    """programs.tsv rows (decimal key, sha256) when the pack ends with the manifest"""
    rows = []
    for line in tail.decode("ascii", "replace").split("\r\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not re.fullmatch(r"[0-9]{4,24}", parts[0]) or not re.fullmatch(r"[0-9a-f]{64}", parts[1]):
            return []
        rows.append(tuple(parts))
    return rows


# --------------------------------------------------------------------------------------------------------------------
# The two tables
# --------------------------------------------------------------------------------------------------------------------
def find_index_table(img, offsets, sizes):
    """{id: (offset, size)} from the 12 byte records, found by the (offset, size) of the second module"""
    d = img.data
    pattern = struct.pack("<II", offsets[1], sizes[1])
    pos = d.find(pattern)

    while pos != -1:
        first = pos - 4 - 12  # a record is {id, offset, size}: the id of record 1 is just before its offset

        if first >= 0 and pos % 4 == 0 and first + 12 * len(offsets) <= len(d):
            first_id = struct.unpack_from("<I", d, first)[0]
            table = {}

            for i in range(len(offsets)):
                rid, off, size = struct.unpack_from("<III", d, first + 12 * i)

                if (rid, off, size) != (first_id + i, offsets[i], sizes[i]):
                    table = None
                    break

                table[rid] = (off, size)

            if table:
                return table

        pos = d.find(pattern, pos + 1)

    raise ExtractError("the SPIR-V index table was not found")


def read_hex_string(img, va):
    off = img.va_to_offset(va)
    text = img.data[off:off + 64]
    if not re.fullmatch(rb"[0-9a-f]{64}", text) or img.data[off + 64:off + 65] != b"\0":
        raise ExtractError("a program record does not point at a SHA-256 string")
    return text.decode("ascii")


def find_program_table(img, ids):
    """Program records, found by the first ids at a 40 byte stride"""
    d = img.data
    first = struct.pack("<I", ids[0])
    pos = d.find(first)
    while pos != -1:
        if all(pos + 40 * i + 4 <= len(d) and struct.unpack_from("<I", d, pos + 40 * i)[0] == ids[i]
               for i in range(min(4, len(ids)))):
            records = []
            try:
                for i, rid in enumerate(ids):
                    # The resource id is at +8, not the start of ProgramAsset.
                    # Starting here would silently read the NEXT row's DXIL hash.
                    at = pos - 8 + 40 * i
                    if at < 0 or at + 40 > len(d):
                        raise ExtractError("truncated program record")
                    f = struct.unpack_from("<QIIQIIII", d, at)
                    if f[1] != rid or f[2] != 0:
                        raise ExtractError("record %d is not in order" % i)
                    records.append(dict(
                        id=f[1], sha256=read_hex_string(img, f[3]), bytes=f[4], push=f[5], bind_start=f[6],
                        bind_count=f[7], source=read_hex_string(img, f[0]) if f[0] else None))
                return records
            except ExtractError:
                pass
        pos = d.find(first, pos + 1)
    raise ExtractError("the SPIR-V program table was not found")


# --------------------------------------------------------------------------------------------------------------------
def binding_text(rows):
    """"binding:type,binding:type", "-" without any"""
    return ",".join("%d:%d" % row for row in rows) or "-"


def manifest_line(rec, local, origin, keys):
    return "\t".join([
        str(rec["id"]), rec["sha256"], str(rec["bytes"]), str(rec["push"]), str(rec["bind_count"]),
        "x".join(str(v) for v in local) if local else "-", rec["source"] or "-", origin, keys,
        binding_text(rec["binding_rows"])])


def read_programs(data):
    """Reads the program set of an MFGUnlocker.asi image and checks it, without writing anything.

    Returns a dict: rows ((record, SPIR-V module, local size, origin, port keys) in id order), manifest (the bytes of
    programs.tsv), pack_id (the RCDATA resource the modules came from), image (the parsed PE) and notice."""
    img = Image(data)
    packs = {}
    for res_id, blob in img.rcdata().items():
        decoded = lzma_alone(blob)
        if decoded:
            packs[res_id] = decoded

    spirv_packs = {rid: p for rid, p in packs.items() if p[:4] == SPIRV_MAGIC}
    dxil_packs = {rid: p for rid, p in packs.items() if p[:4] == b"DXBC"}
    if not spirv_packs:
        raise ExtractError("no SPIR-V pack among the compressed resources")

    # The XeFG program pack is the one whose modules all have a single 'main' compute entry point
    chosen = None
    for rid, pack in sorted(spirv_packs.items()):
        offsets = split_modules(pack)
        ends = offsets[1:] + [len(pack)]
        try:
            ok = all(spirv_summary(pack[o:e])[0] and [n for _, n in spirv_summary(pack[o:e])[0]] == ["main"]
                     for o, e in zip(offsets, ends))
        except ExtractError:
            ok = False
        if ok and (chosen is None or len(offsets) > len(chosen[2])):
            chosen = (rid, pack, offsets)
    if not chosen:
        raise ExtractError("no SPIR-V pack made of 'main' compute programs")

    pack_id, pack, offsets = chosen
    sizes = [e - o for o, e in zip(offsets, offsets[1:] + [len(pack)])]

    index = find_index_table(img, offsets, sizes)
    ids = sorted(index)
    records = find_program_table(img, ids)

    # DXIL the image carries: hash -> program keys of the pinned set, and the replacement programs
    port_hashes, port_keys = {}, {}
    for rid, p in dxil_packs.items():
        blobs, tail = dxbc_blobs(p)
        for b in blobs:
            port_hashes[sha256(b)] = rid
        for key, h in read_program_manifest(tail):
            port_keys.setdefault(h, []).append(key)

    rows = []
    for rec in records:
        off, size = index[rec["id"]]
        module = pack[off:off + size]
        if sha256(module) != rec["sha256"]:
            raise ExtractError("program %d does not match the SHA-256 the image lists" % rec["id"])
        if size != rec["bytes"]:
            raise ExtractError("program %d size differs from its record" % rec["id"])
        entries, local, bindings = spirv_summary(module)
        if [n for _, n in entries] != ["main"] or bindings != rec["bind_count"]:
            raise ExtractError("program %d does not match its record (entry %s, %d bindings, expected %d)"
                               % (rec["id"], entries, bindings, rec["bind_count"]))
        rec["binding_rows"] = spirv_bindings(module)
        if len(rec["binding_rows"]) != rec["bind_count"]:
            raise ExtractError("program %d: %d descriptor types read, its record says %d bindings"
                               % (rec["id"], len(rec["binding_rows"]), rec["bind_count"]))
        src = rec["source"]
        origin = "unhashed" if not src else "port" if src in port_keys else "port-extra" if src in port_hashes else "intel"
        rows.append((rec, module, local, origin, ",".join(port_keys.get(src, [])) or "-"))

    starts = [r[0]["bind_start"] for r in rows]
    counts = [r[0]["bind_count"] for r in rows]
    if any(starts[i + 1] != starts[i] + counts[i] for i in range(len(rows) - 1)):
        raise ExtractError("the binding ranges are not contiguous")

    # The image's own binding table has to hold exactly the rows read from the SPIR-V, one program after the other
    table = b"".join(struct.pack("<HH", binding, kind) for r in rows for binding, kind in r[0]["binding_rows"])
    table_offset = img.data.find(table) if table else -1

    lines = [manifest_line(rec, local, origin, keys) for rec, module, local, origin, keys in rows]
    manifest = ("\r\n".join(lines) + "\r\n").encode("ascii")

    input_copy = find_input_copy(data)
    if sorted(input_copy) != list(range(len(INPUT_COPY_SHA256))):
        raise ExtractError("the input-copy shaders the host needs were not all found (%d of %d)"
                           % (len(input_copy), len(INPUT_COPY_SHA256)))

    return dict(rows=rows, manifest=manifest, pack_id=pack_id, pack_size=len(pack), image=img,
                notice=img.rcdata().get(104), binding_table=table_offset, binding_rows=len(table) // 4,
                input_copy=input_copy)


def origin_counts(rows):
    counts = {}
    for row in rows:
        counts[row[3]] = counts.get(row[3], 0) + 1
    return counts


def describe_build(data, img):
    """(ProductVersion, FileVersion, known) of an MFGUnlocker.asi image"""
    digest = sha256(data)
    version = img.version_strings()
    known = KNOWN_BUILDS.get(digest)
    product = version.get("ProductVersion") or (known[0] if known else "unknown")
    file_version = version.get("FileVersion") or (known[1] if known else "unknown")
    return product, file_version, known is not None


def extract(asi_path, output, only_known):
    with open(asi_path, "rb") as f:
        data = f.read()
    digest = sha256(data)

    program_set = read_programs(data)
    img = program_set["image"]
    rows, manifest = program_set["rows"], program_set["manifest"]
    product, file_version, known = describe_build(data, img)

    print("MFGUnlocker.asi %s (file version %s), sha256 %s" % (product, file_version, digest))
    if not known:
        message = ("this build is not one of the %d the extractor was run on; every program is still verified against "
                   "the SHA-256 table in the image" % len(KNOWN_BUILDS))
        if only_known:
            raise ExtractError(message + ". Drop --only-known to use it anyway")
        print("note:", message, file=sys.stderr)

    print("SPIR-V pack: resource %d, %d bytes, %d modules" % (program_set["pack_id"], program_set["pack_size"], len(rows)))
    if program_set["binding_table"] >= 0:
        print("binding table: %d rows, the image's own table at file offset 0x%X holds exactly these"
              % (program_set["binding_rows"], program_set["binding_table"]))
    else:
        print("note: the image's binding table was not found, the descriptor types come from the SPIR-V alone",
              file=sys.stderr)

    os.makedirs(output, exist_ok=True)
    for rec, module, local, origin, keys in rows:
        with open(os.path.join(output, "%d.spv" % rec["id"]), "wb") as f:
            f.write(module)
    for variant, module in sorted(program_set["input_copy"].items()):
        with open(os.path.join(output, "input_copy_%d.spv" % variant), "wb") as f:
            f.write(module)
    with open(os.path.join(output, "programs.tsv"), "wb") as f:
        f.write(manifest)
    with open(os.path.join(output, "LICENSE"), "wb") as f:
        f.write(LICENSE_TEXT.replace("\n", "\r\n").encode("ascii"))

    if program_set["notice"]:
        with open(os.path.join(output, "XEFG-PROGRAMS-NOTICE.txt"), "wb") as f:
            f.write(program_set["notice"])

    counts_by_origin = origin_counts(rows)
    summary = ", ".join("%s %d" % (name, counts_by_origin[name]) for name in sorted(counts_by_origin))
    source = [
        "MFG Unlocker build: %s (file version %s)" % (product, file_version),
        "MFGUnlocker.asi sha256: %s" % digest,
        "programs: %d (%s)" % (len(rows), summary),
        "programs.tsv sha256: %s" % sha256(manifest),
        "input-copy shaders: %d, the SHA-256 their generator pins" % len(program_set["input_copy"]),
        "binding types: %s" % ("match the image's own table" if program_set["binding_table"] >= 0
                               else "read from the SPIR-V, not found in the image's table"),
        "extracted by tools/ExtractXeFgVulkanPrograms.py",
    ]
    with open(os.path.join(output, "SOURCE.txt"), "wb") as f:
        f.write(("\r\n".join(source) + "\r\n").encode("ascii"))

    print("wrote %d programs and %d input-copy shaders to %s" % (len(rows), len(program_set["input_copy"]), output))
    print("origins:", counts_by_origin)
    print("programs.tsv sha256:", sha256(manifest))
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asi", help="MFGUnlocker.asi")
    parser.add_argument("--output", required=True, help="folder to write programs.tsv and the .spv files to")
    parser.add_argument("--only-known", action="store_true", help="refuse an MFGUnlocker.asi that isn't in KNOWN_BUILDS")
    parser.add_argument("--any-build", action="store_true", help=argparse.SUPPRESS)  # the default now, kept for old callers
    args = parser.parse_args()

    try:
        extract(args.asi, args.output, args.only_known)
    except ExtractError as error:
        print("error:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
