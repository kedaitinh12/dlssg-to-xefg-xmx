#!/usr/bin/env python3
"""Fails unless a built dll exports every function its sources declare as exported.

    python tools/CheckExports.py <dll> <source.cpp> [<source.cpp> ...]
    python tools/CheckExports.py --list <source.cpp> [<source.cpp> ...]

A source declares an export with NGX_EXPORT or __declspec(dllexport) in front of the function. The export table is read
from the PE file itself, so this needs neither Visual Studio's dumpbin nor the build tree, and it catches a source file
that dropped out of the project: that compiles and links fine, and then the dll exports nothing from it.

Standard library only.
"""
import re
import struct
import sys

DECLARATION = re.compile(r'(?:NGX_EXPORT|__declspec\s*\(\s*dllexport\s*\))\s+[^();{}]*?(\w+)\s*\(')


def declared_exports(path):
    """Names of the functions a source marks as exported, in file order."""
    with open(path, encoding='utf-8-sig') as source:
        text = source.read()
    text = re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)   # comments
    text = re.sub(r'(?m)^[ \t]*#.*$', '', text)                  # preprocessor lines, the NGX_EXPORT definition included
    return [match.group(1) for match in DECLARATION.finditer(text)]


def pe_exports(path):
    """Names in the export table of a PE32 or PE32+ file."""
    with open(path, 'rb') as image:
        data = image.read()
    if data[:2] != b'MZ' or len(data) < 0x40:
        raise SystemExit(f'{path}: not a PE file')
    pe = struct.unpack_from('<I', data, 0x3C)[0]
    if data[pe:pe + 4] != b'PE\0\0':
        raise SystemExit(f'{path}: no PE signature')
    section_count = struct.unpack_from('<H', data, pe + 6)[0]
    optional_size = struct.unpack_from('<H', data, pe + 20)[0]
    optional = pe + 24
    magic = struct.unpack_from('<H', data, optional)[0]
    if magic not in (0x10B, 0x20B):
        raise SystemExit(f'{path}: unknown optional header 0x{magic:X}')
    export_rva = struct.unpack_from('<I', data, optional + (112 if magic == 0x20B else 96))[0]
    if not export_rva:
        return set()

    sections = []
    first = optional + optional_size
    for i in range(section_count):
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from('<IIII', data, first + 40 * i + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_pointer))

    def offset(rva):
        for address, size, raw in sections:
            if address <= rva < address + size:
                return rva - address + raw
        raise SystemExit(f'{path}: RVA 0x{rva:X} is in no section')

    table = offset(export_rva)
    name_count = struct.unpack_from('<I', data, table + 24)[0]
    names = offset(struct.unpack_from('<I', data, table + 32)[0])
    exported = set()
    for i in range(name_count):
        start = offset(struct.unpack_from('<I', data, names + 4 * i)[0])
        exported.add(data[start:data.index(b'\0', start)].decode('ascii'))
    return exported


def main(argv):
    if len(argv) >= 3 and argv[1] == '--list':
        for source in argv[2:]:
            for name in declared_exports(source):
                print(name)
        return 0
    if len(argv) < 3:
        print(__doc__.strip().split('\n\n')[0], file=sys.stderr)
        print('usage: CheckExports.py <dll> <source.cpp> [<source.cpp> ...]', file=sys.stderr)
        return 2

    dll, sources = argv[1], argv[2:]
    declared = {}
    for source in sources:
        for name in declared_exports(source):
            declared.setdefault(name, source)
    if not declared:
        print('No exported functions are declared in the sources, nothing to check.', file=sys.stderr)
        return 2

    exported = pe_exports(dll)
    missing = [name for name in declared if name not in exported]
    for name in missing:
        print(f'MISSING {name} (declared in {declared[name]})')
    print(f'{len(declared) - len(missing)} of {len(declared)} declared exports are in {dll}')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
