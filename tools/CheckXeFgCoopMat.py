#!/usr/bin/env python3
"""Reject missing, duplicate or corrupted cooperative-matrix kernels and program aliases."""
import argparse
import csv
import hashlib
from pathlib import Path
import struct
from GenerateXeFgCoopMat import KERNELS


def validate_spirv(path, digest, size):
    data = path.read_bytes()
    if len(data) < 20 or len(data) % 4 or struct.unpack_from('<I', data)[0] != 0x07230203:
        raise ValueError(f'{path.name}: invalid SPIR-V header or length')
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'{path.name}: size or SHA-256 mismatch')


def check(root):
    expected = {row[0] for row in KERNELS}
    expected_aliases = {prog: (row[0], row[1], row[2], row[3], row[7], row[8])
                        for row in KERNELS for prog in row[-1]}
    if len(expected) != 16 or len(expected_aliases) != 17:
        raise ValueError('unexpected kernel or program count')
    rows = {}
    for line in (root / 'coopmat_layers.tsv').read_text(encoding='utf-8').splitlines():
        name, digest, size = line.split()
        if name in rows or name not in expected:
            raise ValueError(f'unexpected or duplicate kernel: {name}')
        rows[name] = (digest, int(size))
    if set(rows) != expected:
        raise ValueError(f'missing kernels: {sorted(expected - set(rows))}')
    for name, (digest, size) in rows.items():
        validate_spirv(root / f'{name}.spv', digest, size)
    seen = set()
    with (root / 'coopmat_programs.tsv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if reader.fieldnames != ['program','kernel','family','in','out','tile_h','grid_z','bytes','sha256']:
            raise ValueError('invalid program manifest header')
        for row in reader:
            prog, kernel = row['program'], row['kernel']
            info = (kernel, *(int(row[key]) for key in ('family','in','out','tile_h','grid_z')))
            if prog in seen or expected_aliases.get(prog) != info:
                raise ValueError(f'unexpected or duplicate alias: {prog}')
            digest, size = rows[kernel]
            if (row['sha256'], int(row['bytes'])) != (digest, size):
                raise ValueError(f'{prog}: alias manifest does not match kernel')
            validate_spirv(root / f'{prog}.spv', digest, size)
            seen.add(prog)
    if seen != set(expected_aliases):
        raise ValueError(f'missing aliases: {sorted(set(expected_aliases) - seen)}')
    print(f'OK: {len(rows)} cooperative-matrix kernels and {len(seen)} aliases, sizes and SHA-256 verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        check(args.directory)
        xmx_dir = args.directory / 'xmx_8x8x32'
        if xmx_dir.is_dir():
            check(xmx_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'error: {exc}\n')
