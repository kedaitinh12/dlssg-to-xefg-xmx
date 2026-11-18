#!/usr/bin/env python3
"""Reject incomplete, empty or corrupted cooperative-matrix build output."""
import argparse
import hashlib
from pathlib import Path
import struct
from GenerateXeFgCoopMat import KERNELS


def check(root: Path) -> None:
    expected = {row[0] for row in KERNELS}
    rows = {}
    for line in (root / 'coopmat_layers.tsv').read_text(encoding='utf-8').splitlines():
        name, digest, size = line.split()
        if name in rows or name not in expected:
            raise ValueError(f'unexpected or duplicate kernel: {name}')
        rows[name] = (digest, int(size))
    if set(rows) != expected:
        raise ValueError(f'missing kernels: {sorted(expected - set(rows))}')
    for name, (digest, size) in rows.items():
        data = (root / f'{name}.spv').read_bytes()
        if len(data) < 20 or len(data) % 4 or struct.unpack_from('<I', data)[0] != 0x07230203:
            raise ValueError(f'{name}: invalid SPIR-V header or length')
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'{name}: size or SHA-256 mismatch')
    print(f'OK: {len(rows)} cooperative-matrix shaders, sizes and SHA-256 verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        check(args.directory)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'error: {exc}\n')
