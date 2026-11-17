#!/usr/bin/env python3
"""Windows loader smoke test; does not initialize NGX/OptiScaler or a GPU."""
import ctypes
import os
from pathlib import Path
import re
import sys

if sys.platform != 'win32':
    raise SystemExit('Run this test on Windows x64, not Linux.')
if ctypes.sizeof(ctypes.c_void_p) != 8:
    raise SystemExit('Use 64-bit Python.')
root = Path(__file__).resolve().parents[2]
dll = Path(sys.argv[1]).resolve(strict=True)
exports = {'OptiPluginInit', 'InitializeASI'}
for name in ('Exports.cpp', 'ExportsVulkan.cpp'):
    exports.update(re.findall(r'\b(NVSDK_NGX_\w+)\s*\(', (root / 'src' / name).read_text()))
with os.add_dll_directory(str(dll.parent)):
    module = ctypes.WinDLL(str(dll))
    for name in sorted(exports):
        getattr(module, name)
print(f'PASS: Windows loaded {dll.name}; resolved {len(exports)} exports. No GPU calls made.')
