#!/usr/bin/env python3
"""Checks tools/CheckXeFgBuild.py without a libxess_fg.dll (Intel's, so it isn't in the repository).

    python3 tests/xefg_build/check_tool.py

What the tool says about a build is only as good as its reading of this repository's own tables and of the file, so this
checks both ends that need no Intel data:

 1. its zstd decoder reproduces the known answers (--selftest), which is also what says libzstd or the zstandard package works;
 2. it still reads the tables it compares a build with: the builds of src/XeFG.cpp, the layer table of cuda_layers.h, the
    families of RuntimeLayout.generated.h (a change of their format would otherwise make it check nothing) and the models
    list of programs.tsv;
 3. what isn't a PE file, or is a damaged one, ends with exit code 2 and a message, never a traceback, and a file that is a PE
    file with nothing in it ends with 1.

Exit code 0 when everything holds. The run against a real libxess_fg.dll is `tools/CheckXeFgBuild.py` itself (build.yml does it
for third_party/libxess_fg.dll when there is one).
"""
import importlib.util
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "CheckXeFgBuild.py"

failures = []


def check(ok, text):
    print(("  ok    " if ok else "  FAIL  ") + text)

    if not ok:
        failures.append(text)


def read(what, function, *arguments):
    """What a table reader returns, or a failed check (the tool ends with fail() when it can't find its table)."""
    try:
        return function(*arguments)
    except (SystemExit, Exception) as error:
        check(False, f"{what}: {type(error).__name__} {error}")
        return None


def run(*args):
    done = subprocess.run([sys.executable, "-B", "-I", str(TOOL), *args], capture_output=True, text=True, timeout=120)
    return done.returncode, done.stdout, done.stderr


spec = importlib.util.spec_from_file_location("CheckXeFgBuild", TOOL)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

print("1. the zstd decoder")
code, out, err = run("--selftest")
check(code == 0 and "zstd decoder: ok" in out, f"--selftest: exit {code}, {(out or err).strip()}")

print("\n2. the tables of the repository")
builds = read("src/XeFG.cpp", tool.known_builds, ROOT) or []
check({b["stamp"] for b in builds} >= {0x69CB0F4D, 0x6A95ED0F}, f"src/XeFG.cpp: {len(builds)} builds, 1.3.1 and 1.3.4 among them")
check(bool(builds) and all(len(b["api"]) == 6 and all(b["api"].values()) for b in builds),
      "src/XeFG.cpp: six function addresses for each")

rows = read("cuda_layers.h", tool.layer_table, ROOT) or []
programs = {row[0] for row in rows}
intel = [row for row in rows if row[0] not in tool.PORTED_FLOW_PROGRAMS]
check(len(rows) == 17, f"cuda_layers.h: {len(rows)} layers (17 expected, the mask 1FFFF of the route)")
check(tool.PORTED_FLOW_PROGRAMS <= programs, "cuda_layers.h: the ported flow programs the tool leaves out are all in the table")
check(len(intel) == 13, f"cuda_layers.h: {len(intel)} Intel programs (13 expected)")
check(all(len(row[1]) == 64 and set(row[1]) <= set("0123456789abcdef") for row in rows), "cuda_layers.h: a SHA-256 for each")


class Nothing:
    """A file image with nothing in it: every witness of the layout families is searched for and none is found."""
    sections = []
    size_of_image = 0

    def read(self, rva, size):
        return None

    def functions(self):
        return []

    def in_range(self, *arguments):
        return False


results, chosen = read("RuntimeLayout.generated.h", tool.layout_discover, Nothing(), ROOT) or ({}, None)
check({"fg78", "fg68", "fg95"} <= set(results), f"RuntimeLayout.generated.h: families {', '.join(sorted(results))}")
# A family whose arrays can't be read is rejected at the step "start"; one that was read and found nowhere, at a witness
check(bool(results) and all(error and error.startswith("witness 0: ") and "matched 0 times" in error for error in results.values())
      and chosen is None, "RuntimeLayout.generated.h: every family's witnesses are read (and none is found in nothing)")

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    header = "\t".join(["id", "spv", "size", "cb", "nres", "group", "dxil", "kind", "keys", "bindings"])
    (folder / "programs.tsv").write_text("\n".join([
        header,
        "\t".join(["a1", "s1", "1", "0", "1", "8", "d1", "intel", "k", "b"]),
        "\t".join(["a2", "s2", "1", "0", "1", "8", "d2", "port", "k", "b"]),
        "\t".join(["a3", "s3", "1", "0", "1", "8", "d3", "intel", "k", "b"]),
        "\t".join(["a4", "s4", "1", "0", "1", "8", "d4", "unhashed", "k", "b"]),
    ]) + "\n", encoding="utf-8")
    check(tool.vulkan_intel_programs(folder) == [("a1", "d1"), ("a3", "d3")], "programs.tsv: the Intel programs and their SHA-256")

print("\n3. files that aren't what the tool expects")

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)

    plain = folder / "plain.bin"
    plain.write_bytes(b"not a PE file " * 100)
    code, out, err = run(str(plain))
    check(code == 2 and "not a PE file" in err and "Traceback" not in err, f"a file that isn't a PE file: exit {code}, {err.strip()}")

    code, out, err = run(str(folder / "missing.dll"))
    check(code == 2 and "isn't a file" in err and "Traceback" not in err, f"a file that isn't there: exit {code}, {err.strip()}")

    code, out, err = run()
    check(code == 2 and "Traceback" not in err, f"no arguments: exit {code}")

    # A PE header with no sections (damaged), and one whose only section lies beyond the end of the file (nothing in it)
    header = bytearray(0x400)
    header[0:2] = b"MZ"
    struct.pack_into("<I", header, 0x3C, 0x80)
    header[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHI", header, 0x84, 0x8664, 0, 0x12345678)
    struct.pack_into("<H", header, 0x80 + 20, 0xF0)
    struct.pack_into("<H", header, 0x80 + 24, 0x20B)

    damaged = folder / "no_sections.bin"
    damaged.write_bytes(bytes(header))
    code, out, err = run(str(damaged))
    check(code == 2 and "damaged PE file" in err and "Traceback" not in err, f"a PE header without sections: exit {code}, {err.strip()}")

    hollow = bytearray(header)
    struct.pack_into("<HHI", hollow, 0x84, 0x8664, 1, 0x12345678)
    section = 0x80 + 24 + 0xF0
    hollow[section:section + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", hollow, section + 8, 0x100000, 0x1000, 0x100000, 0x2000)
    struct.pack_into("<I", hollow, section + 36, 0x60000020)
    path = folder / "hollow.bin"
    path.write_bytes(bytes(hollow))
    code, out, err = run(str(path))
    check(code == 1 and "Traceback" not in err and "problem(s)" in out, f"a PE file with nothing in it: exit {code}, says what it lacks")

print()

if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)

print("all checks passed")
