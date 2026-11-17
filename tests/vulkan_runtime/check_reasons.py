#!/usr/bin/env python3
"""Every reason code the Vulkan runtime and its hooks can give has a text in VulkanPolicy.h.

There are two tables. Describe's Rows say why a device can't run XeMFG: the codes are string literals in VulkanHost.cpp
(Refuse(physical, "code")) and VulkanRuntime.cpp (the reason of the device, what Blocker() and NotPreparedReason() return).
DescribeTensor's TensorRows say why the tensor-core route isn't installed on a device that can: the codes are literals in
xvk_tensor.cpp (MFG-Unlocker's), VulkanTensor.cpp and InstallTensorRoute() of VulkanRuntime.cpp. The check reads those
sources and the tables and sees that they agree: no code without a text, no text twice, and no row that nothing can produce
except the ones kept for VulkanRuntimeOff builds and for the menu.

The reasons a single layer gives for staying with its SPIR-V program (dispatch_shape, bindings_incomplete, view_type, ...) go
to the log and not to a table: they aren't a state of the route.

    python3 check_reasons.py [SRC_DIR]
"""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "src")
policy = (src / "vulkan" / "VulkanPolicy.h").read_text(encoding="utf-8")


def table(name):
    # (?<![A-Za-z]): Rows is also the end of TensorRows
    found = re.search(r"(?<![A-Za-z])" + name + r"\[\]\s*=\s*\{(.*?)\n\t\t\};", policy, re.S)
    if not found:
        sys.exit(f"no {name}[] table in VulkanPolicy.h")
    return re.findall(r'\{\s*"([a-z_0-9]+)"\s*,\s*"', found.group(1))


def read(name):
    return (src / "vulkan" / name).read_text(encoding="utf-8")


def codes(text, regexes):
    found = set()
    for rx in regexes:
        for match in re.findall(rx, text):
            found.update(m for m in (match if isinstance(match, tuple) else (match,)) if m)
    return found


# The tensor-core route's install function of the runtime is cut out of VulkanRuntime.cpp: its codes are the tensor table's
runtime = read("VulkanRuntime.cpp")
start = runtime.index("void InstallTensorRoute(")
end = runtime.index("\n\t}\n", start) + 4
tensor_in_runtime = runtime[start:end]
runtime = runtime[:start] + runtime[end:]

device_used = codes(read("VulkanHost.cpp"), [r'Refuse\(\s*\w+\s*,\s*"([a-z_0-9]+)"'])
device_used |= codes(runtime, [
    r'\breason\s*=\s*"([a-z_0-9]+)"',
    r'\breturn\s+"([a-z_0-9]+)"\s*;',
    r'\b(?:gDeviceReason|d\.reason)\s*[!=]=\s*"([a-z_0-9]+)"',
])

tensor_used = codes(read("xvk_tensor.cpp"), [
    r'\breason\s*=\s*"([a-z_0-9]+)"',
    r'\breason\s*=\s*\w+\s*\?\s*"([a-z_0-9]+)"\s*:\s*"([a-z_0-9]+)"',
    r'reason\\":\\"([a-z_0-9]+)\\"',  # the report of a faulting driver call, JSON in a string literal
])
tensor_used |= codes(read("VulkanTensor.cpp"), [
    r'\breason\s*=\s*"([a-z_0-9]+)"',
    r'\bRefused\(\s*\w+\s*,\s*"([a-z_0-9]+)"',
])
tensor_used |= codes(tensor_in_runtime, [r'\breason\s*=\s*"([a-z_0-9]+)"'])

problems = []
summary = []

# The runtime reports no_hooks and no_loader without the sources returning them: they name builds and states the menu shows
for label, name, used, kept in (("device", "Rows", device_used, ("no_hooks", "no_loader")),
                                ("tensor", "TensorRows", tensor_used, ())):
    rows = table(name)
    dupes = sorted({r for r in rows if rows.count(r) > 1})
    missing = sorted(c for c in used if c not in rows)
    spare = sorted(c for c in rows if c not in used and c not in kept)

    if dupes:
        problems.append(f"{label}: rows twice: " + ", ".join(dupes))
    if missing:
        problems.append(f"{label}: codes without a text: " + ", ".join(missing))
    if spare:
        problems.append(f"{label}: texts nothing uses: " + ", ".join(spare))

    summary.append(f"{label} {len(rows)} texts, {len(used)} codes in the sources")

print("; ".join(summary))
if problems:
    print("FAIL " + "; ".join(problems))
    sys.exit(1)
print("ok")
