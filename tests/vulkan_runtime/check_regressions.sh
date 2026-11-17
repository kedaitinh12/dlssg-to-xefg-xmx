#!/usr/bin/env bash
# Linux regression tests, not a Windows or GPU validation.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
printf '#include "Windows.h"\n' > "$build/windows.h"
common=(-std=c++20 -g -fsanitize=address,undefined -fno-sanitize-recover=all
    -I "$root/src" -I "$here/stubs")
g++ "${common[@]}" "$here/feature_chain_test.cpp" -o "$build/feature-chain"
"$build/feature-chain"
# ASan retains unused global provider tables and their Windows dependencies.
# The copy harness uses UBSan; the feature-chain test above uses both sanitizers.
g++ "${common[@]}" -fno-sanitize=address -I "$build" -ffunction-sections -fdata-sections -Wl,--gc-sections \
    '-D__declspec(x)=' -DNOMINMAX -DUNICODE -D_UNICODE -DWIN32_LEAN_AND_MEAN \
    -DVK_USE_PLATFORM_WIN32_KHR -DVK_NO_PROTOTYPES "$here/copy_test.cpp" -o "$build/copy"
"$build/copy"
