#!/usr/bin/env bash
# Syntax checks the Vulkan runtime's Windows-only sources on Linux, against the stubs in tests/vulkan_runtime/stubs
# (the parts of the Win32, D3D12, WRL and Vulkan APIs the sources use) and the real MinHook header.
#
# What it proves: the sources parse, every name they use is declared, calls and structure members match the
# declarations, and the values VulkanRuntime.cpp static_asserts hold. What it can't prove: that the stubs are the real
# APIs (the build on Windows is the check of that), or anything about how the code behaves. Nothing is linked or run.
#
# Usage: bash tests/vulkan_runtime/check_syntax.sh [compiler ...]   (default: every one of clang++ and g++ found)
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

compilers=("$@")
if [ ${#compilers[@]} -eq 0 ]; then
	for c in clang++ g++; do
		command -v "$c" > /dev/null 2>&1 && compilers+=("$c")
	done
fi
if [ ${#compilers[@]} -eq 0 ]; then
	echo "no C++ compiler found" >&2
	exit 2
fi

sources=(
	src/vulkan/VulkanHost.cpp
	src/vulkan/VulkanRuntime.cpp
	src/vulkan/VulkanTensor.cpp
	src/vulkan/VulkanCoopMat.cpp
	src/vulkan/xvk_extra.h
	src/vulkan/xvk_tensor.h
	src/vulkan/xvk_coopmat.h
	src/MfgPort/Cuda/cuda_route.h
	src/MfgPort/Cuda/cuda_layers.h
	src/MfgPort/Cuda/cuda_vulkan.cpp
	src/MfgPort/Wmma/wmma_route.h
	tests/vulkan_runtime/tensor_api_check.cpp
	src/VulkanProvider.cpp
	src/ExportsVulkan.cpp
	src/Plugin.cpp
)

# xvk.h includes <windows.h> in lower case, and a stubs folder can't have that next to Windows.h: a checkout on Windows, or an
# archive unpacked there, has one file for both names. So the forwarder is made here, outside the tree.
lower="$(mktemp -d)"
trap 'rm -rf "$lower"' EXIT
printf '#pragma once\n#include "Windows.h"\n' > "$lower/windows.h"

failed=0
for cxx in "${compilers[@]}"; do
	flags=(-std=c++20 -fsyntax-only -Wall -Wextra -Wno-missing-field-initializers -Wno-unknown-pragmas
		-Wno-ignored-attributes -Wno-cast-function-type
		-DNOMINMAX -DUNICODE -D_UNICODE -DWIN32_LEAN_AND_MEAN -DVK_USE_PLATFORM_WIN32_KHR -DVK_NO_PROTOTYPES
		-I "$lower" -I "$here/stubs" -I "$root/third_party/minhook/include" -I "$root/src" -I "$root/src/vulkan")
	if [ -n "${EXTRA_CXXFLAGS:-}" ]; then
		# shellcheck disable=SC2206
		flags+=(${EXTRA_CXXFLAGS})
	fi
	case "$cxx" in
		*clang*) flags+=(-fms-extensions -fdeclspec -Wno-microsoft -Wno-pragma-pack -Wno-pragma-once-outside-header) ;;
		*) flags+=("-D__declspec(attribute)=") ;; # GCC has no __declspec
	esac
	for s in "${sources[@]}"; do
		# A header is checked by including it into an empty file
		case "$s" in
			*.h) input=(-x c++ /dev/null -include "$root/$s") ;;
			*) input=("$root/$s") ;;
		esac
		if "$cxx" "${flags[@]}" "${input[@]}"; then
			echo "ok   $(basename "$cxx")  $s"
		else
			echo "FAIL $(basename "$cxx")  $s"
			failed=1
		fi
	done
done
exit $failed
