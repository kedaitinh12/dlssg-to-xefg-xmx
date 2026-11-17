// Just enough of the Win32 API for src/MfgKernels.h and src/MfgPort/Sha256.h to compile on Linux
#pragma once
#include <cstddef>
#include <cstdint>

struct HINSTANCE__;
using HMODULE = HINSTANCE__*;
using ULONG = uint32_t;
using PUCHAR = unsigned char*;
using NTSTATUS = long;
using LPCWSTR = const wchar_t*;
