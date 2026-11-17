// Just enough of the Win32 API for VulkanProvider.cpp to compile and run on Linux against a fake process
#pragma once
#include <cstdint>
#include <cstddef>

#define WINAPI
#define MAX_PATH 260
#define CP_UTF8 65001

using DWORD = uint32_t;
using BOOL = int;
using UINT = unsigned int;
using ULONGLONG = unsigned long long;
using HANDLE = void*;
struct HINSTANCE__;
using HMODULE = HINSTANCE__*;
using LPCWSTR = const wchar_t*;
using FARPROC = long long (*)();

// The tests implement these
extern "C++"
{
	FARPROC GetProcAddress(HMODULE module, const char* name);
	DWORD GetModuleFileNameW(HMODULE module, wchar_t* path, DWORD size);
	HANDLE GetCurrentProcess();
	ULONGLONG GetTickCount64();
	int WideCharToMultiByte(UINT codePage, DWORD flags, const wchar_t* wide, int wideCount, char* narrow, int narrowCount,
		const char* defaultChar, BOOL* usedDefault);
}
