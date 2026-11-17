#pragma once
#include "Windows.h"

extern "C++" BOOL EnumProcessModules(HANDLE process, HMODULE* modules, DWORD bytes, DWORD* needed);
