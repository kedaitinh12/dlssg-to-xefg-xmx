// Wine / Proton (Linux) runtime detection helper (MFG-Unlocker winequeue.1 pattern).
// Zero overhead on native Windows: caches GetProcAddress(ntdll, "wine_get_version") once per process.
#pragma once

#include <Windows.h>

namespace WineCompat
{
	inline bool IsWineOrProton() noexcept
	{
		static const bool underWine = []() noexcept {
			const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
			return ntdll && GetProcAddress(ntdll, "wine_get_version") != nullptr;
		}();
		return underWine;
	}
} // namespace WineCompat
