#pragma once

// Windows side of the .session marker shared by the early loader and the main DLL: which exe this process is, and
// whether the process a marker names is still running. The decisions themselves live in EarlyLoaderCore.h.

#include "EarlyLoaderCore.h"

#include <Windows.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace EarlyLoader
{
	inline std::wstring CurrentExePathW()
	{
		std::wstring path(MAX_PATH, L'\0');
		for (;;)
		{
			const DWORD len = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
			if (len == 0)
				return {};
			if (len < path.size())
			{
				path.resize(len);
				return path;
			}
			if (path.size() >= 32768)
				return {};
			path.resize(path.size() * 2);
		}
	}

	inline bool IsCurrentProcessHelper()
	{
		return IsHelperProcessExe(CurrentExePathW());
	}

	inline std::string NarrowForCompare(const std::wstring& wide)
	{
		if (wide.empty())
			return {};
		const int len = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0,
			nullptr, nullptr);
		if (len <= 0)
			return {};
		std::string out(static_cast<size_t>(len), '\0');
		WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), out.data(), len, nullptr, nullptr);
		return out;
	}

	// The marker's pid is a running process of the marker's exe. A pid we may not open is taken as running (another
	// user's or a protected process holds it); a pid reused by a different program counts as gone.
	inline bool IsMarkerOwnerAlive(const SessionMarkerInfo& marker)
	{
		if (marker.pid == 0)
			return false;

		HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, marker.pid);
		if (!process)
			return GetLastError() == ERROR_ACCESS_DENIED;

		bool alive = false;
		DWORD exitCode = 0;
		if (GetExitCodeProcess(process, &exitCode) && exitCode == STILL_ACTIVE)
		{
			alive = true;
			std::wstring image(MAX_PATH, L'\0');
			DWORD size = static_cast<DWORD>(image.size());
			if (QueryFullProcessImageNameW(process, 0, image.data(), &size))
			{
				image.resize(size);
				alive = SameExeName(marker.exe, NarrowForCompare(image));
			}
		}
		CloseHandle(process);
		return alive;
	}

	inline SessionMarkerInfo ReadSessionMarker(const std::filesystem::path& markerPath)
	{
		std::ifstream in(markerPath);
		if (!in.is_open())
			return {};
		std::string line;
		std::getline(in, line);
		if (line.size() > 256)
			line.resize(256);
		return ParseSessionMarker(line);
	}
} // namespace EarlyLoader
