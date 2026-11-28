// Just enough of the Win32 API to syntax check the Vulkan runtime's sources on Linux (tests/vulkan_runtime/check_syntax.sh).
// Nothing here runs, the declarations are the real ones' shapes.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cwchar>

#define WINAPI
#define CALLBACK
#define APIENTRY
#define MAX_PATH 260
#define CP_UTF8 65001
#define __stdcall
#define _M_X64 1

#define VOID void
using DWORD = uint32_t;
using BOOL = int;
using UINT = unsigned int;
using WORD = uint16_t;
using ULONGLONG = unsigned long long;
using LONG = long;
using HANDLE = void*;
using LPVOID = void*;
using LPCVOID = const void*;
using SIZE_T = size_t;
using LPCWSTR = const wchar_t*;
using LPWSTR = wchar_t*;
using LPCSTR = const char*;
using LPSTR = char*;
using HRESULT = int32_t;
struct HINSTANCE__;
using HMODULE = HINSTANCE__*;
using HINSTANCE = HINSTANCE__*;
struct HWND__;
using HWND = HWND__*;
using FARPROC = long long (*)();

struct LUID
{
	DWORD LowPart;
	LONG HighPart;
};

#define TRUE 1
#define FALSE 0
#define S_OK ((HRESULT)0)
#define E_FAIL ((HRESULT)0x80004005L)
#define E_INVALIDARG ((HRESULT)0x80070057L)
#define E_OUTOFMEMORY ((HRESULT)0x8007000EL)
#define FAILED(hr) (((HRESULT)(hr)) < 0)
#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)

#define LOAD_LIBRARY_AS_DATAFILE 0x00000002
#define LOAD_LIBRARY_AS_IMAGE_RESOURCE 0x00000020
#define LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE 0x00000040

#define GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT 0x00000002
#define GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS 0x00000004

struct GUID
{
	uint32_t Data1;
	uint16_t Data2;
	uint16_t Data3;
	uint8_t Data4[8];
};
using IID = GUID;
using REFIID = const IID&;
using REFGUID = const GUID&;

inline constexpr IID IID_Stub{};
#define __uuidof(x) IID_Stub
#define IID_PPV_ARGS(ppType) IID_Stub, reinterpret_cast<void**>(ppType)

extern "C++"
{
	FARPROC GetProcAddress(HMODULE module, const char* name);
	HMODULE GetModuleHandleW(LPCWSTR name);
	BOOL GetModuleHandleExW(DWORD flags, LPCWSTR nameOrAddress, HMODULE* module);
	HMODULE LoadLibraryW(LPCWSTR name);
	HMODULE LoadLibraryExW(LPCWSTR name, HANDLE file, DWORD flags);
	DWORD GetModuleFileNameW(HMODULE module, wchar_t* path, DWORD size);
	DWORD GetEnvironmentVariableW(LPCWSTR name, LPWSTR value, DWORD size);
	BOOL SetEnvironmentVariableW(LPCWSTR name, LPCWSTR value);
	#define ERROR_ALREADY_EXISTS 183L
	struct SYSTEMTIME
	{
		WORD wYear;
		WORD wMonth;
		WORD wDayOfWeek;
		WORD wDay;
		WORD wHour;
		WORD wMinute;
		WORD wSecond;
		WORD wMilliseconds;
	};
	void GetLocalTime(SYSTEMTIME* lpSystemTime);
	BOOL IsDebuggerPresent();
	void OutputDebugStringA(const char* lpOutputString);
	DWORD GetLastError();
	DWORD GetCurrentProcessId();
	DWORD GetCurrentThreadId();
	HANDLE GetCurrentProcess();
	#define MEM_COMMIT 0x00001000u
	#define PAGE_NOACCESS 0x01u
	#define PAGE_READONLY 0x02u
	#define PAGE_READWRITE 0x04u
	#define PAGE_WRITECOPY 0x08u
	#define PAGE_EXECUTE_READ 0x20u
	#define PAGE_EXECUTE_READWRITE 0x40u
	#define PAGE_EXECUTE_WRITECOPY 0x80u
	#define PAGE_GUARD 0x100u
	struct MEMORY_BASIC_INFORMATION
	{
		void* BaseAddress;
		void* AllocationBase;
		DWORD AllocationProtect;
		size_t RegionSize;
		DWORD State;
		DWORD Protect;
		DWORD Type;
	};
	size_t VirtualQuery(const void* lpAddress, MEMORY_BASIC_INFORMATION* lpBuffer, size_t dwLength);
	HANDLE CreateMutexW(void* lpMutexAttributes, BOOL bInitialOwner, LPCWSTR lpName);
	union LARGE_INTEGER { int64_t QuadPart; };
	BOOL QueryPerformanceCounter(LARGE_INTEGER* lpPerformanceCount);
	BOOL QueryPerformanceFrequency(LARGE_INTEGER* lpFrequency);
	BOOL SetWaitableTimer(HANDLE hTimer, const LARGE_INTEGER* lpDueTime, long lPeriod,
		void* pfnCompletionRoutine, void* lpArgToCompletionRoutine, BOOL fResume);
	DWORD WaitForSingleObject(HANDLE hHandle, DWORD dwMilliseconds);
	BOOL CloseHandle(HANDLE hObject);
	// The .session marker's owner check (src/loader/SessionMarkerWin.h)
	#define PROCESS_QUERY_LIMITED_INFORMATION 0x1000
	#define STILL_ACTIVE 259L
	#define ERROR_ACCESS_DENIED 5L
	HANDLE OpenProcess(DWORD dwDesiredAccess, BOOL bInheritHandle, DWORD dwProcessId);
	BOOL GetExitCodeProcess(HANDLE hProcess, DWORD* lpExitCode);
	BOOL QueryFullProcessImageNameW(HANDLE hProcess, DWORD dwFlags, wchar_t* lpExeName, DWORD* lpdwSize);
	#define FILE_SHARE_READ 0x00000001
	#define FILE_SHARE_WRITE 0x00000002
	#define FILE_SHARE_DELETE 0x00000004
	#define FILE_READ_ATTRIBUTES 0x0080
	#define OPEN_EXISTING 3
	#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
	struct BY_HANDLE_FILE_INFORMATION
	{
		DWORD dwVolumeSerialNumber;
		DWORD nFileIndexHigh;
		DWORD nFileIndexLow;
	};
	HANDLE CreateFileW(LPCWSTR lpFileName, DWORD dwDesiredAccess, DWORD dwShareMode,
		void* lpSecurityAttributes, DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes, HANDLE hTemplateFile);
	BOOL GetFileInformationByHandle(HANDLE hFile, BY_HANDLE_FILE_INFORMATION* lpFileInformation);
	void Sleep(DWORD dwMilliseconds);
	inline void YieldProcessor() {}
	inline BOOL SwitchToThread() { return 0; }
	ULONGLONG GetTickCount64();
	int WideCharToMultiByte(UINT codePage, DWORD flags, const wchar_t* wide, int wideCount, char* narrow, int narrowCount,
		const char* defaultChar, BOOL* usedDefault);
	#define ENUM_CURRENT_SETTINGS ((DWORD)-1)
	struct DEVMODEW
	{
		WORD dmSize;
		DWORD dmDisplayFrequency;
	};
	struct HMONITOR__;
	using HMONITOR = HMONITOR__*;
	#define MONITOR_DEFAULTTONEAREST 0x00000002
	#define CCHDEVICENAME 32
	struct MONITORINFOEXW
	{
		DWORD cbSize;
		wchar_t szDevice[CCHDEVICENAME];
	};
	HWND GetForegroundWindow();
	HMONITOR MonitorFromWindow(HWND hwnd, DWORD dwFlags);
	BOOL GetMonitorInfoW(HMONITOR hMonitor, MONITORINFOEXW* lpmi);
	BOOL EnumDisplaySettingsW(LPCWSTR lpszDeviceName, DWORD iModeNum, DEVMODEW* lpDevMode);
	int _wcsicmp(const wchar_t* a, const wchar_t* b);

	// std::filesystem::path::c_str() is wchar_t on Windows and char on the Linux this is checked on, these two let the
	// sources' calls with a path's c_str() parse here. They are not Win32 functions
	int _wcsicmp(const char* a, const wchar_t* b);
	HMODULE LoadLibraryW(const char* name);
}
