#include "Log.h"
#include "Util.h"

#include <Windows.h>

#include <fstream>
#include <mutex>

void Log::Write(std::string_view level, std::string_view message)
{
	static std::mutex mutex;
	static std::ofstream file(ThisModulePath().replace_extension(L".log"), std::ios::trunc);

	SYSTEMTIME now;
	GetLocalTime(&now);
	auto line = std::format("[{:02}:{:02}:{:02}.{:03}] [{}] {}\n", now.wHour, now.wMinute, now.wSecond,
		now.wMilliseconds, level, message);

	std::scoped_lock lock(mutex);
	if (IsDebuggerPresent())
		OutputDebugStringA(line.c_str());

	file.write(line.data(), static_cast<std::streamsize>(line.size()));
	if (level == "ERROR" || level == "WARN")
		file.flush();
}
