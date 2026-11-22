// Minimal DXGI 1.4 stub for Linux syntax/regression checks (FrameGenerator.h & VulkanRuntime.cpp VRAM guard).
#pragma once
#include "d3d12.h"

enum DXGI_MEMORY_SEGMENT_GROUP
{
	DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0,
	DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL = 1,
};

struct DXGI_QUERY_VIDEO_MEMORY_INFO
{
	UINT64 Budget;
	UINT64 CurrentUsage;
	UINT64 AvailableForReservation;
	UINT64 CurrentReservation;
};

struct IDXGIAdapter1 : IUnknown
{
};

struct IDXGIAdapter3 : IDXGIAdapter1
{
	virtual HRESULT QueryVideoMemoryInfo(UINT, DXGI_MEMORY_SEGMENT_GROUP, DXGI_QUERY_VIDEO_MEMORY_INFO*) { return E_FAIL; }
};

struct IDXGIFactory4 : IUnknown
{
	virtual HRESULT EnumAdapterByLuid(LUID, REFIID, void**) { return E_FAIL; }
};

inline HRESULT CreateDXGIFactory1(REFIID, void**) { return E_FAIL; }
