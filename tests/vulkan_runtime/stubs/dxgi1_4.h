// Minimal DXGI 1.4 stub for Linux syntax/regression checks (FrameGenerator.h uses ComPtr<IDXGIAdapter3>).
#pragma once
#include "d3d12.h"

struct IDXGIAdapter3 : IUnknown {};
