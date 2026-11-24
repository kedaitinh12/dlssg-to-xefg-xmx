// The subset of D3D12 and DXGI the Vulkan runtime's sources and the headers they include use, with the real layouts
// where the sources build a structure by position (tests/vulkan_runtime/check_syntax.sh).
#pragma once
#include "Windows.h"

#include <cstdint>

using UINT64 = uint64_t;
using UINT16 = uint16_t;
using UINT8 = uint8_t;

enum DXGI_FORMAT : UINT
{
	DXGI_FORMAT_UNKNOWN = 0,
	DXGI_FORMAT_R16G16B16A16_TYPELESS = 9,
	DXGI_FORMAT_R16G16B16A16_FLOAT = 10,
	DXGI_FORMAT_R32G32_TYPELESS = 15,
	DXGI_FORMAT_R32G32_FLOAT = 16,
	DXGI_FORMAT_R10G10B10A2_TYPELESS = 23,
	DXGI_FORMAT_R10G10B10A2_UNORM = 24,
	DXGI_FORMAT_R11G11B10_FLOAT = 26,
	DXGI_FORMAT_R8G8B8A8_TYPELESS = 27,
	DXGI_FORMAT_R8G8B8A8_UNORM = 28,
	DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = 29,
	DXGI_FORMAT_R16G16_TYPELESS = 33,
	DXGI_FORMAT_R16G16_FLOAT = 34,
	DXGI_FORMAT_R32_TYPELESS = 39,
	DXGI_FORMAT_D32_FLOAT = 40,
	DXGI_FORMAT_R32_FLOAT = 41,
	DXGI_FORMAT_R16_TYPELESS = 53,
	DXGI_FORMAT_D16_UNORM = 55,
	DXGI_FORMAT_R16_UNORM = 56,
	DXGI_FORMAT_R8_UINT = 62,
	DXGI_FORMAT_B8G8R8A8_UNORM = 87,
	DXGI_FORMAT_B8G8R8A8_TYPELESS = 90,
	DXGI_FORMAT_B8G8R8A8_UNORM_SRGB = 91,
};

struct DXGI_SAMPLE_DESC
{
	UINT Count;
	UINT Quality;
};

struct IUnknown
{
	virtual unsigned long AddRef() = 0;
	virtual unsigned long Release() = 0;
	virtual HRESULT QueryInterface(REFIID, void**) = 0;
};

enum D3D12_RESOURCE_DIMENSION
{
	D3D12_RESOURCE_DIMENSION_UNKNOWN = 0,
	D3D12_RESOURCE_DIMENSION_BUFFER = 1,
	D3D12_RESOURCE_DIMENSION_TEXTURE2D = 3,
};

enum D3D12_TEXTURE_LAYOUT
{
	D3D12_TEXTURE_LAYOUT_UNKNOWN = 0,
	D3D12_TEXTURE_LAYOUT_ROW_MAJOR = 1,
};

enum D3D12_RESOURCE_FLAGS
{
	D3D12_RESOURCE_FLAG_NONE = 0,
	D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS = 0x4,
};

enum D3D12_RESOURCE_STATES
{
	D3D12_RESOURCE_STATE_COMMON = 0,
	D3D12_RESOURCE_STATE_UNORDERED_ACCESS = 0x8,
	D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE = 0x40,
	D3D12_RESOURCE_STATE_COPY_DEST = 0x400,
	D3D12_RESOURCE_STATE_COPY_SOURCE = 0x800,
};

enum D3D12_HEAP_TYPE
{
	D3D12_HEAP_TYPE_DEFAULT = 1,
	D3D12_HEAP_TYPE_UPLOAD = 2,
	D3D12_HEAP_TYPE_READBACK = 3,
};

enum D3D12_HEAP_FLAGS
{
	D3D12_HEAP_FLAG_NONE = 0,
};

enum D3D12_COMMAND_LIST_TYPE
{
	D3D12_COMMAND_LIST_TYPE_DIRECT = 0,
};

struct D3D12_HEAP_PROPERTIES
{
	D3D12_HEAP_TYPE Type;
	int CPUPageProperty;
	int MemoryPoolPreference;
	UINT CreationNodeMask;
	UINT VisibleNodeMask;
};

struct D3D12_RESOURCE_DESC
{
	D3D12_RESOURCE_DIMENSION Dimension;
	UINT64 Alignment;
	UINT64 Width;
	UINT Height;
	UINT16 DepthOrArraySize;
	UINT16 MipLevels;
	DXGI_FORMAT Format;
	DXGI_SAMPLE_DESC SampleDesc;
	D3D12_TEXTURE_LAYOUT Layout;
	D3D12_RESOURCE_FLAGS Flags;
};

struct D3D12_CLEAR_VALUE;

enum D3D12_RESOURCE_BARRIER_TYPE
{
	D3D12_RESOURCE_BARRIER_TYPE_TRANSITION = 0,
};

enum D3D12_RESOURCE_BARRIER_FLAGS
{
	D3D12_RESOURCE_BARRIER_FLAG_NONE = 0,
};

#define D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES 0xffffffff

struct ID3D12Resource;

struct D3D12_RESOURCE_TRANSITION_BARRIER
{
	ID3D12Resource* pResource;
	UINT Subresource;
	D3D12_RESOURCE_STATES StateBefore;
	D3D12_RESOURCE_STATES StateAfter;
};

struct D3D12_RESOURCE_BARRIER
{
	D3D12_RESOURCE_BARRIER_TYPE Type;
	D3D12_RESOURCE_BARRIER_FLAGS Flags;
	union
	{
		D3D12_RESOURCE_TRANSITION_BARRIER Transition;
	};
};

struct ID3D12Object : IUnknown
{
};

struct D3D12_RANGE
{
	SIZE_T Begin;
	SIZE_T End;
};

struct ID3D12Resource : ID3D12Object
{
	virtual HRESULT Map(UINT, const D3D12_RANGE*, void**) { return E_FAIL; }
	virtual void Unmap(UINT, const D3D12_RANGE*) {}
};

struct ID3D12QueryHeap : ID3D12Object {};
struct ID3D12Heap : ID3D12Object
{
};

struct ID3D12PipelineLibrary : ID3D12Object
{
};

struct ID3D12RootSignature : ID3D12Object
{
};

struct ID3D12PipelineState : ID3D12Object
{
};

struct ID3D12DescriptorHeap : ID3D12Object
{
};

struct ID3D12CommandAllocator : ID3D12Object
{
	virtual HRESULT Reset() = 0;
};

struct ID3D12GraphicsCommandList : ID3D12Object
{
	virtual HRESULT Close() = 0;
	virtual HRESULT Reset(ID3D12CommandAllocator* allocator, void* initialState) = 0;
	virtual void ResourceBarrier(UINT numBarriers, const D3D12_RESOURCE_BARRIER* barriers) = 0;
	virtual void CopyResource(ID3D12Resource* destination, ID3D12Resource* source) = 0;
};

struct ID3D12Device : ID3D12Object
{
	virtual HRESULT CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** allocator) = 0;
	virtual HRESULT CreateCommandList(UINT nodeMask, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator,
		void* initialState, REFIID riid, void** list) = 0;
	virtual HRESULT CreateCommittedResource(const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS flags,
		const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear, REFIID riid,
		void** resource) = 0;
	virtual LUID GetAdapterLuid() { return LUID{}; }
};

enum D3D12_DRED_ENABLEMENT { D3D12_DRED_ENABLEMENT_FORCED_ON };
struct ID3D12DeviceRemovedExtendedDataSettings : IUnknown
{
    virtual void SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT) = 0;
    virtual void SetPageFaultEnablement(D3D12_DRED_ENABLEMENT) = 0;
};
struct ID3D12DeviceRemovedExtendedDataSettings1 : ID3D12DeviceRemovedExtendedDataSettings
{
    virtual void SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT) = 0;
};
