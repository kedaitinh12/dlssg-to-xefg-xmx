// The part of Microsoft::WRL::ComPtr the Vulkan runtime's sources use
#pragma once
#include <utility>

namespace Microsoft::WRL
{
	template <class T> class ComPtr
	{
	public:
		ComPtr() = default;
		ComPtr(std::nullptr_t) {}
		ComPtr(T* other) : ptr(other) { if (ptr) ptr->AddRef(); }
		ComPtr(const ComPtr& other) : ptr(other.ptr) { if (ptr) ptr->AddRef(); }
		ComPtr(ComPtr&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
		~ComPtr() { Reset(); }

		ComPtr& operator=(const ComPtr& other)
		{
			ComPtr(other).Swap(*this);
			return *this;
		}

		ComPtr& operator=(ComPtr&& other) noexcept
		{
			ComPtr(std::move(other)).Swap(*this);
			return *this;
		}

		T* Get() const { return ptr; }
		T* operator->() const { return ptr; }
		T** operator&()
		{
			Reset();
			return &ptr;
		}
		T** GetAddressOf() { return &ptr; }
		explicit operator bool() const { return ptr != nullptr; }

		unsigned long Reset()
		{
			T* old = ptr;
			ptr = nullptr;
			return old ? old->Release() : 0;
		}

		void Swap(ComPtr& other) { std::swap(ptr, other.ptr); }

	private:
		T* ptr = nullptr;
	};
}
