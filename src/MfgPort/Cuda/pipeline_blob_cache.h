// In-memory DX12 compute PSO blob cache across XeFG context recreations (ported from MFG-Unlocker V1.7.2).
// Speeds up re-enabling FG or changing resolution/settings on DX12 with zero per-frame hot-path cost.
#pragma once

#include "../Sha256.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace xefg_cuda
{
	inline constexpr GUID kPipelineCacheRootSignature =
		{ 0xe8a622f0, 0xf77a, 0x431f, { 0x9d, 0x71, 0x7e, 0x43, 0x1b, 0xd8, 0x90, 0xa2 } };
	inline constexpr size_t kMaxPipelineInputBytes = 1u << 20;

	struct PipelineRootSignature
	{
		sha256_util::Digest digest{};
		UINT nodeMask{};
		bool operator==(const PipelineRootSignature&) const = default;
	};
	static_assert(sizeof(PipelineRootSignature) == 36);

	inline void RecordPipelineRootSignature(ID3D12RootSignature* root, UINT nodeMask, const void* bytes, SIZE_T size) noexcept
	{
		if (!root || !bytes || !size || size > kMaxPipelineInputBytes)
			return;
		PipelineRootSignature signature{};
		signature.nodeMask = nodeMask;
		if (sha256_util::Compute(bytes, size, signature.digest))
			root->SetPrivateData(kPipelineCacheRootSignature, sizeof(signature), &signature);
	}

	class PipelineBlobCache
	{
	public:
		// Call when a XeFG context is retired so cached blobs become eligible for reuse on the next context.
		void ContextRetired() noexcept
		{
			std::lock_guard lock(mutex_);
			++generation_;
		}

		template <class Create>
		HRESULT CreateCompute(const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, REFIID iid, void** out, Create create) noexcept
		{
			D3D12_COMPUTE_PIPELINE_STATE_DESC cachedDesc{};
			Key key{};
			if (!desc || !out || iid != __uuidof(ID3D12PipelineState) || !Describe(desc, cachedDesc, key))
				return create(desc, out);

			BlobPtr blob;
			bool remember = false;
			uint64_t generation = 0;
			{
				std::lock_guard lock(mutex_);
				generation = generation_;
				const Entry* entry = Find(key);
				if (entry && entry->generation < generation)
					blob = entry->blob;
				else if (!entry && entries_.size() < kMaxEntries)
					remember = true;
			}

			if (blob)
			{
				cachedDesc.CachedPSO.pCachedBlob = blob->data();
				cachedDesc.CachedPSO.CachedBlobSizeInBytes = blob->size();
				const HRESULT hr = create(&cachedDesc, out);
				if (SUCCEEDED(hr))
					return hr;
				// Driver rejected stale blob (e.g. driver/flags change): fall back to cold compile.
				{
					std::lock_guard lock(mutex_);
					Erase(key);
				}
				return create(desc, out);
			}

			const HRESULT hr = create(desc, out);
			if (FAILED(hr) || !remember || !*out)
				return hr;

			auto* pso = static_cast<ID3D12PipelineState*>(*out);
			Microsoft::WRL::ComPtr<ID3DBlob> rawBlob;
			if (SUCCEEDED(pso->GetCachedBlob(&rawBlob)) && rawBlob &&
				rawBlob->GetBufferPointer() && rawBlob->GetBufferSize() > 0 &&
				rawBlob->GetBufferSize() <= kMaxBlobBytes)
			{
				auto copy = std::make_shared<std::vector<uint8_t>>(rawBlob->GetBufferSize());
				std::memcpy(copy->data(), rawBlob->GetBufferPointer(), rawBlob->GetBufferSize());
				std::lock_guard lock(mutex_);
				if (entries_.size() < kMaxEntries && !Find(key))
					entries_.push_back({ key, std::move(copy), generation });
			}
			return hr;
		}

	private:
		static constexpr size_t kMaxEntries = 256;
		static constexpr size_t kMaxBlobBytes = 4u << 20;

		using BlobPtr = std::shared_ptr<std::vector<uint8_t>>;

		struct Key
		{
			PipelineRootSignature root{};
			sha256_util::Digest csDigest{};
			UINT nodeMask{};
			D3D12_PIPELINE_STATE_FLAGS flags{};
			bool operator==(const Key&) const = default;
		};

		struct Entry
		{
			Key key{};
			BlobPtr blob;
			uint64_t generation{};
		};

		static bool Describe(const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc,
			D3D12_COMPUTE_PIPELINE_STATE_DESC& cachedDesc, Key& key) noexcept
		{
			if (!desc->pRootSignature || !desc->CS.pShaderBytecode || !desc->CS.BytecodeLength ||
				desc->CS.BytecodeLength > kMaxPipelineInputBytes ||
				desc->CachedPSO.pCachedBlob != nullptr)
			{
				return false;
			}
			UINT rootSize = sizeof(PipelineRootSignature);
			if (FAILED(desc->pRootSignature->GetPrivateData(kPipelineCacheRootSignature, &rootSize, &key.root)) ||
				rootSize != sizeof(PipelineRootSignature))
			{
				return false;
			}
			if (!sha256_util::Compute(desc->CS.pShaderBytecode, desc->CS.BytecodeLength, key.csDigest))
				return false;
			key.nodeMask = desc->NodeMask;
			key.flags = desc->Flags;
			cachedDesc = *desc;
			return true;
		}

		const Entry* Find(const Key& key) const noexcept
		{
			for (const auto& e : entries_)
				if (e.key == key)
					return &e;
			return nullptr;
		}

		void Erase(const Key& key) noexcept
		{
			for (auto it = entries_.begin(); it != entries_.end(); ++it)
			{
				if (it->key == key)
				{
					entries_.erase(it);
					return;
				}
			}
		}

		std::mutex mutex_;
		std::vector<Entry> entries_;
		uint64_t generation_ = 0;
	};
} // namespace xefg_cuda
