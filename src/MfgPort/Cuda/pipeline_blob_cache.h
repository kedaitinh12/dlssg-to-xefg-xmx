// In-memory DX12 compute PSO blob cache across XeFG context recreations (ported from MFG-Unlocker V1.7.2).
// Speeds up re-enabling FG or changing resolution/settings on DX12 with zero per-frame hot-path cost.
#pragma once

#include "../Sha256.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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

		void Clear() noexcept
		{
			std::lock_guard lock(mutex_);
			entries_.clear();
			dirty_ = false;
		}

		void LoadFromDisk(uint32_t vendorId, uint32_t deviceId) noexcept
		{
			try
			{
				char disableBuf[8]{};
				const DWORD disableLen = GetEnvironmentVariableA("XEFG_DISABLE_PSO_DISK_CACHE", disableBuf, sizeof(disableBuf));
				if (disableLen == 1 && disableBuf[0] == '1')
					return;
				if (!vendorId || !deviceId)
					return;

				wchar_t localAppData[MAX_PATH]{};
				const DWORD envLen = GetEnvironmentVariableW(L"LOCALAPPDATA", localAppData, MAX_PATH);
				if (envLen == 0 || envLen >= MAX_PATH)
					return;

				const std::filesystem::path cacheDir = std::filesystem::path(localAppData) / L"dlssg-to-xefg";
				std::error_code ec;
				std::filesystem::create_directories(cacheDir, ec);
				char fname[96]{};
				std::snprintf(fname, sizeof(fname), "d3d12_pso_cache_v1_%04x_%04x.bin", vendorId, deviceId);
				const std::filesystem::path path = cacheDir / fname;

				std::lock_guard lock(mutex_);
				if (diskLoaded_ && cachePath_ == path)
					return;
				cachePath_ = path;
				vendorId_ = vendorId;
				deviceId_ = deviceId;
				diskLoaded_ = true;

				const auto fileSz = std::filesystem::file_size(path, ec);
				if (ec || fileSz < sizeof(DiskHeader) || fileSz > kMaxDiskBytes)
					return;

				std::ifstream in(path, std::ios::binary);
				DiskHeader hdr{};
				if (!in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)))
					return;
				if (hdr.magic != kDiskMagic || hdr.version != kDiskVersion ||
					hdr.vendorId != vendorId || hdr.deviceId != deviceId || hdr.count > kMaxEntries)
					return;

				for (uint32_t i = 0; i < hdr.count && entries_.size() < kMaxEntries; ++i)
				{
					DiskEntryHeader eh{};
					if (!in.read(reinterpret_cast<char*>(&eh), sizeof(eh)))
						break;
					if (eh.blobSize == 0 || eh.blobSize > kMaxBlobBytes)
						break;
					auto blob = std::make_shared<std::vector<uint8_t>>(eh.blobSize);
					if (!in.read(reinterpret_cast<char*>(blob->data()), static_cast<std::streamsize>(eh.blobSize)))
						break;
					Key k{};
					std::memcpy(k.root.digest.data(), eh.rootDigest, sizeof(eh.rootDigest));
					k.root.nodeMask = eh.rootNodeMask;
					std::memcpy(k.csDigest.data(), eh.csDigest, sizeof(eh.csDigest));
					k.nodeMask = eh.nodeMask;
					k.flags = static_cast<D3D12_PIPELINE_STATE_FLAGS>(eh.flags);
					if (!Find(k))
						entries_.push_back({ k, std::move(blob), 0ull });
				}
			}
			catch (...)
			{
			}
		}

		void SaveToDisk() noexcept
		{
			try
			{
				std::filesystem::path path;
				uint32_t vendorId = 0, deviceId = 0;
				std::vector<Entry> snapshot;
				{
					std::lock_guard lock(mutex_);
					if (!dirty_ || cachePath_.empty() || !vendorId_ || !deviceId_)
						return;
					dirty_ = false;
					path = cachePath_;
					vendorId = vendorId_;
					deviceId = deviceId_;
					snapshot = entries_;
				}

				size_t totalBytes = sizeof(DiskHeader);
				std::vector<const Entry*> toWrite;
				toWrite.reserve(snapshot.size());
				for (const auto& e : snapshot)
				{
					if (!e.blob || e.blob->empty() || e.blob->size() > kMaxBlobBytes)
						continue;
					const size_t recBytes = sizeof(DiskEntryHeader) + e.blob->size();
					if (totalBytes + recBytes > kMaxDiskBytes)
						break;
					totalBytes += recBytes;
					toWrite.push_back(&e);
				}

				std::filesystem::path tmpPath = path;
				tmpPath += L".tmp";
				{
					std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
					DiskHeader hdr{ kDiskMagic, kDiskVersion, vendorId, deviceId, static_cast<uint32_t>(toWrite.size()) };
					if (!out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr)))
						return;
					for (const Entry* e : toWrite)
					{
						DiskEntryHeader eh{};
						std::memcpy(eh.rootDigest, e->key.root.digest.data(), sizeof(eh.rootDigest));
						eh.rootNodeMask = e->key.root.nodeMask;
						std::memcpy(eh.csDigest, e->key.csDigest.data(), sizeof(eh.csDigest));
						eh.nodeMask = e->key.nodeMask;
						eh.flags = static_cast<uint32_t>(e->key.flags);
						eh.blobSize = static_cast<uint32_t>(e->blob->size());
						if (!out.write(reinterpret_cast<const char*>(&eh), sizeof(eh)) ||
							!out.write(reinterpret_cast<const char*>(e->blob->data()), static_cast<std::streamsize>(e->blob->size())))
							return;
					}
				}
				std::error_code ec;
				std::filesystem::rename(tmpPath, path, ec);
				if (ec)
					std::filesystem::remove(tmpPath, ec);
			}
			catch (...)
			{
			}
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
					dirty_ = true;
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
				{
					entries_.push_back({ key, std::move(copy), generation });
					dirty_ = true;
				}
			}
			return hr;
		}

	private:
		static constexpr size_t kMaxEntries = 256;
		static constexpr size_t kMaxBlobBytes = 4u << 20;
		static constexpr size_t kMaxDiskBytes = 32u << 20;
		static constexpr uint32_t kDiskMagic = 0x4F535058u; // 'XPSO'
		static constexpr uint32_t kDiskVersion = 1u;

		struct DiskHeader
		{
			uint32_t magic{};
			uint32_t version{};
			uint32_t vendorId{};
			uint32_t deviceId{};
			uint32_t count{};
		};

		struct DiskEntryHeader
		{
			uint8_t rootDigest[32]{};
			uint32_t rootNodeMask{};
			uint8_t csDigest[32]{};
			uint32_t nodeMask{};
			uint32_t flags{};
			uint32_t blobSize{};
		};

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
		uint64_t generation_ = 1; // start at 1 so disk-loaded entries (gen=0) are immediately eligible
		std::filesystem::path cachePath_;
		uint32_t vendorId_ = 0;
		uint32_t deviceId_ = 0;
		bool diskLoaded_ = false;
		bool dirty_ = false;
	};
} // namespace xefg_cuda
