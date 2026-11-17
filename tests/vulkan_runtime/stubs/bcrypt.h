// The six CNG calls src/MfgPort/Sha256.h makes, declared and nothing else: tests/vulkan_runtime/check_syntax.sh only parses
// the sources, nothing is linked or run (tests/vulkan_programs/stubs/bcrypt.h has the one that computes).
#pragma once
#include "Windows.h"

#define BCRYPT_SHA256_ALGORITHM L"SHA256"

using ULONG = unsigned int;
using PUCHAR = unsigned char*;
using NTSTATUS = int32_t;
using BCRYPT_ALG_HANDLE = void*;
using BCRYPT_HASH_HANDLE = void*;

extern "C++"
{
	NTSTATUS BCryptOpenAlgorithmProvider(BCRYPT_ALG_HANDLE* algorithm, LPCWSTR id, LPCWSTR implementation, ULONG flags);
	NTSTATUS BCryptCloseAlgorithmProvider(BCRYPT_ALG_HANDLE algorithm, ULONG flags);
	NTSTATUS BCryptCreateHash(BCRYPT_ALG_HANDLE algorithm, BCRYPT_HASH_HANDLE* hash, PUCHAR object, ULONG objectSize,
		PUCHAR secret, ULONG secretSize, ULONG flags);
	NTSTATUS BCryptHashData(BCRYPT_HASH_HANDLE hash, PUCHAR input, ULONG size, ULONG flags);
	NTSTATUS BCryptFinishHash(BCRYPT_HASH_HANDLE hash, PUCHAR output, ULONG size, ULONG flags);
	NTSTATUS BCryptDestroyHash(BCRYPT_HASH_HANDLE hash);
}
