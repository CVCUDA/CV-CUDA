/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Definitions.hpp"

#include <common/ObjectBag.hpp>
#include <common/ValueTests.hpp>
#include <cuda_runtime.h>
#include <nvcv/alloc/Allocator.hpp>

#include <array>
#include <cstddef>
#include <new>
#include <thread>

#include <nvcv/alloc/Fwd.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

namespace {

NVCVMemoryBuffer AllocHost(int64_t size, int32_t align)
{
    return static_cast<NVCVMemoryBuffer>(
        ::operator new (static_cast<size_t>(size), std::align_val_t{static_cast<size_t>(align)}, std::nothrow));
}

void FreeHost(NVCVMemoryBuffer ptr, int32_t align) noexcept
{
    ::operator delete (static_cast<void *>(ptr), std::align_val_t{static_cast<size_t>(align)});
}

template<typename OpaquePointer, typename PointerType>
OpaquePointer OpaqueFromPointer(PointerType *ptr) noexcept
{
    return static_cast<OpaquePointer>(static_cast<void *>(ptr));
}

template<typename PointerType, typename OpaquePointer>
PointerType *PointerFromOpaque(OpaquePointer ptr) noexcept
{
    return static_cast<PointerType *>(static_cast<void *>(ptr));
}

} // namespace

TEST(AllocatorTest, CreateAndUseCustom)
{
    std::array<NVCVResourceAllocator, 2> allocators = {};

    int ctx0 = 100;
    int ctx1 = 200;

    allocators[0].resType         = NVCV_RESOURCE_MEM_HOST;
    allocators[0].ctx             = OpaqueFromPointer<NVCVResourceContext>(&ctx0);
    allocators[0].res.mem.fnAlloc = [](auto ctx, int64_t size, int32_t align)
    {
        *PointerFromOpaque<int>(ctx) += 1;
        return AllocHost(size, align);
    };
    allocators[0].res.mem.fnFree = [](auto ctx, auto ptr, int64_t, int32_t align)
    {
        *PointerFromOpaque<int>(ctx) += 10;
        FreeHost(ptr, align);
    };
    allocators[0].cleanup = [](auto ctx, auto *alloc)
    {
        EXPECT_EQ(ctx, alloc->ctx);
        auto *ctx_int = PointerFromOpaque<int>(ctx);
        EXPECT_EQ(*ctx_int, 111);
        *ctx_int = 0xDEAD;
    };

    allocators[1].resType         = NVCV_RESOURCE_MEM_CUDA;
    allocators[1].ctx             = OpaqueFromPointer<NVCVResourceContext>(&ctx1);
    allocators[1].res.mem.fnAlloc = [](auto ctx, int64_t size, int32_t)
    {
        *PointerFromOpaque<int>(ctx) += 1;
        void *mem = nullptr;
        EXPECT_EQ(cudaMalloc(&mem, size), cudaSuccess);
        return static_cast<NVCVMemoryBuffer>(mem);
    };
    allocators[1].res.mem.fnFree = [](auto ctx, auto ptr, int64_t, int32_t)
    {
        *PointerFromOpaque<int>(ctx) += 10;
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    };
    allocators[1].cleanup = [](auto ctx, auto *alloc)
    {
        EXPECT_EQ(ctx, alloc->ctx);
        auto *ctx_int = PointerFromOpaque<int>(ctx);
        EXPECT_EQ(*ctx_int, 211);
        *ctx_int = 0xBAD;
    };

    NVCVAllocatorHandle halloc = nullptr;
    ASSERT_EQ(nvcvAllocatorConstructCustom(allocators.data(), allocators.size(), &halloc), NVCV_SUCCESS);
    ASSERT_NE(halloc, nullptr);

    int refCount = 0;
    EXPECT_EQ(nvcvAllocatorRefCount(halloc, &refCount), NVCV_SUCCESS);
    EXPECT_EQ(refCount, 1);

    int newRef = 0;
    EXPECT_EQ(nvcvAllocatorIncRef(halloc, &newRef), NVCV_SUCCESS);
    EXPECT_EQ(newRef, 2);
    EXPECT_EQ(nvcvAllocatorDecRef(halloc, &newRef), NVCV_SUCCESS);
    EXPECT_EQ(newRef, 1);

    for (int i = 0; i < 2; i++)
    {
        NVCVResourceAllocator  alloc = {};
        NVCVResourceAllocator &ref   = allocators[i];
        auto                   res   = nvcvAllocatorGet(halloc, ref.resType, &alloc);
        EXPECT_EQ(res, NVCV_SUCCESS);
        if (res != NVCV_SUCCESS)
            continue;
        EXPECT_EQ(alloc.resType, ref.resType)
            << "The free function pointer doesn't match the one passsed to construction.";
        EXPECT_EQ(alloc.ctx, ref.ctx) << "The custom allocator context pointer wass corrupted.";
        EXPECT_EQ(alloc.resType, ref.resType)
            << "Got allocator descriptor for a different resource type than requested.";
        EXPECT_EQ(alloc.res.mem.fnAlloc, ref.res.mem.fnAlloc)
            << "The allocation function pointer doesn't match the one passsed to construction.";
        EXPECT_EQ(alloc.res.mem.fnFree, ref.res.mem.fnFree)
            << "The free function pointer doesn't match the one passsed to construction.";
    }

    NVCVResourceAllocator pinnedAlloc{};
    EXPECT_EQ(nvcvAllocatorGet(halloc, NVCV_RESOURCE_MEM_HOST_PINNED, &pinnedAlloc), NVCV_SUCCESS);

    NVCVMemoryBuffer p0 = nullptr;
    NVCVMemoryBuffer p1 = nullptr;
    NVCVMemoryBuffer p2 = nullptr;
    EXPECT_EQ(nvcvAllocatorAllocHostMemory(halloc, &p0, (1 << 20), 256), NVCV_SUCCESS);
    EXPECT_NE(p0, nullptr);
    EXPECT_EQ(ctx0, 101) << "The custom alloc for host memory wasn't invoked";
    EXPECT_EQ(nvcvAllocatorFreeHostMemory(halloc, p0, (1 << 20), 256), NVCV_SUCCESS);
    EXPECT_EQ(ctx0, 111) << "The custom free for host memory wasn't invoked";

    EXPECT_EQ(nvcvAllocatorAllocCudaMemory(halloc, &p1, (1 << 20), 256), NVCV_SUCCESS);
    EXPECT_NE(p1, nullptr);
    EXPECT_EQ(ctx1, 201) << "The custom alloc for CUDA memory wasn't invoked";
    EXPECT_EQ(nvcvAllocatorFreeCudaMemory(halloc, p1, (1 << 20), 256), NVCV_SUCCESS);
    EXPECT_EQ(ctx1, 211) << "The custom free for CUDA memory wasn't invoked";

    EXPECT_EQ(nvcvAllocatorAllocHostPinnedMemory(halloc, &p2, (1 << 20), 256), NVCV_SUCCESS)
        << "Host pinned allocation failed - default allocator should have been used.";
    EXPECT_NE(p2, nullptr);
    EXPECT_EQ(nvcvAllocatorFreeHostPinnedMemory(halloc, p2, (1 << 20), 256), NVCV_SUCCESS);

    newRef = 1;
    EXPECT_EQ(nvcvAllocatorDecRef(halloc, &newRef), NVCV_SUCCESS);
    EXPECT_EQ(newRef, 0);
    EXPECT_EQ(ctx0, 0xDEAD);
    EXPECT_EQ(ctx1, 0xBAD);
}

// smoke: just to check if it compiles.
TEST(Allocator, smoke_test_default)
{
    nvcv::CustomAllocator myalloc;

    NVCVMemoryBuffer ptrDev        = myalloc.cudaMem().alloc(768, 256);
    NVCVMemoryBuffer ptrHost       = myalloc.hostMem().alloc(160, 16);
    NVCVMemoryBuffer ptrHostPinned = myalloc.hostPinnedMem().alloc(144, 16);

    myalloc.cudaMem().free(ptrDev, 768, 256);
    myalloc.hostMem().free(ptrHost, 160, 16);
    myalloc.hostPinnedMem().free(ptrHostPinned, 144, 16);
}

// smoke: just to check if it compiles.
TEST(Allocator, smoke_test_custom_functors)
{
    int                       devCounter        = 1;
    int                       hostCounter       = 1;
    int                       hostPinnedCounter = 1;
    std::array<std::byte, 16> devBuffers{};
    std::array<std::byte, 16> hostBuffers{};
    std::array<std::byte, 16> hostPinnedBuffers{};

    // clang-format off
    nvcv::CustomAllocator myalloc1
    {
        nvcv::CustomHostMemAllocator
        {
            [&hostCounter, &hostBuffers](int64_t size, int32_t)
            {
                auto ptr = OpaqueFromPointer<NVCVMemoryBuffer>(&hostBuffers[hostCounter]);
                hostCounter += size;
                return ptr;
            },
            [&hostCounter, &hostBuffers](const NVCVMemoryBufferRec *ptr, int64_t size, int32_t)
            {
                hostCounter -= size;
                assert(ptr == OpaqueFromPointer<NVCVMemoryBuffer>(&hostBuffers[hostCounter]));
            }
        },
        nvcv::CustomCudaMemAllocator
        {
            [&devCounter, &devBuffers](int64_t size, int32_t)
            {
                auto ptr = OpaqueFromPointer<NVCVMemoryBuffer>(&devBuffers[devCounter]);
                devCounter += size;
                return ptr;
            },
            [&devCounter, &devBuffers](const NVCVMemoryBufferRec *ptr, int64_t size, int32_t)
            {
                devCounter -= size;
                assert(ptr == OpaqueFromPointer<NVCVMemoryBuffer>(&devBuffers[devCounter]));
            }
        },
        nvcv::CustomHostPinnedMemAllocator
        {
            [&hostPinnedCounter, &hostPinnedBuffers](int64_t size, int32_t)
            {
                auto ptr = OpaqueFromPointer<NVCVMemoryBuffer>(&hostPinnedBuffers[hostPinnedCounter]);
                hostPinnedCounter += size;
                return ptr;
            },
            [&hostPinnedCounter, &hostPinnedBuffers](const NVCVMemoryBufferRec *ptr, int64_t size, int32_t)
            {
                hostPinnedCounter -= size;
                assert(ptr == OpaqueFromPointer<NVCVMemoryBuffer>(&hostPinnedBuffers[hostPinnedCounter]));
            }
        },
    };
    // clang-format on

    ASSERT_EQ(OpaqueFromPointer<NVCVMemoryBuffer>(&hostBuffers[1]), myalloc1.hostMem().alloc(5));
    EXPECT_EQ(6, hostCounter);

    ASSERT_EQ(OpaqueFromPointer<NVCVMemoryBuffer>(&hostPinnedBuffers[1]), myalloc1.hostPinnedMem().alloc(10));
    EXPECT_EQ(11, hostPinnedCounter);

    ASSERT_EQ(OpaqueFromPointer<NVCVMemoryBuffer>(&devBuffers[1]), myalloc1.cudaMem().alloc(7));
    EXPECT_EQ(8, devCounter);

    ASSERT_EQ(OpaqueFromPointer<NVCVMemoryBuffer>(&devBuffers[8]), myalloc1.cudaMem().alloc(2));
    EXPECT_EQ(10, devCounter);

    myalloc1.cudaMem().free(OpaqueFromPointer<NVCVMemoryBuffer>(&devBuffers[8]), 2);
    EXPECT_EQ(8, devCounter);

    myalloc1.cudaMem().free(OpaqueFromPointer<NVCVMemoryBuffer>(&devBuffers[1]), 7);
    EXPECT_EQ(1, devCounter);
}

TEST(AllocatorTest, smoke_user_pointer)
{
    std::array<NVCVResourceAllocator, 1> allocators = {};

    int ctx0 = 100;

    allocators[0].resType         = NVCV_RESOURCE_MEM_HOST;
    allocators[0].ctx             = OpaqueFromPointer<NVCVResourceContext>(&ctx0);
    allocators[0].res.mem.fnAlloc = [](auto ctx, int64_t size, int32_t align)
    {
        *PointerFromOpaque<int>(ctx) += 1;
        return AllocHost(size, align);
    };
    allocators[0].res.mem.fnFree = [](auto ctx, auto ptr, int64_t, int32_t align)
    {
        *PointerFromOpaque<int>(ctx) += 10;
        FreeHost(ptr, align);
    };
    allocators[0].cleanup = [](auto ctx, auto *alloc)
    {
        EXPECT_EQ(ctx, alloc->ctx);
        auto *ctx_int = PointerFromOpaque<int>(ctx);
        *ctx_int      = 0xDEAD;
    };

    NVCVAllocatorHandle halloc = nullptr;
    ASSERT_EQ(nvcvAllocatorConstructCustom(allocators.data(), allocators.size(), &halloc), NVCV_SUCCESS);
    ASSERT_NE(halloc, nullptr);

    NVCVUserPointer userPtr;
    ASSERT_EQ(nvcvAllocatorGetUserPointer(halloc, &userPtr), NVCV_SUCCESS);
    EXPECT_EQ(nullptr, userPtr);

    int             userData    = 0x123;
    NVCVUserPointer expectedPtr = OpaqueFromPointer<NVCVUserPointer>(&userData);
    ASSERT_EQ(nvcvAllocatorSetUserPointer(halloc, expectedPtr), NVCV_SUCCESS);
    ASSERT_EQ(nvcvAllocatorGetUserPointer(halloc, &userPtr), NVCV_SUCCESS);
    EXPECT_EQ(expectedPtr, userPtr);

    ASSERT_EQ(nvcvAllocatorSetUserPointer(halloc, nullptr), NVCV_SUCCESS);
    ASSERT_EQ(nvcvAllocatorGetUserPointer(halloc, &userPtr), NVCV_SUCCESS);
    EXPECT_EQ(nullptr, userPtr);

    int newRef = 1;
    EXPECT_EQ(nvcvAllocatorDecRef(halloc, &newRef), NVCV_SUCCESS);
    EXPECT_EQ(newRef, 0);
}

TEST(AllocatorTest, invalid_arguments_api_calls)
{
    std::array<NVCVResourceAllocator, 2> allocators = {};

    allocators[0].resType         = NVCV_RESOURCE_MEM_HOST;
    allocators[0].res.mem.fnAlloc = [](auto, int64_t size, int32_t align)
    {
        return AllocHost(size, align);
    };
    allocators[0].res.mem.fnFree = [](auto, auto ptr, int64_t, int32_t align)
    {
        FreeHost(ptr, align);
    };
    allocators[1].resType         = NVCV_RESOURCE_MEM_CUDA;
    allocators[1].res.mem.fnAlloc = [](auto, int64_t size, int32_t)
    {
        void *mem = nullptr;
        EXPECT_EQ(cudaMalloc(&mem, size), cudaSuccess);
        return static_cast<NVCVMemoryBuffer>(mem);
    };
    allocators[1].res.mem.fnFree = [](auto, auto ptr, int64_t, int32_t)
    {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    };
    NVCVAllocatorHandle halloc = nullptr;
    // 1. Pointer to output handle must not be NULL
    EXPECT_EQ(nvcvAllocatorConstructCustom(allocators.data(), allocators.size(), nullptr), NVCV_ERROR_INVALID_ARGUMENT);
    ASSERT_EQ(nvcvAllocatorConstructCustom(allocators.data(), allocators.size(), &halloc), NVCV_SUCCESS);
    ASSERT_NE(halloc, nullptr);

    // 2. Pointer to output user pointer cannot be NULL
    EXPECT_EQ(nvcvAllocatorGetUserPointer(halloc, nullptr), NVCV_ERROR_INVALID_ARGUMENT);

    // 3. Pointer to output buffer must not be NULL
    EXPECT_EQ(nvcvAllocatorAllocHostMemory(halloc, nullptr, (1 << 10), 256), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(nvcvAllocatorAllocHostPinnedMemory(halloc, nullptr, (1 << 10), 256), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(nvcvAllocatorAllocCudaMemory(halloc, nullptr, (1 << 10), 256), NVCV_ERROR_INVALID_ARGUMENT);

    // 4. allocHostMem
    NVCVMemoryBuffer p0 = nullptr;
    EXPECT_EQ(nvcvAllocatorAllocHostMemory(halloc, &p0, -1, 256), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(nvcvAllocatorAllocHostMemory(halloc, &p0, (1 << 10), 3), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(nvcvAllocatorAllocHostMemory(halloc, &p0, 128, 256), NVCV_ERROR_INVALID_ARGUMENT);

    // 5. allocHostPinnedMem
    EXPECT_EQ(nvcvAllocatorAllocHostPinnedMemory(halloc, &p0, -1, 256), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(nvcvAllocatorAllocHostPinnedMemory(halloc, &p0, (1 << 10), 3), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(nvcvAllocatorAllocHostPinnedMemory(halloc, &p0, 128, 256), NVCV_ERROR_INVALID_ARGUMENT);

    // 6. allocHostPinnedMem
    EXPECT_EQ(nvcvAllocatorAllocCudaMemory(halloc, &p0, -1, 256), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(nvcvAllocatorAllocCudaMemory(halloc, &p0, (1 << 10), 3), NVCV_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(nvcvAllocatorAllocCudaMemory(halloc, &p0, 128, 256), NVCV_ERROR_INVALID_ARGUMENT);

    int newRef = 1;
    EXPECT_EQ(nvcvAllocatorDecRef(halloc, &newRef), NVCV_SUCCESS);
    EXPECT_EQ(newRef, 0);
}

TEST(AllocatorTest, customAllocator_constructor_negative)
{
    std::array<NVCVResourceAllocator, 1> invalidFnAllocAllocator         = {};
    std::array<NVCVResourceAllocator, 1> invalidFnFreeAllocator          = {};
    std::array<NVCVResourceAllocator, 2> duplicatedResourceTypeAllocator = {};

    // 1. allocation function must not be NULL
    invalidFnAllocAllocator[0].resType        = NVCV_RESOURCE_MEM_HOST;
    invalidFnAllocAllocator[0].res.mem.fnFree = [](auto, auto ptr, int64_t, int32_t align)
    {
        FreeHost(ptr, align);
    };
    NVCVAllocatorHandle halloc = nullptr;

    EXPECT_EQ(nvcvAllocatorConstructCustom(invalidFnAllocAllocator.data(), invalidFnAllocAllocator.size(), &halloc),
              NVCV_ERROR_INVALID_ARGUMENT);

    // 2. deallocation function must not be NULL
    invalidFnFreeAllocator[0].resType         = NVCV_RESOURCE_MEM_CUDA;
    invalidFnFreeAllocator[0].res.mem.fnAlloc = [](auto, int64_t size, int32_t)
    {
        void *mem = nullptr;
        EXPECT_EQ(cudaMalloc(&mem, size), cudaSuccess);
        return static_cast<NVCVMemoryBuffer>(mem);
    };
    EXPECT_EQ(nvcvAllocatorConstructCustom(invalidFnFreeAllocator.data(), invalidFnFreeAllocator.size(), &halloc),
              NVCV_ERROR_INVALID_ARGUMENT);

    // 3. duplicated resource type
    duplicatedResourceTypeAllocator[0].resType         = NVCV_RESOURCE_MEM_HOST;
    duplicatedResourceTypeAllocator[0].res.mem.fnAlloc = [](auto, int64_t size, int32_t align)
    {
        return AllocHost(size, align);
    };
    duplicatedResourceTypeAllocator[0].res.mem.fnFree = [](auto, auto ptr, int64_t, int32_t align)
    {
        FreeHost(ptr, align);
    };
    duplicatedResourceTypeAllocator[1].resType         = NVCV_RESOURCE_MEM_HOST;
    duplicatedResourceTypeAllocator[1].res.mem.fnAlloc = [](auto, int64_t size, int32_t align)
    {
        return AllocHost(size, align);
    };
    duplicatedResourceTypeAllocator[1].res.mem.fnFree = [](auto, auto ptr, int64_t, int32_t align)
    {
        FreeHost(ptr, align);
    };
    EXPECT_EQ(nvcvAllocatorConstructCustom(duplicatedResourceTypeAllocator.data(),
                                           duplicatedResourceTypeAllocator.size(), &halloc),
              NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(AllocatorTest, get_name)
{
    EXPECT_STREQ("NVCV_RESOURCE_MEM_CUDA", nvcvResourceTypeGetName(NVCV_RESOURCE_MEM_CUDA));
    EXPECT_STREQ("NVCV_RESOURCE_MEM_HOST", nvcvResourceTypeGetName(NVCV_RESOURCE_MEM_HOST));
    EXPECT_STREQ("NVCV_RESOURCE_MEM_HOST_PINNED", nvcvResourceTypeGetName(NVCV_RESOURCE_MEM_HOST_PINNED));
    EXPECT_STREQ("Unexpected error retrieving NVCVResourceType string representation",
                 nvcvResourceTypeGetName(static_cast<NVCVResourceType>(255)));
}
