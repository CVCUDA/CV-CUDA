/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <malloc.h>
#include <nvcv/alloc/Allocator.hpp>

#include <thread>

#include <nvcv/alloc/Fwd.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

TEST(AllocatorTest, CreateAndUseCustom)
{
    NVCVResourceAllocator allocators[2] = {};

    int ctx0 = 100, ctx1 = 200;

    allocators[0].resType         = NVCV_RESOURCE_MEM_HOST;
    allocators[0].ctx             = &ctx0;
    allocators[0].res.mem.fnAlloc = [](void *ctx, int64_t size, int32_t align)
    {
        *(int *)ctx += 1;
        return memalign(align, size);
    };
    allocators[0].res.mem.fnFree = [](void *ctx, void *ptr, int64_t size, int32_t align)
    {
        *(int *)ctx += 10;
        free(ptr);
    };
    allocators[0].cleanup = [](void *ctx, NVCVResourceAllocator *alloc)
    {
        EXPECT_EQ(ctx, alloc->ctx);
        int *ctx_int = static_cast<int *>(ctx);
        EXPECT_EQ(*ctx_int, 111);
        *ctx_int = 0xDEAD;
    };

    allocators[1].resType         = NVCV_RESOURCE_MEM_CUDA;
    allocators[1].ctx             = &ctx1;
    allocators[1].res.mem.fnAlloc = [](void *ctx, int64_t size, int32_t align)
    {
        *(int *)ctx += 1;
        void *mem;
        EXPECT_EQ(cudaMalloc(&mem, size), cudaSuccess);
        return mem;
    };
    allocators[1].res.mem.fnFree = [](void *ctx, void *ptr, int64_t size, int32_t align)
    {
        *(int *)ctx += 10;
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    };
    allocators[1].cleanup = [](void *ctx, NVCVResourceAllocator *alloc)
    {
        EXPECT_EQ(ctx, alloc->ctx);
        int *ctx_int = static_cast<int *>(ctx);
        EXPECT_EQ(*ctx_int, 211);
        *ctx_int = 0xBAD;
    };

    NVCVAllocatorHandle halloc = nullptr;
    ASSERT_EQ(nvcvAllocatorConstructCustom(allocators, 2, &halloc), NVCV_SUCCESS);
    ASSERT_NE(halloc, nullptr);

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

    void *p0 = nullptr, *p1 = nullptr, *p2 = nullptr;
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

    int newRef = 1;
    EXPECT_EQ(nvcvAllocatorDecRef(halloc, &newRef), NVCV_SUCCESS);
    EXPECT_EQ(newRef, 0);
    EXPECT_EQ(ctx0, 0xDEAD);
    EXPECT_EQ(ctx1, 0xBAD);
}

// smoke: just to check if it compiles.
TEST(Allocator, smoke_test_default)
{
    nvcv::CustomAllocator myalloc;

    void *ptrDev        = myalloc.cudaMem().alloc(768, 256);
    void *ptrHost       = myalloc.hostMem().alloc(160, 16);
    void *ptrHostPinned = myalloc.hostPinnedMem().alloc(144, 16);

    myalloc.cudaMem().free(ptrDev, 768, 256);
    myalloc.hostMem().free(ptrHost, 160, 16);
    myalloc.hostPinnedMem().free(ptrHostPinned, 144, 16);
}

// smoke: just to check if it compiles.
TEST(Allocator, smoke_test_custom_functors)
{
    int devCounter        = 1;
    int hostCounter       = 1;
    int hostPinnedCounter = 1;

    // clang-format off
    nvcv::CustomAllocator myalloc1
    {
        nvcv::CustomHostMemAllocator
        {
            [&hostCounter](int64_t size, int32_t align)
            {
                void *ptr = reinterpret_cast<void *>(hostCounter);
                hostCounter += size;
                return ptr;
            },
            [&hostCounter](void *ptr, int64_t size, int32_t align)
            {
                hostCounter -= size;
                assert(hostCounter == reinterpret_cast<ptrdiff_t>(ptr));
            }
        },
        nvcv::CustomCudaMemAllocator
        {
            [&devCounter](int64_t size, int32_t align)
            {
                void *ptr = reinterpret_cast<void *>(devCounter);
                devCounter += size;
                return ptr;
            },
            [&devCounter](void *ptr, int64_t size, int32_t align)
            {
                devCounter -= size;
                assert(devCounter == reinterpret_cast<ptrdiff_t>(ptr));
            }
        },
        nvcv::CustomHostPinnedMemAllocator
        {
            [&hostPinnedCounter](int64_t size, int32_t align)
            {
                void *ptr = reinterpret_cast<void *>(hostPinnedCounter);
                hostPinnedCounter += size;
                return ptr;
            },
            [&hostPinnedCounter](void *ptr, int64_t size, int32_t align)
            {
                hostPinnedCounter -= size;
                assert(hostPinnedCounter == reinterpret_cast<ptrdiff_t>(ptr));
            }
        },
    };
    // clang-format on

    ASSERT_EQ((void *)1, myalloc1.hostMem().alloc(5));
    EXPECT_EQ(6, hostCounter);

    ASSERT_EQ((void *)1, myalloc1.hostPinnedMem().alloc(10));
    EXPECT_EQ(11, hostPinnedCounter);

    ASSERT_EQ((void *)1, myalloc1.cudaMem().alloc(7));
    EXPECT_EQ(8, devCounter);

    ASSERT_EQ((void *)8, myalloc1.cudaMem().alloc(2));
    EXPECT_EQ(10, devCounter);

    myalloc1.cudaMem().free((void *)8, 2);
    EXPECT_EQ(8, devCounter);

    myalloc1.cudaMem().free((void *)1, 7);
    EXPECT_EQ(1, devCounter);
}
