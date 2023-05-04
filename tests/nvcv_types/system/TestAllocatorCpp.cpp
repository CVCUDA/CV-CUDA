/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <malloc.h>
#include <nvcv/alloc/Allocator.hpp>

#include <cassert>

namespace n = nvcv;

TEST(AllocatorTest, FromEmpty)
{
    // Use thread-local variables because they don't need to be captured
    thread_local bool    alloc_called, free_called;
    thread_local int64_t allocated_size;
    thread_local void   *allocated_ptr;

    alloc_called   = false;
    free_called    = false;
    allocated_size = -1;
    allocated_ptr  = nullptr;

    n::CustomMemAllocator<n::HostMemAllocator> alloc(
        [](int64_t size, int32_t align)
        {
            alloc_called   = true;
            allocated_size = size;
            allocated_ptr  = memalign(align, size);
            return allocated_ptr;
        },
        [](void *mem, int64_t size, int32_t align)
        {
            free_called = true;
            EXPECT_EQ(allocated_ptr, mem);
            free(mem);
        });

    EXPECT_FALSE(alloc.needsCleanup());
    void *ctx = alloc.cdata().ctx;
    EXPECT_EQ(ctx, nullptr);

    auto &mem_alloc = alloc.cdata().res.mem;

    void *ptr = mem_alloc.fnAlloc(ctx, 123, 16);
    EXPECT_TRUE(alloc_called);
    EXPECT_EQ(allocated_size, 123);
    EXPECT_EQ(ptr, allocated_ptr);
    mem_alloc.fnFree(ctx, ptr, 123, 16);
    EXPECT_TRUE(free_called);
}

TEST(AllocatorTest, FromSmall)
{
    // Use thread-local variables because they don't need to be captured
    thread_local bool alloc_called, free_called;
    alloc_called = false;
    free_called  = false;

    int16_t c1 = 123, c2 = 321;

    n::CustomMemAllocator<n::HostMemAllocator> alloc(
        [c1](int64_t size, int32_t align)
        {
            alloc_called = true;
            EXPECT_EQ(c1, 123);
            return memalign(align, size);
        },
        [c2](void *mem, int64_t size, int32_t align)
        {
            free_called = true;
            EXPECT_EQ(c2, 321);
            free(mem);
        });

    EXPECT_FALSE(alloc.needsCleanup());
    void *ctx = alloc.cdata().ctx;
    EXPECT_NE(ctx, nullptr);

    auto &mem_alloc = alloc.cdata().res.mem;

    void *ptr = mem_alloc.fnAlloc(ctx, 123, 16);
    EXPECT_TRUE(alloc_called);
    mem_alloc.fnFree(ctx, ptr, 123, 16);
    EXPECT_TRUE(free_called);
}

TEST(AllocatorTest, FromDuplicate)
{
    // Use thread-local variables because they don't need to be captured
    thread_local bool alloc_called, free_called;
    alloc_called = false;
    free_called  = false;

    intptr_t c = 0x12345678;

    n::CustomMemAllocator<n::HostMemAllocator> alloc(
        [c](int64_t size, int32_t align)
        {
            alloc_called = true;
            EXPECT_EQ(c, 0x12345678);
            return memalign(align, size);
        },
        [c](void *mem, int64_t size, int32_t align)
        {
            free_called = true;
            EXPECT_EQ(c, 0x12345678);
            free(mem);
        });

    EXPECT_FALSE(alloc.needsCleanup());
    void *ctx = alloc.cdata().ctx;
    EXPECT_NE(ctx, nullptr);

    auto &mem_alloc = alloc.cdata().res.mem;

    void *ptr = mem_alloc.fnAlloc(ctx, 123, 16);
    EXPECT_TRUE(alloc_called);
    mem_alloc.fnFree(ctx, ptr, 123, 16);
    EXPECT_TRUE(free_called);
}

TEST(AllocatorTest, FromComplexType)
{
    // Use thread-local variables because they don't need to be captured
    thread_local bool alloc_called, free_called;
    alloc_called = false;
    free_called  = false;

    thread_local bool destroyed;
    destroyed = false;

    struct Dummy
    {
        ~Dummy()
        {
            val       = -1;
            destroyed = true;
        }

        intptr_t val = 0x12345678;
    };

    auto p = std::make_shared<Dummy>();

    {
        n::CustomHostMemAllocator alloc(
            [p](int64_t size, int32_t align)
            {
                alloc_called = true;
                EXPECT_EQ(p->val, 0x12345678);
                return memalign(align, size);
            },
            [p](void *mem, int64_t size, int32_t align)
            {
                free_called = true;
                EXPECT_EQ(p->val, 0x12345678);
                free(mem);
            });
        p.reset();
        EXPECT_FALSE(destroyed);

        EXPECT_TRUE(alloc.needsCleanup());
        void *ctx = alloc.cdata().ctx;
        EXPECT_NE(ctx, nullptr);

        auto &mem_alloc = alloc.cdata().res.mem;

        void *ptr = mem_alloc.fnAlloc(ctx, 123, 16);
        EXPECT_TRUE(alloc_called);
        mem_alloc.fnFree(ctx, ptr, 123, 16);
        EXPECT_TRUE(free_called);
    }
    EXPECT_TRUE(destroyed);
}

TEST(AllocatorTest, ConstructCustom)
{
    struct Status
    {
        bool host_alloc_called;
        bool host_free_called;
        bool cuda_alloc_called;
        bool cuda_free_called;
    };

    thread_local Status status;
    status = {};

    auto ca = CreateCustomAllocator(n::CustomHostMemAllocator(
                                        [](int64_t size, int32_t align)
                                        {
                                            status.host_alloc_called = true;
                                            return memalign(align, size);
                                        },
                                        [](void *mem, int64_t size, int32_t align)
                                        {
                                            status.host_free_called = true;
                                            return free(mem);
                                        }),
                                    n::CustomCudaMemAllocator(
                                        [](int64_t size, int32_t align)
                                        {
                                            status.cuda_alloc_called = true;
                                            void *mem;
                                            EXPECT_EQ(cudaMalloc(&mem, size), cudaSuccess);
                                            return mem;
                                        },
                                        [](void *mem, int64_t size, int32_t align)
                                        {
                                            status.cuda_free_called = true;
                                            EXPECT_EQ(cudaFree(mem), cudaSuccess);
                                        }));

    ASSERT_FALSE(status.cuda_alloc_called);
    ASSERT_FALSE(status.cuda_free_called);
    ASSERT_FALSE(status.host_alloc_called);
    ASSERT_FALSE(status.host_free_called);

    void *cumem = ca.cudaMem().alloc(256);
    EXPECT_TRUE(status.cuda_alloc_called);
    ca.cudaMem().free(cumem, 256);
    EXPECT_TRUE(status.cuda_free_called);

    auto *hmem = ca.hostMem().alloc(256);
    EXPECT_TRUE(status.host_alloc_called);
    ca.hostMem().free(hmem, 256);
    EXPECT_TRUE(status.host_free_called);
}

TEST(AllocatorTest, ConstructCustomWithDeleter)
{
    struct Status
    {
        bool host_alloc_called;
        bool host_free_called;
        bool cuda_alloc_called;
        bool cuda_free_called;
    };

    thread_local Status status;
    status = {};

    thread_local bool destroyed;
    destroyed = false;

    struct Dummy
    {
        ~Dummy()
        {
            val       = -1;
            destroyed = true;
        }

        intptr_t val = 0x12345678;
    };

    auto sh = std::make_shared<Dummy>();

    auto ca = CreateCustomAllocator(n::CustomHostMemAllocator(
                                        [sh](int64_t size, int32_t align)
                                        {
                                            status.host_alloc_called = true;
                                            return memalign(align, size);
                                        },
                                        [sh](void *mem, int64_t size, int32_t align)
                                        {
                                            status.host_free_called = true;
                                            return free(mem);
                                        }),
                                    n::CustomCudaMemAllocator(
                                        [sh](int64_t size, int32_t align)
                                        {
                                            status.cuda_alloc_called = true;
                                            void *mem;
                                            EXPECT_EQ(cudaMalloc(&mem, size), cudaSuccess);
                                            return mem;
                                        },
                                        [sh](void *mem, int64_t size, int32_t align)
                                        {
                                            status.cuda_free_called = true;
                                            EXPECT_EQ(cudaFree(mem), cudaSuccess);
                                        }));
    ASSERT_GT(sh.use_count(), 1);
    sh.reset();

    ASSERT_FALSE(destroyed);

    ASSERT_FALSE(status.cuda_alloc_called);
    ASSERT_FALSE(status.cuda_free_called);
    ASSERT_FALSE(status.host_alloc_called);
    ASSERT_FALSE(status.host_free_called);

    void *cumem = ca.cudaMem().alloc(256);
    EXPECT_TRUE(status.cuda_alloc_called);
    ca.cudaMem().free(cumem, 256);
    EXPECT_TRUE(status.cuda_free_called);

    auto *hmem = ca.hostMem().alloc(256);
    EXPECT_TRUE(status.host_alloc_called);
    ca.hostMem().free(hmem, 256);
    EXPECT_TRUE(status.host_free_called);

    ca.reset();
    ASSERT_TRUE(destroyed);
}
