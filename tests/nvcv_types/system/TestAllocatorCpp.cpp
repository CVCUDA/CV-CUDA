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

#include <nvcv/alloc/Allocator.hpp>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>

namespace n = nvcv;

namespace {

std::byte *AllocHost(int64_t size, int32_t align)
{
    return static_cast<std::byte *>(
        ::operator new (static_cast<size_t>(size), std::align_val_t{static_cast<size_t>(align)}, std::nothrow));
}

template<typename PointerType>
void FreeHost(PointerType *ptr, int32_t align) noexcept
{
    ::operator delete (ptr, std::align_val_t{static_cast<size_t>(align)});
}

} // namespace

TEST(AllocatorTest, FromEmpty)
{
    // Use thread-local variables because they don't need to be captured
    thread_local bool             alloc_called;
    thread_local bool             free_called;
    thread_local int64_t          allocated_size;
    thread_local NVCVMemoryBuffer allocated_ptr;

    alloc_called   = false;
    free_called    = false;
    allocated_size = -1;
    allocated_ptr  = nullptr;

    n::CustomMemAllocator<n::HostMemAllocator> alloc(
        [](int64_t size, int32_t align)
        {
            alloc_called   = true;
            allocated_size = size;
            allocated_ptr  = static_cast<NVCVMemoryBuffer>(static_cast<void *>(AllocHost(size, align)));
            return allocated_ptr;
        },
        [](NVCVMemoryBuffer mem, int64_t, int32_t align)
        {
            free_called = true;
            EXPECT_EQ(allocated_ptr, mem);
            FreeHost(mem, align);
        });

    EXPECT_FALSE(alloc.needsCleanup());
    NVCVResourceContext ctx = alloc.cdata().ctx;
    EXPECT_EQ(ctx, nullptr);

    auto &mem_alloc = alloc.cdata().res.mem;

    NVCVMemoryBuffer ptr = mem_alloc.fnAlloc(ctx, 123, 16);
    EXPECT_TRUE(alloc_called);
    EXPECT_EQ(allocated_size, 123);
    EXPECT_EQ(ptr, allocated_ptr);
    mem_alloc.fnFree(ctx, ptr, 123, 16);
    EXPECT_TRUE(free_called);
}

TEST(AllocatorTest, FromSmall)
{
    // Use thread-local variables because they don't need to be captured
    thread_local bool alloc_called;
    thread_local bool free_called;
    alloc_called = false;
    free_called  = false;

    int16_t c1 = 123;
    int16_t c2 = 321;

    n::CustomMemAllocator<n::HostMemAllocator> alloc(
        [c1](int64_t size, int32_t align)
        {
            alloc_called = true;
            EXPECT_EQ(c1, 123);
            return AllocHost(size, align);
        },
        [c2](NVCVMemoryBuffer mem, int64_t, int32_t align)
        {
            free_called = true;
            EXPECT_EQ(c2, 321);
            FreeHost(mem, align);
        });

    EXPECT_FALSE(alloc.needsCleanup());
    NVCVResourceContext ctx = alloc.cdata().ctx;
    EXPECT_NE(ctx, nullptr);

    auto &mem_alloc = alloc.cdata().res.mem;

    NVCVMemoryBuffer ptr = mem_alloc.fnAlloc(ctx, 123, 16);
    EXPECT_TRUE(alloc_called);
    mem_alloc.fnFree(ctx, ptr, 123, 16);
    EXPECT_TRUE(free_called);
}

TEST(AllocatorTest, FromDuplicate)
{
    struct Status
    {
        bool       alloc_called = false;
        bool       free_called  = false;
        intptr_t   value        = 0x12345678;
        std::byte *allocated    = nullptr;
    };

    Status status;

    struct DuplicateFunctor
    {
        Status *status;

        std::byte *operator()(int64_t size, int32_t align) const
        {
            status->alloc_called = true;
            EXPECT_EQ(status->value, 0x12345678);
            status->allocated = AllocHost(size, align);
            return status->allocated;
        }

        void operator()(NVCVMemoryBuffer mem, int64_t, int32_t align) const
        {
            status->free_called = true;
            EXPECT_EQ(status->value, 0x12345678);
            EXPECT_EQ(status->allocated, reinterpret_cast<std::byte *>(mem));
            FreeHost(mem, align);
        }
    };

    n::CustomMemAllocator<n::HostMemAllocator> alloc(DuplicateFunctor{&status}, DuplicateFunctor{&status});

    EXPECT_FALSE(alloc.needsCleanup());
    NVCVResourceContext ctx = alloc.cdata().ctx;
    EXPECT_NE(ctx, nullptr);

    auto &mem_alloc = alloc.cdata().res.mem;

    NVCVMemoryBuffer ptr = mem_alloc.fnAlloc(ctx, 123, 16);
    EXPECT_TRUE(status.alloc_called);
    mem_alloc.fnFree(ctx, ptr, 123, 16);
    EXPECT_TRUE(status.free_called);
}

TEST(AllocatorTest, FromDuplicateDifferentTypesMustNotShareStorage)
{
    struct Status
    {
        bool       alloc_called = false;
        bool       free_called  = false;
        std::byte *allocated    = nullptr;
    };

    Status status;

    struct AllocFunctor
    {
        Status *status;

        std::byte *operator()(int64_t size, int32_t align) const
        {
            status->alloc_called = true;
            status->allocated    = AllocHost(size, align);
            return status->allocated;
        }
    };

    struct FreeFunctor
    {
        Status *status;

        void operator()(NVCVMemoryBuffer mem, int64_t, int32_t align) const
        {
            status->free_called = true;
            EXPECT_EQ(status->allocated, reinterpret_cast<std::byte *>(mem));
            FreeHost(mem, align);
        }
    };

    static_assert(sizeof(AllocFunctor) == sizeof(FreeFunctor), "Test requires equal-size functors");
    static_assert(std::is_trivially_copyable_v<AllocFunctor>, "Test requires a trivial alloc functor");
    static_assert(std::is_trivially_copyable_v<FreeFunctor>, "Test requires a trivial free functor");

    AllocFunctor allocFn{&status};
    FreeFunctor  freeFn{&status};
    ASSERT_EQ(0, std::memcmp(&allocFn, &freeFn, sizeof(allocFn)));

    n::CustomMemAllocator<n::HostMemAllocator> alloc(AllocFunctor{&status}, FreeFunctor{&status});

    EXPECT_TRUE(alloc.needsCleanup());
    NVCVResourceContext ctx = alloc.cdata().ctx;
    EXPECT_NE(ctx, nullptr);

    auto &mem_alloc = alloc.cdata().res.mem;

    NVCVMemoryBuffer ptr = mem_alloc.fnAlloc(ctx, 123, 16);
    EXPECT_TRUE(status.alloc_called);
    mem_alloc.fnFree(ctx, ptr, 123, 16);
    EXPECT_TRUE(status.free_called);
}

TEST(AllocatorTest, FromComplexType)
{
    // Use thread-local variables because they don't need to be captured
    thread_local bool alloc_called;
    thread_local bool free_called;
    alloc_called = false;
    free_called  = false;

    thread_local bool destroyed;
    destroyed = false;

    struct Dummy
    {
        Dummy() = default;

        Dummy(const Dummy &)            = delete;
        Dummy(Dummy &&)                 = delete;
        Dummy &operator=(const Dummy &) = delete;
        Dummy &operator=(Dummy &&)      = delete;

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
                return AllocHost(size, align);
            },
            [p](NVCVMemoryBuffer mem, int64_t, int32_t align)
            {
                free_called = true;
                EXPECT_EQ(p->val, 0x12345678);
                FreeHost(mem, align);
            });
        p.reset();
        EXPECT_FALSE(destroyed);

        EXPECT_TRUE(alloc.needsCleanup());
        NVCVResourceContext ctx = alloc.cdata().ctx;
        EXPECT_NE(ctx, nullptr);

        auto &mem_alloc = alloc.cdata().res.mem;

        NVCVMemoryBuffer ptr = mem_alloc.fnAlloc(ctx, 123, 16);
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
                                            return AllocHost(size, align);
                                        },
                                        [](NVCVMemoryBuffer mem, int64_t, int32_t align)
                                        {
                                            status.host_free_called = true;
                                            FreeHost(mem, align);
                                        }),
                                    n::CustomCudaMemAllocator(
                                        [](int64_t size, int32_t)
                                        {
                                            status.cuda_alloc_called = true;
                                            void *mem                = nullptr;
                                            EXPECT_EQ(cudaMalloc(&mem, size), cudaSuccess);
                                            return static_cast<NVCVMemoryBuffer>(mem);
                                        },
                                        [](NVCVMemoryBuffer mem, int64_t, int32_t)
                                        {
                                            status.cuda_free_called = true;
                                            EXPECT_EQ(cudaFree(mem), cudaSuccess);
                                        }));

    ASSERT_FALSE(status.cuda_alloc_called);
    ASSERT_FALSE(status.cuda_free_called);
    ASSERT_FALSE(status.host_alloc_called);
    ASSERT_FALSE(status.host_free_called);

    NVCVMemoryBuffer cumem = ca.cudaMem().alloc(256);
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
        Dummy() = default;

        Dummy(const Dummy &)            = delete;
        Dummy(Dummy &&)                 = delete;
        Dummy &operator=(const Dummy &) = delete;
        Dummy &operator=(Dummy &&)      = delete;

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
                                            return AllocHost(size, align);
                                        },
                                        [sh](NVCVMemoryBuffer mem, int64_t, int32_t align)
                                        {
                                            status.host_free_called = true;
                                            FreeHost(mem, align);
                                        }),
                                    n::CustomCudaMemAllocator(
                                        [sh](int64_t size, int32_t)
                                        {
                                            status.cuda_alloc_called = true;
                                            void *mem                = nullptr;
                                            EXPECT_EQ(cudaMalloc(&mem, size), cudaSuccess);
                                            return static_cast<NVCVMemoryBuffer>(mem);
                                        },
                                        [sh](NVCVMemoryBuffer mem, int64_t, int32_t)
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

    NVCVMemoryBuffer cumem = ca.cudaMem().alloc(256);
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
