/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/alloc/Allocator.h>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <thread>

#include <nvcv/alloc/Fwd.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

// WIP: just to check if it compiles.
TEST(Allocator, wip_test_default)
{
    nvcv::CustomAllocator myalloc;

    void *ptrDev        = myalloc.cudaMem().alloc(768, 256);
    void *ptrHost       = myalloc.hostMem().alloc(160, 16);
    void *ptrHostPinned = myalloc.hostPinnedMem().alloc(144, 16);

    myalloc.cudaMem().free(ptrDev, 768, 256);
    myalloc.hostMem().free(ptrHost, 160, 16);
    myalloc.hostPinnedMem().free(ptrHostPinned, 144, 16);
}

// WIP: just to check if it compiles.
TEST(Allocator, wip_test_custom_functors)
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

// WIP: just to check if it compiles.
TEST(Allocator, wip_test_custom_object)
{
    class MyCudaAlloc : public nvcv::ICudaMemAllocator
    {
    private:
        void *doAlloc(int64_t size, int32_t align) override
        {
            void *ptr;
            cudaMalloc(&ptr, size);
            return ptr;
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            cudaFree(ptr);
        }
    };

    class MyHostAlloc : public nvcv::IHostMemAllocator
    {
    private:
        void *doAlloc(int64_t size, int32_t align) override
        {
            return ::malloc(size);
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            ::free(ptr);
        }
    };

    nvcv::CustomAllocator myalloc1{MyHostAlloc{}, MyCudaAlloc{}};
}

TEST(Allocator, wip_test_custom_object_functor)
{
    class MyCudaAlloc
    {
    public:
        void *alloc(int64_t size, int32_t align)
        {
            void *ptr;
            cudaMalloc(&ptr, size);
            return ptr;
        }

        void dealloc(void *ptr, int64_t size, int32_t align) noexcept
        {
            cudaFree(ptr);
        }
    };

    class MyHostAlloc
    {
    public:
        void *alloc(int64_t size, int32_t align)
        {
            return ::malloc(size);
        }

        void dealloc(void *ptr, int64_t size, int32_t align) noexcept
        {
            ::free(ptr);
        }
    };

    auto myCudaAlloc = std::make_shared<MyCudaAlloc>();
    auto myHostAlloc = std::make_shared<MyHostAlloc>();

    // clang-format off
    nvcv::CustomAllocator myalloc1
    {
        nvcv::CustomHostMemAllocator{
            [myHostAlloc](int64_t size, int32_t align)
            {
                return myHostAlloc->alloc(size, align);
            },
            [myHostAlloc](void *ptr, int64_t size, int32_t align)
            {
                return myHostAlloc->dealloc(ptr, size, align);
            }
        },
        nvcv::CustomCudaMemAllocator{
            [myCudaAlloc](int64_t size, int32_t align)
            {
                return myCudaAlloc->alloc(size, align);
            },
            [myCudaAlloc](void *ptr, int64_t size, int32_t align)
            {
                return myCudaAlloc->dealloc(ptr, size, align);
            }
        },
    };
    // clang-format on
}

// WIP: just to check if it compiles.
TEST(Allocator, wip_test_custom_object_ref)
{
    class MyHostAlloc : public nvcv::IHostMemAllocator
    {
    private:
        void *doAlloc(int64_t size, int32_t align) override
        {
            return ::malloc(size);
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            ::free(ptr);
        }
    };

    MyHostAlloc myHostAlloc;

    nvcv::CustomAllocator myalloc1{
        std::ref(myHostAlloc),
    };

    nvcv::CustomAllocator myalloc2{std::ref(myHostAlloc)};

    auto myalloc3 = nvcv::CreateCustomAllocator(std::ref(myHostAlloc));

    EXPECT_EQ(&myHostAlloc, dynamic_cast<MyHostAlloc *>(&myalloc3.hostMem()));
    EXPECT_EQ(nullptr, dynamic_cast<MyHostAlloc *>(&myalloc3.cudaMem()));
}

class MyAsyncAlloc : public nvcv::ICudaMemAllocator
{
public:
    void setStream(cudaStream_t stream)
    {
        m_stream = stream;
    }

private:
    void *doAlloc(int64_t size, int32_t align) override
    {
        void *ptr;
        EXPECT_EQ(cudaSuccess, cudaMallocAsync(&ptr, size, m_stream));
        return ptr;
    }

    void doFree(void *ptr, int64_t size, int32_t align) noexcept override
    {
        EXPECT_EQ(cudaSuccess, cudaFreeAsync(ptr, m_stream));
    }

    cudaStream_t m_stream = 0;
};

TEST(Allocator, wip_test_dali_stream_async)
{
    cudaStream_t stream1, stream2;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream1));
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream2));

    auto fn = [](cudaStream_t stream)
    {
        thread_local MyAsyncAlloc          myAsyncAlloc;
        thread_local nvcv::CustomAllocator myalloc{std::ref(myAsyncAlloc)};

        myAsyncAlloc.setStream(stream);

        void *ptr = myalloc.cudaMem().alloc(123, 5);
        myalloc.cudaMem().free(ptr, 123, 5);
    };

    std::thread thread1(fn, stream1);
    std::thread thread2(fn, stream2);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream1));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream2));

    thread1.join();
    thread2.join();

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

TEST(Allocator, wip_double_destroy_noop)
{
    NVCVAllocatorHandle handle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvAllocatorConstructCustom(nullptr, 0, &handle));

    nvcvAllocatorDestroy(handle);

    void *ptr;
    NVCV_ASSERT_STATUS(NVCV_ERROR_INVALID_ARGUMENT, nvcvAllocatorFreeHostMemory(handle, &ptr, 16, 16));

    nvcvAllocatorDestroy(handle); // no-op, already destroyed
}

TEST(Allocator, wip_user_pointer)
{
    nvcv::CustomAllocator alloc;
    EXPECT_EQ(nullptr, alloc.userPointer());

    alloc.setUserPointer((void *)0x123);
    EXPECT_EQ((void *)0x123, alloc.userPointer());

    alloc.setUserPointer(nullptr);
    EXPECT_EQ(nullptr, alloc.userPointer());
}

TEST(Allocator, wip_cast)
{
    nvcv::CustomAllocator alloc;

    EXPECT_EQ(&alloc, nvcv::StaticCast<nvcv::CustomAllocator<> *>(alloc.handle()));
    EXPECT_EQ(&alloc, nvcv::StaticCast<nvcv::IAllocator *>(alloc.handle()));

    EXPECT_EQ(&alloc, &nvcv::StaticCast<nvcv::CustomAllocator<>>(alloc.handle()));
    EXPECT_EQ(&alloc, &nvcv::StaticCast<nvcv::IAllocator>(alloc.handle()));

    EXPECT_EQ(&alloc, nvcv::DynamicCast<nvcv::CustomAllocator<> *>(alloc.handle()));
    EXPECT_EQ(&alloc, nvcv::DynamicCast<nvcv::IAllocator *>(alloc.handle()));

    EXPECT_EQ(&alloc, &nvcv::DynamicCast<nvcv::CustomAllocator<>>(alloc.handle()));
    EXPECT_EQ(&alloc, &nvcv::DynamicCast<nvcv::IAllocator>(alloc.handle()));

    EXPECT_EQ(nullptr, nvcv::DynamicCast<nvcv::IAllocator *>(nullptr));
    EXPECT_THROW(nvcv::DynamicCast<nvcv::IAllocator>(nullptr), std::bad_cast);

    // Now when we create the object via C API

    NVCVAllocatorHandle handle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvAllocatorConstructCustom(nullptr, 0, &handle));

    // Size of the internal buffer used to store the WrapHandle object
    // we might have to create for containers allocated via C API.
    // This value must never decrease, or else it'll break ABI compatibility.
    uintptr_t max = 512;

    EXPECT_GE(max, sizeof(nvcv::detail::WrapHandle<nvcv::IAllocator>)) << "Must be big enough for the WrapHandle";

    void *cxxPtr = &max;
    ASSERT_EQ(NVCV_SUCCESS, nvcvAllocatorGetUserPointer((NVCVAllocatorHandle)(((uintptr_t)handle) | 1), &cxxPtr));
    ASSERT_NE(&max, cxxPtr) << "Pointer must have been changed";

    // Buffer too big, bail.
    max    = 513;
    cxxPtr = &max;
    ASSERT_EQ(NVCV_ERROR_INTERNAL, nvcvAllocatorGetUserPointer((NVCVAllocatorHandle)(((uintptr_t)handle) | 1), &cxxPtr))
        << "Required WrapHandle buffer storage should have been too big";

    nvcv::IAllocator *palloc = nvcv::StaticCast<nvcv::IAllocator *>(handle);
    ASSERT_NE(nullptr, palloc);
    EXPECT_EQ(handle, palloc->handle());

    EXPECT_EQ(palloc, nvcv::DynamicCast<nvcv::IAllocator *>(handle));
    EXPECT_EQ(nullptr, nvcv::DynamicCast<nvcv::CustomAllocator<> *>(handle));

    nvcvAllocatorDestroy(handle);
}

// disabled temporary while the API isn't stable
#if 0

// Parameter space validity tests ============================================

/********************************************
 *         nvcvMemAllocatorCreate
 *******************************************/

class MemAllocatorCreateParamTest
    : public t::TestWithParam<std::tuple<test::Param<"handle", bool>,        // 0
                                         test::Param<"numCustomAllocs", int> // 1
                                             NVCVStatus>>                    // 2
{
public:
    MemAllocatorCreateParamTest()
        : m_goldStatus(std::get<1>(GetParam()))
    {
        if (std::get<0>(GetParam()))
        {
            EXPECT_EQ(NVCV_SUCCESS, nvcvMemAllocatorCreate(&m_paramHandle));
        }
        else
        {
            m_paramHandle = nullptr;
        }
    }

    ~MemAllocatorCreateParamTest()
    {
        nvcvMemAllocatorDestroy(m_paramHandle);
    }

protected:
    NVCVMemAllocator m_paramHandle;
    NVCVStatus       m_goldStatus;
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Positive_handle, MemAllocatorCreateParamTest,
                              test::Value(true) * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_handle, MemAllocatorCreateParamTest,
                              test::Value(false) * NVCV_ERROR_INVALID_ARGUMENT);

// clang-format on

TEST_P(MemAllocatorCreateParamTest, stream)
{
    NVCVMemAllocator handle = nullptr;
    EXPECT_EQ(m_goldStatus, nvcvMemAllocatorCreate(m_paramHandle ? &handle : nullptr));

    nvcvMemAllocatorDestroy(handle); // to avoid memleaks
}

/********************************************
 *         nvcvMemAllocatorDestroy
 *******************************************/

using MemAllocatorDestroyParamTest = MemAllocatorCreateParamTest;

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Positive_handle, MemAllocatorDestroyParamTest,
                              test::ValueList{true, false} * NVCV_SUCCESS);

// clang-format on

TEST_P(MemAllocatorDestroyParamTest, stream)
{
    // Must not crash or assert, but we can't test that without using
    // googletest's Death Tests, as it involves forking the process.
    nvcvMemAllocatorDestroy(m_paramHandle);
    m_paramHandle = nullptr;
}

/********************************************
 *    nvcvMemAllocatorSetCustomAllocator
 *******************************************/

class MemAllocatorSetAllocatorParamTest
    : public t::TestWithParam<std::tuple<test::Param<"handle", bool, true>,                     // 0
                                         test::Param<"memtype", NVCVMemoryType, NVCV_MEM_HOST>, // 1
                                         test::Param<"fnMalloc", bool, true>,                   // 2
                                         test::Param<"fnFree", bool, true>,                     // 3
                                         test::Param<"ctx", bool, false>,                       // 4
                                         NVCVStatus>>                                           // 5
{
public:
    MemAllocatorSetAllocatorParamTest()
        : m_paramMemType(std::get<1>(GetParam()))
        , m_goldStatus(std::get<5>(GetParam()))
    {
        if (std::get<0>(GetParam()))
        {
            EXPECT_EQ(NVCV_SUCCESS, nvcvMemAllocatorCreate(&m_paramHandle));
        }

        if (std::get<2>(GetParam()))
        {
            // Dummy implementations
            static auto fnMalloc = [](void *, int64_t, int32_t, uint32_t) -> void *
            {
                return nullptr;
            };
            m_paramFnAllocMem = fnMalloc;
        }

        if (std::get<3>(GetParam()))
        {
            static auto fnFree = [](void *, void *, int64_t, int32_t, uint32_t) -> void {
            };
            m_paramFnFreeMem = fnFree;
        }

        if (std::get<4>(GetParam()))
        {
            m_paramContext = this;
        }
    }

    ~MemAllocatorSetAllocatorParamTest()
    {
        nvcvMemAllocatorDestroy(m_paramHandle);
    }

protected:
    NVCVMemAllocator m_paramHandle     = nullptr;
    NVCVMemAllocFunc m_paramFnAllocMem = nullptr;
    ;
    NVCVMemFreeFunc m_paramFnFreeMem = nullptr;
    void           *m_paramContext   = nullptr;
    NVCVMemoryType  m_paramMemType;
    NVCVStatus      m_goldStatus;
};

static test::ValueList g_ValidMemTypes = {NVCV_MEM_HOST, NVCV_MEM_CUDA, NVCV_MEM_CUDA_PINNED};

static test::ValueList g_InvalidMemTypes = {
    (NVCVMemoryType)-1,
    (NVCVMemoryType)NVCV_NUM_MEMORY_TYPES,
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Positive_handle, MemAllocatorSetAllocatorParamTest,
                              test::Value(true) * Dup<4>(test::ValueDefault())
                              * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_handle, MemAllocatorSetAllocatorParamTest,
                              test::Value(false) * Dup<4>(test::ValueDefault())
                              * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Positive_memtype, MemAllocatorSetAllocatorParamTest,
                              Dup<1>(test::ValueDefault()) * g_ValidMemTypes * Dup<3>(test::ValueDefault())
                              * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_memtype, MemAllocatorSetAllocatorParamTest,
                              Dup<1>(test::ValueDefault()) * g_InvalidMemTypes * Dup<3>(test::ValueDefault())
                              * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Positive_fnMalloc, MemAllocatorSetAllocatorParamTest,
                              Dup<2>(test::ValueDefault()) * test::Value(true) * Dup<2>(test::ValueDefault())
                              * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_fnMalloc, MemAllocatorSetAllocatorParamTest,
                              Dup<2>(test::ValueDefault()) * test::Value(false) * Dup<2>(test::ValueDefault())
                              * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Positive_fnFree, MemAllocatorSetAllocatorParamTest,
                              Dup<3>(test::ValueDefault()) * test::Value(true) * Dup<1>(test::ValueDefault())
                              * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_fnFree, MemAllocatorSetAllocatorParamTest,
                              Dup<3>(test::ValueDefault()) * test::Value(false) * Dup<1>(test::ValueDefault())
                              * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Positive_context, MemAllocatorSetAllocatorParamTest,
                              Dup<4>(test::ValueDefault()) * test::ValueList{true,false}
                              * NVCV_SUCCESS);

// clang-format on

TEST_P(MemAllocatorSetAllocatorParamTest, test)
{
    EXPECT_EQ(m_goldStatus, nvcvMemAllocatorSetCustomAllocator(m_paramHandle, m_paramMemType, m_paramFnAllocMem,
                                                               m_paramFnFreeMem, m_paramContext));
}

// Execution tests ===========================================

class MemAllocatorCreateExecTest : public t::Test
{
protected:
    test::ObjectBag m_bag;
};

TEST_F(MemAllocatorCreateExecTest, handle_filled_in)
{
    NVCVMemAllocator handle = nullptr;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMemAllocatorCreate(&handle));
    m_bag.insert(handle);

    EXPECT_NE(nullptr, handle);
}

#endif
