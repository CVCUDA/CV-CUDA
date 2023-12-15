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

#include <cvcuda/priv/WorkspaceAllocator.hpp>

#define EXPECT_PTR_EQ(a, b) EXPECT_EQ((const void *)(a), (const void *)(b))

TEST(WorkspaceMemAllocatorTest, Get)
{
    alignas(64) char     base[64];
    cvcuda::WorkspaceMem wm{};
    wm.req  = {64, 64};
    wm.data = base;

    cvcuda::WorkspaceMemAllocator wa(wm);
    EXPECT_PTR_EQ(wa.get<char>(3), base + 0);
    EXPECT_PTR_EQ(wa.get<int32_t>(3), base + 4);
    EXPECT_PTR_EQ(wa.get<float>(), base + 16);
    EXPECT_PTR_EQ(wa.get<float>(1, 16), base + 32);
    EXPECT_PTR_EQ(wa.get<float>(4), base + 48);
}

TEST(WorkspaceMemAllocatorTest, ExceedWorkspaceSize)
{
    alignas(64) char     base[64];
    cvcuda::WorkspaceMem wm{};
    wm.req  = {64, 64};
    wm.data = base;

    cvcuda::WorkspaceMemAllocator wa(wm);
    EXPECT_PTR_EQ(wa.get<double>(4), base + 0);
    EXPECT_PTR_EQ(wa.get<float>(7), base + 32);
    EXPECT_PTR_EQ(wa.allocated(), 60);
    EXPECT_THROW(wa.get<float>(2), nvcv::Exception);
    EXPECT_PTR_EQ(wa.get<float>(1), base + 60);
    EXPECT_THROW(wa.get<char>(1), nvcv::Exception);
}

TEST(WorkspaceAllocatorTest, Get)
{
    alignas(64) char  base[64];
    alignas(64) char  pinnedBase[64];
    cvcuda::Workspace ws{};
    ws.hostMem.req    = {64, 64};
    ws.hostMem.data   = base;
    ws.pinnedMem.req  = {64, 64};
    ws.pinnedMem.data = pinnedBase;

    cvcuda::WorkspaceAllocator wa(ws);
    EXPECT_PTR_EQ(wa.getHost<double>(4), base + 0);
    EXPECT_PTR_EQ(wa.getHost<float>(7), base + 32);
    EXPECT_PTR_EQ(wa.getPinned<double>(4), pinnedBase + 0);
    EXPECT_EQ(wa.hostMem.allocated(), 60);
    EXPECT_EQ(wa.pinnedMem.allocated(), 32);
    EXPECT_THROW(wa.getHost<float>(2), nvcv::Exception);
    EXPECT_PTR_EQ(wa.getHost<float>(1), base + 60);
    EXPECT_THROW(wa.getHost<char>(1), nvcv::Exception);
}

TEST(WorkspaceMemAllocatorTest, AcquireRelease)
{
    alignas(64) char     base[64];
    cvcuda::WorkspaceMem wm{};
    wm.req  = {64, 64};
    wm.data = base;
    ASSERT_EQ(cudaEventCreateWithFlags(&wm.ready, cudaEventDisableTiming), cudaSuccess);

    EXPECT_NO_THROW({
        cvcuda::WorkspaceMemAllocator wa(wm, cudaStream_t(0));
        EXPECT_PTR_EQ(wa.get(32), base);
    });

    EXPECT_NO_THROW({ cvcuda::WorkspaceMemAllocator wa(wm, cudaStream_t(0)); });

    EXPECT_NO_THROW({
        cvcuda::WorkspaceMemAllocator wa(wm, cudaStream_t(0));
        wa.acquire(std::nullopt);
        EXPECT_PTR_EQ(wa.get(32), base);
    });

    EXPECT_THROW(
        {
            cvcuda::WorkspaceMemAllocator wa(wm, std::nullopt, std::nullopt);
            EXPECT_PTR_EQ(wa.get(32), base);
            wa.acquire(std::nullopt);
        },
        std::logic_error)
        << "acquire after get should be an error";

    EXPECT_THROW(
        {
            cvcuda::WorkspaceMemAllocator wa(wm, std::nullopt, std::nullopt);
            wa.acquire(std::nullopt);
            wa.acquire(std::nullopt);
        },
        std::logic_error)
        << "double acquire should be an error";

    EXPECT_THROW(
        {
            cvcuda::WorkspaceMemAllocator wa(wm, std::nullopt, std::nullopt);
            wa.release(std::nullopt);
            EXPECT_PTR_EQ(wa.get(32), base);
        },
        std::logic_error)
        << "get after release should be an error";

    EXPECT_THROW(
        {
            cvcuda::WorkspaceMemAllocator wa(wm, std::nullopt, std::nullopt);
            wa.release(std::nullopt);
            wa.acquire(std::nullopt);
        },
        std::logic_error)
        << "acquire after release should be an error";

    EXPECT_THROW(
        {
            cvcuda::WorkspaceMemAllocator wa(wm, std::nullopt, std::nullopt);
            wa.release(std::nullopt);
            wa.release(std::nullopt);
        },
        std::logic_error)
        << "double release should be an error";

    ASSERT_EQ(cudaEventDestroy(wm.ready), cudaSuccess);
}

TEST(WorkspaceMemAllocatorTest, Sync)
{
    void  *_junk;
    size_t junk_size = 100 << 20;
    ASSERT_EQ(cudaMalloc(&_junk, junk_size), cudaSuccess);
    std::unique_ptr<void, void (*)(void *)> junk(_junk, [](void *p) { EXPECT_EQ(cudaFree(p), cudaSuccess); });

    alignas(64) char     base[64];
    cvcuda::WorkspaceMem wm{};
    wm.req  = {64, 64};
    wm.data = base;
    ASSERT_EQ(cudaEventCreateWithFlags(&wm.ready, cudaEventDisableTiming), cudaSuccess);

    // this is supposed to last long enough to be reliably "not ready"
    auto hog = [&]()
    {
        for (int i = 0; i < 256; i++)
        {
            ASSERT_EQ(cudaMemset(junk.get(), i, junk_size), cudaSuccess);
        }
    };

    EXPECT_NO_THROW({
        hog();
        ASSERT_EQ(cudaEventRecord(wm.ready, 0), cudaSuccess);
        {
            cvcuda::WorkspaceMemAllocator wa(wm, cudaStream_t(0));
            EXPECT_EQ(cudaEventQuery(wm.ready), cudaErrorNotReady); // no sync yet
        }
        EXPECT_EQ(cudaEventQuery(wm.ready), cudaErrorNotReady); // no sync necessary
    }) << "No memory was requested, no sync is necessary, no error expected";

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_NO_THROW({
        ASSERT_EQ(cudaEventRecord(wm.ready, 0), cudaSuccess);
        {
            cvcuda::WorkspaceMemAllocator wa(wm, cudaStream_t(0));
            EXPECT_PTR_EQ(wa.get(32), base);
            hog();
        }
        EXPECT_EQ(cudaEventQuery(wm.ready), cudaErrorNotReady); // device sync only
    }) << "Acquire and release properly called, no exception should be raised";

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_NO_THROW({
        hog();
        ASSERT_EQ(cudaEventRecord(wm.ready, 0), cudaSuccess);
        {
            cvcuda::WorkspaceMemAllocator wa(wm, std::nullopt, std::nullopt);
            EXPECT_EQ(cudaEventQuery(wm.ready), cudaErrorNotReady); // no sync yet
            EXPECT_PTR_EQ(wa.get(32), base);
            EXPECT_EQ(cudaEventQuery(wm.ready), cudaSuccess); // sync in get
        }
    }) << "Acquire and release properly called, no exception should be raised";

    ASSERT_EQ(cudaEventDestroy(wm.ready), cudaSuccess);
}
