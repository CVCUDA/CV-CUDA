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

#include <cvcuda/priv/WorkspaceAllocator.hpp>
#include <nvcv/util/CheckError.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <future>

#define EXPECT_PTR_EQ(a, b) EXPECT_EQ((const void *)(a), (const void *)(b))

namespace {

class BlockedCudaStream
{
public:
    BlockedCudaStream()
        : m_release(m_releasePromise.get_future().share())
        , m_started(m_startedPromise.get_future())
    {
        NVCV_CHECK_THROW(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
        try
        {
            NVCV_CHECK_THROW(cudaLaunchHostFunc(m_stream, Wait, this));
            m_started.wait();
        }
        catch (...)
        {
            (void)cudaStreamDestroy(m_stream);
            throw;
        }
    }

    ~BlockedCudaStream()
    {
        release();
        (void)cudaStreamSynchronize(m_stream);
        (void)cudaStreamDestroy(m_stream);
    }

    BlockedCudaStream(const BlockedCudaStream &)            = delete;
    BlockedCudaStream &operator=(const BlockedCudaStream &) = delete;

    cudaStream_t get() const
    {
        return m_stream;
    }

    void release()
    {
        if (!m_released)
        {
            m_releasePromise.set_value();
            m_released = true;
        }
    }

private:
    static void CUDART_CB Wait(void *context)
    {
        auto &gate = *static_cast<BlockedCudaStream *>(context);
        gate.m_startedPromise.set_value();
        gate.m_release.wait();
    }

    cudaStream_t             m_stream{};
    std::promise<void>       m_releasePromise;
    std::shared_future<void> m_release;
    std::promise<void>       m_startedPromise;
    std::future<void>        m_started;
    bool                     m_released = false;
};

} // namespace

TEST(AllocateWorkspaceTest, AllocationFailureUsesPublicException)
{
    cvcuda::WorkspaceRequirements req{};
    req.cudaMem = {1, 256};

    try
    {
        auto ws = cvcuda::AllocateWorkspace(req);
        FAIL() << "Expected workspace allocation to fail";
    }
    catch (const nvcv::Exception &e)
    {
        EXPECT_EQ(nvcv::Status::ERROR_INVALID_ARGUMENT, e.code());
    }
}

TEST(WorkspaceMemAllocatorTest, Get)
{
    alignas(64) std::array<char, 64> base;
    cvcuda::WorkspaceMem             wm{};
    wm.req  = {64, 64};
    wm.data = base.data();

    cvcuda::WorkspaceMemAllocator wa(wm);
    EXPECT_PTR_EQ(wa.get<char>(3), base.data() + 0);
    EXPECT_PTR_EQ(wa.get<int32_t>(3), base.data() + 4);
    EXPECT_PTR_EQ(wa.get<float>(), base.data() + 16);
    EXPECT_PTR_EQ(wa.get<float>(1, 16), base.data() + 32);
    EXPECT_PTR_EQ(wa.get<float>(4), base.data() + 48);
}

TEST(WorkspaceMemAllocatorTest, ExceedWorkspaceSize)
{
    alignas(64) std::array<char, 64> base;
    cvcuda::WorkspaceMem             wm{};
    wm.req  = {64, 64};
    wm.data = base.data();

    cvcuda::WorkspaceMemAllocator wa(wm);
    EXPECT_PTR_EQ(wa.get<double>(4), base.data() + 0);
    EXPECT_PTR_EQ(wa.get<float>(7), base.data() + 32);
    EXPECT_PTR_EQ(wa.allocated(), 60);
    EXPECT_THROW(wa.get<float>(2), nvcv::Exception);
    EXPECT_PTR_EQ(wa.get<float>(1), base.data() + 60);
    EXPECT_THROW(wa.get<char>(1), nvcv::Exception);
}

TEST(WorkspaceAllocatorTest, Get)
{
    alignas(64) std::array<char, 64> base;
    alignas(64) std::array<char, 64> pinnedBase;
    cvcuda::Workspace                ws{};
    ws.hostMem.req    = {64, 64};
    ws.hostMem.data   = base.data();
    ws.pinnedMem.req  = {64, 64};
    ws.pinnedMem.data = pinnedBase.data();

    cvcuda::WorkspaceAllocator wa(ws);
    EXPECT_PTR_EQ(wa.getHost<double>(4), base.data() + 0);
    EXPECT_PTR_EQ(wa.getHost<float>(7), base.data() + 32);
    EXPECT_PTR_EQ(wa.getPinned<double>(4), pinnedBase.data() + 0);
    EXPECT_EQ(wa.hostMem.allocated(), 60);
    EXPECT_EQ(wa.pinnedMem.allocated(), 32);
    EXPECT_THROW(wa.getHost<float>(2), nvcv::Exception);
    EXPECT_PTR_EQ(wa.getHost<float>(1), base.data() + 60);
    EXPECT_THROW(wa.getHost<char>(1), nvcv::Exception);
}

TEST(WorkspaceMemAllocatorTest, AcquireRelease)
{
    alignas(64) std::array<char, 64> base;
    cvcuda::WorkspaceMem             wm{};
    wm.req  = {64, 64};
    wm.data = base.data();
    ASSERT_EQ(cudaEventCreateWithFlags(&wm.ready, cudaEventDisableTiming), cudaSuccess);

    EXPECT_NO_THROW({
        cvcuda::WorkspaceMemAllocator wa(wm, cudaStream_t(0));
        EXPECT_PTR_EQ(wa.get(32), base.data());
    });

    EXPECT_NO_THROW({ cvcuda::WorkspaceMemAllocator wa(wm, cudaStream_t(0)); });

    EXPECT_NO_THROW({
        cvcuda::WorkspaceMemAllocator wa(wm, cudaStream_t(0));
        wa.acquire(std::nullopt);
        EXPECT_PTR_EQ(wa.get(32), base.data());
    });

    EXPECT_THROW(
        {
            cvcuda::WorkspaceMemAllocator wa(wm, std::nullopt, std::nullopt);
            EXPECT_PTR_EQ(wa.get(32), base.data());
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
            EXPECT_PTR_EQ(wa.get(32), base.data());
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
    alignas(64) std::array<char, 64> base;
    cvcuda::WorkspaceMem             wm{};
    wm.req  = {64, 64};
    wm.data = base.data();
    ASSERT_EQ(cudaEventCreateWithFlags(&wm.ready, cudaEventDisableTiming), cudaSuccess);

    EXPECT_NO_THROW({
        BlockedCudaStream gate;
        ASSERT_EQ(cudaEventRecord(wm.ready, gate.get()), cudaSuccess);
        {
            cvcuda::WorkspaceMemAllocator wa(wm, gate.get());
            EXPECT_EQ(cudaEventQuery(wm.ready), cudaErrorNotReady); // no sync yet
        }
        EXPECT_EQ(cudaEventQuery(wm.ready), cudaErrorNotReady); // no sync necessary
    }) << "No memory was requested, no sync is necessary, no error expected";

    EXPECT_NO_THROW({
        ASSERT_EQ(cudaEventRecord(wm.ready, 0), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(wm.ready), cudaSuccess);
        BlockedCudaStream gate;
        {
            cvcuda::WorkspaceMemAllocator wa(wm, gate.get());
            EXPECT_PTR_EQ(wa.get(32), base.data());
        }
        EXPECT_EQ(cudaEventQuery(wm.ready), cudaErrorNotReady); // device sync only
    }) << "Acquire and release properly called, no exception should be raised";

    EXPECT_NO_THROW({
        std::future<char *> sync;
        BlockedCudaStream   gate;
        ASSERT_EQ(cudaEventRecord(wm.ready, gate.get()), cudaSuccess);
        {
            cvcuda::WorkspaceMemAllocator wa(wm, std::nullopt, std::nullopt);
            EXPECT_EQ(cudaEventQuery(wm.ready), cudaErrorNotReady); // no sync yet
            sync = std::async(std::launch::async, [&wa] { return wa.get(32); });
            EXPECT_EQ(sync.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
            gate.release();
            EXPECT_PTR_EQ(sync.get(), base.data());
            EXPECT_EQ(cudaEventQuery(wm.ready), cudaSuccess); // sync in get
        }
    }) << "Acquire and release properly called, no exception should be raised";

    ASSERT_EQ(cudaEventDestroy(wm.ready), cudaSuccess);
}
