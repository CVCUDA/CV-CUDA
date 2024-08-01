/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cvcuda/util/StreamId.hpp>

#include <thread>

TEST(StreamIdTest, RegularAndDefault)
{
    cudaStream_t stream1 = 0, stream2 = 0;
    (void)cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    (void)cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    if (!stream1 || !stream2)
    {
        if (stream1)
            (void)cudaStreamDestroy(stream1);
        if (stream2)
            (void)cudaStreamDestroy(stream2);
        FAIL() << "Could not create two CUDA streams";
    }

    uint64_t id1 = nvcv::util::GetCudaStreamIdHint(stream1);
    uint64_t id2 = nvcv::util::GetCudaStreamIdHint(stream2);
    uint64_t id3 = nvcv::util::GetCudaStreamIdHint(0);
    EXPECT_NE(id1, id2);
    EXPECT_NE(id1, id3);
    EXPECT_NE(id2, id3);
    (void)cudaStreamDestroy(stream1);
    (void)cudaStreamDestroy(stream2);
}

/** Tests that distinct streams with the same handle get different IDs
 */
TEST(StreamIdTest, HandleReuse)
{
    if (!nvcv::util::IsCudaStreamIdHintUnambiguous())
        GTEST_SKIP() << "This platform doesn't have an unambiguous CUDA stream id\n";

    struct CudaDeleter
    {
        void operator()(void *p)
        {
            cudaFree(p);
        }
    };

    auto CudaAlloc = [](size_t size)
    {
        void *ret = nullptr;
        cudaMalloc(&ret, size);
        return ret;
    };

    size_t                             bufSize = 256 << 20; // 256MiB
    std::unique_ptr<void, CudaDeleter> mem(CudaAlloc(bufSize));

    cudaEvent_t e;
    (void)cudaEventCreateWithFlags(&e, cudaEventDisableTiming);

    bool done        = false;
    int  maxAttempts = 10;
    for (int i = 0; i < maxAttempts; i++)
    {
        (void)cudaDeviceSynchronize();
        cudaStream_t stream1 = 0, stream2 = 0;
        (void)cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
        uint64_t id1 = nvcv::util::GetCudaStreamIdHint(stream1);
        for (int i = 0; i < 10; i++) cudaMemsetAsync(mem.get(), i, bufSize, stream1);
        cudaEventRecord(e, stream1);
        if (stream1)
            (void)cudaStreamDestroy(stream1);
        (void)cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
        bool     stillRunning = (cudaEventQuery(e) == cudaErrorNotReady);
        uint64_t id2          = nvcv::util::GetCudaStreamIdHint(stream2);
        if (stream2)
            (void)cudaStreamDestroy(stream2);
        if (stream1 != stream2)
            continue; // no handle reuse - retry
        if (!stillRunning)
            continue; // the stream wasn't running - the ID may be the same without any harm
        EXPECT_NE(id1, id2);
        done = true;
        break;
    }

    (void)cudaEventDestroy(e);

    if (!done)
        GTEST_SKIP() << "Could not trigger handle reuse - no way to conduct the test";
}

TEST(StreamIdTest, PerThreadDefault)
{
    const int                N = 4;
    std::vector<std::thread> threads(N);
    std::vector<uint64_t>    ids(N);
    for (int i = 0; i < N; i++)
    {
        threads[i] = std::thread(
            [&, i]()
            {
                (void)cudaFree(0); // create/assign a context
                ids[i] = nvcv::util::GetCudaStreamIdHint(cudaStreamPerThread);
            });
    }
    for (int i = 0; i < N; i++) threads[i].join();
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++)
            EXPECT_NE(ids[i], ids[j]) << "Per-thread streams for threads " << i << " and " << j << " do not differ.";
}
