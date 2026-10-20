/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TESTS_COMMON_DEVICEBUFFER_HPP
#define NVCV_TESTS_COMMON_DEVICEBUFFER_HPP

#include <cuda_runtime.h>

#include <cstddef>

namespace nvcv::test {

/**
 * A device allocation that frees on scope exit.
 *
 * gtest's ASSERT_* returns from the enclosing function, so a cudaFree written at the end of a test
 * body is skipped on exactly the paths that fail -- which is where leaking matters most, because a
 * failing test is the one that gets rerun. Construction cannot throw; get() returns nullptr if the
 * allocation failed, so callers ASSERT_NE on it.
 */
template<typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t count)
    {
        // Keep zero-sized buffers null so the callers' existing ASSERT_NE checks also reject
        // vacuous probe lists.
        if (count == 0)
        {
            return;
        }
        if (cudaMalloc(&m_ptr, count * sizeof(T)) != cudaSuccess)
        {
            m_ptr = nullptr;
            // Consume the error rather than leaving it latched: the caller reports the failure
            // through the null pointer, and a stale error would otherwise surface at whichever
            // cudaGetLastError() runs next and blame the wrong operation.
            (void)cudaGetLastError();
        }
    }

    ~DeviceBuffer()
    {
        if (m_ptr != nullptr)
        {
            (void)cudaFree(m_ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer &)            = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() const
    {
        return m_ptr;
    }

private:
    T *m_ptr = nullptr;
};

/**
 * A CUDA stream that destroys on scope exit, for the same reason DeviceBuffer frees on scope exit:
 * an ASSERT_* between creating the stream and destroying it returns from the enclosing function
 * and skips the destroy.
 */
class StreamGuard
{
public:
    StreamGuard()
    {
        if (cudaStreamCreate(&m_stream) != cudaSuccess)
        {
            m_stream = nullptr;
            (void)cudaGetLastError(); // see DeviceBuffer: do not leave the error latched
        }
    }

    ~StreamGuard()
    {
        if (m_stream != nullptr)
        {
            (void)cudaStreamDestroy(m_stream);
        }
    }

    StreamGuard(const StreamGuard &)            = delete;
    StreamGuard &operator=(const StreamGuard &) = delete;

    cudaStream_t get() const
    {
        return m_stream;
    }

private:
    cudaStream_t m_stream = nullptr;
};

// A one-int device result seeded with a value the parity kernels never write. Freshly mapped
// device memory usually reads back as zero, which is also the success value, so a kernel that
// never runs would otherwise report success.
inline constexpr int kResultUnwritten = -1;

inline cudaError_t SeedResult(int *devResult)
{
    return cudaMemcpy(devResult, &kResultUnwritten, sizeof(int), cudaMemcpyHostToDevice);
}

} // namespace nvcv::test

#endif // NVCV_TESTS_COMMON_DEVICEBUFFER_HPP
