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

#ifndef NVCV_PYTHON_PRIV_BUFFER_ALLOCATOR_HPP
#define NVCV_PYTHON_PRIV_BUFFER_ALLOCATOR_HPP

#include <common/CheckError.hpp>
#include <nvcv/alloc/Allocator.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

namespace nvcvpy::priv {

struct BufferAllocation
{
    std::shared_ptr<std::byte> data;
    int64_t                    capacity;
    int                        device;
};

inline void FreeBuffer(std::byte *ptr, int device)
{
    int         previous = -1;
    cudaError_t status   = cudaGetDevice(&previous);
    util::CheckLog(status);
    if (status != cudaSuccess)
        return;
    status = cudaSetDevice(device);
    util::CheckLog(status);
    if (status == cudaSuccess)
    {
        util::CheckLog(cudaFree(ptr));
        util::CheckLog(cudaSetDevice(previous));
    }
}

inline std::shared_ptr<BufferAllocation> AllocateBuffer(int64_t capacity)
{
    int device;
    util::CheckThrow(cudaGetDevice(&device));
    void *ptr = nullptr;
    util::CheckThrow(cudaMalloc(&ptr, capacity));
    std::shared_ptr<std::byte> buffer(static_cast<std::byte *>(ptr), [device](std::byte *p) { FreeBuffer(p, device); });
    return std::make_shared<BufferAllocation>(BufferAllocation{std::move(buffer), capacity, device});
}

// Each descriptor gets a fresh allocator whose first request binds it to the
// retained buffer. Later requests through the public native allocator API must
// remain independent allocations.
inline nvcv::Allocator CreateBufferAllocator(std::shared_ptr<BufferAllocation> buffer)
{
    struct State
    {
        explicit State(std::shared_ptr<BufferAllocation> buffer_)
            : buffer(std::move(buffer_))
        {
        }

        std::shared_ptr<BufferAllocation> buffer;
        std::atomic_bool                  supplied{false};
    };

    auto state    = std::make_shared<State>(std::move(buffer));
    auto allocate = [state](int64_t size, int32_t alignment)
    {
        if (!state->supplied.exchange(true))
        {
            if (size > state->buffer->capacity
                || reinterpret_cast<uintptr_t>(state->buffer->data.get()) % alignment != 0)
            {
                throw std::bad_alloc();
            }
            return state->buffer->data.get();
        }

        int previous;
        util::CheckThrow(cudaGetDevice(&previous));
        util::CheckThrow(cudaSetDevice(state->buffer->device));
        void       *ptr    = nullptr;
        cudaError_t status = cudaMalloc(&ptr, size);
        // Restore before reporting allocation failure as well as on success.
        util::CheckLog(cudaSetDevice(previous));
        util::CheckThrow(status);
        if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0)
        {
            FreeBuffer(static_cast<std::byte *>(ptr), state->buffer->device);
            throw std::bad_alloc();
        }
        return static_cast<std::byte *>(ptr);
    };
    auto release = [state](NVCVMemoryBuffer ptr, int64_t, int32_t)
    {
        auto retained = static_cast<NVCVMemoryBuffer>(static_cast<void *>(state->buffer->data.get()));
        if (ptr != nullptr && ptr != retained)
        {
            FreeBuffer(static_cast<std::byte *>(static_cast<void *>(ptr)), state->buffer->device);
        }
    };
    return nvcv::CustomAllocator(nvcv::CustomCudaMemAllocator(std::move(allocate), std::move(release)));
}

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_BUFFER_ALLOCATOR_HPP
