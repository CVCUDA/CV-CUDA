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

#include "DefaultAllocator.hpp"

#include <cuda_runtime.h>
#include <nvcv/Version.h>
#include <util/CheckError.hpp>

#include <algorithm>
#include <cstdlib> // for aligned_alloc

namespace nvcv::priv {

void *DefaultAllocator::doAllocHostMem(int64_t size, int32_t align)
{
    return std::aligned_alloc(align, size);
}

void DefaultAllocator::doFreeHostMem(void *ptr, int64_t size, int32_t align) noexcept
{
    (void)size;
    (void)align;
    std::free(ptr);
}

void *DefaultAllocator::doAllocHostPinnedMem(int64_t size, int32_t align)
{
    void *ptr = nullptr;
    NVCV_CHECK_THROW(::cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    // TODO: can we do better than this?
    if (reinterpret_cast<uintptr_t>(ptr) % align != 0)
    {
        NVCV_CHECK_LOG(::cudaFreeHost(ptr));
        throw Exception(NVCV_ERROR_INTERNAL, "Can't allocate %ld bytes of CUDA memory with alignment at %d bytes", size,
                        align);
    }
    return ptr;
}

void DefaultAllocator::doFreeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept
{
    (void)size;
    (void)align;

    NVCV_CHECK_LOG(::cudaFreeHost(ptr));
}

void *DefaultAllocator::doAllocCudaMem(int64_t size, int32_t align)
{
    void *ptr = nullptr;
    NVCV_CHECK_THROW(::cudaMalloc(&ptr, size));

    // TODO: can we do better than this?
    if (reinterpret_cast<uintptr_t>(ptr) % align != 0)
    {
        NVCV_CHECK_LOG(::cudaFree(ptr));
        throw Exception(NVCV_ERROR_INTERNAL, "Can't allocate %ld bytes of CUDA memory with alignment at %d bytes", size,
                        align);
    }
    return ptr;
}

void DefaultAllocator::doFreeCudaMem(void *ptr, int64_t size, int32_t align) noexcept
{
    (void)size;
    (void)align;

    NVCV_CHECK_LOG(::cudaFree(ptr));
}

} // namespace nvcv::priv
