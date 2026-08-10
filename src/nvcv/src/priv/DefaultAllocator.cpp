/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/util/CheckError.hpp>

#include <algorithm>
#include <new>

namespace nvcv::priv {

NVCVMemoryBuffer DefaultAllocator::doAllocHostMem(int64_t size, int32_t align)
{
    return static_cast<NVCVMemoryBuffer>(
        ::operator new (static_cast<size_t>(size), std::align_val_t{static_cast<size_t>(align)}));
}

void DefaultAllocator::doFreeHostMem(NVCVMemoryBuffer ptr, int64_t size, int32_t align) noexcept
{
    (void)size;
    ::operator delete (ptr, std::align_val_t{static_cast<size_t>(align)});
}

NVCVMemoryBuffer DefaultAllocator::doAllocHostPinnedMem(int64_t size, int32_t align)
{
    void *ptr = nullptr;
    NVCV_CHECK_THROW(::cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    // REVISIT: can we do better than this?
    if (reinterpret_cast<uintptr_t>(ptr) % align != 0)
    {
        NVCV_CHECK_LOG(::cudaFreeHost(ptr));
        throw Exception(NVCV_ERROR_INTERNAL, "Can't allocate %ld bytes of CUDA memory with alignment at %d bytes", size,
                        align);
    }
    return static_cast<NVCVMemoryBuffer>(ptr);
}

void DefaultAllocator::doFreeHostPinnedMem(NVCVMemoryBuffer ptr, int64_t size, int32_t align) noexcept
{
    (void)size;
    (void)align;

    NVCV_CHECK_LOG(::cudaFreeHost(ptr));
}

NVCVMemoryBuffer DefaultAllocator::doAllocCudaMem(int64_t size, int32_t align)
{
    void *ptr = nullptr;
    NVCV_CHECK_THROW(::cudaMalloc(&ptr, size));

    // REVISIT: can we do better than this?
    if (reinterpret_cast<uintptr_t>(ptr) % align != 0)
    {
        NVCV_CHECK_LOG(::cudaFree(ptr));
        throw Exception(NVCV_ERROR_INTERNAL, "Can't allocate %ld bytes of CUDA memory with alignment at %d bytes", size,
                        align);
    }
    return static_cast<NVCVMemoryBuffer>(ptr);
}

void DefaultAllocator::doFreeCudaMem(NVCVMemoryBuffer ptr, int64_t size, int32_t align) noexcept
{
    (void)size;
    (void)align;

    NVCV_CHECK_LOG(::cudaFree(ptr));
}

NVCVResourceAllocator DefaultAllocator::doGet(NVCVResourceType resType)
{
    NVCVResourceAllocator custAllocator = {};
    custAllocator.ctx                   = static_cast<NVCVResourceContext>(static_cast<void *>(this));
    custAllocator.resType               = resType;

    switch (resType)
    {
    case NVCV_RESOURCE_MEM_HOST:
        static auto defAllocHostMem = [](NVCVResourceContext ctx, int64_t size, int32_t align)
        {
            auto *self = static_cast<DefaultAllocator *>(static_cast<void *>(ctx));
            return self->allocHostMem(size, align);
        };
        static auto defFreeHostMem = [](NVCVResourceContext ctx, NVCVMemoryBuffer ptr, int64_t size, int32_t align)
        {
            auto *self = static_cast<DefaultAllocator *>(static_cast<void *>(ctx));
            return self->freeHostMem(ptr, size, align);
        };
        custAllocator.res.mem.fnAlloc = defAllocHostMem;
        custAllocator.res.mem.fnFree  = defFreeHostMem;
        break;

    case NVCV_RESOURCE_MEM_CUDA:
        static auto defAllocCudaMem = [](NVCVResourceContext ctx, int64_t size, int32_t align)
        {
            auto *self = static_cast<DefaultAllocator *>(static_cast<void *>(ctx));
            return self->allocCudaMem(size, align);
        };
        static auto defFreeCudaMem = [](NVCVResourceContext ctx, NVCVMemoryBuffer ptr, int64_t size, int32_t align)
        {
            auto *self = static_cast<DefaultAllocator *>(static_cast<void *>(ctx));
            return self->freeCudaMem(ptr, size, align);
        };
        custAllocator.res.mem.fnAlloc = defAllocCudaMem;
        custAllocator.res.mem.fnFree  = defFreeCudaMem;
        break;

    case NVCV_RESOURCE_MEM_HOST_PINNED:
        static auto defAllocHostPinnedMem = [](NVCVResourceContext ctx, int64_t size, int32_t align)
        {
            auto *self = static_cast<DefaultAllocator *>(static_cast<void *>(ctx));
            return self->allocHostPinnedMem(size, align);
        };
        static auto defFreeHostPinnedMem
            = [](NVCVResourceContext ctx, NVCVMemoryBuffer ptr, int64_t size, int32_t align)
        {
            auto *self = static_cast<DefaultAllocator *>(static_cast<void *>(ctx));
            return self->freeHostPinnedMem(ptr, size, align);
        };
        custAllocator.res.mem.fnAlloc = defAllocHostPinnedMem;
        custAllocator.res.mem.fnFree  = defFreeHostPinnedMem;
        break;

    default:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Unknown resource type: " << resType << ".";
    }

    return custAllocator;
}

} // namespace nvcv::priv
