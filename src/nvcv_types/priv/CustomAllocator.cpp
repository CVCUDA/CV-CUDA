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

#include "CustomAllocator.hpp"

#include "DefaultAllocator.hpp"

#include <cuda_runtime.h>
#include <nvcv/Version.h>
#include <util/CheckError.hpp>

#include <algorithm>
#include <cstdlib> // for aligned_alloc

namespace nvcv::priv {

CustomAllocator::CustomAllocator(const NVCVCustomAllocator *customAllocators, int32_t numCustomAllocators)
{
    if (customAllocators == nullptr && numCustomAllocators != 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Array to custom allocators must not be NULL if custom allocator count is > 0");
    }

    // First fill in the custom allocators passed by the user

    uint32_t filledMap = 0;
    static_assert(NVCV_NUM_RESOURCE_TYPES <= 32);

    for (int i = 0; i < numCustomAllocators; ++i)
    {
        NVCV_ASSERT(customAllocators != nullptr);

        const NVCVCustomAllocator &custAlloc = customAllocators[i];

        bool valid = false;
        switch (custAlloc.resType)
        {
        case NVCV_RESOURCE_MEM_HOST:
        case NVCV_RESOURCE_MEM_HOST_PINNED:
        case NVCV_RESOURCE_MEM_CUDA:
            if (custAlloc.res.mem.fnAlloc == nullptr)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "Custom memory allocation function for type " << custAlloc.resType << " must not be NULL";
            }
            if (custAlloc.res.mem.fnFree == nullptr)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "Custom memory deallocation function for type " << custAlloc.resType << " must not be NULL";
            }

            valid = true;
            break;
        }

        if (!valid)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Memory type '%d' is not understood", (int)custAlloc.resType);
        }

        if (filledMap & (1 << custAlloc.resType))
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Custom memory allocator for type " << custAlloc.resType << " is already defined";
        }

        m_allocators[custAlloc.resType] = custAlloc;
        filledMap |= 1 << custAlloc.resType;
    }

    // Now go through all allocators, find the ones that aren't customized
    // and set them to corresponding default allocator.

    static IAllocator &defAllocator = GetDefaultAllocator();

    for (int i = 0; i < NVCV_NUM_RESOURCE_TYPES; ++i)
    {
        NVCVCustomAllocator &custAllocator = m_allocators[i];

        // Resource allocator already defined?
        if (filledMap & (1 << i))
        {
            continue; // skip it
        }

        // Context not needed
        custAllocator.ctx = nullptr;

        custAllocator.resType = static_cast<NVCVResourceType>(i);
        filledMap |= (1 << i);

        switch (static_cast<NVCVResourceType>(i))
        {
        case NVCV_RESOURCE_MEM_HOST:
            static auto defAllocHostMem = [](void *ctx, int64_t size, int32_t align)
            {
                return defAllocator.allocHostMem(size, align);
            };
            static auto defFreeHostMem = [](void *ctx, void *ptr, int64_t size, int32_t align)
            {
                (void)size;
                (void)align;
                return defAllocator.freeHostMem(ptr, size, align);
            };
            custAllocator.res.mem.fnAlloc = defAllocHostMem;
            custAllocator.res.mem.fnFree  = defFreeHostMem;
            break;

        case NVCV_RESOURCE_MEM_CUDA:
            static auto defAllocCudaMem = [](void *ctx, int64_t size, int32_t align)
            {
                return defAllocator.allocCudaMem(size, align);
            };
            static auto defFreeCudaMem = [](void *ctx, void *ptr, int64_t size, int32_t align)
            {
                (void)size;
                (void)align;
                return defAllocator.freeCudaMem(ptr, size, align);
            };
            custAllocator.res.mem.fnAlloc = defAllocCudaMem;
            custAllocator.res.mem.fnFree  = defFreeCudaMem;
            break;

        case NVCV_RESOURCE_MEM_HOST_PINNED:
            static auto defAllocHostPinnedMem = [](void *ctx, int64_t size, int32_t align)
            {
                return defAllocator.allocHostPinnedMem(size, align);
            };
            static auto defFreeHostPinnedMem = [](void *ctx, void *ptr, int64_t size, int32_t align)
            {
                (void)size;
                (void)align;
                return defAllocator.freeHostPinnedMem(ptr, size, align);
            };
            custAllocator.res.mem.fnAlloc = defAllocHostPinnedMem;
            custAllocator.res.mem.fnFree  = defFreeHostPinnedMem;
            break;
        }
    }

    NVCV_ASSERT((filledMap & ((1 << NVCV_NUM_RESOURCE_TYPES) - 1)) == ((1 << NVCV_NUM_RESOURCE_TYPES) - 1)
                && "Some allocators weren't filled in");
}

// Host Memory ------------------

void *CustomAllocator::doAllocHostMem(int64_t size, int32_t align)
{
    NVCVCustomAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_HOST];
    NVCV_ASSERT(custom.res.mem.fnAlloc != nullptr);
    return custom.res.mem.fnAlloc(custom.ctx, size, align);
}

void CustomAllocator::doFreeHostMem(void *ptr, int64_t size, int32_t align) noexcept
{
    NVCVCustomAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_HOST];
    NVCV_ASSERT(custom.res.mem.fnFree != nullptr);
    return custom.res.mem.fnFree(custom.ctx, ptr, size, align);
}

// Host Pinned Memory ------------------

void *CustomAllocator::doAllocHostPinnedMem(int64_t size, int32_t align)
{
    NVCVCustomAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_HOST_PINNED];
    NVCV_ASSERT(custom.res.mem.fnAlloc != nullptr);
    return custom.res.mem.fnAlloc(custom.ctx, size, align);
}

void CustomAllocator::doFreeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept
{
    NVCVCustomAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_HOST_PINNED];
    NVCV_ASSERT(custom.res.mem.fnFree != nullptr);
    return custom.res.mem.fnFree(custom.ctx, ptr, size, align);
}

// Cuda Memory ------------------

void *CustomAllocator::doAllocCudaMem(int64_t size, int32_t align)
{
    NVCVCustomAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_CUDA];
    NVCV_ASSERT(custom.res.mem.fnAlloc != nullptr);
    return custom.res.mem.fnAlloc(custom.ctx, size, align);
}

void CustomAllocator::doFreeCudaMem(void *ptr, int64_t size, int32_t align) noexcept
{
    NVCVCustomAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_CUDA];
    NVCV_ASSERT(custom.res.mem.fnFree != nullptr);
    return custom.res.mem.fnFree(custom.ctx, ptr, size, align);
}

} // namespace nvcv::priv
