/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CustomAllocator::CustomAllocator(const NVCVResourceAllocator *customAllocators, int32_t numCustomAllocators)
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

        const NVCVResourceAllocator &custAlloc = customAllocators[i];

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

    m_customAllocatorMask = filledMap;

    // Now go through all allocators, find the ones that aren't customized
    // and set them to the corresponding default allocator.

    for (int i = 0; i < NVCV_NUM_RESOURCE_TYPES; ++i)
    {
        NVCVResourceAllocator &custAllocator = m_allocators[i];

        // Resource allocator already defined?
        if (filledMap & (1 << i))
        {
            continue; // skip it
        }

        filledMap |= (1 << i);
        custAllocator = GetDefaultAllocator().get((NVCVResourceType)i);
    }

    NVCV_ASSERT((filledMap & ((1 << NVCV_NUM_RESOURCE_TYPES) - 1)) == ((1 << NVCV_NUM_RESOURCE_TYPES) - 1)
                && "Some allocators weren't filled in");
}

CustomAllocator::~CustomAllocator()
{
    for (NVCVResourceAllocator &alloc : m_allocators)
    {
        if (alloc.cleanup)
        {
            alloc.cleanup(alloc.ctx, &alloc);
        }
    }
}

NVCVResourceAllocator CustomAllocator::doGet(NVCVResourceType resType)
{
    NVCV_ASSERT(static_cast<unsigned>(resType) < NVCV_NUM_RESOURCE_TYPES);
    return m_allocators[resType];
}

// Host Memory ------------------

void *CustomAllocator::doAllocHostMem(int64_t size, int32_t align)
{
    NVCVResourceAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_HOST];
    NVCV_ASSERT(custom.res.mem.fnAlloc != nullptr);
    return custom.res.mem.fnAlloc(custom.ctx, size, align);
}

void CustomAllocator::doFreeHostMem(void *ptr, int64_t size, int32_t align) noexcept
{
    NVCVResourceAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_HOST];
    NVCV_ASSERT(custom.res.mem.fnFree != nullptr);
    return custom.res.mem.fnFree(custom.ctx, ptr, size, align);
}

// Host Pinned Memory ------------------

void *CustomAllocator::doAllocHostPinnedMem(int64_t size, int32_t align)
{
    NVCVResourceAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_HOST_PINNED];
    NVCV_ASSERT(custom.res.mem.fnAlloc != nullptr);
    return custom.res.mem.fnAlloc(custom.ctx, size, align);
}

void CustomAllocator::doFreeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept
{
    NVCVResourceAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_HOST_PINNED];
    NVCV_ASSERT(custom.res.mem.fnFree != nullptr);
    return custom.res.mem.fnFree(custom.ctx, ptr, size, align);
}

// Cuda Memory ------------------

void *CustomAllocator::doAllocCudaMem(int64_t size, int32_t align)
{
    NVCVResourceAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_CUDA];
    NVCV_ASSERT(custom.res.mem.fnAlloc != nullptr);
    return custom.res.mem.fnAlloc(custom.ctx, size, align);
}

void CustomAllocator::doFreeCudaMem(void *ptr, int64_t size, int32_t align) noexcept
{
    NVCVResourceAllocator &custom = m_allocators[NVCV_RESOURCE_MEM_CUDA];
    NVCV_ASSERT(custom.res.mem.fnFree != nullptr);
    return custom.res.mem.fnFree(custom.ctx, ptr, size, align);
}

} // namespace nvcv::priv
