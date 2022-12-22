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

#include "IAllocator.hpp"

#include "IContext.hpp"

#include <util/Math.hpp>

namespace nvcv::priv {

void *IAllocator::allocHostMem(int64_t size, int32_t align)
{
    if (size < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Host memory allocator size must be >= 0, not %ld", size);
    }

    if (!util::IsPowerOfTwo(align))
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment when allocating host memory must be a power of two, not %d", align);
    }

    if (util::RoundUp(size, align) != size)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Host memory allocator size must be an integral multiple of alignment %d, not %ld", align,
                        size);
    }

    return doAllocHostMem(size, align);
}

void IAllocator::freeHostMem(void *ptr, int64_t size, int32_t align) noexcept
{
    doFreeHostMem(ptr, size, align);
}

void *IAllocator::allocHostPinnedMem(int64_t size, int32_t align)
{
    if (size < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Host-pinned memory allocator size must be >= 0, not %ld", size);
    }

    if (!util::IsPowerOfTwo(align))
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment when allocating host-pinned memory must be a power of two, not %d", align);
    }

    if (util::RoundUp(size, align) != size)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Host memory allocator size must be an integral multiple of alignment %d, not %ld", align,
                        size);
    }

    return doAllocHostPinnedMem(size, align);
}

void IAllocator::freeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept
{
    doFreeHostPinnedMem(ptr, size, align);
}

void *IAllocator::allocCudaMem(int64_t size, int32_t align)
{
    if (size < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Device memory allocator size must be >= 0, not %ld", size);
    }

    if (!util::IsPowerOfTwo(align))
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment when allocating device memory must be a power of two, not %d", align);
    }

    if (util::RoundUp(size, align) != size)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Host memory allocator size must be an integral multiple of alignment %d, not %ld", align,
                        size);
    }

    return doAllocCudaMem(size, align);
}

void IAllocator::freeCudaMem(void *ptr, int64_t size, int32_t align) noexcept
{
    doFreeCudaMem(ptr, size, align);
}

priv::IAllocator &GetDefaultAllocator()
{
    return GlobalContext().allocDefault();
}

priv::IAllocator &GetAllocator(NVCVAllocatorHandle handle)
{
    if (handle == nullptr)
    {
        return GetDefaultAllocator();
    }
    else
    {
        return priv::ToStaticRef<priv::IAllocator>(handle);
    }
}

} // namespace nvcv::priv
