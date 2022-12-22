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

#ifndef NVCV_IRESOURCEALLOCATOR_HPP
#define NVCV_IRESOURCEALLOCATOR_HPP

#include <cassert>
#include <cstddef> // for std::max_align_t

/**
 * @file IResourceAllocator.hpp
 *
 * @brief Defines C++ interface for resource allocation.
 */

namespace nvcv {

class IResourceAllocator
{
public:
    virtual ~IResourceAllocator() = 0;
};

inline IResourceAllocator::~IResourceAllocator() {}

class IMemAllocator : public IResourceAllocator
{
public:
    static constexpr int DEFAULT_ALIGN = alignof(std::max_align_t);

    using AllocFunc = void *(int64_t size, int32_t align);
    using FreeFunc  = void(void *ptr, int64_t size, int32_t align);

    void *alloc(int64_t size, int32_t align = DEFAULT_ALIGN);
    void  free(void *ptr, int64_t size, int32_t align = DEFAULT_ALIGN) noexcept;

private:
    // NVI pattern
    virtual void *doAlloc(int64_t size, int32_t align)                    = 0;
    virtual void  doFree(void *ptr, int64_t size, int32_t align) noexcept = 0;
};

class IHostMemAllocator : public virtual IMemAllocator
{
};

class IHostPinnedMemAllocator : public virtual IMemAllocator
{
};

class ICudaMemAllocator : public virtual IMemAllocator
{
};

// Implementation ----------------------

inline void *IMemAllocator::alloc(int64_t size, int32_t align)
{
    void *ptr = doAlloc(size, align);
    assert(ptr != nullptr && "nvcv::IMemAllocator::alloc post-condition failed");
    return ptr;
}

inline void IMemAllocator::free(void *ptr, int64_t size, int32_t align) noexcept
{
    doFree(ptr, size, align);
}

} // namespace nvcv

#endif // NVCV_IRESOURCEALLOCATOR_HPP
