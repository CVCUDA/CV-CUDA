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

/**
 * @file IAllocator.hpp
 *
 * @brief Defines the public C++ interface to allocators.
 */

#ifndef NVCV_ALLOC_IALLOCATOR_HPP
#define NVCV_ALLOC_IALLOCATOR_HPP

#include "../Casts.hpp"

#include <cstdint>

#include "Fwd.hpp" // for NVCVAllocator

namespace nvcv {

// Helper class to explicitly assign
// address alignments.
class MemAlignment
{
public:
    MemAlignment() = default;

    int32_t baseAddr() const
    {
        return m_baseAddrAlignment;
    }

    int32_t rowAddr() const
    {
        return m_rowAddrAlignment;
    }

    MemAlignment &baseAddr(int32_t alignment)
    {
        m_baseAddrAlignment = alignment;
        return *this;
    }

    MemAlignment &rowAddr(int32_t alignment)
    {
        m_rowAddrAlignment = alignment;
        return *this;
    }

private:
    int32_t m_baseAddrAlignment = 0;
    int32_t m_rowAddrAlignment  = 0;
};

class IAllocator
{
public:
    using HandleType    = NVCVAllocatorHandle;
    using BaseInterface = IAllocator;

    virtual ~IAllocator() = default;

    NVCVAllocatorHandle handle() const noexcept;
    static IAllocator  *cast(HandleType h);

    IHostMemAllocator       &hostMem();
    IHostPinnedMemAllocator &hostPinnedMem();
    ICudaMemAllocator       &cudaMem();

    void  setUserPointer(void *ptr);
    void *userPointer() const;

private:
    // Using the NVI pattern.
    virtual NVCVAllocatorHandle doGetHandle() const = 0;

    virtual IHostMemAllocator       &doGetHostMemAllocator()       = 0;
    virtual IHostPinnedMemAllocator &doGetHostPinnedMemAllocator() = 0;
    virtual ICudaMemAllocator       &doGetCudaMemAllocator()       = 0;
};

inline NVCVAllocatorHandle IAllocator::handle() const noexcept
{
    return doGetHandle();
}

inline IHostMemAllocator &IAllocator::hostMem()
{
    return doGetHostMemAllocator();
}

inline IHostPinnedMemAllocator &IAllocator::hostPinnedMem()
{
    return doGetHostPinnedMemAllocator();
}

inline ICudaMemAllocator &IAllocator::cudaMem()
{
    return doGetCudaMemAllocator();
}

inline void IAllocator::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvAllocatorSetUserPointer(this->handle(), ptr));
}

inline void *IAllocator::userPointer() const
{
    void *ptr;
    detail::CheckThrow(nvcvAllocatorGetUserPointer(this->handle(), &ptr));
    return ptr;
}

inline IAllocator *IAllocator::cast(HandleType h)
{
    return detail::CastImpl<IAllocator>(&nvcvAllocatorGetUserPointer, &nvcvAllocatorSetUserPointer, h);
}

} // namespace nvcv

// Needed for casts
#include "AllocatorWrapHandle.hpp"

#endif // NVCV_ALLOC_IMEMALLOCATOR_HPP
