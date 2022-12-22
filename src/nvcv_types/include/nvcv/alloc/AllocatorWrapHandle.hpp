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
 * @file Allocator.hpp
 *
 * @brief Defines the public C++ implementation of allocators that wraps C allocators.
 */

#ifndef NVCV_ALLOCATOR_WRAP_HANDLE_HPP
#define NVCV_ALLOCATOR_WRAP_HANDLE_HPP

#include "../detail/CheckError.hpp"
#include "IAllocator.hpp"
#include "IResourceAllocator.hpp"

#include <functional>
#include <initializer_list>
#include <memory>

namespace nvcv {

namespace detail {

template<class IFACE>
class WrapHandle;

template<>
class WrapHandle<IAllocator> final : public virtual IAllocator
{
public:
    explicit WrapHandle(NVCVAllocatorHandle handle)
        : m_handle(handle)
        , m_allocHostMem(handle)
        , m_allocHostPinnedMem(handle)
        , m_allocCudaMem(handle)
    {
    }

private:
    class HostMemAllocator final : public IHostMemAllocator
    {
    public:
        HostMemAllocator(NVCVAllocatorHandle handle)
            : m_handle(handle)
        {
        }

    private:
        NVCVAllocatorHandle m_handle;

        void *doAlloc(int64_t size, int32_t align) override
        {
            void *ptr;
            detail::CheckThrow(nvcvAllocatorAllocHostMemory(m_handle, &ptr, size, align));
            return ptr;
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            nvcvAllocatorFreeHostMemory(m_handle, ptr, size, align);
        }
    };

    class HostPinnedMemAllocator final : public IHostPinnedMemAllocator
    {
    public:
        HostPinnedMemAllocator(NVCVAllocatorHandle handle)
            : m_handle(handle)
        {
        }

    private:
        NVCVAllocatorHandle m_handle;

        void *doAlloc(int64_t size, int32_t align) override
        {
            void *ptr;
            detail::CheckThrow(nvcvAllocatorAllocHostPinnedMemory(m_handle, &ptr, size, align));
            return ptr;
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            nvcvAllocatorFreeHostPinnedMemory(m_handle, ptr, size, align);
        }
    };

    class CudaMemAllocator final : public ICudaMemAllocator
    {
    public:
        CudaMemAllocator(NVCVAllocatorHandle handle)
            : m_handle(handle)
        {
        }

    private:
        NVCVAllocatorHandle m_handle;

        void *doAlloc(int64_t size, int32_t align) override
        {
            void *ptr;
            detail::CheckThrow(nvcvAllocatorAllocCudaMemory(m_handle, &ptr, size, align));
            return ptr;
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            nvcvAllocatorFreeCudaMemory(m_handle, ptr, size, align);
        }
    };

    NVCVAllocatorHandle m_handle;

    HostMemAllocator       m_allocHostMem;
    HostPinnedMemAllocator m_allocHostPinnedMem;
    CudaMemAllocator       m_allocCudaMem;

    NVCVAllocatorHandle doGetHandle() const noexcept override
    {
        return m_handle;
    }

    IHostMemAllocator &doGetHostMemAllocator() override
    {
        return m_allocHostMem;
    }

    IHostPinnedMemAllocator &doGetHostPinnedMemAllocator() override
    {
        return m_allocHostPinnedMem;
    }

    ICudaMemAllocator &doGetCudaMemAllocator() override
    {
        return m_allocCudaMem;
    }
};

} // namespace detail

// For API backward-compatibility
using AllocatorWrapHandle = detail::WrapHandle<IAllocator>;

} // namespace nvcv

#endif // NVCV_ALLOCATOR_WRAP_HANDLE_HPP
