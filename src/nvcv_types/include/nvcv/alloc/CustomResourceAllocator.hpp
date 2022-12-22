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

#ifndef NVCV_CUSTOMRESOURCEALLOCATOR_HPP
#define NVCV_CUSTOMRESOURCEALLOCATOR_HPP

/**
 * @file CustomResourceAllocator.hpp
 *
 * @brief Defines C++ implementation of custom resource allocation.
 */

#include "IResourceAllocator.hpp"

namespace nvcv {

// Definition ------------------

namespace detail {

class CustomMemAllocatorImpl : public virtual IMemAllocator
{
public:
    using Interface = IMemAllocator;

    using AllocFunc = std::function<Interface::AllocFunc>;
    using FreeFunc  = std::function<Interface::FreeFunc>;

    CustomMemAllocatorImpl(AllocFunc alloc, FreeFunc free)
        : m_alloc(std::move(alloc))
        , m_free(std::move(free))
    {
    }

private:
    AllocFunc m_alloc;
    FreeFunc  m_free;

    void *doAlloc(int64_t size, int32_t align) override
    {
        return m_alloc(size, align);
    }

    void doFree(void *ptr, int64_t size, int32_t align) noexcept override
    {
        return m_free(ptr, size, align);
    }
};

} // namespace detail

class CustomHostMemAllocator final
    : public virtual IHostMemAllocator
    , private detail::CustomMemAllocatorImpl
{
public:
    using Interface = IHostMemAllocator;

    using detail::CustomMemAllocatorImpl::CustomMemAllocatorImpl;
};

class CustomHostPinnedMemAllocator final
    : public virtual IHostPinnedMemAllocator
    , private detail::CustomMemAllocatorImpl
{
public:
    using Interface = IHostPinnedMemAllocator;

    using detail::CustomMemAllocatorImpl::CustomMemAllocatorImpl;
};

class CustomCudaMemAllocator final
    : public virtual ICudaMemAllocator
    , private detail::CustomMemAllocatorImpl
{
public:
    using Interface = ICudaMemAllocator;

    using detail::CustomMemAllocatorImpl::CustomMemAllocatorImpl;
};

} // namespace nvcv

#endif // NVCV_CUSTOMRESOURCEALLOCATOR_HPP
