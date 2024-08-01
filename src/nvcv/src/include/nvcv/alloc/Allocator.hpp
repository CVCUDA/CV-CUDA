/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_ALLOC_ALLOCATOR_HPP
#define NVCV_ALLOC_ALLOCATOR_HPP

#include "../CoreResource.hpp"
#include "../detail/Callback.hpp"
#include "../detail/CompilerUtils.h"
#include "../detail/TypeTraits.hpp"
#include "Allocator.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>

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

/** A base class that encapsulates an NVCVResourceAllocator struct
 *
 * This class is a convenience wrapper around NVCVResourceAllocator. The derived classes expose
 * additional functionality, specific to the resource type being allocated.
 *
 * @warning ResourceAllocator does not own the context object pointed to by cdata().ctx.
 *          The destruction of ResourceAllocator does not call the cleanup function.
 */
class ResourceAllocator
{
public:
    ResourceAllocator() = default;

    explicit ResourceAllocator(const NVCVResourceAllocator &alloc)
        : m_data(alloc)
    {
    }

    /** Returns the underlying allocator descriptor
     */
    const NVCVResourceAllocator &cdata() const &
    {
        return m_data;
    }

    /** Returns the underlying allocator descriptor
     */
    NVCVResourceAllocator cdata() &&
    {
        return m_data;
    }

    /** Casts the resource allocator to a compatible type.
     */
    template<typename Derived>
    Derived cast() const
    {
        static_assert(std::is_base_of<ResourceAllocator, Derived>::value,
                      "The requested type does not inherit from ResourceAllocator");
        static_assert(std::is_constructible<Derived, NVCVResourceAllocator>::value,
                      "The requested type must be constructible from NVCVResourceAllocator");
        return Derived(m_data);
    }

    static constexpr bool IsCompatibleKind(NVCVResourceType)
    {
        return true;
    }

protected:
    NVCVResourceAllocator &data() &
    {
        return m_data;
    }

    NVCVResourceAllocator m_data{};
};

/** Encapculates a memory allocator (NVCV_RESOURCE_MEM_*)
 */
class MemAllocator : public ResourceAllocator
{
public:
    using ResourceAllocator::ResourceAllocator;

    static constexpr int DEFAULT_ALIGN = alignof(std::max_align_t);

    /** Calls the allocation function from the underlying descriptor
     */
    void *alloc(int64_t size, int32_t align = DEFAULT_ALIGN)
    {
        return m_data.res.mem.fnAlloc(m_data.ctx, size, align);
    }

    /** Calls the deallocation function from the underlying descriptor
     */
    void free(void *ptr, int64_t size, int32_t align = DEFAULT_ALIGN) noexcept
    {
        m_data.res.mem.fnFree(m_data.ctx, ptr, size, align);
    }

    static constexpr bool IsCompatibleKind(NVCVResourceType resType)
    {
        return resType == NVCV_RESOURCE_MEM_HOST || resType == NVCV_RESOURCE_MEM_HOST_PINNED
            || resType == NVCV_RESOURCE_MEM_CUDA;
    }
};

namespace detail {

/** Provides a common implementation for different memory allocator wrappers
 */
template<NVCVResourceType KIND>
class MemAllocatorWithKind : public MemAllocator
{
public:
    static constexpr NVCVResourceType kResourceType = KIND;

    MemAllocatorWithKind() = default;

    MemAllocatorWithKind(const NVCVResourceAllocator &data);

    static constexpr bool IsCompatibleKind(NVCVResourceType resType)
    {
        return resType == kResourceType;
    }
};

} // namespace detail

/** Encapsulates a host memory allocator descriptor
 */
class HostMemAllocator : public detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_HOST>
{
    using Impl = detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_HOST>;
    using Impl::Impl;
};

/** Encapsulates a host pinned memory allocator descriptor
 */
class HostPinnedMemAllocator : public detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_HOST_PINNED>
{
    using Impl = detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_HOST_PINNED>;
    using Impl::Impl;
};

/** Encapsulates a CUDA memory allocator descriptor
 */
class CudaMemAllocator : public detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_CUDA>
{
    using Impl = detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_CUDA>;
    using Impl::Impl;
};

NVCV_IMPL_SHARED_HANDLE(Allocator);

/** Represents a reference to an allocator object.
 *
 * The allocator object defines functions for allocating various resources, including
 * different kinds of memory.
 *
 * A custom allocator can be created via `nvcv::CustomAllocator` helper class.
 */
class Allocator : public CoreResource<NVCVAllocatorHandle, Allocator>
{
public:
    using CoreResource<NVCVAllocatorHandle, Allocator>::CoreResource;

    HostMemAllocator       hostMem() const;
    HostPinnedMemAllocator hostPinnedMem() const;
    CudaMemAllocator       cudaMem() const;

    ResourceAllocator get(NVCVResourceType resType) const;

    template<typename ResAlloc>
    ResAlloc get() const;
};

///////////////////////////////////////////////
// Custom allocators

/** Marshals a set of allocation/deallocation functions as NVCVResourceAllocator
 *
 * @note This class should not be used directly. Use one of the following typedefs:
 *       - CustomHostMemAllocator
 *       - CustomHostPinnedMemAllocator
 *       - CustomCudaMemAllocator
 *
 * A `CustomMemAllocator` is passed as a constructor argument to `CustomAllocator`.
 *
 * @tparam AllocatorType  the type of the allocator (one of: HostMemAllocator, HostPinnedMemAllocator, CudaMemAllocator)
 */
template<typename AllocatorType>
class CustomMemAllocator
{
private:
    template<typename Callable>
    struct has_trivial_copy_and_destruction
        : std::integral_constant<bool, std::is_trivially_copyable<Callable>::value
                                           && std::is_trivially_destructible<Callable>::value>
    {
    };

    template<typename Callable>
    struct by_value
        : std::integral_constant<bool, has_trivial_copy_and_destruction<Callable>::value
                                           && sizeof(Callable) <= sizeof(void *)
                                           && alignof(Callable) <= alignof(void *)>
    {
    };

    template<typename T>
    static constexpr size_t DataSize()
    {
        return std::is_empty<T>::value ? 0 : sizeof(T);
    }

public:
    /**  Constructs a custom memory allocator from a pair of alloc/free functions
     *
     * Usage:
     *
     * ```
     * nvcv::CustomHostMemAllocator alloc(
     *     [&alloc](int64_t size, int32_t align)
     *     {
     *         return alloc.allocate(size, align);
     *     },
     *     [&alloc](void *mem, int64_t size, int32_t align)
     *     {
     *         alloc.free(mem, size, align);
     *     });
     * ```
     *
     * @note When used with lambda functions, the user is responsible for ensuring
     *       that the allocator object doesn't outlive the variables captured by reference.
     */
    template<typename AllocFunction, typename FreeFunction,
             typename = detail::EnableIf_t<detail::IsInvocableR<void *, AllocFunction, int64_t, int32_t>::value>,
             typename = detail::EnableIf_t<detail::IsInvocableR<void, FreeFunction, void *, int64_t, int32_t>::value>>
    CustomMemAllocator(AllocFunction &&alloc, FreeFunction &&free);

    // TODO(michalz): Add a way of constructing a custom allocator without using lambdas/captures, e.g.
    //                from an object that matches the allocator concept.

    CustomMemAllocator(CustomMemAllocator &&other)
    {
        *this = std::move(other);
    }

    ~CustomMemAllocator()
    {
        reset();
    }

    bool needsCleanup() const noexcept
    {
        return m_data.cleanup != nullptr;
    }

    /** Gets the underlying allocator descriptor.
     */
    const NVCVResourceAllocator &cdata() const &noexcept
    {
        return m_data;
    }

    /** Removes the underlying allocator descriptor, passing the ownership to the caller.
     */
    NVCV_NODISCARD NVCVResourceAllocator release() noexcept
    {
        NVCVResourceAllocator ret = {};
        std::swap(ret, m_data);
        return ret;
    }

    /** Replaces the currently owned descriptor with the one passed in the argument.
     *
     * The ownership of the descriptor is transfered to `CustomMemAllocator`.
     */
    void reset(NVCVResourceAllocator &&alloc) noexcept
    {
        reset();
        std::swap(m_data, alloc);
    }

    /** Clears the allocator descriptor, performing cleanup, if necessary.
     */
    void reset() noexcept
    {
        if (m_data.cleanup)
            m_data.cleanup(m_data.ctx, &m_data);
        m_data = {};
    }

    /** Moves the descriptor from another CustomMemAllocator to this one.
     *
     * If the previously owned descriptor requries cleanup, it is performed.
     * After the call, `other` has its descriptor cleared.
     */
    CustomMemAllocator &operator=(CustomMemAllocator &&other) noexcept
    {
        reset(other.release());
        return *this;
    }

private:
    template<typename...>
    friend class CustomAllocator;

    template<typename AllocFunction, typename FreeFunction>
    void Construct(AllocFunction &&alloc, FreeFunction &&free, std::true_type);

    template<typename AllocFunction, typename FreeFunction>
    void Construct(AllocFunction &&alloc, FreeFunction &&free, std::false_type);

    template<typename AllocFunction, typename FreeFunction>
    void ConstructFromDuplicateValues(AllocFunction &&alloc, FreeFunction &&free, std::true_type);

#if __cplusplus < 201703L
    template<typename AllocFunction, typename FreeFunction>
    void ConstructByOneValue(AllocFunction &&, FreeFunction &&, std::true_type, std::false_type)
    {
        assert(!"should never get here");
    }
#endif

    NVCVResourceAllocator m_data{};
};

using CustomHostMemAllocator       = CustomMemAllocator<HostMemAllocator>;
using CustomHostPinnedMemAllocator = CustomMemAllocator<HostPinnedMemAllocator>;
using CustomCudaMemAllocator       = CustomMemAllocator<CudaMemAllocator>;

/** A helper clas for defining custom allocators.
 *
 * @note Direct use of this class is recommended only in C++ 17 and newer.
 *       For older standards, use CreateCustomAllocator function insted.
 *
 * This class aggregates custom resource allocators.
 *
 * @tparam ResourceAllocators
 */
template<typename... ResourceAllocators>
class CustomAllocator final : public Allocator
{
public:
    explicit CustomAllocator(ResourceAllocators &&...allocators);

    ~CustomAllocator()
    {
        preDestroy();
    }

private:
    static constexpr bool kHasReferences
        = detail::Disjunction<detail::IsRefWrapper<detail::RemoveRef_t<ResourceAllocators>>...>::value;

    template<bool hasReferences = kHasReferences>
    detail::EnableIf_t<hasReferences> preDestroy()
    {
        if (this->reset() != 0)
            throw std::logic_error(
                "The allocator context contains references. The handle must not outlive the context.");
    }

    template<bool hasReferences = kHasReferences>
    detail::EnableIf_t<!hasReferences> preDestroy() noexcept
    {
    }
};

/** Constructs a `CustomAllocator` from a set of resource allocators.
 */
template<typename... ResourceAllocators>
CustomAllocator<ResourceAllocators...> CreateCustomAllocator(ResourceAllocators &&...allocators)
{
    return CustomAllocator<ResourceAllocators...>{std::move(allocators)...};
}

} // namespace nvcv

#include "AllocatorImpl.hpp"

#endif // NVCV_ALLOC_ALLOCATOR_HPP
