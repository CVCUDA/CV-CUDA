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

#ifndef NVCV_ALLOC_ALLOCATOR_IMPL_HPP
#define NVCV_ALLOC_ALLOCATOR_IMPL_HPP

namespace nvcv {

//////////////////////////////////////////////////////////////////////////////
// Allocator

inline HostMemAllocator Allocator::hostMem() const
{
    return get<HostMemAllocator>();
}

inline HostPinnedMemAllocator Allocator::hostPinnedMem() const
{
    return get<HostPinnedMemAllocator>();
}

inline CudaMemAllocator Allocator::cudaMem() const
{
    return get<CudaMemAllocator>();
}

inline ResourceAllocator Allocator::get(NVCVResourceType resType) const
{
    NVCVResourceAllocator data;
    detail::CheckThrow(nvcvAllocatorGet(handle(), resType, &data));
    return ResourceAllocator(data);
}

template<typename ResAlloc>
ResAlloc Allocator::get() const
{
    static_assert(std::is_base_of<ResourceAllocator, ResAlloc>::value,
                  "The requested resource allocator type is not derived from ResourceAllocator.");
    NVCVResourceAllocator data;
    detail::CheckThrow(nvcvAllocatorGet(handle(), ResAlloc::kResourceType, &data));
    return ResAlloc(data);
}

//////////////////////////////////////////////////////////////////////////////
// CustomMemAllocator

template<typename AllocatorType>
template<typename AllocFunction, typename FreeFunction, typename, typename>
CustomMemAllocator<AllocatorType>::CustomMemAllocator(AllocFunction &&alloc, FreeFunction &&free)
{
    static_assert(!std::is_lvalue_reference<AllocFunction>::value && !std::is_lvalue_reference<FreeFunction>::value,
                  "The allocation and deallocation functions must not be L-Value references. Use std::ref "
                  "if a reference is required. Note that using references will place additional requirements "
                  "on the lifetime of the function objects.");

    using T            = std::tuple<AllocFunction, FreeFunction>;
    const bool trivial = has_trivial_copy_and_destruction<AllocFunction>::value
                      && has_trivial_copy_and_destruction<FreeFunction>::value;

    const bool tuple_by_value = trivial && sizeof(T) <= sizeof(void *) && alignof(T) <= alignof(void *);

    const bool construct_from_one_value_if_equal = trivial && by_value<AllocFunction>::value
                                                && by_value<FreeFunction>::value
                                                && sizeof(AllocFunction) == sizeof(FreeFunction);

    // Can we fit the tuple inside a single pointer? If yes, go for it!
    if NVCV_IF_CONSTEXPR (tuple_by_value)
    {
        Construct(std::forward<AllocFunction>(alloc), std::forward<FreeFunction>(free),
                  std::integral_constant<bool, tuple_by_value>());
    }
    // Are the two callables trivial and do they context objects coincide? If yes, use that object and reinterpret the data
    // This might be useful in a common case where both alloc and free are lambdas that capture only one - and the same - pointer-like value.
    else if NVCV_IF_CONSTEXPR (construct_from_one_value_if_equal)
    {
        if (!std::memcmp(&alloc, &free, std::min(DataSize<AllocFunction>(), DataSize<FreeFunction>())))
            ConstructFromDuplicateValues(std::forward<AllocFunction>(alloc), std::forward<FreeFunction>(free),
                                         std::integral_constant<bool, construct_from_one_value_if_equal>());
        else
            Construct(std::forward<AllocFunction>(alloc), std::forward<FreeFunction>(free),
                      std::integral_constant<bool, tuple_by_value>());
    }
    // Back to square one - need to dynamically allocate the objects - we still have _one_ context object for two functions;
    // with std::function we'd end up dynamically allocating a pair of std::functions, each of which could dynamically allocate
    // - so we're still good; possibly down from 3 dynamic allocations to one
    else
    {
        Construct(std::forward<AllocFunction>(alloc), std::forward<FreeFunction>(free),
                  std::integral_constant<bool, tuple_by_value>());
    }
    m_data.resType = AllocatorType::kResourceType;
}

template<typename AllocatorType>
template<typename AllocFunction, typename FreeFunction>
void CustomMemAllocator<AllocatorType>::Construct(AllocFunction &&alloc, FreeFunction &&free, std::true_type)
{
    using T = std::tuple<AllocFunction, FreeFunction>; // TODO - use something that's trivially copyable
    T ctx{std::move(alloc), std::move(free)};
    static_assert(sizeof(T) <= sizeof(void *), "Internal error - this should never be invoked with a type that large.");

    m_data.res.mem.fnAlloc = [](void *c, int64_t size, int32_t align) -> void *
    {
        T     *target   = reinterpret_cast<T *>(&c);
        auto &&callable = std::get<0>(*target);
        return callable(size, align);
    };
    m_data.res.mem.fnFree = [](void *c, void *ptr, int64_t size, int32_t align)
    {
        T     *target   = reinterpret_cast<T *>(&c);
        auto &&callable = std::get<1>(*target);
        callable(ptr, size, align);
    };

    m_data.cleanup = nullptr;

    if NVCV_IF_CONSTEXPR (!std::is_empty<T>::value)
        std::memcpy(&m_data.ctx, &ctx, DataSize<T>());
}

template<typename AllocatorType>
template<typename AllocFunction, typename FreeFunction>
void CustomMemAllocator<AllocatorType>::Construct(AllocFunction &&alloc, FreeFunction &&free, std::false_type)
{
    using T = std::tuple<AllocFunction, FreeFunction>;
    std::unique_ptr<T> ctx(new T{std::move(alloc), std::move(free)});
    auto               cleanup = [](void *ctx, NVCVResourceAllocator *) noexcept
    {
        delete (T *)ctx;
    };

    m_data.res.mem.fnAlloc = [](void *c, int64_t size, int32_t align) -> void *
    {
        return std::get<0>(*static_cast<T *>(c))(size, align);
    };
    m_data.res.mem.fnFree = [](void *c, void *ptr, int64_t size, int32_t align)
    {
        std::get<1> (*static_cast<T *>(c))(ptr, size, align);
    };

    m_data.cleanup = cleanup;
    m_data.ctx     = ctx.release();
}

template<typename AllocatorType>
template<typename AllocFunction, typename FreeFunction>
void CustomMemAllocator<AllocatorType>::ConstructFromDuplicateValues(AllocFunction &&alloc, FreeFunction &&free,
                                                                     std::true_type)
{
    static_assert(std::is_trivially_copyable<AllocFunction>::value || std::is_empty<AllocFunction>::value,
                  "Internal error - should not pick this overload");
    static_assert(std::is_trivially_copyable<FreeFunction>::value || std::is_empty<FreeFunction>::value,
                  "Internal error - should not pick this overload");
    m_data.res.mem.fnAlloc = [](void *c, int64_t size, int32_t align) -> void *
    {
        AllocFunction *alloc = reinterpret_cast<AllocFunction *>(&c);
        return (*alloc)(size, align);
    };
    m_data.res.mem.fnFree = [](void *c, void *ptr, int64_t size, int32_t align)
    {
        FreeFunction *free = reinterpret_cast<FreeFunction *>(&c);
        (*free)(ptr, size, align);
    };

    m_data.cleanup = nullptr;
    m_data.ctx     = nullptr;
    if (DataSize<AllocFunction>() >= DataSize<FreeFunction>())
        std::memcpy(&m_data.ctx, &alloc, DataSize<AllocFunction>());
    else
        std::memcpy(&m_data.ctx, &free, DataSize<FreeFunction>());
}

//////////////////////////////////////////////////////////////////////////////
// CustomAllocator

template<typename... ResourceAllocators>
inline CustomAllocator<ResourceAllocators...>::CustomAllocator(ResourceAllocators &&...allocators)
{
    NVCVResourceAllocator data[] = {allocators.cdata()...};
    NVCVAllocatorHandle   h      = {};
    detail::CheckThrow(nvcvAllocatorConstructCustom(data, sizeof...(allocators), &h));
    // void-cast the (nodiscard) result of the allocators - we know what we're doing here...
    int dummy[] = {((void)allocators.release(), 0)...};
    (void)dummy;
    reset(std::move(h));
}

namespace detail {

template<NVCVResourceType KIND>
MemAllocatorWithKind<KIND>::MemAllocatorWithKind(const NVCVResourceAllocator &data)
    : MemAllocator(data)
{
    if (!IsCompatibleKind(data.resType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Incompatible allocated resource type.");
    }
}
} // namespace detail

} // namespace nvcv

#endif // NVCV_ALLOC_ALLOCATOR_IMPL_HPP
