/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_DETAIL_UNIQUE_OBJ_HPP
#define NVCV_DETAIL_UNIQUE_OBJ_HPP

#include <memory>
#include <utility>

namespace nvcv { namespace detail {

template<typename T>
struct UniqueObjDeleter
{
    void operator()(T *ptr) const noexcept
    {
        if (ptr != nullptr)
        {
            std::allocator<T> alloc;
            std::allocator_traits<std::allocator<T>>::destroy(alloc, ptr);
            std::allocator_traits<std::allocator<T>>::deallocate(alloc, ptr, 1);
        }
    }
};

template<typename T>
using UniqueObj = std::unique_ptr<T, UniqueObjDeleter<T>>;

template<typename T, typename... Args>
UniqueObj<T> MakeUniqueObj(Args &&...args)
{
    std::allocator<T> alloc;
    T                *ptr = std::allocator_traits<std::allocator<T>>::allocate(alloc, 1);

    try
    {
        std::allocator_traits<std::allocator<T>>::construct(alloc, ptr, std::forward<Args>(args)...);
    }
    catch (...)
    {
        std::allocator_traits<std::allocator<T>>::deallocate(alloc, ptr, 1);
        throw;
    }

    return UniqueObj<T>(ptr);
}

}} // namespace nvcv::detail

#endif // NVCV_DETAIL_UNIQUE_OBJ_HPP
