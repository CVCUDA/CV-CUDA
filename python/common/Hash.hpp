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

#ifndef NVCV_PYTHON_HASH_HPP
#define NVCV_PYTHON_HASH_HPP

#include <nvcv/Size.hpp>

#include <functional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <vector>

namespace nvcvpy::util {

template<class T>
requires(!std::is_enum_v<T> && std::is_default_constructible_v<std::hash<T>>) size_t ComputeHash(const T &a)
{
    return std::hash<T>{}(a);
}

template<class T>
requires(std::is_enum_v<T>) size_t ComputeHash(const T &a)
{
    using Base = typename std::underlying_type<T>::type;

    return std::hash<Base>{}(static_cast<Base>(a));
}

template<std::ranges::range R>
size_t ComputeHash(const R &a);

template<class... TT>
size_t ComputeHash(const std::tuple<TT...> &a);

template<class HEAD, class... TAIL>
requires(sizeof...(TAIL) >= 1) size_t ComputeHash(const HEAD &a, const TAIL &...aa)
{
    return ComputeHash(a) ^ (ComputeHash(aa...) << 1);
}

template<std::ranges::range R>
size_t ComputeHash(const R &a)
{
    size_t hash = ComputeHash(std::ranges::size(a));
    for (const auto &v : a)
    {
        hash = ComputeHash(hash, v);
    }
    return hash;
}

// Hashing for tuples ---------------------
namespace detail {
template<std::size_t... IDX, class T>
size_t ComputeHashTupleHelper(std::index_sequence<IDX...>, const T &a)
{
    return ComputeHash(std::get<IDX>(a)...);
}
} // namespace detail

template<class... TT>
size_t ComputeHash(const std::tuple<TT...> &a)
{
    return detail::ComputeHashTupleHelper(std::index_sequence_for<TT...>(), a);
}

inline size_t ComputeHash()
{
    return ComputeHash(612 /* any value works */);
}

} // namespace nvcvpy::util

namespace nvcv {

inline size_t ComputeHash(const Size2D &s)
{
    using nvcvpy::util::ComputeHash;
    return ComputeHash(s.w, s.h);
}

} // namespace nvcv

#endif // NVCV_PYTHON_HASH_HPP
