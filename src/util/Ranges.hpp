/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_UTIL_RANGES_HPP
#define NVCV_UTIL_RANGES_HPP

#include <iterator>
#include <type_traits>

namespace nvcv::util::ranges {

namespace detail {

void begin(...);

template<class R, int N>
R *begin(R (&&r)[N]);

template<class R, int N>
R *end(R (&&r)[N]);

template<class T>
constexpr bool HasBegin()
{
    using detail::begin;
    using std::begin;
    return !std::is_same_v<decltype(begin(std::declval<T>())), void>;
}

void end(...);

template<class T>
constexpr bool HasEnd()
{
    using detail::end;
    using std::end;
    return !std::is_same_v<decltype(end(std::declval<T>())), void>;
}

} // namespace detail

template<class T>
constexpr bool IsRange = detail::HasBegin<T>() && detail::HasEnd<T>();

template<class R, int N>
auto Begin(R (&r)[N])
{
    return r;
}

template<class R, int N>
auto End(R (&r)[N])
{
    return r + N;
}

template<class R>
auto Begin(R &&r)
{
    using std::begin;
    return begin(r);
}

template<class R>
auto End(R &&r)
{
    using std::end;
    return end(r);
}

template<class R>
auto Data(R &&r)
{
    return &*Begin(r);
}

template<class R>
auto Size(const R &r)
{
    using std::distance;
    return distance(Begin(r), End(r));
}

template<class T>
using RangeValue = std::remove_reference_t<decltype(*Begin(std::declval<T>()))>;

namespace detail {
template<class T>
constexpr bool IsRandomAccessRange()
{
    if constexpr (IsRange<T>)
    {
        return std::is_same_v<typename std::iterator_traits<
                                  std::remove_reference_t<decltype(Begin(std::declval<T>()))>>::iterator_category,
                              std::random_access_iterator_tag>;
    }
    else
    {
        return false;
    }
}
} // namespace detail

template<class T>
constexpr bool IsRandomAccessRange = detail::IsRandomAccessRange<T>();

} // namespace nvcv::util::ranges

#endif // NVCV_UTIL_RANGES_HPP
