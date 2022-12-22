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

#ifndef NVCV_UTIL_MATH_HPP
#define NVCV_UTIL_MATH_HPP

#include "Compiler.hpp"
#include "Metaprogramming.hpp"

#include <cassert>
#include <type_traits>

namespace nvcv::util {

template<class T, class U, class = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
NVCV_CUDA_HOST_DEVICE constexpr T RoundUp(T value, U multiple)
{
    return (value + multiple - 1) / multiple * multiple;
}

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
NVCV_CUDA_HOST_DEVICE constexpr bool IsPowerOfTwo(T value)
{
    return (value & (value - 1)) == 0;
}

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
NVCV_CUDA_HOST_DEVICE constexpr auto RoundUpNextPowerOfTwo(T x)
{
    assert(x >= 0);

    // Source: Hacker's Delight 1st ed, p.48,
    // adapted for any integer size.

    x = x - 1;
    // all constants, compiler can unroll it
    for (size_t i = 1; i < sizeof(T) * 8; i <<= 1)
    {
        x = x | (x >> i);
    }

    if constexpr (std::is_same_v<T, int32_t>)
    {
        return (int64_t)x + 1;
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        return (uint64_t)x + 1;
    }
    else
    {
        return x + 1;
    }
}

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
NVCV_CUDA_HOST_DEVICE constexpr auto DivUp(T num, TypeIdentity<T> den)
{
    assert(num >= 0);
    assert(den > 0);

    return (num + (den - 1)) / den;
}

template<class T, class U, class = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
NVCV_CUDA_HOST_DEVICE constexpr auto RoundUpPowerOfTwo(T value, U multiple)
{
    assert(value >= 0);
    assert(multiple >= 0);

    assert(IsPowerOfTwo(multiple));

    // Source: Hacker's Delight 1st ed, p.45,

    return (value + (multiple - 1)) & -multiple;
}

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
constexpr int ILog2(T value)
{
    assert(value > 0);

    if constexpr (sizeof(T) <= sizeof(unsigned))
    {
        return sizeof(unsigned) * 8 - __builtin_clz(value) - 1;
    }
    else if constexpr (sizeof(T) <= sizeof(unsigned long))
    {
        return sizeof(unsigned long) * 8 - __builtin_clzl(value) - 1;
    }
    else if constexpr (sizeof(T) <= sizeof(unsigned long long))
    {
        return sizeof(unsigned long long) * 8 - __builtin_clzll(value) - 1;
    }
    else
    {
        static_assert(sizeof(T) == 0, "Type too large");
        return 0;
    }
}

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
NVCV_CUDA_HOST_DEVICE constexpr auto DivUpPowerOfTwo(T num, TypeIdentity<T> den)
{
    assert(num >= 0);
    assert(den > 0);
    assert(IsPowerOfTwo(den));

    return (num >> ILog2(den)) + !!(num & (den - 1));
}

} // namespace nvcv::util

#endif // NVCV_UTIL_MATH_HPP
