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

#ifndef NVCV_CUDA_DETAIL_RANGE_CAST_IMPL_HPP
#define NVCV_CUDA_DETAIL_RANGE_CAST_IMPL_HPP

// Internal implementation of range cast functionality.
// Not to be used directly.

#include "MathWrappersImpl.hpp" // for RoundImpl, etc.
#include "Metaprogramming.hpp"  // for TypeTraits, etc.
#include "SaturateCastImpl.hpp" // for BaseSaturateImpl, etc.

namespace nvcv::cuda::detail {

template<typename T, typename U>
inline __host__ __device__ T RangeCastImpl(U u)
{
    if constexpr (std::is_floating_point_v<U> && std::is_floating_point_v<T> && sizeof(U) > sizeof(T))
    {
        // any-float -> any-float, big -> small
        return u <= -TypeTraits<T>::max ? -TypeTraits<T>::max
                                        : (u >= TypeTraits<T>::max ? TypeTraits<T>::max : static_cast<T>(u));
    }
    else if constexpr (std::is_floating_point_v<U> && std::is_integral_v<T> && std::is_signed_v<T>)
    {
        // any-float -> any-integral-signed
        return u >= U{1} ? TypeTraits<T>::max
                         : (u <= U{-1} ? -TypeTraits<T>::max : RoundImpl<T, U>(TypeTraits<T>::max * u));
    }
    else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && std::is_floating_point_v<T>)
    {
        // any-integral-signed -> any-float
        constexpr T invmax = T{1} / TypeTraits<U>::max;

        T out = static_cast<T>(u) * invmax;
        return out < T{-1} ? T{-1} : out;
    }
    else if constexpr (std::is_floating_point_v<U> && std::is_integral_v<T> && std::is_unsigned_v<T>)
    {
        // any-float -> any-integral-unsigned
        return u >= U{1} ? TypeTraits<T>::max : (u <= U{0} ? T{0} : RoundImpl<T, U>(TypeTraits<T>::max * u));
    }
    else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && std::is_floating_point_v<T>)
    {
        // any-integral-unsigned -> any-float
        constexpr T invmax = T{1} / TypeTraits<U>::max;
        return static_cast<T>(u) * invmax;
    }
    else if constexpr (std::is_integral_v<U> && std::is_integral_v<T>)
    {
        // any-integral -> any-integral, range cast reduces to saturate cast
        return BaseSaturateCastImpl<T, U>(u);
    }
    else
    {
        // any-float -> any-float, small -> big and equal, range cast reduces to none
        return u;
    }
}

} // namespace nvcv::cuda::detail

#endif // NVCV_CUDA_DETAIL_RANGE_CAST_IMPL_HPP
