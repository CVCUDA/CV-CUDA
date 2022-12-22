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

#ifndef NVCV_CUDA_DETAIL_MATH_WRAPPERS_IMPL_HPP
#define NVCV_CUDA_DETAIL_MATH_WRAPPERS_IMPL_HPP

// Internal implementation of math wrapppers functionalities.
// Not to be used directly.

#include <cmath> // for std::round, etc.

namespace nvcv::cuda::detail {

#ifdef __CUDA_ARCH__

template<typename T, typename U>
__device__ __forceinline__ T DeviceRoundImpl(U u)
{
    if constexpr (std::is_same_v<U, float>)
    {
        if constexpr (std::is_same_v<T, float> || sizeof(T) < 4)
        {
            return static_cast<T>(rintf(u));
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            return __float2int_rn(u);
        }
        else if constexpr (std::is_same_v<T, unsigned int>)
        {
            return __float2uint_rn(u);
        }
        else if constexpr (std::is_same_v<T, long int>)
        {
            if constexpr (sizeof(long int) == sizeof(int))
            {
                return __float2int_rn(u);
            }
            else
            {
                return __float2ll_rd(u + 0.5f);
            }
        }
        else if constexpr (std::is_same_v<T, unsigned long int>)
        {
            if constexpr (sizeof(unsigned long int) == sizeof(unsigned int))
            {
                return __float2uint_rn(u);
            }
            else
            {
                return __float2ull_rd(u + 0.5f);
            }
        }
        else if constexpr (std::is_same_v<T, long long int>)
        {
            return __float2ll_rd(u + 0.5f);
        }
        else if constexpr (std::is_same_v<T, unsigned long long int>)
        {
            return __float2ull_rd(u + 0.5f);
        }
    }
    else if constexpr (std::is_same_v<U, double>)
    {
        if constexpr (std::is_same_v<T, double> || sizeof(T) < 4)
        {
            return static_cast<T>(rint(u));
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return __double2float_rn(u);
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            return __double2int_rn(u);
        }
        else if constexpr (std::is_same_v<T, unsigned int>)
        {
            return __double2uint_rn(u);
        }
        else if constexpr (std::is_same_v<T, long int>)
        {
            if constexpr (sizeof(long int) == sizeof(int))
            {
                return __double2int_rn(u);
            }
            else
            {
                return __double2ll_rd(u + 0.5f);
            }
        }
        else if constexpr (std::is_same_v<T, unsigned long int>)
        {
            if constexpr (sizeof(unsigned long int) == sizeof(unsigned int))
            {
                return __double2uint_rn(u);
            }
            else
            {
                return __double2ull_rd(u + 0.5f);
            }
        }
        else if constexpr (std::is_same_v<T, long long int>)
        {
            return __double2ll_rd(u + 0.5f);
        }
        else if constexpr (std::is_same_v<T, unsigned long long int>)
        {
            return __double2ull_rd(u + 0.5f);
        }
    }
    else
    {
        return static_cast<T>(u);
    }
}

template<typename U>
__device__ __forceinline__ U DeviceMinImpl(U a, U b)
{
    if constexpr (std::is_same_v<U, unsigned int>)
    {
        return ::umin(a, b);
    }
    else if constexpr (std::is_same_v<U, unsigned long long int>)
    {
        return ::ullmin(a, b);
    }
    else if constexpr (std::is_same_v<U, long long int>)
    {
        return ::llmin(a, b);
    }
    else
    {
        return ::min(a, b);
    }
}

template<typename U>
__device__ __forceinline__ U DeviceMaxImpl(U a, U b)
{
    if constexpr (std::is_same_v<U, unsigned int>)
    {
        return ::umax(a, b);
    }
    else if constexpr (std::is_same_v<U, unsigned long long int>)
    {
        return ::ullmax(a, b);
    }
    else if constexpr (std::is_same_v<U, long long int>)
    {
        return ::llmax(a, b);
    }
    else
    {
        return ::max(a, b);
    }
}

template<typename U>
__device__ __forceinline__ U DeviceExpImpl(U u)
{
    if constexpr (std::is_same_v<U, float>)
    {
        return __expf(u);
    }
    else if constexpr (sizeof(U) <= 4)
    {
        return static_cast<U>(__expf(static_cast<float>(u)));
    }
    else
    {
        return ::exp(u);
    }
}

template<typename U>
__device__ __forceinline__ U DeviceSqrtImpl(U u)
{
    if constexpr (std::is_same_v<U, float>)
    {
        return __fsqrt_rn(u);
    }
    else if constexpr (std::is_same_v<U, double>)
    {
        return __dsqrt_rn(u);
    }
    else if constexpr (sizeof(U) <= 4)
    {
        return static_cast<U>(__fsqrt_rn(static_cast<float>(u)));
    }
    else
    {
        return static_cast<U>(__dsqrt_rn(static_cast<double>(u)));
    }
}

template<typename U>
__device__ __forceinline__ U DeviceAbsImpl(U u)
{
    if constexpr (std::is_same_v<U, int>)
    {
        return ::abs(u);
    }
    else if constexpr (std::is_same_v<U, long int>)
    {
        return ::labs(u);
    }
    else if constexpr (std::is_same_v<U, long long int>)
    {
        return ::llabs(u);
    }
    else
    {
        return ::abs(u);
    }
}

#endif

template<typename T, typename U>
inline __host__ __device__ T RoundImpl(U u)
{
#ifdef __CUDA_ARCH__
    return DeviceRoundImpl<T, U>(u);
#else
    return std::round(u);
#endif
}

template<typename U>
inline __host__ __device__ U MinImpl(U a, U b)
{
#ifdef __CUDA_ARCH__
    return DeviceMinImpl(a, b);
#else
    return std::min(a, b);
#endif
}

template<typename U>
inline __host__ __device__ U MaxImpl(U a, U b)
{
#ifdef __CUDA_ARCH__
    return DeviceMaxImpl(a, b);
#else
    return std::max(a, b);
#endif
}

template<typename U>
inline __host__ __device__ U ExpImpl(U u)
{
#ifdef __CUDA_ARCH__
    return DeviceExpImpl(u);
#else
    return std::exp(u);
#endif
}

template<typename U>
inline __host__ __device__ U SqrtImpl(U u)
{
#ifdef __CUDA_ARCH__
    return DeviceSqrtImpl(u);
#else
    return std::sqrt(u);
#endif
}

template<typename U>
inline __host__ __device__ U AbsImpl(U u)
{
    if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U>)
    {
        return u;
    }
    else
    {
#ifdef __CUDA_ARCH__
        return DeviceAbsImpl(u);
#else
        return std::abs(u);
#endif
    }
}

} // namespace nvcv::cuda::detail

#endif // NVCV_CUDA_DETAIL_MATH_WRAPPERS_IMPL_HPP
