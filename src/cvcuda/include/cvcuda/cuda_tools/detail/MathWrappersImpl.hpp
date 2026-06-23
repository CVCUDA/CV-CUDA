/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cfenv> // for FE_TONEAREST, etc.
#include <cmath> // for std::round, etc.

namespace nvcv::cuda::detail {

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

template<typename T, typename U, int RM = FE_TONEAREST>
__device__ __forceinline__ T DeviceRoundImpl(U u)
{
    if constexpr (std::is_same_v<U, float>)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            if constexpr (RM == FE_TONEAREST)
                return rintf(u);
            else if constexpr (RM == FE_DOWNWARD)
                return floorf(u);
            else if constexpr (RM == FE_UPWARD)
                return ceilf(u);
            else if constexpr (RM == FE_TOWARDZERO)
                return truncf(u);
        }
        else if constexpr (std::is_same_v<T, int> || (sizeof(T) < 4 && std::is_integral_v<T> && std::is_signed_v<T>))
        {
            if constexpr (RM == FE_TONEAREST)
                return static_cast<T>(__float2int_rn(u));
            else if constexpr (RM == FE_DOWNWARD)
                return static_cast<T>(__float2int_rd(u));
            else if constexpr (RM == FE_UPWARD)
                return static_cast<T>(__float2int_ru(u));
            else if constexpr (RM == FE_TOWARDZERO)
                return static_cast<T>(__float2int_rz(u));
        }
        else if constexpr (std::is_same_v<
                               T, unsigned int> || (sizeof(T) < 4 && std::is_integral_v<T> && std::is_unsigned_v<T>))
        {
            if constexpr (RM == FE_TONEAREST)
                return static_cast<T>(__float2uint_rn(u));
            else if constexpr (RM == FE_DOWNWARD)
                return static_cast<T>(__float2uint_rd(u));
            else if constexpr (RM == FE_UPWARD)
                return static_cast<T>(__float2uint_ru(u));
            else if constexpr (RM == FE_TOWARDZERO)
                return static_cast<T>(__float2uint_rz(u));
        }
        else if constexpr (std::is_same_v<T, long int>)
        {
            if constexpr (sizeof(long int) == sizeof(int))
            {
                if constexpr (RM == FE_TONEAREST)
                    return __float2int_rn(u);
                else if constexpr (RM == FE_DOWNWARD)
                    return __float2int_rd(u);
                else if constexpr (RM == FE_UPWARD)
                    return __float2int_ru(u);
                else if constexpr (RM == FE_TOWARDZERO)
                    return __float2int_rz(u);
            }
            else
            {
                if constexpr (RM == FE_TONEAREST)
                    return __float2ll_rn(u);
                else if constexpr (RM == FE_DOWNWARD)
                    return __float2ll_rd(u);
                else if constexpr (RM == FE_UPWARD)
                    return __float2ll_ru(u);
                else if constexpr (RM == FE_TOWARDZERO)
                    return __float2ll_rz(u);
            }
        }
        else if constexpr (std::is_same_v<T, unsigned long int>)
        {
            if constexpr (sizeof(unsigned long int) == sizeof(unsigned int))
            {
                if constexpr (RM == FE_TONEAREST)
                    return __float2uint_rn(u);
                else if constexpr (RM == FE_DOWNWARD)
                    return __float2uint_rd(u);
                else if constexpr (RM == FE_UPWARD)
                    return __float2uint_ru(u);
                else if constexpr (RM == FE_TOWARDZERO)
                    return __float2uint_rz(u);
            }
            else
            {
                if constexpr (RM == FE_TONEAREST)
                    return __float2ull_rn(u);
                else if constexpr (RM == FE_DOWNWARD)
                    return __float2ull_rd(u);
                else if constexpr (RM == FE_UPWARD)
                    return __float2ull_ru(u);
                else if constexpr (RM == FE_TOWARDZERO)
                    return __float2ull_rz(u);
            }
        }
        else if constexpr (std::is_same_v<T, long long int>)
        {
            if constexpr (RM == FE_TONEAREST)
                return __float2ll_rn(u);
            else if constexpr (RM == FE_DOWNWARD)
                return __float2ll_rd(u);
            else if constexpr (RM == FE_UPWARD)
                return __float2ll_ru(u);
            else if constexpr (RM == FE_TOWARDZERO)
                return __float2ll_rz(u);
        }
        else if constexpr (std::is_same_v<T, unsigned long long int>)
        {
            if constexpr (RM == FE_TONEAREST)
                return __float2ull_rn(u);
            else if constexpr (RM == FE_DOWNWARD)
                return __float2ull_rd(u);
            else if constexpr (RM == FE_UPWARD)
                return __float2ull_ru(u);
            else if constexpr (RM == FE_TOWARDZERO)
                return __float2ull_rz(u);
        }
        else
        {
            assert(false && "Undefined round types");
        }
    }
    else if constexpr (std::is_same_v<U, double>)
    {
        if constexpr (std::is_same_v<T, double>)
        {
            if constexpr (RM == FE_TONEAREST)
                return rint(u);
            else if constexpr (RM == FE_DOWNWARD)
                return floor(u);
            else if constexpr (RM == FE_UPWARD)
                return ceil(u);
            else if constexpr (RM == FE_TOWARDZERO)
                return trunc(u);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            if constexpr (RM == FE_TONEAREST)
                return __double2float_rn(u);
            else if constexpr (RM == FE_DOWNWARD)
                return __double2float_rd(u);
            else if constexpr (RM == FE_UPWARD)
                return __double2float_ru(u);
            else if constexpr (RM == FE_TOWARDZERO)
                return __double2float_rz(u);
        }
        else if constexpr (std::is_same_v<T, int> || (sizeof(T) < 4 && std::is_integral_v<T> && std::is_signed_v<T>))
        {
            if constexpr (RM == FE_TONEAREST)
                return static_cast<T>(__double2int_rn(u));
            else if constexpr (RM == FE_DOWNWARD)
                return static_cast<T>(__double2int_rd(u));
            else if constexpr (RM == FE_UPWARD)
                return static_cast<T>(__double2int_ru(u));
            else if constexpr (RM == FE_TOWARDZERO)
                return static_cast<T>(__double2int_rz(u));
        }
        else if constexpr (std::is_same_v<
                               T, unsigned int> || (sizeof(T) < 4 && std::is_integral_v<T> && std::is_unsigned_v<T>))
        {
            if constexpr (RM == FE_TONEAREST)
                return static_cast<T>(__double2uint_rn(u));
            else if constexpr (RM == FE_DOWNWARD)
                return static_cast<T>(__double2uint_rd(u));
            else if constexpr (RM == FE_UPWARD)
                return static_cast<T>(__double2uint_ru(u));
            else if constexpr (RM == FE_TOWARDZERO)
                return static_cast<T>(__double2uint_rz(u));
        }
        else if constexpr (std::is_same_v<T, long int>)
        {
            if constexpr (sizeof(long int) == sizeof(int))
            {
                if constexpr (RM == FE_TONEAREST)
                    return __double2int_rn(u);
                else if constexpr (RM == FE_DOWNWARD)
                    return __double2int_rd(u);
                else if constexpr (RM == FE_UPWARD)
                    return __double2int_ru(u);
                else if constexpr (RM == FE_TOWARDZERO)
                    return __double2int_rz(u);
            }
            else
            {
                if constexpr (RM == FE_TONEAREST)
                    return __double2ll_rn(u);
                else if constexpr (RM == FE_DOWNWARD)
                    return __double2ll_rd(u);
                else if constexpr (RM == FE_UPWARD)
                    return __double2ll_ru(u);
                else if constexpr (RM == FE_TOWARDZERO)
                    return __double2ll_rz(u);
            }
        }
        else if constexpr (std::is_same_v<T, unsigned long int>)
        {
            if constexpr (sizeof(unsigned long int) == sizeof(unsigned int))
            {
                if constexpr (RM == FE_TONEAREST)
                    return __double2uint_rn(u);
                else if constexpr (RM == FE_DOWNWARD)
                    return __double2uint_rd(u);
                else if constexpr (RM == FE_UPWARD)
                    return __double2uint_ru(u);
                else if constexpr (RM == FE_TOWARDZERO)
                    return __double2uint_rz(u);
            }
            else
            {
                if constexpr (RM == FE_TONEAREST)
                    return __double2ull_rn(u);
                else if constexpr (RM == FE_DOWNWARD)
                    return __double2ull_rd(u);
                else if constexpr (RM == FE_UPWARD)
                    return __double2ull_ru(u);
                else if constexpr (RM == FE_TOWARDZERO)
                    return __double2ull_rz(u);
            }
        }
        else if constexpr (std::is_same_v<T, long long int>)
        {
            if constexpr (RM == FE_TONEAREST)
                return __double2ll_rn(u);
            else if constexpr (RM == FE_DOWNWARD)
                return __double2ll_rd(u);
            else if constexpr (RM == FE_UPWARD)
                return __double2ll_ru(u);
            else if constexpr (RM == FE_TOWARDZERO)
                return __double2ll_rz(u);
        }
        else if constexpr (std::is_same_v<T, unsigned long long int>)
        {
            if constexpr (RM == FE_TONEAREST)
                return __double2ull_rn(u);
            else if constexpr (RM == FE_DOWNWARD)
                return __double2ull_rd(u);
            else if constexpr (RM == FE_UPWARD)
                return __double2ull_ru(u);
            else if constexpr (RM == FE_TOWARDZERO)
                return __double2ull_rz(u);
        }
        else
        {
            assert(false && "Undefined round types");
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
#if defined(__HIP_DEVICE_COMPILE__)
    // HIP's global namespace has no typed umin/ullmin/llmin; the ternary lowers to
    // the same integer-min instruction the CUDA builtins emit. The comparison is
    // spelled exactly like the host std::min fallback (b < a ? b : a) so the device
    // result is identical to the host reference for every input, including the
    // NaN / signed-zero cases the morphology tests exercise with raw-byte floats.
    return b < a ? b : a;
#else
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
#endif
}

template<typename U>
__device__ __forceinline__ U DeviceMaxImpl(U a, U b)
{
#if defined(__HIP_DEVICE_COMPILE__)
    // Spelled exactly like the host std::max fallback (a < b ? b : a) so the device
    // max matches the host reference exactly on NaN / signed-zero inputs.
    return a < b ? b : a;
#else
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
#endif
}

template<typename U, typename S>
__device__ __forceinline__ U DevicePowImpl(U x, S y)
{
    if constexpr ((std::is_same_v<U, float> && std::is_same_v<S, float>) || (std::is_same_v<U, float> && sizeof(S) <= 4)
                  || (sizeof(U) <= 4 && sizeof(S) <= 4))
    {
        if (x >= 0.f)
        {
            return __powf(x, y);
        }
        else
        {
            return powf(x, y);
        }
    }
    else
    {
        return ::pow(x, y);
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
#if defined(__HIP_DEVICE_COMPILE__)
    // gfx90a's single-precision __fsqrt_rn is not always correctly rounded (it can
    // be 1 ULP off, e.g. sqrt(93606.0f)), whereas CUDA's sqrt.rn.f32 and the host
    // std::sqrt are correctly rounded. Route 32-bit sqrt through the correctly
    // rounded f64 sqrt and round once to keep results identical to the CUDA
    // build and the CPU references (CDNA has fast f64 sqrt).
    if constexpr (std::is_same_v<U, double>)
    {
        return __dsqrt_rn(u);
    }
    else
    {
        return static_cast<U>(__dsqrt_rn(static_cast<double>(u)));
    }
#else
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
#endif
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

// CUDA does round-to-nearest-even, C/C++ functions that do the same are roundeven*, however they are available
// only in recent C23 which may not be readily available, therefore we need our own implementation of roundeven
template<typename U>
inline __host__ U RoundEvenImpl(U u)
{
    U rounded = std::round(u);
    if (std::abs(rounded - u) == U(0.5))
    {
        if (static_cast<int64_t>(rounded) & 1)
        {
            rounded -= std::copysign(U(1.0), u);
        }
    }
    return rounded;
}

template<typename T, typename U, int RM = FE_TONEAREST>
inline __host__ __device__ T RoundImpl(U u)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return DeviceRoundImpl<T, U, RM>(u);
#else
    // In host we use C++ to do round depending on round mode by selecting at compile time the correct function:
    // round is to nearest; floor is downward; ceil is upward; and trunc is towards zero.
    if constexpr (RM == FE_TONEAREST)
    {
        return RoundEvenImpl(u);
    }
    else if constexpr (RM == FE_DOWNWARD)
    {
        return std::floor(u);
    }
    else if constexpr (RM == FE_UPWARD)
    {
        return std::ceil(u);
    }
    else if constexpr (RM == FE_TOWARDZERO)
    {
        return std::trunc(u);
    }
#endif
}

template<typename U>
inline __host__ __device__ U MinImpl(U a, U b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return DeviceMinImpl(a, b);
#else
    return std::min(a, b);
#endif
}

template<typename U>
inline __host__ __device__ U MaxImpl(U a, U b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return DeviceMaxImpl(a, b);
#else
    return std::max(a, b);
#endif
}

template<typename U, typename S>
inline __host__ __device__ U PowImpl(U x, S y)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return DevicePowImpl(x, y);
#else
    return std::pow(x, y);
#endif
}

template<typename U>
inline __host__ __device__ U ExpImpl(U u)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return DeviceExpImpl(u);
#else
    return std::exp(u);
#endif
}

template<typename U>
inline __host__ __device__ U SqrtImpl(U u)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
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
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return DeviceAbsImpl(u);
#else
        return std::abs(u);
#endif
    }
}

template<typename U, typename S>
inline __host__ __device__ U ClampImpl(U u, S lo, S hi)
{
    return u <= lo ? lo : (u >= hi ? hi : u);
}

} // namespace nvcv::cuda::detail

#endif // NVCV_CUDA_DETAIL_MATH_WRAPPERS_IMPL_HPP
