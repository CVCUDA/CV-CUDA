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

#ifndef NVCV_CUDA_DETAIL_SATURATE_CAST_IMPL_HPP
#define NVCV_CUDA_DETAIL_SATURATE_CAST_IMPL_HPP

// Internal implementation of range cast functionality.
// Not to be used directly.

#include "MathWrappersImpl.hpp" // for RoundImpl, etc.
#include "Metaprogramming.hpp"  // for TypeTraits, etc.

namespace nvcv::cuda::detail {

// The base saturate cast implementation can be used by host- or device-side calls

template<typename T, typename U>
inline __host__ __device__ T BaseSaturateCastImpl(U u)
{
    constexpr bool SmallToBig = sizeof(U) <= sizeof(T);
    constexpr bool BigToSmall = sizeof(U) > sizeof(T);

    // To silence spurious warnings with gcc-11.1 (-Wunused-but-set-variable)
    (void)SmallToBig;
    (void)BigToSmall;

    if constexpr (std::is_floating_point_v<U> && std::is_integral_v<T>)
    {
        // any-float -> any-integral
        constexpr U minT = static_cast<U>(TypeTraits<T>::min);
        constexpr U maxT = static_cast<U>(TypeTraits<T>::max);

        U out = RoundImpl<U, U>(u);

        out = out < minT ? minT : (out > maxT ? maxT : out);

        return static_cast<T>(out);
    }
    else if constexpr (std::is_integral_v<
                           U> && std::is_signed_v<U> && std::is_integral_v<T> && std::is_unsigned_v<T> && SmallToBig)
    {
        // any-integral-signed -> any-integral-unsigned, small -> big and equal
        return u <= 0 ? 0 : static_cast<T>(u);
    }
    else if constexpr (
        std::is_integral_v<
            U> && std::is_integral_v<T> && ((std::is_signed_v<U> && std::is_signed_v<T>) || (std::is_unsigned_v<U> && std::is_unsigned_v<T>))
        && BigToSmall)
    {
        // any-integral-signed -> any-integral-signed, big -> small
        // any-integral-unsigned -> any-integral-unsigned, big -> small
        return u <= TypeTraits<T>::min ? TypeTraits<T>::min
                                       : (u >= TypeTraits<T>::max ? TypeTraits<T>::max : static_cast<T>(u));
    }
    else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && std::is_integral_v<T> && std::is_signed_v<T>)
    {
        // any-integral-unsigned -> any-integral-signed
        return u >= TypeTraits<T>::max ? TypeTraits<T>::max : static_cast<T>(u);
    }
    else if constexpr (std::is_integral_v<
                           U> && std::is_signed_v<U> && std::is_integral_v<T> && std::is_unsigned_v<T> && BigToSmall)
    {
        // any-integral-signed -> any-integral-unsigned, big -> small
        return u <= static_cast<U>(TypeTraits<T>::min)
                 ? TypeTraits<T>::min
                 : (u >= static_cast<U>(TypeTraits<T>::max) ? TypeTraits<T>::max : static_cast<T>(u));
    }
    else
    {
        // For the cases below, saturate cast reduces to none
        // any-integral-signed -> any-integral-signed, small -> big and equal
        // any-integral-unsigned -> any-integral-unsigned, small -> big and equal
        // any -> any-float
        return u;
    }
}

// Leaf-node saturate cast implementation accepts all-combination types and is always used in host-side calls and
// sometimes used in device-side calls, i.e. when there is no fast specialization to be used instead (see below)

template<typename T, typename U>
inline __host__ __device__ T SaturateCastImpl(U u)
{
    return BaseSaturateCastImpl<T>(u);
}

// Internal-node saturate cast implementations are specialized for each combination of target type and source type
// and uses inline PTX assembly in device-side calls and the base saturate cast implementation in host-side calls

#ifdef __CUDA_ARCH__
#    define NVCV_CUDA_SAT_BODY(TARGET_TYPE, PTX_TYPE, PTX_ASM) \
        PTX_TYPE out = 0;                                      \
        PTX_ASM;                                               \
        return out
#else
#    define NVCV_CUDA_SAT_BODY(TARGET_TYPE, PTX_TYPE, PTX_ASM) return BaseSaturateCastImpl<TARGET_TYPE>(u)
#endif

#define NVCV_CUDA_SAT_DEF(TARGET_TYPE, SOURCE_TYPE, PTX_TYPE, PTX_ASM)                                        \
    template<>                                                                                                \
    __host__ __device__ __forceinline__ TARGET_TYPE SaturateCastImpl<TARGET_TYPE, SOURCE_TYPE>(SOURCE_TYPE u) \
    {                                                                                                         \
        NVCV_CUDA_SAT_BODY(TARGET_TYPE, PTX_TYPE, PTX_ASM);                                                   \
    }

// clang-format off

NVCV_CUDA_SAT_DEF(unsigned char, signed char, unsigned int, asm("cvt.sat.u8.s8 %0, %1;" : "=r"(out) : "r"(static_cast<int>(u))))
NVCV_CUDA_SAT_DEF(unsigned char, short, unsigned int, asm("cvt.sat.u8.s16 %0, %1;" : "=r"(out) : "h"(u)))
NVCV_CUDA_SAT_DEF(unsigned char, unsigned short, unsigned int, asm("cvt.sat.u8.u16 %0, %1;" : "=r"(out) : "h"(u)))
NVCV_CUDA_SAT_DEF(unsigned char, int, unsigned int, asm("cvt.sat.u8.s32 %0, %1;" : "=r"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(unsigned char, unsigned int, unsigned int, asm("cvt.sat.u8.u32 %0, %1;" : "=r"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(unsigned char, float, unsigned int, asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(out) : "f"(u)))
NVCV_CUDA_SAT_DEF(unsigned char, double, unsigned int, asm("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(out) : "d"(u)))

NVCV_CUDA_SAT_DEF(signed char, unsigned char, unsigned int, asm("cvt.sat.s8.u8 %0, %1;" : "=r"(out) : "r"(static_cast<unsigned int>(u))))
NVCV_CUDA_SAT_DEF(signed char, short, unsigned int, asm("cvt.sat.s8.s16 %0, %1;" : "=r"(out) : "h"(u)))
NVCV_CUDA_SAT_DEF(signed char, unsigned short, unsigned int, asm("cvt.sat.s8.u16 %0, %1;" : "=r"(out) : "h"(u)))
NVCV_CUDA_SAT_DEF(signed char, int, unsigned int, asm("cvt.sat.s8.s32 %0, %1;" : "=r"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(signed char, unsigned int, unsigned int, asm("cvt.sat.s8.u32 %0, %1;" : "=r"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(signed char, float, unsigned int, asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(out) : "f"(u)))
NVCV_CUDA_SAT_DEF(signed char, double, unsigned int, asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(out) : "d"(u)))

NVCV_CUDA_SAT_DEF(unsigned short, signed char, unsigned short, asm("cvt.sat.u16.s8 %0, %1;" : "=h"(out) : "r"(static_cast<int>(u))))
NVCV_CUDA_SAT_DEF(unsigned short, short, unsigned short, asm("cvt.sat.u16.s16 %0, %1;" : "=h"(out) : "h"(u)))
NVCV_CUDA_SAT_DEF(unsigned short, int, unsigned short, asm("cvt.sat.u16.s32 %0, %1;" : "=h"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(unsigned short, unsigned int, unsigned short, asm("cvt.sat.u16.u32 %0, %1;" : "=h"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(unsigned short, float, unsigned short, asm("cvt.rni.sat.u16.f32 %0, %1;" : "=h"(out) : "f"(u)))
NVCV_CUDA_SAT_DEF(unsigned short, double, unsigned short, asm("cvt.rni.sat.u16.f64 %0, %1;" : "=h"(out) : "d"(u)))

NVCV_CUDA_SAT_DEF(short, unsigned short, short, asm("cvt.sat.s16.u16 %0, %1;" : "=h"(out) : "h"(u)))
NVCV_CUDA_SAT_DEF(short, int, short, asm("cvt.sat.s16.s32 %0, %1;" : "=h"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(short, unsigned int, short, asm("cvt.sat.s16.u32 %0, %1;" : "=h"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(short, float, short, asm("cvt.rni.sat.s16.f32 %0, %1;" : "=h"(out) : "f"(u)))
NVCV_CUDA_SAT_DEF(short, double, short, asm("cvt.rni.sat.s16.f64 %0, %1;" : "=h"(out) : "d"(u)))

NVCV_CUDA_SAT_DEF(int, unsigned int, int, asm("cvt.sat.s32.u32 %0, %1;" : "=r"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(int, float, int, out = __float2int_rn(u))
NVCV_CUDA_SAT_DEF(int, double, int, out = __double2int_rn(u))

NVCV_CUDA_SAT_DEF(unsigned int, signed char, unsigned int, asm("cvt.sat.u32.s8 %0, %1;" : "=r"(out) : "r"(static_cast<int>(u))))
NVCV_CUDA_SAT_DEF(unsigned int, short, unsigned int, asm("cvt.sat.u32.s16 %0, %1;" : "=r"(out) : "h"(u)))
NVCV_CUDA_SAT_DEF(unsigned int, int, unsigned int, asm("cvt.sat.u32.s32 %0, %1;" : "=r"(out) : "r"(u)))
NVCV_CUDA_SAT_DEF(unsigned int, float, unsigned int, out = __float2uint_rn(u))
NVCV_CUDA_SAT_DEF(unsigned int, double, unsigned int, out = __double2uint_rn(u))

// clang-format on

#undef NVCV_CUDA_SAT_DEF
#undef NVCV_CUDA_SAT_BODY

} // namespace nvcv::cuda::detail

#endif // NVCV_CUDA_DETAIL_SATURATE_CAST_IMPL_HPP
