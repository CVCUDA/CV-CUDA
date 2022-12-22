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

/**
 * @file MathWrappers.hpp
 *
 * @brief Defines math wrappers over CUDA intrinsics/functions in device and standard C++ counterparts in host.
 */

#ifndef NVCV_CUDA_MATH_WRAPPERS_HPP
#define NVCV_CUDA_MATH_WRAPPERS_HPP

#include "TypeTraits.hpp"              // for Require, etc.
#include "detail/MathWrappersImpl.hpp" // for MathWrappersImpl, etc.

namespace nvcv::cuda {

namespace detail {

template<typename T, typename U, typename RT>
inline __host__ __device__ RT RoundImpl(U u)
{
    RT out{};

#pragma unroll
    for (int e = 0; e < nvcv::cuda::NumElements<RT>; ++e)
    {
        GetElement(out, e) = RoundImpl<T, BaseType<U>>(GetElement(u, e));
    }

    return out;
}

} // namespace detail

/**
 * @defgroup NVCV_CPP_CUDATOOLS_MATHWRAPPERS Math wrappers
 * @{
 */

/**
 * @brief Metafunction to round all elements of the input
 *
 * @details This function rounds all elements of the input and returns the result with the same type as the input.
 * Optionally, the base type of the result may be specified by the template argument type \p T.  For instance, a
 * float4 can have its 4 elements rounded into a float4 result, or to a different result type, such as T=int, where
 * the result will be int4 with the rounded results (see example below).  It is a requirement of round that the
 * input source type has type traits and the optional result type \p T is a regular C type.
 *
 * @code
 * using FloatType = MakeType<float, 4>;
 * FloatType res = ...;
 * FloatType float_rounded = round(res);
 * ConvertBaseTypeTo<int, FloatType> int_rounded = round<int>(res);
 * @endcode
 *
 * @tparam U Type of the source value (with 1 to 4 elements) passed as argument
 * @tparam T Optional type that defines the result of the round
 *
 * @param[in] u Source value to round all elements with its same type or \p T
 *
 * @return The value with all elements rounded
 */
template<typename T, typename U, typename RT = ConvertBaseTypeTo<T, U>,
         class = Require<HasTypeTraits<T, U> && !IsCompound<T>>>
inline __host__ __device__ std::enable_if_t<!std::is_same_v<T, U>, RT> round(U u)
{
    return detail::RoundImpl<T, U, RT>(u);
}

template<typename U>
inline __host__ __device__ U round(U u)
{
    return detail::RoundImpl<BaseType<U>, U, U>(u);
}

#define NVCV_CUDA_BINARY_SIMD(TYPE_U, INTRINSIC)                  \
    constexpr(std::is_same_v<U, TYPE_U>)                          \
    {                                                             \
        unsigned int r_a = *reinterpret_cast<unsigned int *>(&a); \
        unsigned int r_b = *reinterpret_cast<unsigned int *>(&b); \
        unsigned int ret = INTRINSIC(r_a, r_b);                   \
        return *reinterpret_cast<TYPE_U *>(&ret);                 \
    }

/**
 * @brief Metafunction to compute the minimum of two inputs per element
 *
 * @details This function finds the minimum of two inputs per element and returns the result with the same type as
 * the input.  For instance, two int4 inputs {1, 2, 3, 4} and {4, 3, 2, 1} yield the minimum {1, 2, 2, 1} as int4
 * as well (see example below).  It is a requirement of min that the input source type has type traits.
 *
 * @code
 * using IntType = MakeType<int, 4>;
 * IntType a = {1, 2, 3, 4}, b = {4, 3, 2, 1};
 * IntType ab_min = min(a, b); // = {1, 2, 2, 1}
 * @endcode
 *
 * @tparam U Type of the two source arguments and the return type
 *
 * @param[in] u Input value to compute \f$ min(x_a, x_b) \f$ where \f$ x_a \f$ (\f$ x_b \f$) is each element of
 *              \f$ a \f$ (\f$ b \f$)
 *
 * @return The return value with one minimum per element
 */
template<typename U, class = Require<HasTypeTraits<U>>>
inline __host__ __device__ U min(U a, U b)
{
    // clang-format off
#ifdef __CUDA_ARCH__
    if NVCV_CUDA_BINARY_SIMD (short2, __vmins2)
    else if NVCV_CUDA_BINARY_SIMD (char4, __vmins4)
    else if NVCV_CUDA_BINARY_SIMD (ushort2, __vminu2)
    else if NVCV_CUDA_BINARY_SIMD (uchar4, __vminu4)
    else
#endif
    {
        U out{};
#pragma unroll
        for (int e = 0; e < nvcv::cuda::NumElements<U>; ++e)
        {
            GetElement(out, e) = detail::MinImpl(GetElement(a, e), GetElement(b, e));
        }
        return out;
    }
    // clang-format on
}

/**
 * @brief Metafunction to compute the maximum of two inputs per element
 *
 * @details This function finds the maximum of two inputs per element and returns the result with the same type as
 * the input.  For instance, two int4 inputs {1, 2, 3, 4} and {4, 3, 2, 1} yield the maximum {4, 3, 3, 4} as int4
 * as well (see example below).  It is a requirement of max that the input source type has type traits.
 *
 * @code
 * using IntType = MakeType<int, 4>;
 * IntType a = {1, 2, 3, 4}, b = {4, 3, 2, 1};
 * IntType ab_max = max(a, b); // = {4, 3, 3, 4}
 * @endcode
 *
 * @tparam U Type of the two source arguments and the return type
 *
 * @param[in] u Input value to compute \f$ max(x_a, x_b) \f$ where \f$ x_a \f$ (\f$ x_b \f$) is each element of
 *              \f$ a \f$ (\f$ b \f$)
 *
 * @return The return value with maximums per element
 */
template<typename U, class = Require<HasTypeTraits<U>>>
inline __host__ __device__ U max(U a, U b)
{
    // clang-format off
#ifdef __CUDA_ARCH__
    if NVCV_CUDA_BINARY_SIMD (short2, __vmaxs2)
    else if NVCV_CUDA_BINARY_SIMD (char4, __vmaxs4)
    else if NVCV_CUDA_BINARY_SIMD (ushort2, __vmaxu2)
    else if NVCV_CUDA_BINARY_SIMD (uchar4, __vmaxu4)
    else
#endif
    {
        U out{};
#pragma unroll
        for (int e = 0; e < nvcv::cuda::NumElements<U>; ++e)
        {
            GetElement(out, e) = detail::MaxImpl(GetElement(a, e), GetElement(b, e));
        }
        return out;
    }
    // clang-format on
}

#undef NVCV_CUDA_BINARY_SIMD

/**
 * @brief Metafunction to compute the natural (base e) exponential of all elements of the input
 *
 * @details This function computes the natural (base e) exponential of all elements of the input and returns the
 * result with the same type as the input.  It is a requirement of exp that the input source type has type traits.
 *
 * @tparam U Type of the source argument and the return type
 *
 * @param[in] u Input value to compute \f$ e^x \f$ where \f$ x \f$ is each element of \f$ u \f$
 *
 * @return The return value with all elements as the result of the natural (base e) exponential
 */
template<typename U, class = Require<HasTypeTraits<U>>>
inline __host__ __device__ U exp(U u)
{
    U out{};

#pragma unroll
    for (int e = 0; e < nvcv::cuda::NumElements<U>; ++e)
    {
        GetElement(out, e) = detail::ExpImpl(GetElement(u, e));
    }

    return out;
}

/**
 * @brief Metafunction to compute the square root of all elements of the input
 *
 * @details This function computes the square root of all elements of the input and returns the result with the
 * same type as the input.  It is a requirement of sqrt that the input source type has type traits.
 *
 * @tparam U Type of the source argument and the return type
 *
 * @param[in] u Input value to compute \f$ \sqrt{x} \f$ where \f$ x \f$ is each element of \f$ u \f$
 *
 * @return The return value with all elements as the result of the square root
 */
template<typename U, class = Require<HasTypeTraits<U>>>
inline __host__ __device__ U sqrt(U u)
{
    U out{};

#pragma unroll
    for (int e = 0; e < nvcv::cuda::NumElements<U>; ++e)
    {
        GetElement(out, e) = detail::SqrtImpl(GetElement(u, e));
    }

    return out;
}

#define NVCV_CUDA_UNARY_SIMD(TYPE_U, INTRINSIC)                   \
    constexpr(std::is_same_v<U, TYPE_U>)                          \
    {                                                             \
        unsigned int r_u = *reinterpret_cast<unsigned int *>(&u); \
        unsigned int ret = INTRINSIC(r_u);                        \
        return *reinterpret_cast<TYPE_U *>(&ret);                 \
    }

/**
 * @brief Metafunction to compute the absolute value of all elements of the input
 *
 * @details This function computes the absolute value of all elements of the input and returns the result with the
 * same type as the input.  For instance, an int4 input {-1, 2, -3, 4} yields the absolute {1, 2, 3, 4} as int4 as
 * well (see example below).  It is a requirement of abs that the input source type has type traits.
 *
 * @code
 * using IntType = MakeType<int, 4>;
 * IntType a = {-1, 2, -3, 4};
 * IntType a_abs = abs(a); // = {1, 2, 3, 4}
 * @endcode
 *
 * @tparam U Type of the source argument and the return type
 *
 * @param[in] u Input value to compute \f$ |x| \f$ where \f$ x \f$ is each element of \f$ u \f$
 *
 * @return The return value with the absolute of all elements
 */
template<typename U, class = Require<HasTypeTraits<U>>>
inline __host__ __device__ U abs(U u)
{
    // clang-format off
#ifdef __CUDA_ARCH__
    if NVCV_CUDA_UNARY_SIMD (short2, __vabsss2)
    else if NVCV_CUDA_UNARY_SIMD (char4, __vabsss4)
    else
#endif
    {
        U out{};
#pragma unroll
        for (int e = 0; e < nvcv::cuda::NumElements<U>; ++e)
        {
            GetElement(out, e) = detail::AbsImpl(GetElement(u, e));
        }
        return out;
    }
    // clang-format on
}

#undef NVCV_CUDA_UNARY_SIMD

/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_MATH_WRAPPERS_HPP
