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

/**
 * @file MathOps.hpp
 *
 * @brief Defines math operations.
 */

#ifndef NVCV_CUDA_MATH_OPS_HPP
#define NVCV_CUDA_MATH_OPS_HPP

#include "StaticCast.hpp" // for StaticCast, etc.
#include "TypeTraits.hpp" // for Require, etc.

#include <utility> // for std::declval, etc.

namespace nvcv::cuda::detail {

// clang-format off

// Metavariable to check if two types are compound and have the same number of components.
template<class T, class U, class = Require<HasTypeTraits<T, U>>>
constexpr bool IsSameCompound = IsCompound<T> && TypeTraits<T>::components == TypeTraits<U>::components;

// Metavariable to check that at least one type is of compound type out of two types.
// If both are compound type, then it is checked that both have the same number of components.
template<typename T, typename U, class = Require<HasTypeTraits<T, U>>>
constexpr bool OneIsCompound =
    (TypeTraits<T>::components == 0 && TypeTraits<U>::components >= 1) ||
    (TypeTraits<T>::components >= 1 && TypeTraits<U>::components == 0) ||
    IsSameCompound<T, U>;

// Metavariable to check if a type is of integral type.
template<typename T, class = Require<HasTypeTraits<T>>>
constexpr bool IsIntegral = std::is_integral_v<typename TypeTraits<T>::base_type>;

// Metavariable to require that at least one type is of compound type out of two integral types.
// If both are compound type, then it is required that both have the same number of components.
template<typename T, typename U, class = Require<HasTypeTraits<T, U>>>
constexpr bool OneIsCompoundAndBothAreIntegral = OneIsCompound<T, U> && IsIntegral<T> && IsIntegral<U>;

// Metavariable to require that a type is a CUDA compound of integral type.
template<typename T, class = Require<HasTypeTraits<T>>>
constexpr bool IsIntegralCompound = IsIntegral<T> && IsCompound<T>;

// clang-format on

} // namespace nvcv::cuda::detail

/**
 * Operators on CUDA compound types resembling the same operator on corresponding regular C type.
 *
 * This whole group defines a set of arithmetic and bitwise operators defined on CUDA compound types.
 * They work the same way as the corresponding regular C type.  For instance, three int3 a, b and c, will accept
 * the operation a += b * c (see example below).  Furthermore, the operators accept mixed operands as CUDA compound
 * and regular C types, e.g. two int3 a and b and one int c will accept the operation a += b * c, where the scalar
 * c propagates its value for all components of b in the multiplication and the int3 result in the assignment to a.
 *
 * @defgroup NVCV_CPP_CUDATOOLS_MATHOPERATORS Math operators
 * @{
 *
 * @code
 * using DataType = ...;
 * DataType pix = ...;
 * float kernel = ...;
 * ConvertBaseTypeTo<float, DataType> res = {0};
 * res += kernel * pix;
 * @endcode
 *
 * @tparam T Type of the first CUDA compound or regular C type operand.
 * @tparam U Type of the second CUDA compound or regular C type operand.
 *
 * @param[in] a First operand.
 * @param[in] b Second operand.
 *
 * @return Return value of applying the operator on \p a and \p b.
 */

#define NVCV_CUDA_UNARY_OPERATOR(OPERATOR, REQUIREMENT)                                                          \
    template<typename T, class = nvcv::cuda::Require<REQUIREMENT<T>>>                                            \
    inline __host__ __device__ auto operator OPERATOR(T a)                                                       \
    {                                                                                                            \
        using RT = nvcv::cuda::ConvertBaseTypeTo<decltype(OPERATOR std::declval<nvcv::cuda::BaseType<T>>()), T>; \
        if constexpr (nvcv::cuda::NumElements<RT> == 1)                                                          \
            return RT{OPERATOR a.x};                                                                             \
        else if constexpr (nvcv::cuda::NumElements<RT> == 2)                                                     \
            return RT{OPERATOR a.x, OPERATOR a.y};                                                               \
        else if constexpr (nvcv::cuda::NumElements<RT> == 3)                                                     \
            return RT{OPERATOR a.x, OPERATOR a.y, OPERATOR a.z};                                                 \
        else if constexpr (nvcv::cuda::NumElements<RT> == 4)                                                     \
            return RT{OPERATOR a.x, OPERATOR a.y, OPERATOR a.z, OPERATOR a.w};                                   \
    }

NVCV_CUDA_UNARY_OPERATOR(-, nvcv::cuda::IsCompound)
NVCV_CUDA_UNARY_OPERATOR(+, nvcv::cuda::IsCompound)
NVCV_CUDA_UNARY_OPERATOR(~, nvcv::cuda::detail::IsIntegralCompound)

#undef NVCV_CUDA_UNARY_OPERATOR

// The element-wise body is shared between the generic operator and, on HIP, a
// more-specialized compound-compound overload (see below). HIP's vector types
// (HIP_vector_type) ship their own operator+(vec<T,n>, U) and operator+(U,
// vec<T,n>); for a mixed pair like float3 + uchar3 both bind and they are
// equally specialized, so the call is ambiguous (a HIP header limitation).
// CV-CUDA's generic operator(T, U) is less specialized than HIP's and loses
// partial ordering, so on HIP we also emit operator(vec<T1,n>, vec<T2,n>),
// which is more specialized than HIP's pair and wins, while same-type vec+vec
// still resolves to HIP's own (correct) operator and vec+scalar is unchanged.
#define NVCV_CUDA_BINARY_OPERATOR_BODY(OPERATOR)                                                                \
    {                                                                                                           \
        using RT = nvcv::cuda::MakeType<                                                                        \
            decltype(std::declval<nvcv::cuda::BaseType<T>>() OPERATOR std::declval<nvcv::cuda::BaseType<U>>()), \
            nvcv::cuda::NumComponents<T> == 0 ? nvcv::cuda::NumComponents<U> : nvcv::cuda::NumComponents<T>>;   \
        if constexpr (nvcv::cuda::NumComponents<T> == 0)                                                        \
        {                                                                                                       \
            if constexpr (nvcv::cuda::NumElements<RT> == 1)                                                     \
                return RT{a OPERATOR b.x};                                                                      \
            else if constexpr (nvcv::cuda::NumElements<RT> == 2)                                                \
                return RT{a OPERATOR b.x, a OPERATOR b.y};                                                      \
            else if constexpr (nvcv::cuda::NumElements<RT> == 3)                                                \
                return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z};                                      \
            else if constexpr (nvcv::cuda::NumElements<RT> == 4)                                                \
                return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z, a OPERATOR b.w};                      \
        }                                                                                                       \
        else if constexpr (nvcv::cuda::NumComponents<U> == 0)                                                   \
        {                                                                                                       \
            if constexpr (nvcv::cuda::NumElements<RT> == 1)                                                     \
                return RT{a.x OPERATOR b};                                                                      \
            else if constexpr (nvcv::cuda::NumElements<RT> == 2)                                                \
                return RT{a.x OPERATOR b, a.y OPERATOR b};                                                      \
            else if constexpr (nvcv::cuda::NumElements<RT> == 3)                                                \
                return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b};                                      \
            else if constexpr (nvcv::cuda::NumElements<RT> == 4)                                                \
                return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b, a.w OPERATOR b};                      \
        }                                                                                                       \
        else                                                                                                    \
        {                                                                                                       \
            if constexpr (nvcv::cuda::NumElements<RT> == 1)                                                     \
                return RT{a.x OPERATOR b.x};                                                                    \
            else if constexpr (nvcv::cuda::NumElements<RT> == 2)                                                \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y};                                                  \
            else if constexpr (nvcv::cuda::NumElements<RT> == 3)                                                \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z};                                \
            else if constexpr (nvcv::cuda::NumElements<RT> == 4)                                                \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z, a.w OPERATOR b.w};              \
        }                                                                                                       \
    }

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
// HIP_vector_type also ships operator(vec<T,n>, U) / operator(U, vec<T,n>) with a
// templated scalar U; for a float/double scalar these keep the vector's integer
// element type (truncating) and win partial ordering over CV-CUDA's promoting
// operator(T, U). Emit vec<>+float and vec<>+double overloads (concrete scalar
// types, hence more specialized than HIP's templated U) so CV-CUDA's promoting
// body is selected. The compound-compound overload covers the mixed-vector case.
// Direct element-wise bodies: the shared BODY macro syntax-checks `b.x` in
// untaken if-constexpr branches, which is a hard error when the scalar arg has a
// concrete (non-dependent) type, so the scalar overloads spell the loop out.
#define NVCV_CUDA_BINARY_OPERATOR_HIP_SCALAR(OPERATOR, SCALAR)                                                  \
    template<typename T1, unsigned int N,                                                                      \
             class = nvcv::cuda::Require<nvcv::cuda::IsCompound<HIP_vector_type<T1, N>>>>                       \
    inline __host__ __device__ auto operator OPERATOR(HIP_vector_type<T1, N> a, SCALAR b)                       \
    {                                                                                                           \
        using RT = nvcv::cuda::MakeType<decltype(std::declval<T1>() OPERATOR std::declval<SCALAR>()), N>;       \
        if constexpr (N == 1)                                                                                  \
            return RT{a.x OPERATOR b};                                                                          \
        else if constexpr (N == 2)                                                                             \
            return RT{a.x OPERATOR b, a.y OPERATOR b};                                                          \
        else if constexpr (N == 3)                                                                             \
            return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b};                                          \
        else                                                                                                   \
            return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b, a.w OPERATOR b};                          \
    }                                                                                                           \
    template<typename T2, unsigned int N,                                                                      \
             class = nvcv::cuda::Require<nvcv::cuda::IsCompound<HIP_vector_type<T2, N>>>>                       \
    inline __host__ __device__ auto operator OPERATOR(SCALAR a, HIP_vector_type<T2, N> b)                       \
    {                                                                                                           \
        using RT = nvcv::cuda::MakeType<decltype(std::declval<SCALAR>() OPERATOR std::declval<T2>()), N>;       \
        if constexpr (N == 1)                                                                                  \
            return RT{a OPERATOR b.x};                                                                          \
        else if constexpr (N == 2)                                                                             \
            return RT{a OPERATOR b.x, a OPERATOR b.y};                                                          \
        else if constexpr (N == 3)                                                                             \
            return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z};                                          \
        else                                                                                                   \
            return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z, a OPERATOR b.w};                          \
    }
// HIP_vector_type ships its own operator OP for every pair (vec,vec),
// (const vec&, U), (U, const vec&) with U unconstrained and the vector by const
// reference. Two cases need CV-CUDA's promoting semantics instead of HIP's:
//   * a mixed-element vector pair (e.g. float3 OP uchar3): HIP's (vec,vec) needs
//     both element types identical, so both HIP's (const vec&, U) and (U, const
//     vec&) are viable and tie, and either truncates to one operand's element
//     type. A both-operands-concrete overload is more specialized than either
//     HIP form by signature alone, so it wins without a constraint; the SFINAE
//     enable_if(!same element type) keeps same-type vec OP vec on HIP's own
//     operator. This is the only case the operator kernels exercise, so it is
//     emitted unconditionally (C++17-valid).
//   * a vector OP dim3 (dim3 is an NVCV compound but not a HIP_vector_type):
//     HIP's (const vec&, U=dim3) is chosen and make_vector_type<T,n>(dim3) is
//     ill-formed. dim3 is the only NVCV compound that is not a HIP_vector_type
//     (and is always 3 components), so a forward overload with a CONCRETE dim3
//     operand (more specialized than HIP's templated U) wins unambiguously --
//     for the integral-only operators (%,&,|,^,<<,>>) HIP's own operator is
//     itself enable_if-constrained, so a requires-clause could not break the
//     tie, but a concrete parameter type still does. The 3-element result body
//     is spelled out (the shared body's if-constexpr branches reference .w,
//     which is a hard error on the concrete 3-element dim3 even when discarded).
//     The mirror covers dim3 OP vec; both are plain C++17.
#define NVCV_CUDA_BINARY_OPERATOR_HIP_DIM3(OPERATOR)                                                            \
    template<typename T1, class = nvcv::cuda::Require<nvcv::cuda::IsCompound<HIP_vector_type<T1, 3>>>>          \
    inline __host__ __device__ auto operator OPERATOR(const HIP_vector_type<T1, 3> &a, dim3 b)                  \
    {                                                                                                           \
        using RT = nvcv::cuda::MakeType<decltype(std::declval<T1>() OPERATOR std::declval<unsigned int>()), 3>;\
        return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z};                                        \
    }                                                                                                           \
    template<typename T2, class = nvcv::cuda::Require<nvcv::cuda::IsCompound<HIP_vector_type<T2, 3>>>>          \
    inline __host__ __device__ auto operator OPERATOR(dim3 a, const HIP_vector_type<T2, 3> &b)                  \
    {                                                                                                           \
        using RT = nvcv::cuda::MakeType<decltype(std::declval<unsigned int>() OPERATOR std::declval<T2>()), 3>;\
        return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z};                                        \
    }
#define NVCV_CUDA_BINARY_OPERATOR_HIP(OPERATOR)                                                                 \
    template<typename T1, typename T2, unsigned int N,                                                          \
             class = nvcv::cuda::Require<!std::is_same_v<T1, T2>>>                                              \
    inline __host__ __device__ auto operator OPERATOR(const HIP_vector_type<T1, N> &a,                          \
                                                      const HIP_vector_type<T2, N> &b)                          \
    {                                                                                                           \
        using T = HIP_vector_type<T1, N>;                                                                       \
        using U = HIP_vector_type<T2, N>;                                                                       \
        NVCV_CUDA_BINARY_OPERATOR_BODY(OPERATOR)                                                                \
    }                                                                                                           \
    NVCV_CUDA_BINARY_OPERATOR_HIP_DIM3(OPERATOR)                                                                \
    NVCV_CUDA_BINARY_OPERATOR_HIP_SCALAR(OPERATOR, float)                                                       \
    NVCV_CUDA_BINARY_OPERATOR_HIP_SCALAR(OPERATOR, double)
#else
#define NVCV_CUDA_BINARY_OPERATOR_HIP(OPERATOR)
#endif

#define NVCV_CUDA_BINARY_OPERATOR(OPERATOR, REQUIREMENT)                                                        \
    template<typename T, typename U, class = nvcv::cuda::Require<REQUIREMENT<T, U>>>                            \
    inline __host__ __device__ auto operator OPERATOR(T a, U b) NVCV_CUDA_BINARY_OPERATOR_BODY(OPERATOR)        \
    NVCV_CUDA_BINARY_OPERATOR_HIP(OPERATOR)                                                                     \
    template<typename T, typename U, class = nvcv::cuda::Require<nvcv::cuda::IsCompound<T>>>                    \
    inline __host__ __device__ T &operator OPERATOR##=(T &a, U b)                                               \
    {                                                                                                           \
        return a = nvcv::cuda::StaticCast<nvcv::cuda::BaseType<T>>(a OPERATOR b);                               \
    }

NVCV_CUDA_BINARY_OPERATOR(-, nvcv::cuda::detail::OneIsCompound)
NVCV_CUDA_BINARY_OPERATOR(+, nvcv::cuda::detail::OneIsCompound)
NVCV_CUDA_BINARY_OPERATOR(*, nvcv::cuda::detail::OneIsCompound)
NVCV_CUDA_BINARY_OPERATOR(/, nvcv::cuda::detail::OneIsCompound)
NVCV_CUDA_BINARY_OPERATOR(%, nvcv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(&, nvcv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(|, nvcv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(^, nvcv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(<<, nvcv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(>>, nvcv::cuda::detail::OneIsCompoundAndBothAreIntegral)

#undef NVCV_CUDA_BINARY_OPERATOR

template<typename T, typename U, class = nvcv::cuda::Require<nvcv::cuda::detail::IsSameCompound<T, U>>>
inline __host__ __device__ bool operator==(T a, U b)
{
    if constexpr (nvcv::cuda::NumElements<T> >= 1)
        if (a.x != b.x)
            return false;
    if constexpr (nvcv::cuda::NumElements<T> >= 2)
        if (a.y != b.y)
            return false;
    if constexpr (nvcv::cuda::NumElements<T> >= 3)
        if (a.z != b.z)
            return false;
    if constexpr (nvcv::cuda::NumElements<T> == 4)
        if (a.w != b.w)
            return false;
    return true;
}

template<typename T, typename U, class = nvcv::cuda::Require<nvcv::cuda::detail::IsSameCompound<T, U>>>
inline __host__ __device__ bool operator!=(T a, U b)
{
    return !(a == b);
}

// HIP_vector_type ships operator==/!= for (vec,vec), (const vec&, U), (U, const
// vec&). A mixed pair like int3 == long3 makes both HIP (const vec&, U) and (U,
// const vec&) viable (tie), and vec == dim3 is ill-formed inside
// make_vector_type(dim3). Same structure and rationale as the arithmetic fix: a
// both-operands-concrete overload (more specialized than either HIP form) for
// the mixed-element vector pair (SFINAE-excluding same element type so HIP's own
// operator keeps that), plus a concrete-dim3 forward and mirror. All C++17.
// Bodies are spelled out (a concrete operand trips the if-constexpr probe of the
// shared arithmetic body).
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
#define NVCV_CUDA_HIP_COMPARE_BODY(RET_ON_DIFF, RET_DEFAULT)                                                    \
    {                                                                                                           \
        bool eq = true;                                                                                         \
        if constexpr (N >= 1)                                                                                  \
            eq = eq && (a.x == b.x);                                                                            \
        if constexpr (N >= 2)                                                                                  \
            eq = eq && (a.y == b.y);                                                                            \
        if constexpr (N >= 3)                                                                                  \
            eq = eq && (a.z == b.z);                                                                            \
        if constexpr (N == 4)                                                                                  \
            eq = eq && (a.w == b.w);                                                                            \
        return eq ? RET_DEFAULT : RET_ON_DIFF;                                                                  \
    }
#define NVCV_CUDA_HIP_COMPARE(OPERATOR, RET_ON_DIFF, RET_DEFAULT)                                               \
    template<typename T1, typename T2, unsigned int N,                                                          \
             class = nvcv::cuda::Require<!std::is_same_v<T1, T2>>>                                              \
    inline __host__ __device__ bool operator OPERATOR(const HIP_vector_type<T1, N> &a,                          \
                                                      const HIP_vector_type<T2, N> &b)                          \
        NVCV_CUDA_HIP_COMPARE_BODY(RET_ON_DIFF, RET_DEFAULT)                                                    \
    template<typename T1, class = nvcv::cuda::Require<nvcv::cuda::IsCompound<HIP_vector_type<T1, 3>>>>          \
    inline __host__ __device__ bool operator OPERATOR(const HIP_vector_type<T1, 3> &a, dim3 b)                  \
    {                                                                                                           \
        bool eq = (a.x == b.x) && (a.y == b.y) && (a.z == b.z);                                                 \
        return eq ? RET_DEFAULT : RET_ON_DIFF;                                                                  \
    }                                                                                                           \
    template<typename T2, class = nvcv::cuda::Require<nvcv::cuda::IsCompound<HIP_vector_type<T2, 3>>>>          \
    inline __host__ __device__ bool operator OPERATOR(dim3 a, const HIP_vector_type<T2, 3> &b)                  \
    {                                                                                                           \
        bool eq = (a.x == b.x) && (a.y == b.y) && (a.z == b.z);                                                 \
        return eq ? RET_DEFAULT : RET_ON_DIFF;                                                                  \
    }

NVCV_CUDA_HIP_COMPARE(==, false, true)
NVCV_CUDA_HIP_COMPARE(!=, true, false)

#undef NVCV_CUDA_HIP_COMPARE
#undef NVCV_CUDA_HIP_COMPARE_BODY
#endif

namespace nvcv::cuda {

template<typename T, typename U, class = nvcv::cuda::Require<nvcv::cuda::detail::IsSameCompound<T, U>>>
inline __host__ __device__ auto dot(T a, U b)
{
    using PT = decltype(std::declval<nvcv::cuda::BaseType<T>>() * std::declval<nvcv::cuda::BaseType<U>>());
    using RT = decltype(std::declval<PT>() + std::declval<PT>());

    if constexpr (nvcv::cuda::NumComponents<T> == 1)
        return RT{a.x * b.x};
    else if constexpr (nvcv::cuda::NumComponents<T> == 2)
        return RT{a.x * b.x + a.y * b.y};
    else if constexpr (nvcv::cuda::NumComponents<T> == 3)
        return RT{a.x * b.x + a.y * b.y + a.z * b.z};
    else if constexpr (nvcv::cuda::NumComponents<T> == 4)
        return RT{a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w};
}

} // namespace nvcv::cuda

/**@}*/

#endif // NVCV_CUDA_MATH_OPS_HPP
