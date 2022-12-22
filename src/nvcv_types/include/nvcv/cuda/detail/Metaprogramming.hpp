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

#ifndef NVCV_CUDA_DETAIL_METAPROGRAMMING_HPP
#define NVCV_CUDA_DETAIL_METAPROGRAMMING_HPP

// Internal implementation of meta-programming functionalities.
// Not to be used directly.

#include <cuda_runtime.h> // for uchar1, etc.

#include <cfloat>      // for FLT_MIN, etc.
#include <climits>     // for CHAR_MIN, etc.
#include <type_traits> // for std::remove_const, etc.

namespace nvcv::cuda::detail {

template<class FROM, class TO>
struct CopyConstness
{
    using type = typename std::remove_const_t<TO>;
};

template<class FROM, class TO>
struct CopyConstness<const FROM, TO>
{
    using type = typename std::add_const_t<TO>;
};

// Metatype to copy the const FROM type TO type.
template<class FROM, class TO>
using CopyConstness_t = typename CopyConstness<FROM, TO>::type;

// Metatype to add information to regular C types and CUDA compound types.
template<class T>
struct TypeTraits;

#define NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE, BASE_TYPE, COMPONENTS, ELEMENTS, MIN_VAL, MAX_VAL) \
    template<>                                                                                  \
    struct TypeTraits<COMPOUND_TYPE>                                                            \
    {                                                                                           \
        using base_type                       = BASE_TYPE;                                      \
        static constexpr int       components = COMPONENTS;                                     \
        static constexpr int       elements   = ELEMENTS;                                       \
        static constexpr char      name[]     = #COMPOUND_TYPE;                                 \
        static constexpr base_type min        = MIN_VAL;                                        \
        static constexpr base_type max        = MAX_VAL;                                        \
    }

NVCV_CUDA_TYPE_TRAITS(dim3, unsigned int, 3, 3, 0, UINT_MAX);

NVCV_CUDA_TYPE_TRAITS(unsigned char, unsigned char, 0, 1, 0, UCHAR_MAX);
NVCV_CUDA_TYPE_TRAITS(signed char, signed char, 0, 1, SCHAR_MIN, SCHAR_MAX);
#if CHAR_MIN == 0
NVCV_CUDA_TYPE_TRAITS(char, unsigned char, 0, 1, 0, UCHAR_MAX);
#else
NVCV_CUDA_TYPE_TRAITS(char, signed char, 0, 1, SCHAR_MIN, SCHAR_MAX);
#endif
NVCV_CUDA_TYPE_TRAITS(short, short, 0, 1, SHRT_MIN, SHRT_MAX);
NVCV_CUDA_TYPE_TRAITS(unsigned short, unsigned short, 0, 1, 0, USHRT_MAX);
NVCV_CUDA_TYPE_TRAITS(int, int, 0, 1, INT_MIN, INT_MAX);
NVCV_CUDA_TYPE_TRAITS(unsigned int, unsigned int, 0, 1, 0, UINT_MAX);
NVCV_CUDA_TYPE_TRAITS(long, long, 0, 1, LONG_MIN, LONG_MAX);
NVCV_CUDA_TYPE_TRAITS(unsigned long, unsigned long, 0, 1, 0, ULONG_MAX);
NVCV_CUDA_TYPE_TRAITS(long long, long long, 0, 1, LLONG_MIN, LLONG_MAX);
NVCV_CUDA_TYPE_TRAITS(unsigned long long, unsigned long long, 0, 1, 0, ULLONG_MAX);
NVCV_CUDA_TYPE_TRAITS(float, float, 0, 1, FLT_MIN, FLT_MAX);
NVCV_CUDA_TYPE_TRAITS(double, double, 0, 1, DBL_MIN, DBL_MAX);

#define NVCV_CUDA_TYPE_TRAITS_1_TO_4(COMPOUND_TYPE, BASE_TYPE, MIN_VAL, MAX_VAL) \
    NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE##1, BASE_TYPE, 1, 1, MIN_VAL, MAX_VAL);  \
    NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE##2, BASE_TYPE, 2, 2, MIN_VAL, MAX_VAL);  \
    NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE##3, BASE_TYPE, 3, 3, MIN_VAL, MAX_VAL);  \
    NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE##4, BASE_TYPE, 4, 4, MIN_VAL, MAX_VAL)

NVCV_CUDA_TYPE_TRAITS_1_TO_4(char, signed char, SCHAR_MIN, SCHAR_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(uchar, unsigned char, 0, UCHAR_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(short, short, SHRT_MIN, SHRT_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(ushort, unsigned short, 0, USHRT_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(int, int, INT_MIN, INT_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(uint, unsigned int, 0, UINT_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(long, long, LONG_MIN, LONG_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(ulong, unsigned long, 0, ULONG_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(longlong, long long, LLONG_MIN, LLONG_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(ulonglong, unsigned long long, 0, ULLONG_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(float, float, FLT_MIN, FLT_MAX);
NVCV_CUDA_TYPE_TRAITS_1_TO_4(double, double, DBL_MIN, DBL_MAX);

#undef NVCV_CUDA_TYPE_TRAITS_1_TO_4
#undef NVCV_CUDA_TYPE_TRAITS

template<class T>
struct TypeTraits<const T> : TypeTraits<T>
{
};

template<class T>
struct TypeTraits<volatile T> : TypeTraits<T>
{
};

template<class T>
struct TypeTraits<const volatile T> : TypeTraits<T>
{
};

// Metatype to make a type given a base type and a number of components.
template<class T, int C>
struct MakeType;

#define NVCV_CUDA_MAKE_TYPE(BASE_TYPE, COMPONENTS, COMPOUND_TYPE) \
    template<>                                                    \
    struct MakeType<BASE_TYPE, COMPONENTS>                        \
    {                                                             \
        using type = COMPOUND_TYPE;                               \
    }

#define NVCV_CUDA_MAKE_TYPE_0_TO_4(BASE_TYPE, COMPOUND_TYPE) \
    NVCV_CUDA_MAKE_TYPE(BASE_TYPE, 0, BASE_TYPE);            \
    NVCV_CUDA_MAKE_TYPE(BASE_TYPE, 1, COMPOUND_TYPE##1);     \
    NVCV_CUDA_MAKE_TYPE(BASE_TYPE, 2, COMPOUND_TYPE##2);     \
    NVCV_CUDA_MAKE_TYPE(BASE_TYPE, 3, COMPOUND_TYPE##3);     \
    NVCV_CUDA_MAKE_TYPE(BASE_TYPE, 4, COMPOUND_TYPE##4)

#if CHAR_MIN == 0
NVCV_CUDA_MAKE_TYPE_0_TO_4(char, uchar);
#else
NVCV_CUDA_MAKE_TYPE_0_TO_4(char, char);
#endif
NVCV_CUDA_MAKE_TYPE_0_TO_4(unsigned char, uchar);
NVCV_CUDA_MAKE_TYPE_0_TO_4(signed char, char);
NVCV_CUDA_MAKE_TYPE_0_TO_4(unsigned short, ushort);
NVCV_CUDA_MAKE_TYPE_0_TO_4(short, short);
NVCV_CUDA_MAKE_TYPE_0_TO_4(unsigned int, uint);
NVCV_CUDA_MAKE_TYPE_0_TO_4(int, int);
NVCV_CUDA_MAKE_TYPE_0_TO_4(unsigned long, ulong);
NVCV_CUDA_MAKE_TYPE_0_TO_4(long, long);
NVCV_CUDA_MAKE_TYPE_0_TO_4(unsigned long long, ulonglong);
NVCV_CUDA_MAKE_TYPE_0_TO_4(long long, longlong);
NVCV_CUDA_MAKE_TYPE_0_TO_4(float, float);
NVCV_CUDA_MAKE_TYPE_0_TO_4(double, double);

#undef NVCV_CUDA_MAKE_TYPE_0_TO_4
#undef NVCV_CUDA_MAKE_TYPE

template<class T, int C>
struct MakeType<const T, C>
{
    using type = const typename MakeType<T, C>::type;
};

template<class T, int C>
struct MakeType<volatile T, C>
{
    using type = volatile typename MakeType<T, C>::type;
};

template<class T, int C>
struct MakeType<const volatile T, C>
{
    using type = const volatile typename MakeType<T, C>::type;
};

template<class T, int C>
using MakeType_t = typename detail::MakeType<T, C>::type;

// Metatype to convert the base type of a target type to another base type.
template<class BT, class T>
struct ConvertBaseTypeTo
{
    using type = MakeType_t<BT, TypeTraits<T>::components>;
};

template<class BT, class T>
struct ConvertBaseTypeTo<BT, const T>
{
    using type = const MakeType_t<BT, TypeTraits<T>::components>;
};

template<class BT, class T>
struct ConvertBaseTypeTo<BT, volatile T>
{
    using type = volatile MakeType_t<BT, TypeTraits<T>::components>;
};

template<class BT, class T>
struct ConvertBaseTypeTo<BT, const volatile T>
{
    using type = const volatile MakeType_t<BT, TypeTraits<T>::components>;
};

template<class BT, class T>
using ConvertBaseTypeTo_t = typename ConvertBaseTypeTo<BT, T>::type;

// Metatype to check if a type has type traits associated with it.
// If T does not have TypeTrait<T>, value is false, otherwise value is true.
template<typename T, typename = void>
struct HasTypeTraits_t : std::false_type
{
};

template<typename T>
struct HasTypeTraits_t<T, std::void_t<typename TypeTraits<T>::base_type>> : std::true_type
{
};

} // namespace nvcv::cuda::detail

#endif // NVCV_CUDA_DETAIL_METAPROGRAMMING_HPP
