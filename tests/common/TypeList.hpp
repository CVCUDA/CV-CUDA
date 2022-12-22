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

#ifndef NVCV_TEST_COMMON_TYPELIST_HPP
#define NVCV_TEST_COMMON_TYPELIST_HPP

#include "ValueList.hpp"

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>
#include <util/Compiler.hpp>

// Utilities for creating parameters for typed tests on GoogleTest
// We support both typed and (constexpr) value parameters. In order to work them
// seamslessly, we wrap the value into type Value
//
// Types is used to define type list, it's just an alias to ::testing::Types:
// using Types = util::Types<int,char,float>;
//
// Values is used to define compile-time value lists
// using Values = util::Values<'c',3.0,100>;
//
// You can declare your tests using a template fixture:
//
// template <class T>
// class TestFixture : ::testing::Test
// {
// };
//
// Inside your test function, you can access the parameters like this:
//
// TEST(TestFixture, mytest)
// {
//      using MEM = GetType<TypeParam,0>; // the first type element
//      constexpr auto VALUE = GetValue<TypeParam,1>; // second value element
// }
//
// You can compose complicated type/value arguments using Concat and Combine and RemoveIf
//
// using Types = Combine<Types<int,float>,Values<0,4>>;
// creates the parameters <int,0> <int,4> <float,0> <float,4>
//
// Concat is useful to concatenate parameter lists created with Combine
//
// using Types = Concat<Combine<Types<int,float>,ValueType<0>>,
//                      Combine<Types<char,double>,ValueType<1,2>>>;
// creates the parameters <int,0> <float,0> <char,1> <char,2> <double,1> <double,2>
//
// RemoveIf can be used to remove some parameters that match a given predicate:
//
// using Types = RemoveIf<AllSame, Combine<Types<int,char>, Types<int,char>>>;
// creates the parameters <int,char>,<char,int>

namespace testing::internal {

// googletest-1.10.0 has no ::testing::Types<>,
// but we need one. Let's make a specialization that will help us out.
template<>
struct ProxyTypeList<>
{
    using type = None;
};

} // namespace testing::internal

namespace nvcv::test::type {

// Types -----------------------------------------

using ::testing::Types;

template<class T, int D>
struct GetTypeImpl
{
    static_assert(D == 0, "Out of bounds");
    using type = T;
};

template<class... T, int D>
struct GetTypeImpl<::testing::internal::Types<T...>, D>
{
    static_assert(D < sizeof...(T), "Out of bounds");

    using type = typename GetTypeImpl<typename ::testing::internal::Types<T...>::Tail, D - 1>::type;
};

template<class... ARGS>
struct GetTypeImpl<::testing::internal::Types<ARGS...>, 0>
{
    static_assert(sizeof...(ARGS) > 0, "Out of bounds");

    using type = typename ::testing::internal::Types<ARGS...>::Head;
};

template<class... ARGS, int D>
struct GetTypeImpl<Types<ARGS...>, D> : GetTypeImpl<typename Types<ARGS...>::type, D>
{
};

template<class TUPLE, int D>
using GetType = typename GetTypeImpl<TUPLE, D>::type;

// GetSize -------------------------------
// returns the size (number of elements) of the type list

template<class TUPLE>
struct GetSizeImpl;

template<class... TYPES>
struct GetSizeImpl<Types<TYPES...>>
{
    static constexpr auto value = sizeof...(TYPES);
};

template<class TUPLE>
constexpr auto GetSize = GetSizeImpl<TUPLE>::value;

// Values -----------------------------------------

// NOTE: be aware that gcc-7.x has a bug where V's type will be
// wrong if the value was already instantiated with another type.
// gcc-8.0 fixes it. clang-6.0.0 doesn't have this bug.
// decltype(GetValue<Values<1,3ul,1>>) == 'const int' instead of 'const unsigned long'
// Ref: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=79092
// One WAR would be to define Value without using auto, i.e.
// Value<class T, T V>, and creating a macro that uses decltype
// to define 'class T', so that we don't have to write the type
// ourselves. Since the minimum compiler version we support doesn't
// have this bug, we're fine, but let's leave a reminder here in case
// we feel like supporting again an old compiler.

#if (defined(NVCV_GCC_VERSION) && NVCV_GCC_VERSION < 80000) \
    || (defined(NVCV_CLANG_VERSION) && NVCV_CLANG_VERSION < 60000)
#    error Compiler not supported, cryptic compiler errors will ensue
#endif

template<auto V>
struct Value
{
    static constexpr auto value = V;
};

template<auto... ARGS>
struct ValuesImpl
{
    using type = Types<Value<ARGS>...>;
};

template<auto... ARGS>
using Values = typename ValuesImpl<ARGS...>::type;

template<class TUPLE, int D>
constexpr auto GetValue = GetType<TUPLE, D>::value;

// Concat -----------------------------------------

namespace detail {
template<class A, class B>
struct Concat2;

template<class... T, class... U>
struct Concat2<Types<T...>, Types<U...>>
{
    using type = Types<T..., U...>;
};
} // namespace detail

template<class... T>
struct ConcatImpl;

template<class HEAD1, class HEAD2, class... TAIL>
struct ConcatImpl<HEAD1, HEAD2, TAIL...>
{
    using type = typename ConcatImpl<typename detail::Concat2<HEAD1, HEAD2>::type, TAIL...>::type;
};

template<class A>
struct ConcatImpl<A>
{
    using type = A;
};

template<class... A>
struct ConcatImpl<Types<A...>>
{
    using type = Types<A...>;
};

template<>
struct ConcatImpl<>
{
    using type = Types<>;
};

template<class... T>
using Concat = typename ConcatImpl<T...>::type;

// Subset -------------------------------

template<class T, int BEG, int CUR, int END>
struct SubsetImpl : SubsetImpl<Types<T>, BEG, CUR, END>
{
};

template<int BEG>
struct SubsetImpl<Types<>, BEG, 0, BEG>
{
    using type = Types<>;
};

// inside the range?
template<class HEAD, class... TAIL, int CUR, int END>
struct SubsetImpl<Types<HEAD, TAIL...>, CUR, CUR, END>
{
    static_assert(CUR < END);
    using type = Concat<Types<HEAD>, typename SubsetImpl<Types<TAIL...>, CUR + 1, CUR + 1, END>::type>;
};

// range ended
template<class HEAD, class... TAIL, int CUR>
struct SubsetImpl<Types<HEAD, TAIL...>, CUR, CUR, CUR>
{
    using type = Types<>;
};

// before range?
template<class HEAD, class... TAIL, int BEG, int CUR, int END>
struct SubsetImpl<Types<HEAD, TAIL...>, BEG, CUR, END>
{
    static_assert(BEG >= 0);
    static_assert(BEG <= END);
    using type = typename SubsetImpl<Types<TAIL...>, BEG, CUR + 1, END>::type;
};

template<class T, int BEG, int END>
using Subset = typename SubsetImpl<T, BEG, 0, END>::type;

// Flatten -----------------------------------------

template<class T>
struct FlattenImpl;

template<>
struct FlattenImpl<Types<>>
{
    using type = Types<>;
};

template<class HEAD, class... TAIL>
struct FlattenImpl<Types<HEAD, TAIL...>>
{
    using type = Concat<Types<HEAD>, typename FlattenImpl<Types<TAIL...>>::type>;
};

template<class... HEAD, class... TAIL>
struct FlattenImpl<Types<::testing::internal::ProxyTypeList<HEAD...>, TAIL...>>
{
    using type = typename FlattenImpl<Types<HEAD..., TAIL...>>::type;
};

// MSVC16/2019 won't match the above HEAD... variadic template. We have to
// expand it manually. Let's do it for 5 types, it should be enough for now.
template<class T1, class... TAIL>
struct FlattenImpl<Types<Types<T1>, TAIL...>>
{
    using type = typename FlattenImpl<Types<T1, TAIL...>>::type;
};

template<class T1, class T2, class... TAIL>
struct FlattenImpl<Types<Types<T1, T2>, TAIL...>>
{
    using type = typename FlattenImpl<Types<T1, T2, TAIL...>>::type;
};

template<class T1, class T2, class T3, class... TAIL>
struct FlattenImpl<Types<Types<T1, T2, T3>, TAIL...>>
{
    using type = typename FlattenImpl<Types<T1, T2, T3, TAIL...>>::type;
};

template<class T1, class T2, class T3, class T4, class... TAIL>
struct FlattenImpl<Types<Types<T1, T2, T3, T4>, TAIL...>>
{
    using type = typename FlattenImpl<Types<T1, T2, T3, T4, TAIL...>>::type;
};

template<class T1, class T2, class T3, class T4, class T5, class... TAIL>
struct FlattenImpl<Types<Types<T1, T2, T3, T4, T5>, TAIL...>>
{
    using type = typename FlattenImpl<Types<T1, T2, T3, T4, T5, TAIL...>>::type;
};
// End of MSVC16/2019 HACK

template<class T>
using Flatten = typename FlattenImpl<T>::type;

// Combine -----------------------------------------

namespace detail {
// prepend T in TUPLE
template<class T, class TUPLE>
struct Prepend1;

template<class T, class... ARGS>
struct Prepend1<T, Types<ARGS...>>
{
    using type = Flatten<Types<T, ARGS...>>;
};

template<class T, class TUPLES>
struct Prepend;

// Prepend T in all TUPLES
template<class T, class... TUPLES>
struct Prepend<T, Types<TUPLES...>>
{
    using type = Types<typename Prepend1<T, TUPLES>::type...>;
};

// skip empty tuples
template<class T, class... TUPLES>
struct Prepend<T, Types<Types<>, TUPLES...>> : Prepend<T, Types<TUPLES...>>
{
};
} // namespace detail

template<class... ARGS>
struct CombineImpl;

template<>
struct CombineImpl<>
{
    using type = Types<>;
};

template<class... ARGS>
struct CombineImpl<Types<ARGS...>>
{
    using type = Types<Types<ARGS>...>;
};

template<class... AARGS, class... TAIL>
struct CombineImpl<Types<AARGS...>, TAIL...>
{
    using type = Concat<typename detail::Prepend<AARGS, typename CombineImpl<TAIL...>::type>::type...>;
};

// to make it easy for the user when there's only one element to be joined
template<class T, class... TAIL>
struct CombineImpl<T, TAIL...> : CombineImpl<Types<T>, TAIL...>
{
};

template<class... ARGS>
using Combine = typename CombineImpl<ARGS...>::type;

// AllSame -----------------------------------------

namespace detail {
template<class... ITEMS>
struct AllSame : std::false_type
{
};

// degenerate case
template<class A>
struct AllSame<A> : std::true_type
{
};

template<class A>
struct AllSame<A, A> : std::true_type
{
};

template<class HEAD, class... TAIL>
struct AllSame<HEAD, HEAD, TAIL...> : AllSame<HEAD, TAIL...>
{
};

template<class... ITEMS>
struct AllSame<Types<ITEMS...>> : AllSame<ITEMS...>
{
};

} // namespace detail

struct AllSame
{
    template<class... ITEMS>
    using Call = detail::AllSame<ITEMS...>;
};

// Exists ---------------------------------

// Do a linear search to find NEEDLE in HAYSACK
template<class NEEDLE, class HAYSACK>
struct ExistsImpl;

// end case, no more types to check
template<class NEEDLE>
struct ExistsImpl<NEEDLE, Types<>> : std::false_type
{
};

// next one matches
template<class NEEDLE, class... TAIL>
struct ExistsImpl<NEEDLE, Types<NEEDLE, TAIL...>> : std::true_type
{
};

// next one doesn't match
template<class NEEDLE, class HEAD, class... TAIL>
struct ExistsImpl<NEEDLE, Types<HEAD, TAIL...>> : ExistsImpl<NEEDLE, Types<TAIL...>>
{
};

template<class NEEDLE, class HAYSACK>
constexpr bool Exists = ExistsImpl<NEEDLE, HAYSACK>::value;

// ContainedIn -----------------------------------------

template<class HAYSACK>
struct ContainedIn
{
    template<class NEEDLE>
    using Call = ExistsImpl<NEEDLE, HAYSACK>;
};

// RemoveIf -----------------------------------------

template<class PRED, class TUPLE>
struct RemoveIfImpl;

template<class PRED>
struct RemoveIfImpl<PRED, Types<>>
{
    using type = Types<>;
};

template<class PRED, class HEAD, class... TAIL>
struct RemoveIfImpl<PRED, Types<HEAD, TAIL...>>
{
    using type = Concat<typename std::conditional<PRED::template Call<HEAD>::value, Types<>, Types<HEAD>>::type,
                        typename RemoveIfImpl<PRED, Types<TAIL...>>::type>;
};

template<class PRED, class TUPLE>
using RemoveIf = typename RemoveIfImpl<PRED, TUPLE>::type;

// Transform --------------------------------

template<class XFORM, class ITEM>
struct TransformImpl
{
    using type = Types<typename XFORM::template Call<ITEM>>;
};

template<class XFORM, class... ITEMS>
struct TransformImpl<XFORM, Types<ITEMS...>>
{
    using type = Types<typename XFORM::template Call<ITEMS>...>;
};

template<class XFORM, class TYPES>
using Transform = typename TransformImpl<XFORM, TYPES>::type;

// Rep --------------------------------

namespace detail {
template<class T, int N, class RES>
struct Rep;

template<class T, int N, class... ITEMS>
struct Rep<T, N, Types<ITEMS...>>
{
    using type = typename Rep<T, N - 1, Types<T, ITEMS...>>::type;
};

template<class T, class... ITEMS>
struct Rep<T, 0, Types<ITEMS...>>
{
    using type = Types<ITEMS...>;
};
} // namespace detail

template<int N>
struct Rep
{
    template<class T>
    using Call = typename detail::Rep<T, N, Types<>>::type;
};

// Append --------------------------------

template<class TYPES, class... ITEMS>
struct AppendImpl;

template<class... HEAD, class... TAIL>
struct AppendImpl<Types<HEAD...>, TAIL...>
{
    using type = Types<HEAD..., TAIL...>;
};

template<class TYPES, class... ITEMS>
using Append = typename AppendImpl<TYPES, ITEMS...>::type;

// Apply --------------------------------

template<template<class> class XFORM>
struct Apply
{
    template<class T>
    using Call = typename XFORM<T>::type;
};

// Deref --------------------------------

template<template<class> class XFORM>
struct Deref
{
    template<class T>
    using Call = Transform<Apply<XFORM>, T>;
};

// Remove -------------------------------------------
// remove items from tuple given by their indices

namespace detail {
template<class TUPLE, int CUR, int... IDXs>
struct Remove;

// nothing else to do?
template<class... ITEMS, int CUR>
struct Remove<Types<ITEMS...>, CUR>
{
    using type = Types<ITEMS...>;
};

// index match current item?
template<class HEAD, class... TAIL, int CUR, int... IDXTAIL>
struct Remove<Types<HEAD, TAIL...>, CUR, CUR, IDXTAIL...>
{
    // remove it, and recurse into the remaining items
    using type = typename Remove<Types<TAIL...>, CUR + 1, IDXTAIL...>::type;
};

// index doesn't match current item?
template<class HEAD, class... TAIL, int CUR, int IDXHEAD, int... IDXTAIL>
struct Remove<Types<HEAD, TAIL...>, CUR, IDXHEAD, IDXTAIL...>
{
    static_assert(sizeof...(TAIL) + 1 > IDXHEAD - CUR, "Index out of bounds");

    // add current item to output and recurse into the remaining items
    using type = Concat<Types<HEAD>, typename Remove<Types<TAIL...>, CUR + 1, IDXHEAD, IDXTAIL...>::type>;
};
} // namespace detail

template<class TUPLE, int... IDXs>
struct RemoveImpl
{
    using type = typename detail::Remove<TUPLE, 0, IDXs...>::type;
};

template<class TUPLE, int... IDXs>
using Remove = typename RemoveImpl<TUPLE, IDXs...>::type;

// Unique --------------------------------

namespace detail {
template<class... ITEMS>
struct Unique;

template<>
struct Unique<>
{
    using type = Types<>;
};

template<class HEAD, class... TAIL>
struct Unique<HEAD, TAIL...>
{
    using type = Concat<std::conditional_t<Exists<HEAD, Types<TAIL...>>, Types<>, Types<HEAD>>,
                        typename Unique<TAIL...>::type>;
};

} // namespace detail

template<class TYPES>
struct UniqueImpl;

template<class... ITEMS>
struct UniqueImpl<Types<ITEMS...>>
{
    using type = typename detail::Unique<ITEMS...>::type;
};

template<class TYPES>
using Unique = typename UniqueImpl<TYPES>::type;

// Zip --------------------------------

template<class... T>
struct ZipImpl;

template<class HEAD, class... TAIL>
struct ZipImpl<HEAD, TAIL...>
{
private:
    template<size_t IDX>
    struct Impl2
    {
        using type = Types<GetType<HEAD, IDX>, GetType<TAIL, IDX>...>;
    };

    template<class IDXs>
    struct Impl1;

    template<size_t... IDXs>
    struct Impl1<std::index_sequence<IDXs...>>
    {
        using type = Types<typename Impl2<IDXs>::type...>;
    };

public:
    using type = typename Impl1<std::make_index_sequence<GetSize<HEAD>>>::type;
};

template<>
struct ZipImpl<>
{
    using type = Types<>;
};

template<class... T>
using Zip = typename ZipImpl<T...>::type;

// SetDifference -------------------------------------------------

template<class T, class U>
using SetDifference = RemoveIf<ContainedIn<U>, T>;

// Contains ------------------------------
// check if value is in Values container

template<class T>
constexpr bool Contains(Types<>, T size)
{
    return false;
}

template<class T, auto HEAD, auto... TAIL>
constexpr bool Contains(Types<Value<HEAD>, Value<TAIL>...>, T needle)
{
    if (HEAD == needle)
    {
        return true;
    }
    else
    {
        return Contains(Types<Value<TAIL>...>(), needle);
    }
}

inline ValueList<> ToValueList(Types<>)
{
    return {};
}

template<auto... VV>
auto ToValueList(Types<Value<VV>...>)
{
    using T = std::common_type_t<decltype(VV)...>;
    static_assert(!std::is_same_v<T, void>, "Values don't have a common type");

    return ValueList<T>{static_cast<T>(VV)...};
}

template<class T>
auto ToValueList()
{
    return ToValueList(T());
}

} // namespace nvcv::test::type

namespace nvcv::test {
using type::ToValueList;
using type::Types;
using type::Values;
} // namespace nvcv::test

#endif // NVCV_TEST_COMMON_TYPELIST_HPP
