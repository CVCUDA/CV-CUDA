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

#include "TypeList.hpp"

#include <gtest/gtest.h>

namespace nvcv::test::type {

/* Here we have an implementation of typed tests that is optimized
 * when there are parameters that are integral constants. If we use
 * gtests' TYPED_TEST, it'll instantiate different template tests even
 * for different integral constant parameters. That's a lot of templates.
 *
 * Instead, we split the processing into compile- and run-time. During
 * compile time we instantiate only enough template tests to cover all
 * types, and at run time we use gtest's facilities to actually create
 * the tests for type parameter combination, but passing the integral
 * constants as normal (run-time) variables, as std::tuples.
 *
 * Each test then is associated with two types: a type list
 * where the position of an integral constant is substituted with
 * ValueSlot<integral type>, and the types are the test parameter types.
 * The other is a std::tuple with the same size as the type list, but
 * each type is replaced by "TypeSlot", and the value is the actual integral
 * type.
 *
 * The data structure that stores all the parameters is a tuple of ParamInfo
 * for a given unique type combination, and it stores all value combinations
 * for that type.
 */

namespace t = ::testing;

namespace detail {
// Placeholder for a value, stores the integral constant type
template<class T>
struct ValueSlot
{
};

// Placeholder for a type parameter in a tuple.
struct TypeSlot
{
};

// Given a parameter type list, replaces values by ValueSlot<T>,
// keeping the types intact. To be used with Transform.
struct FilterOutValue
{
    template<class T>
    static T *impl(T *);

    template<auto V>
    static ValueSlot<decltype(V)> *impl(Value<V> *);

    template<class T>
    using Call = std::remove_reference_t<decltype(*impl((T *)nullptr))>;
};

// Given a parameter type list, replaces types with TypeSlot,
// and values by the integral constant type.
// To be used with Transform.
struct FilterOutType
{
    template<class T>
    static TypeSlot impl(T *)
    {
        return {};
    }

    template<class T>
    static T impl(ValueSlot<T> *);

    template<auto V>
    static decltype(V) impl(Value<V> *)
    {
        return V;
    };

    template<class T>
    using Call = decltype(impl((T *)nullptr));
};

// Returns whether a type represents a value or not.
template<class T>
struct IsValueImpl : public std::false_type
{
};

template<class T>
struct IsValueImpl<ValueSlot<T>> : public std::true_type
{
};
template<class T>
constexpr bool IsValue = IsValueImpl<T>::value;

// Transforms a parameter type list into a tuple type.
template<class PARAM>
struct ToTupleImpl
{
    using type = std::tuple<TypeSlot>;
};

template<auto V>
struct ToTupleImpl<Value<V>>
{
    using type = std::tuple<decltype(V)>;
};

template<class... ARG>
struct ToTupleImpl<Types<ARG...>>
{
    template<class T>
    static TypeSlot impl(T *);

    template<auto V>
    static decltype(V) impl(Value<V> *);

    using type = std::tuple<decltype(impl((ARG *)nullptr))...>;
};
template<class T>
using ToTuple = typename ToTupleImpl<T>::type;

// Transforms a parameter into a tuple
template<class... T>
auto make_tuple(Types<T...> *)
{
    return std::make_tuple(FilterOutType::impl((T *)nullptr)...);
}

template<auto V>
auto make_tuple(Value<V> *)
{
    return std::make_tuple(V);
}

template<class T>
auto make_tuple(T *)
{
    return std::make_tuple(TypeSlot());
}

// Returns a string that represents the parameter.
// Ex.: for Types<Value<1>, Value<2>, float, double>, it returns
// "<1,2,float,double>
template<bool FIRST = true, class T>
void GetParamName(std::ostream &out, T *)
{
    if (!FIRST)
    {
        out << ',';
    }
    out << ::testing::internal::GetTypeName<T>();
}

template<bool FIRST = true, auto V>
inline void GetParamName(std::ostream &out, Value<V> *)
{
    if (!FIRST)
    {
        out << ',';
    }
    out << V;
}

template<class HEAD, class... TAIL>
std::string GetParamName(Types<HEAD, TAIL...> *)
{
    std::ostringstream out;
    out << '<';

    GetParamName(out, (HEAD *)nullptr);
    if constexpr (sizeof...(TAIL) > 0)
    {
        (GetParamName<false>(out, (TAIL *)nullptr), ...);
    }

    out << '>';

    return out.str();
}

template<class T>
std::string GetParamName(T *)
{
    return GetParamName((Types<T> *)nullptr);
}

template<class T>
std::string GetParamName(T)
{
    return GetParamName((T *)nullptr);
}

// Stores all values for the given unique TYPES combination
// Also stores the string representation of the whole parameter.
template<class TYPES, class TUPLE>
struct ParamInfo
{
    std::vector<std::pair<TUPLE, std::string>> values;
};

// Given a list of parameters returns a tuple with a list of unique
// type elements of ParamInfo.
template<class TUPLE, class... UT, class... PARAM>
auto make_parameters_tuple(Types<UT...>, Types<PARAM...>)
{
    using T = std::tuple<ParamInfo<UT, TUPLE>...>;
    T ret;

    (std::get<ParamInfo<Transform<FilterOutValue, PARAM>, TUPLE>>(ret).values.emplace_back(
         make_tuple((PARAM *)nullptr), GetParamName((PARAM *)nullptr)),
     ...);
    return ret;
}

// Helper class to return the type I of the TypeParam list, but checking
// if it's an actual type (and not a value)
template<class TypeParam, int I>
struct GetTypeImpl
{
    using type = GetType<TypeParam, I>;
    static_assert(!IsValue<type>, "Test parameter isn't a type");
};

// So that users can pass list of types or values directly
template<class... T>
struct WrapParams
{
    using type = Types<T...>;
};

template<class... T>
struct WrapParams<Types<T...>>
{
    using type = Types<T...>;
};

// Register at run time all the values for a given TYPE.
template<template<class> class FIXTURE, int... IDX, class TYPE, class TUPLE>
void RegisterTests(const ParamInfo<TYPE, TUPLE> &data, const char *casename, const char *testname, const char *file,
                   int line, int &base)
{
    for (size_t i = 0; i < data.values.size(); ++i)
    {
        std::ostringstream ss;
        ss << testname << '/' << base++;
        ::testing::RegisterTest(casename, ss.str().c_str(), nullptr, data.values[i].second.c_str(), __FILE__, __LINE__,
                                [value = data.values[i].first]() -> typename FIXTURE<TYPE>::BaseFixture *
                                {
                                    auto fix = std::make_unique<FIXTURE<TYPE>>(value);
                                    if (fix->ShouldSkip(TYPE(), value))
                                    {
                                        class Skip final : public FIXTURE<TYPE>::BaseFixture
                                        {
                                        public:
                                            virtual void SetUp() override
                                            {
                                                GTEST_SKIP();
                                            };
                                            virtual void TestBody() override
                                            {
                                                FAIL() << "Should not be executed";
                                            }
                                        };
                                        return new Skip;
                                    }
                                    else
                                    {
                                        return fix.release();
                                    }
                                });
    }
}

// Go through all parameters and register the tests.
template<template<class> class FIXTURE, size_t... IDX, class DATA>
void RegisterTests(const DATA &data, const char *casename, const char *testname, const char *file, int line,
                   std::index_sequence<IDX...>)
{
    int base = 0;
    (void)base;
    (RegisterTests<FIXTURE>(std::get<IDX>(data), casename, testname, file, line, base), ...);
}

template<class PARAMS>
struct GetUniqueTypes;

template<class... PARAM>
struct GetUniqueTypes<Types<PARAM...>>
{
    using type = type::Unique<Types<type::Transform<FilterOutValue, PARAM>...>>;
};

template<class T, class DEP>
struct MakeDependent
{
    using type = typename std::enable_if_t<sizeof(DEP *) != 0, T>;
};
} // namespace detail

#define NVCV_MIXTYPED_TEST_SUITE_F(CaseName, ...)                                                                   \
    using CaseName##_Types = typename ::nvcv::test::type::detail::WrapParams<__VA_ARGS__>::type;                    \
    using CaseName##_Tuple = ::nvcv::test::type::detail::ToTuple<::nvcv::test::type::GetType<CaseName##_Types, 0>>; \
    const auto CaseName##_TupleData = ::nvcv::test::type::detail::make_parameters_tuple<CaseName##_Tuple>(          \
        typename ::nvcv::test::type::detail::GetUniqueTypes<CaseName##_Types>::type(), CaseName##_Types());

#define NVCV_MIXTYPED_TEST_SUITE(CaseName, ...) \
    class CaseName : public ::testing::Test     \
    {                                           \
    };                                          \
    NVCV_MIXTYPED_TEST_SUITE_F(CaseName, __VA_ARGS__)

#define NVCV_MIXTYPED_TEST(CaseName, TestName)                                                                    \
    template<class T>                                                                                             \
    class CaseName##TestName##_Fixture final : public CaseName                                                    \
    {                                                                                                             \
    public:                                                                                                       \
        using BaseFixture = CaseName;                                                                             \
        using TypeParam   = T;                                                                                    \
        using Tuple       = CaseName##_Tuple;                                                                     \
        CaseName##TestName##_Fixture(const Tuple &p)                                                              \
            : m_params(p)                                                                                         \
        {                                                                                                         \
        }                                                                                                         \
        virtual void TestBody() override;                                                                         \
        template<int I>                                                                                           \
        using GetType = typename ::nvcv::test::type::detail::GetTypeImpl<TypeParam, I>::type;                     \
        template<int I>                                                                                           \
        auto &&GetValue() const                                                                                   \
        {                                                                                                         \
            static_assert(!std::is_same_v<std::tuple_element_t<I, Tuple>, ::nvcv::test::type::detail::TypeSlot>,  \
                          "Test parameter isn't a value");                                                        \
            return std::get<I>(m_params);                                                                         \
        }                                                                                                         \
                                                                                                                  \
    private:                                                                                                      \
        template<class C>                                                                                         \
        static constexpr std::true_type parentHasShouldSkip(decltype(&C::ShouldSkip));                            \
        template<class C>                                                                                         \
        static constexpr std::true_type parentHasShouldSkip(decltype(&C::template ShouldSkip<TypeParam, Tuple>)); \
        template<class C>                                                                                         \
        static constexpr std::false_type parentHasShouldSkip(...);                                                \
        static constexpr bool            ParentHasShouldSkip                                                      \
            = std::is_same_v<decltype(parentHasShouldSkip<CaseName>(nullptr)), std::true_type>;                   \
                                                                                                                  \
    public:                                                                                                       \
        template<class DUMMY = void>                                                                              \
        bool ShouldSkip(const TypeParam &a, const Tuple &b) const                                                 \
        {                                                                                                         \
            using AUX = typename ::nvcv::test::type::detail::MakeDependent<CaseName, DUMMY>::type;                \
            if constexpr (ParentHasShouldSkip)                                                                    \
            {                                                                                                     \
                return AUX::ShouldSkip(a, b);                                                                     \
            }                                                                                                     \
            else                                                                                                  \
            {                                                                                                     \
                return false;                                                                                     \
            }                                                                                                     \
        }                                                                                                         \
                                                                                                                  \
    private:                                                                                                      \
        const Tuple m_params;                                                                                     \
    };                                                                                                            \
    static auto dummy_##CaseName##TestName = []()                                                                 \
    {                                                                                                             \
        const auto &TupleData = CaseName##_TupleData;                                                             \
        ::nvcv::test::type::detail::RegisterTests<CaseName##TestName##_Fixture>(                                  \
            CaseName##_TupleData, #CaseName, #TestName, __FILE__, __LINE__,                                       \
            std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<decltype(TupleData)>>>());         \
        return nullptr;                                                                                           \
    }();                                                                                                          \
    template<class NVCV_TypeParam>                                                                                \
    void CaseName##TestName##_Fixture<NVCV_TypeParam>::TestBody()

} // namespace nvcv::test::type
