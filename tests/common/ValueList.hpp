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

#ifndef NVCV_TEST_COMMON_VALUELIST_HPP
#define NVCV_TEST_COMMON_VALUELIST_HPP

#include <gtest/gtest.h>
#include <util/HashMD5.hpp>

#include <algorithm>
#include <iomanip>
#include <list>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace nvcv::test {

namespace detail {

template<class T>
struct IsTuple : std::false_type
{
};

template<class... TT>
struct IsTuple<std::tuple<TT...>> : std::true_type
{
};

template<class T>
struct IsTuple<T &> : IsTuple<std::decay_t<T>>
{
};

template<class T>
struct IsTuple<T &&> : IsTuple<std::decay_t<T>>
{
};

inline auto JoinTuple()
{
    return std::tuple<>();
}

template<class T, class... TAIL>
auto JoinTuple(T a0, TAIL &&...as);

template<class... TT, class... TAIL>
auto JoinTuple(std::tuple<TT...> a0, TAIL &&...as)
{
    auto rest = JoinTuple(std::forward<TAIL>(as)...);
    return std::tuple_cat(std::move(a0), std::move(rest));
}

template<class T, class... TAIL>
auto JoinTuple(T a0, TAIL &&...as)
{
    return std::tuple_cat(std::make_tuple(std::move(a0)), JoinTuple(std::forward<TAIL>(as)...));
}

template<class T, class... TT>
auto Head(const std::tuple<T, TT...> &t)
{
    return std::get<0>(t);
}

template<size_t... IDX, class... TT>
auto TailImpl(std::index_sequence<IDX...>, const std::tuple<TT...> &t)
{
    return std::make_tuple(std::get<IDX + 1>(t)...);
}

template<class T, class... TT>
auto Tail(const std::tuple<T, TT...> &t)
{
    return TailImpl(std::make_index_sequence<sizeof...(TT)>(), t);
}

template<class T>
struct Identity
{
    using type = T;
};

template<class... TT>
inline std::tuple<> ExtractTuple(std::tuple<TT...> t)
{
    return {};
}

template<int N, int... TAIL, class... TT>
auto ExtractTuple(std::tuple<TT...> t)
{
    return JoinTuple(std::get<N>(t), ExtractTuple<TAIL...>(t));
}

template<int N, class T>
auto ExtractTuple(T t)
{
    static_assert(N == 0, "Index out of range");

    return std::make_tuple(t);
}

template<class T>
auto ExtractTuple(T t)
{
    return std::tuple<>();
}

template<int... NN, class T>
auto MaybeExtractTuple(T t)
{
    if constexpr (sizeof...(NN) == 0)
    {
        return JoinTuple(t);
    }
    else
    {
        return ExtractTuple<NN...>(std::move(t));
    }
}

struct Default
{
    bool operator<(const Default &that) const
    {
        return false;
    }

    bool operator==(const Default &that) const
    {
        return true;
    }
};

template<int IDX, class T, class U>
void ReplaceDefaultsImpl(U &out, const T &in)
{
    using SRC = typename std::tuple_element<IDX, T>::type;
    using DST = typename std::tuple_element<IDX, U>::type;

    if constexpr (!std::is_same_v<SRC, Default>)
    {
        std::get<IDX>(out) = static_cast<DST>(std::get<IDX>(in));
    }
}

template<class T, class U, size_t... IDX>
void ReplaceDefaultsImpl(U &out, const T &in, std::index_sequence<IDX...>)
{
    (ReplaceDefaultsImpl<IDX>(out, in), ...);
}

template<class... UU, class... TT>
requires(std::is_default_constructible_v<std::tuple<UU...>>) std::tuple<UU...> ReplaceDefaults(
    const std::tuple<TT...> &in)
{
    static_assert(sizeof...(TT) == sizeof...(UU));

    std::tuple<UU...> out;
    ReplaceDefaultsImpl(out, in, std::index_sequence_for<TT...>());
    return out;
}

template<class U, class... UU, class T, class... TT>
requires(!std::is_default_constructible_v<std::tuple<U, UU...>>) std::tuple<U, UU...> ReplaceDefaults(
    const std::tuple<T, TT...> &in)
{
    static_assert(sizeof...(TT) == sizeof...(UU));

    if constexpr (std::is_same_v<T, Default>)
    {
        static_assert(std::is_default_constructible_v<U>,
                      "Param type must have an explicit default value as it's not default constructible");

        return JoinTuple(U{}, ReplaceDefaults<UU...>(Tail(in)));
    }
    else
    {
        return JoinTuple(U{std::get<0>(in)}, ReplaceDefaults<UU...>(Tail(in)));
    }
}

} // namespace detail

template<class... TT>
class ValueList
{
public:
    using tuple_value_type = decltype(detail::JoinTuple(std::declval<TT>()...));

    using value_type =
        typename std::conditional_t<std::tuple_size_v<tuple_value_type> == 1, std::tuple_element<0, tuple_value_type>,
                                    detail::Identity<tuple_value_type>>::type;
    using list_type = std::list<value_type>;

    using iterator       = typename list_type::iterator;
    using const_iterator = typename list_type::const_iterator;

    ValueList() = default;

    ValueList(std::initializer_list<value_type> initList)
        : m_list(std::move(initList))
    {
    }

    template<class... UU>
    requires(!std::is_same_v<tuple_value_type, std::tuple<UU...>>) explicit ValueList(const ValueList<UU...> &that)
    {
        for (auto &v : that)
        {
            if constexpr (detail::IsTuple<value_type>::value)
            {
                m_list.emplace_back(detail::ReplaceDefaults<TT...>(detail::JoinTuple(v)));
            }
            else
            {
                m_list.emplace_back(std::get<0>(detail::ReplaceDefaults<TT...>(detail::JoinTuple(v))));
            }
        }
    }

    ValueList(const std::vector<value_type> &v)
    {
        m_list.insert(m_list.end(), v.begin(), v.end());
    }

    auto begin()
    {
        return m_list.begin();
    }

    auto end()
    {
        return m_list.end();
    }

    auto cbegin() const
    {
        return m_list.cbegin();
    }

    auto cend() const
    {
        return m_list.cend();
    }

    auto begin() const
    {
        return m_list.begin();
    }

    auto end() const
    {
        return m_list.end();
    }

    template<class A, class... AA>
    void emplace_back(A &&a0, AA &&...args)
    {
        m_list.emplace_back(detail::JoinTuple(std::forward<A>(a0), std::forward<AA>(args)...));
    }

    auto insert(const_iterator it, value_type v)
    {
        return m_list.insert(it, std::move(v));
    }

    auto push_front(value_type v)
    {
        return m_list.emplace_front(std::move(v));
    }

    auto push_back(value_type v)
    {
        return m_list.emplace_back(std::move(v));
    }

    template<class = void>
    requires(!std::is_same_v<tuple_value_type, value_type>) auto push_back(tuple_value_type v)
    {
        return std::apply([this](auto &...args) { m_list.emplace_back(args...); }, v);
    }

    void concat(ValueList &&other)
    {
        m_list.splice(m_list.end(), std::move(other.m_list));
    }

    void erase(iterator it)
    {
        m_list.erase(it);
    }

    void erase(iterator itbeg, iterator itend)
    {
        m_list.erase(itbeg, itend);
    }

    bool erase(value_type v)
    {
        bool removedAtLeastOne = false;

        for (auto it = m_list.begin(); it != m_list.end();)
        {
            it = std::find(it, m_list.end(), v);
            if (it != m_list.end())
            {
                m_list.erase(it++);
                removedAtLeastOne = true;
            }
        }

        return removedAtLeastOne;
    }

    bool operator==(const ValueList<TT...> &that) const
    {
        return m_list == that.m_list;
    }

    bool operator!=(const ValueList<TT...> &that) const
    {
        return m_list != that.m_list;
    }

    bool exists(const value_type &v) const
    {
        return std::find(m_list.begin(), m_list.end(), v) != m_list.end();
    }

    template<int... NN, class F, class... UU>
    friend ValueList<UU...> UniqueSort(F extractor, ValueList<UU...> a);

    size_t size() const
    {
        return m_list.size();
    }

private:
    list_type m_list;
};

template<class T>
ValueList(std::initializer_list<T>) -> ValueList<T>;

template<class... TT>
ValueList(std::initializer_list<std::tuple<TT...>>) -> ValueList<TT...>;

template<class T>
ValueList(const std::vector<T> &) -> ValueList<T>;

template<class... TT>
ValueList(const std::vector<std::tuple<TT...>> &) -> ValueList<TT...>;

template<class T>
ValueList<T> Value(T v)
{
    return {v};
}

inline ValueList<detail::Default> ValueDefault()
{
    return {detail::Default{}};
}

namespace detail {

template<class T>
struct NormalizeValueList;

template<class... TT>
struct NormalizeValueList<ValueList<TT...>>
{
    using type = ValueList<TT...>;
};

template<class... TT>
struct NormalizeValueList<ValueList<std::tuple<TT...>>>
{
    using type = ValueList<TT...>;
};

template<class T>
struct IsValueList : std::false_type
{
};

template<class... TT>
struct IsValueList<ValueList<TT...>> : std::true_type
{
};

} // namespace detail

template<class T>
constexpr bool IsValueList = detail::IsValueList<T>::value;

// UniqueSort ----------------------------

template<int... NN, class F, class... TT>
ValueList<TT...> UniqueSort(F extractor, ValueList<TT...> a)
{
    a.m_list.sort(
        [&extractor](const auto &a, const auto &b)
        {
            auto ta = apply(extractor, detail::MaybeExtractTuple<NN...>(a));
            auto tb = apply(extractor, detail::MaybeExtractTuple<NN...>(b));

            if (ta == tb)
            {
                return a < b;
            }
            else
            {
                return ta < tb;
            }
        });

    a.m_list.unique(
        [&extractor](const auto &a, const auto &b)
        {
            return apply(extractor, detail::MaybeExtractTuple<NN...>(a))
                == apply(extractor, detail::MaybeExtractTuple<NN...>(b));
        });

    return a;
}

template<int... NN, class... TT>
ValueList<TT...> UniqueSort(ValueList<TT...> a)
{
    return UniqueSort<NN...>([](auto... v) { return std::make_tuple(v...); }, std::move(a));
};

// Concat ----------------------------

template<class T>
ValueList<T> Concat(T v)
{
    return {v};
}

template<class... TT>
auto Concat(ValueList<TT...> v)
{
    return v;
}

template<class... TT>
ValueList<TT...> Concat(std::tuple<TT...> v)
{
    return v;
}

template<class... TT, class... TAIL>
ValueList<TT...> Concat(ValueList<TT...> head, TAIL &&...tail)
{
    typename detail::NormalizeValueList<ValueList<TT...>>::type rest = Concat(std::forward<TAIL>(tail)...);
    head.concat(std::move(rest));
    return head;
}

template<class T, class... TAIL>
ValueList<T> Concat(T v, TAIL &&...tail)
{
    ValueList<T> rest = Concat(std::forward<TAIL>(tail)...);

    rest.push_front(v);
    return rest;
}

// Difference ----------------------------

template<class... TT>
ValueList<TT...> Difference(ValueList<TT...> a, ValueList<TT...> b)
{
    b = UniqueSort(b);

    for (auto it = a.begin(); it != a.end();)
    {
        if (binary_search(b.begin(), b.end(), *it))
        {
            a.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return a;
}

template<class T>
ValueList<T> Difference(ValueList<T> a, T b)
{
    return Difference(a, ValueList<T>{b});
}

template<class T>
ValueList<T> Difference(T a, ValueList<T> b)
{
    return Difference(ValueList<T>{a}, b);
}

// SymmetricDifference  ----------------------------

template<class... TT>
ValueList<TT...> SymmetricDifference(ValueList<TT...> a, ValueList<TT...> b)
{
    return Concat(Difference(a, b), Difference(b, a));
}

template<class T>
ValueList<T> SymmetricDifference(ValueList<T> a, T b)
{
    return SymmetricDifference(a, ValueList<T>{b});
}

template<class T>
ValueList<T> SymmetricDifference(T a, ValueList<T> b)
{
    return SymmetricDifference(ValueList<T>{a}, b);
}

// Intersection  ----------------------------

template<class... TT, class... TAIL>
ValueList<TT...> Intersection(ValueList<TT...> a, ValueList<TT...> b, TAIL &&...tail)
{
    ValueList<TT...> tmp;

    if constexpr (sizeof...(TAIL) > 0)
    {
        tmp = Intersection(std::move(b), std::forward<TAIL>(tail)...);
    }
    else
    {
        tmp = std::move(b);
    }

    tmp = UniqueSort(tmp);

    for (auto it = a.begin(); it != a.end();)
    {
        if (binary_search(tmp.begin(), tmp.end(), *it))
        {
            ++it;
        }
        else
        {
            a.erase(it++);
        }
    }

    return a;
}

template<class T>
ValueList<T> Intersection(ValueList<T> a, T b)
{
    return Intersection(a, ValueList<T>{b});
}

template<class T>
ValueList<T> Intersection(T a, ValueList<T> b)
{
    return Intersection(ValueList<T>{a}, b);
}

// Transform ----------------------------

// xform can return either a single parameter (tuple/value),
// or a value list with multiple parameters, these will all be
// added to the output value list
template<class... TT, class F>
auto Transform(ValueList<TT...> in, F xform)
{
    using R = std::invoke_result_t<F, TT...>;

    typename detail::NormalizeValueList<std::conditional_t<IsValueList<R>, R, ValueList<R>>>::type out;
    for (const auto &v : in)
    {
        if constexpr (IsValueList<R>)
        {
            for (const auto &vv : std::apply(xform, detail::JoinTuple(v)))
            {
                out.push_back(vv);
            }
        }
        else
        {
            out.push_back(std::apply(xform, detail::JoinTuple(v)));
        }
    }

    return out;
}

// Make ----------------------------

namespace detail {

// Check if typename T::value_type exists.
template<class T, class EN = void>
struct HasValueType : std::false_type
{
};

template<class T>
struct HasValueType<T, std::enable_if_t<sizeof(typename T::value_type) != 0>> : std::true_type
{
};

} // namespace detail

template<class OUT, class... TT>
ValueList<OUT> Make(const ValueList<TT...> &in)
{
    ValueList<OUT> out;

    for (auto &v : in)
    {
        if constexpr (std::is_constructible_v<OUT, TT...>)
        {
            out.push_back(std::make_from_tuple<OUT>(detail::JoinTuple(v)));
        }
        else if constexpr (std::is_constructible_v<OUT, std::in_place_t, TT...>)
        {
            // inplace constructor, used with std::optional, std::any, std::variant, etc.
            out.push_back(std::apply([](TT &&...aa) { return OUT{std::in_place, aa...}; }, detail::JoinTuple(v)));
        }
        else if constexpr (detail::HasValueType<OUT>::value)
        {
            // For cases with std::optional and others, but the optional type doesn't have an explicit ctor
            out.push_back(
                std::apply([](TT &&...aa) { return OUT{typename OUT::value_type{aa...}}; }, detail::JoinTuple(v)));
        }
        else
        {
            out.push_back(std::apply([](TT &&...aa) { return OUT{aa...}; }, detail::JoinTuple(v)));
        }
    }

    return out;
}

// Combine ----------------------------

template<class... TT>
auto Combine(ValueList<TT...> a)
{
    return a;
}

template<class T>
ValueList<std::decay_t<T>> Combine(T &&v)
{
    return {v};
}

template<class... TT>
ValueList<std::decay_t<TT>...> Combine(TT &&...v)
{
    return {std::make_tuple(v...)};
}

template<class... TT, class... TAIL>
auto Combine(ValueList<TT...> a, TAIL &&...tail)
{
    auto rest = Combine(std::forward<TAIL>(tail)...);

    typename detail::NormalizeValueList<
        ValueList<decltype(tuple_cat(std::tuple<TT...>(), typename decltype(rest)::tuple_value_type()))>>::type r;

    for (auto ita = a.begin(); ita != a.end(); ++ita)
    {
        for (auto itr = rest.begin(); itr != rest.end(); ++itr)
        {
            r.push_back(detail::JoinTuple(*ita, *itr));
        }
    }

    return r;
}

template<class T, class... TAIL>
auto Combine(T &&a, TAIL &&...tail)
{
    return Combine(ValueList<std::decay_t<T>>{a}, std::forward<TAIL>(tail)...);
}

// Zip  ----------------------------

template<class... TT>
auto Zip(ValueList<TT...> a)
{
    return a;
}

template<class T>
ValueList<std::decay_t<T>> Zip(T &&v)
{
    return {v};
}

template<class... TT>
ValueList<std::decay_t<TT>...> Zip(TT &&...v)
{
    return {std::make_tuple(v...)};
}

template<class... TT, class... TAIL>
auto Zip(ValueList<TT...> a, TAIL &&...tail)
{
    auto rest = Zip(std::forward<TAIL>(tail)...);

    typename detail::NormalizeValueList<
        ValueList<decltype(tuple_cat(std::tuple<TT...>(), typename decltype(rest)::tuple_value_type()))>>::type r;

    if (a.size() == rest.size())
    {
        for (auto ita = a.begin(), itr = rest.begin(); ita != a.end(); ++ita, ++itr)
        {
            r.push_back(detail::JoinTuple(*ita, *itr));
        }
    }
    else
    {
        throw std::logic_error("Zip: value lists can't have different sizes");
    }

    return r;
}

template<class T, class... TAIL>
auto Zip(T &&a, TAIL &&...tail)
{
    return Zip(ValueList<T>{a}, std::forward<TAIL>(tail)...);
}

struct IsSameArgsFunctor
{
private:
    template<class T>
    static bool isSameArgs(T &&)
    {
        return false;
    }

    template<class T, class U>
    static bool isSameArgs(T &&a, U &&b)
    {
        return a == b;
    }

    template<class T, class U, class... TAIL>
    static bool isSameArgs(T &&a, U &&b, TAIL &&...tail)
    {
        if (a == b)
        {
            return isSameArgs(b, tail...);
        }
        else
        {
            return false;
        }
    }

public:
    template<class... TT>
    bool operator()(TT &&...args) const
    {
        return isSameArgs(std::forward<TT>(args)...);
    }
};

inline const IsSameArgsFunctor IsSameArgs;

template<bool MATCH_ALL, class SEQ, class... TT>
struct MatchHelper
{
    static_assert(SEQ::size() == sizeof...(TT));

private:
    // Termination criteria, nothing more to check.
    template<int IDX, class T, class U>
    static bool match(std::integer_sequence<int>, const T &item, const U &needle)
    {
        static_assert(IDX == SEQ::size());
        return true;
    }

    template<int IDX, int N, int... NTAIL, class T, class U>
    bool match(std::integer_sequence<int, N, NTAIL...>, const T &item, const U &needle) const
    {
        return std::get<IDX>(needle) == std::get<N>(detail::JoinTuple(item))
            && match<IDX + 1>(std::integer_sequence<int, NTAIL...>(), item, needle);
    }

    ValueList<TT...> m_needle;

public:
    MatchHelper(TT &&...needle)
        : m_needle({typename ValueList<TT...>::value_type{std::forward<TT>(needle)...}})
    {
    }

    // Matches any value in needle.
    MatchHelper(ValueList<TT...> needle)
        : m_needle(std::move(needle))
    {
    }

    template<class... UU>
    bool operator()(UU &&...args) const
    {
        static_assert(!MATCH_ALL || sizeof...(UU) == sizeof...(TT), "Match arity must match list arity");

        if constexpr (SEQ::size() >= 1)
        {
            for (const auto &m : m_needle)
            {
                if (match<0>(SEQ(), std::make_tuple(std::forward<UU>(args)...), detail::JoinTuple(m)))
                {
                    return true;
                }
            }
            return false;
        }
        else
        {
            return false;
        }
    }
};

template<int N, int... NN, class... TT>
auto Match(TT &&...args)
{
    return MatchHelper<false, std::integer_sequence<int, N, NN...>, TT...>(std::forward<TT>(args)...);
}

template<int N, int... NN, class... TT>
auto Match(ValueList<TT...> list)
{
    return MatchHelper<false, std::integer_sequence<int, N, NN...>, TT...>(std::move(list));
}

template<class... TT>
auto Match(ValueList<TT...> list)
{
    return MatchHelper<true, std::make_integer_sequence<int, sizeof...(TT)>, TT...>(std::move(list));
}

template<class... TT>
auto Match(TT &&...args)
{
    return MatchHelper<true, std::make_integer_sequence<int, sizeof...(TT)>, TT...>(std::forward<TT>(args)...);
}

// RemoveIf<indices> ---------------------------------------------------------
// Remove a parameter from list if a subset of args given by indices match criteria.
template<int I, int... II, class... TT, class F>
ValueList<TT...> RemoveIf(F criteria, test::ValueList<TT...> list)
{
    for (auto it = list.begin(); it != list.end();)
    {
        bool mustRemove;

        if constexpr (detail::IsTuple<decltype(*it)>::value)
        {
            mustRemove = std::apply(criteria, detail::ExtractTuple<I, II...>(*it));
        }
        else
        {
            mustRemove = criteria(*it);
        }

        if (mustRemove)
        {
            list.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return list;
}

// RemoveIf ---------------------------------------------------------
// Remove a parameter from list if it match criteria.
template<class... TT, class F>
ValueList<TT...> RemoveIf(F criteria, test::ValueList<TT...> list)
{
    for (auto it = list.begin(); it != list.end();)
    {
        bool mustRemove;

        if constexpr (detail::IsTuple<decltype(*it)>::value)
        {
            mustRemove = std::apply(criteria, *it);
        }
        else
        {
            mustRemove = criteria(*it);
        }

        if (mustRemove)
        {
            list.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return list;
}

// RemoveIfAny ---------------------------------------------------------
// Remove a parameter from list if any (one or more) arg matches criteria

namespace detail {
template<int... II, class F, class... TT>
bool MustRemoveAny(std::integer_sequence<int, II...>, const F &criteria, const std::tuple<TT...> &v)
{
    static_assert(sizeof...(II) == sizeof...(TT));

    return (false || ... || criteria(std::get<II>(v)));
}
} // namespace detail

template<class... TT, class F>
ValueList<TT...> RemoveIfAny(const F &criteria, test::ValueList<TT...> list)
{
    for (auto it = list.begin(); it != list.end();)
    {
        bool mustRemove;
        if constexpr (detail::IsTuple<decltype(*it)>::value)
        {
            mustRemove = detail::MustRemoveAny(std::make_integer_sequence<int, sizeof...(TT)>(), criteria, *it);
        }
        else
        {
            mustRemove = criteria(*it);
        }

        if (mustRemove)
        {
            list.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return list;
}

// RemoveIfAll ---------------------------------------------
// Remove a parameter from list if each arg matches criteria

namespace detail {
template<int... II, class F, class... TT>
bool MustRemoveAll(std::integer_sequence<int, II...>, const F &criteria, const std::tuple<TT...> &v)
{
    static_assert(sizeof...(II) == sizeof...(TT));

    return (true && ... && criteria(std::get<II>(v)));
}
} // namespace detail

template<class... TT, class F>
ValueList<TT...> RemoveIfAll(const F &criteria, test::ValueList<TT...> list)
{
    for (auto it = list.begin(); it != list.end();)
    {
        bool mustRemove;
        if constexpr (detail::IsTuple<decltype(*it)>::value)
        {
            mustRemove = detail::MustRemoveAll(std::make_integer_sequence<int, sizeof...(TT)>(), criteria, *it);
        }
        else
        {
            mustRemove = criteria(*it);
        }

        if (mustRemove)
        {
            list.erase(it++);
        }
        else
        {
            ++it;
        }
    }

    return list;
}

// SelectIfAny ---------------------------------------------
// Selects a parameter from list if it matches criteria
template<int... II, class... TT, class F>
ValueList<TT...> SelectIf(const F &criteria, test::ValueList<TT...> list)
{
    return RemoveIf<II...>(std::not_fn(criteria), std::move(list));
}

// SelectIfAny ---------------------------------------------
// Selects a parameter from list if any (one or more) arg matches criteria
template<int... II, class... TT, class F>
ValueList<TT...> SelectIfAny(const F &criteria, test::ValueList<TT...> list)
{
    return RemoveIfAll<II...>(std::not_fn(criteria), std::move(list));
}

// SelectIfAll ---------------------------------------------
// Selects a parameter from list if each arg matches criteria
template<int... II, class... TT, class F>
ValueList<TT...> SelectIfAll(const F &criteria, test::ValueList<TT...> list)
{
    return RemoveIfAny<II...>(std::not_fn(criteria), std::move(list));
}

// Create a ValueList of the Nth parameters

template<int... NN, class... TT>
auto Extract(const ValueList<TT...> &v)
{
    using DestTuple = decltype(detail::ExtractTuple<NN...>(detail::JoinTuple(*v.begin())));

    typename detail::NormalizeValueList<ValueList<DestTuple>>::type out;

    for (auto &src : v)
    {
        if constexpr (sizeof...(NN) == 1)
        {
            out.push_back(std::get<0>(detail::ExtractTuple<NN...>(detail::JoinTuple(src))));
        }
        else
        {
            out.push_back(detail::ExtractTuple<NN...>(detail::JoinTuple(src)));
        }
    }

    return out;
}

template<class F>
auto Not(F fn)
{
    return std::not_fn(std::move(fn));
}

namespace detail {
template<class LHS, class RHS>
struct Or
{
    LHS lhs;
    RHS rhs;

    template<class... Args>
    constexpr auto operator()(Args &&...args) &noexcept(noexcept(std::invoke(lhs, args...)
                                                                 || std::invoke(rhs, std::forward<Args>(args)...)))
    {
        return std::invoke(lhs, args...) || std::invoke(rhs, std::forward<Args>(args)...);
    }

    template<class... Args>
    constexpr auto operator()(Args &&...args) const &noexcept(
        noexcept(std::invoke(lhs, args...) || std::invoke(rhs, std::forward<Args>(args)...)))
    {
        return std::invoke(lhs, args...) || std::invoke(rhs, std::forward<Args>(args)...);
    }

    template<class... Args>
    constexpr auto operator()(Args &&...args) &&noexcept(
        noexcept(std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...)))
    {
        return std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...);
    }

    template<class... Args>
    constexpr auto operator()(Args &&...args) const &&noexcept(
        noexcept(std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...)))
    {
        return std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...);
    }
};
} // namespace detail

template<class LHS, class RHS>
constexpr detail::Or<std::decay_t<LHS>, std::decay_t<RHS>> Or(LHS &&lhs, RHS &&rhs)
{
    return {std::forward<LHS>(lhs), std::forward<RHS>(rhs)};
}

namespace detail {
template<class LHS, class RHS>
struct And
{
    LHS lhs;
    RHS rhs;

    template<class... Args>
    constexpr auto operator()(Args &&...args) &noexcept(noexcept(std::invoke(lhs, args...)
                                                                 || std::invoke(rhs, std::forward<Args>(args)...)))
    {
        return std::invoke(lhs, args...) && std::invoke(rhs, std::forward<Args>(args)...);
    }

    template<class... Args>
    constexpr auto operator()(Args &&...args) const &noexcept(
        noexcept(std::invoke(lhs, args...) || std::invoke(rhs, std::forward<Args>(args)...)))
    {
        return std::invoke(lhs, args...) && std::invoke(rhs, std::forward<Args>(args)...);
    }

    template<class... Args>
    constexpr auto operator()(Args &&...args) &&noexcept(
        noexcept(std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...)))
    {
        return std::invoke(std::move(lhs), args...) && std::invoke(std::move(rhs), std::forward<Args>(args)...);
    }

    template<class... Args>
    constexpr auto operator()(Args &&...args) const &&noexcept(
        noexcept(std::invoke(std::move(lhs), args...) || std::invoke(std::move(rhs), std::forward<Args>(args)...)))
    {
        return std::invoke(std::move(lhs), args...) && std::invoke(std::move(rhs), std::forward<Args>(args)...);
    }
};
} // namespace detail

template<class LHS, class RHS>
constexpr detail::And<std::decay_t<LHS>, std::decay_t<RHS>> And(LHS &&lhs, RHS &&rhs)
{
    return {std::forward<LHS>(lhs), std::forward<RHS>(rhs)};
}

template<class... TT>
ValueList<TT...> operator-(const ValueList<TT...> &a, const ValueList<TT...> &b)
{
    return Difference(a, b);
}

template<class T>
ValueList<T> operator-(const ValueList<T> &a, const T &b)
{
    return Difference(a, b);
}

template<class T>
ValueList<T> operator-(const T &a, const ValueList<T> &b)
{
    return Difference(a, b);
}

template<class... TT, class F>
ValueList<TT...> operator-(const ValueList<TT...> &a, const F &fn)
{
    return RemoveIf(fn, a);
}

template<class... TT>
ValueList<TT...> operator|(const ValueList<TT...> &a, const ValueList<TT...> &b)
{
    return Concat(a, b);
}

template<class T>
ValueList<T> operator|(const ValueList<T> &a, const T &b)
{
    return Concat(a, b);
}

template<class T>
ValueList<T> operator|(const T &a, const ValueList<T> &b)
{
    return Concat(a, b);
}

template<class... TT>
ValueList<TT...> operator&(const ValueList<TT...> &a, const ValueList<TT...> &b)
{
    return Intersection(a, b);
}

template<class T>
ValueList<T> operator&(const ValueList<T> &a, const T &b)
{
    return Intersection(a, ValueList<T>{b});
}

template<class T>
ValueList<T> operator&(const T &a, const ValueList<T> &b)
{
    return Intersection(ValueList<T>{a}, b);
}

template<class... TT, class... UU>
auto operator*(const ValueList<TT...> &a, const ValueList<UU...> &b)
{
    return Combine(a, b);
}

template<class... TT, class U>
auto operator*(const ValueList<TT...> &a, const U &b)
{
    return Combine(a, b);
}

template<class T, class... UU>
auto operator*(const T &a, const ValueList<UU...> &b)
{
    return Combine(a, b);
}

template<class... TT, class... UU>
auto operator%(const ValueList<TT...> &a, const ValueList<UU...> &b)
{
    return Zip(a, b);
}

template<class... TT, class... UU>
auto operator^(const ValueList<TT...> &a, const ValueList<UU...> &b)
{
    return SymmetricDifference(a, b);
}

template<class... TT, class U>
auto operator^(const ValueList<TT...> &a, const U &b)
{
    return SymmetricDifference(a, b);
}

template<class T, class... UU>
auto operator^(const T &a, const ValueList<UU...> &b)
{
    return SymmetricDifference(a, b);
}

template<size_t... IDX, class T>
auto DupImpl(const ValueList<T> &v, std::index_sequence<IDX...>)
{
    return Extract<(IDX * 0)...>(v);
}

// Duplicates the value N times
template<int N, class T>
auto Dup(const ValueList<T> &v)
{
    return DupImpl(v, std::make_index_sequence<N>());
}

} // namespace nvcv::test

#endif // NVCV_TEST_COMMON_VALUELIST_HPP
