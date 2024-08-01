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

#ifndef NVCV_OPTIONAL_HPP
#define NVCV_OPTIONAL_HPP

// C++>=17 ?
#if __cplusplus >= 201703L
#    include <new> // for std::launder
#endif

#include "detail/InPlace.hpp"
#include "detail/TypeTraits.hpp"

#include <cassert>
#include <cstddef> // for std::nullptr_t
#include <stdexcept>
#include <type_traits>
#include <utility> // for std::move, std::forward

namespace nvcv {

struct NullOptT
{
};

constexpr NullOptT NullOpt;

/**
 * @brief A container object that may or may not contain a value of a given type.
 *
 * This is a simplified version of the `std::optional` type introduced in C++17.
 * It provides a mechanism to represent non-value states without resorting to
 * pointers, dynamic allocation, or custom 'null' values.
 *
 * @tparam T The type of the value to be stored.
 */
template<class T>
class Optional
{
public:
    using value_type = T;

    /// @brief Default constructor that initializes an empty `Optional`
    Optional() noexcept
        : m_hasValue(false)
    {
    }

    /// @brief Constructs an empty `Optional` using the specified `NullOptT` tag.
    Optional(NullOptT) noexcept
        : Optional()
    {
    }

    /// @brief Copy constructor.
    /// If the other `Optional` contains a value, it will be copied to this `Optional`.
    Optional(const Optional &that)
        : m_hasValue(that.m_hasValue)
    {
        if (m_hasValue)
        {
            new (&m_storage) T(that.value());
        }
    }

    /// @brief Move constructor.
    Optional(Optional &&that) noexcept(std::is_nothrow_move_constructible<T>::value)
        : m_hasValue(that.m_hasValue)
    {
        if (m_hasValue)
        {
            new (&m_storage) T(std::move(that.value()));
            // do not set that.m_hasValue to false as per c++17 standard.
        }
    }

    /**
     * @brief Constructs an `Optional` object by copying the contents of another `Optional` of a different type.
     *
     * This constructor allows for converting between `Optional` objects of different types,
     * provided that the contained type `T` can be constructed from the type `U` of the source `Optional`.
     *
     * @tparam U The contained type of the source `Optional`.
     * @tparam std::is_constructible<T, const U &>::value Ensures that this constructor is only available
     *         if `T` can be constructed from a constant reference to `U`.
     * @param that The source `Optional` object to be copied from.
     */
    template<typename U, detail::EnableIf_t<std::is_constructible<T, const U &>::value, int> = 0>
    Optional(const Optional<U> &that)
        : m_hasValue(that.m_hasValue)
    {
        if (m_hasValue)
        {
            new (&m_storage) T(that.value());
        }
    }

    /**
     * @brief Constructs an `Optional` object by moving the contents of another `Optional` of a different type.
     *
     * This constructor allows for converting between `Optional` objects of different types using move semantics,
     * provided that the contained type `T` can be constructed from the type `U` of the source `Optional` using move construction.
     *
     * It is marked `noexcept` if the move construction of `T` from `U` does not throw exceptions.
     *
     * After the move, the source `Optional` retains its state (i.e., whether it has a value or not) as per the C++17 standard.
     *
     * @tparam U The contained type of the source `Optional`.
     * @tparam std::is_constructible<T, U &&>::value Ensures that this constructor is only available
     *         if `T` can be move-constructed from type `U`.
     * @param that The source `Optional` object to be moved from.
     */
    template<typename U, detail::EnableIf_t<std::is_constructible<T, U &&>::value, int> = 0>
    Optional(Optional<U> &&that) noexcept(std::is_nothrow_constructible<T, U &&>::value)
        : m_hasValue(that.m_hasValue)
    {
        if (m_hasValue)
        {
            new (&m_storage) T(std::move(that.value()));
            // do not set that.m_hasValue to false as per c++17 standard.
        }
    }

    /**
     * @brief Constructs an `Optional` object with a contained value by forwarding the given argument.
     *
     * This constructor is designed for direct value initialization of the `Optional` object's contained value.
     * It employs perfect forwarding to ensure efficiency and flexibility, allowing for both lvalue and rvalue arguments.
     *
     * Importantly, this constructor is selectively disabled in the following scenarios:
     * 1. When the passed argument is of type `detail::InPlaceT`, which is used to signal in-place construction.
     * 2. When the passed argument is another `Optional` object, preventing accidental nesting of `Optional` objects.
     *
     * The use of SFINAE (`detail::EnableIf_t`) ensures that this constructor is only available under appropriate conditions.
     *
     * @tparam U The type of the argument used to initialize the contained value.
     * @tparam std::is_constructible<T, U &&>::value Ensures that the contained type `T` can be constructed from the argument `U`.
     * @tparam !std::is_same<typename std::decay<U>::type, detail::InPlaceT>::value Ensures that the constructor is disabled when the argument is `detail::InPlaceT`.
     * @tparam !std::is_same<typename std::decay<U>::type, Optional<U>>::value Ensures that the constructor is disabled when the argument is another `Optional` object.
     * @param that The argument used to initialize the `Optional` object's contained value.
     */
    template<class U, detail::EnableIf_t<std::is_constructible<T, U &&>::value
                                             && !std::is_same<typename std::decay<U>::type, detail::InPlaceT>::value
                                             && !std::is_same<typename std::decay<U>::type, Optional<U>>::value,
                                         int> = 0>
    Optional(U &&that)
        : m_hasValue(true)
    {
        new (&m_storage) T(std::forward<U>(that));
    }

    /**
     * @brief Constructs an `Optional` object with an in-place constructed contained value.
     *
     * This constructor is designed for in-place construction of the `Optional` object's contained value.
     * The purpose of in-place construction is to construct the value directly within the storage of the `Optional` object,
     * eliminating the need for temporary objects and providing better performance in certain scenarios.
     *
     * The `detail::InPlaceT` tag is used to distinguish this constructor from other overloads and to signal in-place construction.
     *
     * The use of SFINAE (`detail::EnableIf_t`) ensures that this constructor is only available when the contained type `T` can be constructed
     * from the provided argument pack `AA...`.
     *
     * @tparam AA The types of the arguments used for in-place construction of the contained value.
     * @tparam std::is_constructible<T, AA...>::value Ensures that the contained type `T` can be constructed from the argument pack `AA...`.
     * @param std::ignore A tag that indicates in-place construction.
     * @param args... The arguments used for in-place construction of the `Optional` object's contained value.
     */
    template<class... AA, detail::EnableIf_t<std::is_constructible<T, AA...>::value, int> = 0>
    Optional(detail::InPlaceT, AA &&...args)
        : m_hasValue(true)
    {
        new (&m_storage) T(std::forward<AA>(args)...);
    }

    // Dtor
    ~Optional()
    {
        if (m_hasValue)
        {
            this->value().~T();
        }
    }

    /// Comparison operators below
    Optional &operator=(NullOptT) noexcept
    {
        if (m_hasValue)
        {
            this->value().~T();
            m_hasValue = false;
        }
        return *this;
    }

    template<typename U>
    detail::EnableIf_t<std::is_assignable<T &, const U &>::value, Optional &> operator=(const Optional<U> &that)
    {
        if (that.hasValue())
        {
            if (m_hasValue)
            {
                this->value() = that.value();
            }
            else
            {
                new (&m_storage) T(that.value());
                m_hasValue = true;
            }
        }
        else
        {
            if (m_hasValue)
            {
                this->value().~T();
                m_hasValue = false;
            }
        }
        return *this;
    }

    template<typename U>
    detail::EnableIf_t<std::is_assignable<T &, U &&>::value, Optional &> operator=(Optional<U> &&that)
    {
        if (that.hasValue())
        {
            if (m_hasValue)
            {
                this->value() = std::move(that.value());
            }
            else
            {
                new (&m_storage) T(std::move(that.value()));
                m_hasValue = true;
            }
            // do not set that.m_hasValue to false as per c++17 standard.
        }
        else
        {
            if (m_hasValue)
            {
                this->value().~T();
                m_hasValue = false;
            }
        }
        return *this;
    }

    // copy/move assignment
    Optional &operator=(const Optional &that)
    {
        return this->operator=<T>(that);
    }

    Optional &operator=(Optional &&that)
    {
        return this->operator=<T>(std::move(that));
    }

    Optional &operator=(const T &value)
    {
        if (m_hasValue)
        {
            this->value() = value;
        }
        else
        {
            new (&m_storage) T(value);
            m_hasValue = true;
        }
        return *this;
    }

    Optional &operator=(T &&value)
    {
        if (m_hasValue)
        {
            this->value() = std::move(value);
        }
        else
        {
            new (&m_storage) T(std::move(value));
            m_hasValue = true;
        }
        return *this;
    }

    template<class... AA, detail::EnableIf_t<std::is_constructible<T, AA...>::value, int> = 0>
    T &emplace(AA &&...args)
    {
        T *p;
        if (m_hasValue)
        {
            this->value().~T();
            p = new (&m_storage) T(std::forward<AA>(args)...);
        }
        else
        {
            p          = new (&m_storage) T(std::forward<AA>(args)...);
            m_hasValue = true;
        }
        return *p;
    }

    /**
     * @brief Resets the `Optional` to its default (empty) state.
     *
     * If the `Optional` contains a value, the value is destroyed and
     * the `Optional` becomes empty.
     */
    void reset() noexcept
    {
        if (m_hasValue)
        {
            this->value().~T();
            m_hasValue = false;
        }
    }

    /**
     * @brief Swaps the contents of two `Optional` objects.
     *
     * If both objects have values, their values are swapped.
     * If only one object has a value, the value is moved to the other object.
     *
     * @param that Another `Optional` object of the same type.
     */
    void swap(Optional &that)
    {
        if (m_hasValue && that.m_hasValue)
        {
            using std::swap;
            swap(this->value() && that.value());
        }
        else if (!m_hasValue && !that.m_hasValue)
        {
            return;
        }
        else
        {
            Optional *a, *b;
            if (m_hasValue)
            {
                a = this;
                b = &that;
            }
            else
            {
                assert(that.m_hasValue);
                a = &that;
                b = this;
            }
            new (&b->m_storage) T(std::move(a->value()));
            a->value().~T();
            a->m_hasValue = false;
            b->m_hasValue = true;
        }
    }

    /**
     * @brief Checks if the `Optional` contains a value.
     * @return true if the `Optional` has a value, false otherwise.
     */
    bool hasValue() const
    {
        return m_hasValue;
    }

    /**
     * @brief Conversion to bool that checks if the `Optional` contains a value.
     * @return true if the `Optional` has a value, false otherwise.
     */
    explicit operator bool() const
    {
        return m_hasValue;
    }

    /**
     * @brief Returns the contained value.
     * @return A reference to the contained value.
     * @throws std::runtime_error If the `Optional` does not contain a value.
     */
    T &value()
    {
        if (!m_hasValue)
        {
            throw std::runtime_error("Bad optional access");
        }

        T *p = reinterpret_cast<T *>(&m_storage);
#if __cplusplus >= 201703L
        return *std::launder(p);
#else
        return *p;
#endif
    }

    /**
     * @brief Returns the contained value (const version).
     * @return A const reference to the contained value.
     * @throws std::runtime_error If the `Optional` does not contain a value.
     */
    const T &value() const
    {
        if (!m_hasValue)
        {
            throw std::runtime_error("Bad optional access");
        }

        const T *p = reinterpret_cast<const T *>(&m_storage);
#if __cplusplus >= 201703L
        return *std::launder(p);
#else
        return *p;
#endif
    }

    T *operator->()
    {
        return &this->value();
    }

    const T *operator->() const
    {
        return &this->value();
    }

    T &operator*()
    {
        return this->value();
    }

    const T &operator*() const
    {
        return this->value();
    }

private:
    bool                                                       m_hasValue;
    typename std::aligned_storage<sizeof(T), alignof(T)>::type m_storage;
};

template<class T>
bool operator==(const Optional<T> &a, const Optional<T> &b)
{
    if (a && b)
    {
        return *a == b;
    }
    else if (!a && !b)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template<class T>
bool operator==(const Optional<T> &a, NullOptT)
{
    return !a;
}

template<class T>
bool operator==(NullOptT, const Optional<T> &b)
{
    return !b;
}

template<class T>
bool operator==(const Optional<T> &a, std::nullptr_t)
{
    return !a;
}

template<class T>
bool operator==(std::nullptr_t, const Optional<T> &b)
{
    return !b;
}

template<class T>
bool operator==(const Optional<T> &a, const T &b)
{
    return a && *a == b;
}

template<class T>
bool operator==(const T &a, const Optional<T> &b)
{
    return b && a == *b;
}

template<class T>
bool operator!=(const Optional<T> &a, const Optional<T> &b)
{
    return !(a == b);
}

template<class T>
bool operator!=(const Optional<T> &a, std::nullptr_t)
{
    return !(a == nullptr);
}

template<class T>
bool operator!=(std::nullptr_t, const Optional<T> &b)
{
    return !(nullptr == b);
}

template<class T>
bool operator!=(const Optional<T> &a, const T &b)
{
    return !(a == b);
}

template<class T>
bool operator!=(const T &a, const Optional<T> &b)
{
    return !(a == b);
}

} // namespace nvcv

#endif // NVCV_OPTIONAL_HPP
