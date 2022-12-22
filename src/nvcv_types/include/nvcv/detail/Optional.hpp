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

#ifndef NVCV_DETAIL_OPTIONAL_HPP
#define NVCV_DETAIL_OPTIONAL_HPP

// C++>=17 ?
#if __cplusplus >= 201703L
#    include <new> // for std::launder
#endif

#include "InPlace.hpp"

namespace nvcv { namespace detail {

struct NullOptT
{
};

constexpr NullOptT NullOpt;

template<class T>
class Optional
{
public:
    using value_type = T;

    Optional() noexcept
        : m_hasValue(false)
    {
    }

    Optional(NullOptT) noexcept
        : Optional()
    {
    }

    Optional(const Optional &that)
        : m_hasValue(that.m_hasValue)
    {
        if (m_hasValue)
        {
            new (&m_storage) T(that.value());
        }
    }

    Optional(Optional &&that) noexcept(std::is_nothrow_move_constructible<T>::value)
        : m_hasValue(that.m_hasValue)
    {
        if (m_hasValue)
        {
            new (&m_storage) T(std::move(that.value()));
            // do not set that.m_hasValue to false as per c++17 standard.
        }
    }

    template<class U, typename std::enable_if<std::is_constructible<T, U &&>::value
                                                  && !std::is_same<typename std::decay<U>::type, InPlaceT>::value
                                                  && !std::is_same<typename std::decay<U>::type, Optional<U>>::value,
                                              int>::type
                      = 0>
    Optional(U &&that)
        : m_hasValue(true)
    {
        new (&m_storage) T(std::forward<U>(that));
    }

    template<class... AA, typename std::enable_if<std::is_constructible<T, AA...>::value, int>::type = 0>
    Optional(InPlaceT, AA &&...args)
        : m_hasValue(true)
    {
        new (&m_storage) T(std::forward<AA>(args)...);
    }

    ~Optional()
    {
        if (m_hasValue)
        {
            this->value().~T();
        }
    }

    Optional &operator=(NullOptT) noexcept
    {
        if (m_hasValue)
        {
            this->value().~T();
            m_hasValue = false;
        }
    }

    Optional &operator=(const Optional &that)
    {
        if (that.m_hasValue)
        {
            if (m_hasValue)
            {
                this->value() = that.value();
            }
            else
            {
                new (&m_storage) T(that.value());
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

    Optional &operator=(Optional &&that)
    {
        if (that.m_hasValue)
        {
            if (m_hasValue)
            {
                this->value() = std::move(that.value());
            }
            else
            {
                new (&m_storage) T(std::move(that.value()));
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

    template<class... AA, typename std::enable_if<std::is_constructible<T, AA...>::value, int>::type = 0>
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

    void reset() noexcept
    {
        if (m_hasValue)
        {
            this->value().~T();
            m_hasValue = false;
        }
    }

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

    bool hasValue() const
    {
        return m_hasValue;
    }

    explicit operator bool() const
    {
        return m_hasValue;
    }

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
bool operator==(const Optional<T> &a, nullptr_t)
{
    return !a;
}

template<class T>
bool operator==(nullptr_t, const Optional<T> &b)
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
bool operator!=(const Optional<T> &a, nullptr_t)
{
    return !(a == nullptr);
}

template<class T>
bool operator!=(nullptr_t, const Optional<T> &b)
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

}} // namespace nvcv::detail

#endif // NVCV_DETAIL_OPTIONAL_HPP
