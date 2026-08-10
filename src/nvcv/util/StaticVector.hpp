/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_UTIL_STATICVECTOR_HPP
#define NVCV_UTIL_STATICVECTOR_HPP

#include "Assert.h"

#include <algorithm>
#include <cstddef> // for std::byte
#include <initializer_list>
#include <iterator> // for std::reverse_iterator
#include <memory>
#include <new> // for std::bad_alloc
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nvcv::util {

class StaticVectorError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// We want StaticVector<T> to have the same characteristics of
// T, i.e., if T isn't copiable, so isn't StaticVector<T>. In order
// to accomplish it, we have this StaticVectorHelper with a base
// definition, and then partially specialized to handle cases
// where T is copy-constructible and/or copy-assignable.

// Base case, class is move constructible and assignable
template<class T, int N, bool IS_COPY_CONSTRUCTIBLE = std::is_copy_constructible_v<T>,
         bool IS_COPY_ASSIGNABLE = std::is_copy_assignable_v<T>>
class StaticVectorHelper // NOSONAR: small-vector helper keeps STL-like operations together.
{
    static_assert(N >= 0, "StaticVector capacity can't be negative!");

public:
    using value_type      = T;
    using size_type       = size_t;
    using reference       = T &;
    using const_reference = const T &;

    StaticVectorHelper()
        : m_size(0)
    {
        doCheckInvariants();
    }

    explicit StaticVectorHelper(size_t count)
    {
        if (count > static_cast<size_t>(N))
        {
            throw std::bad_alloc();
        }

        m_size = static_cast<int>(count);

        std::uninitialized_default_construct( // NOSONAR: std::ranges overload is C++20.
            this->begin(), this->end());

        doCheckInvariants();
    }

    template<class IT, class = typename std::iterator_traits<IT>::pointer>
    explicit StaticVectorHelper(IT beg, IT end)
    {
        NVCV_ASSERT(beg != nullptr);
        NVCV_ASSERT(end != nullptr);

        auto count = std::distance(beg, end);
        NVCV_ASSERT(count >= 0);

        if (count > static_cast<decltype(count)>(N))
        {
            throw std::bad_alloc();
        }

        m_size = static_cast<int>(count);

        std::uninitialized_copy(beg, end, this->begin());

        doCheckInvariants();
    }

    explicit StaticVectorHelper(size_t count, const T &value)
    {
        if (count > static_cast<size_t>(N))
        {
            throw std::bad_alloc();
        }

        m_size = static_cast<int>(count);

        std::uninitialized_fill(this->begin(), this->end(), value); // NOSONAR: std::ranges overload is C++20.

        doCheckInvariants();
    }

    ~StaticVectorHelper()
    {
        std::destroy(this->begin(), this->end()); // NOSONAR: std::ranges overload is C++20.
    }

    StaticVectorHelper(StaticVectorHelper &&that) noexcept
        : StaticVectorHelper()
    {
        if constexpr (std::is_trivially_copyable_v<T>)
        {
            std::copy(that.begin(), that.end(), this->begin()); // NOSONAR: std::ranges overload is C++20.
        }
        else
        {
            std::uninitialized_move(that.begin(), that.end(), this->end()); // NOSONAR: std::ranges overload is C++20.
        }
        m_size = static_cast<int>(that.size());

        // not setting that's size to 0 on purpose
        // Since we didn't allocate memory for the vector, we're effectively
        // using std::optional's criteria for moving.

        doCheckInvariants();
    }

    StaticVectorHelper &operator=(StaticVectorHelper &&that) noexcept
    {
        if (this != &that)
        {
            if constexpr (std::is_trivially_copyable_v<T>)
            {
                std::copy(that.begin(), that.end(), this->begin()); // NOSONAR: std::ranges overload is C++20.
            }
            else if (this->size() <= that.size())
            {
                std::move(that.begin(), that.begin() + this->size(), this->begin());
                std::uninitialized_move(that.begin() + this->size(), that.end(), this->end());
            }
            else
            {
                std::move(that.begin(), that.end(), this->begin());     // NOSONAR: std::ranges overload is C++20.
                std::destroy(this->begin() + that.size(), this->end()); // NOSONAR: std::ranges overload is C++20.
            }

            m_size = static_cast<int>(that.size());

            // not setting that's size to 0 on purpose
            // Since we didn't allocate memory for the vector, we're effectively
            // using std::optional's criteria for moving.
        }

        doCheckInvariants();
        return *this;
    }

    StaticVectorHelper(std::initializer_list<T> list)
        : StaticVectorHelper()
    {
        if (list.size() > N)
        {
            throw std::bad_alloc();
        }

        // According to the Holy Standard as of C++17, we
        // can't move an item out of an std::initializer_list<T> /facepalm
        std::uninitialized_copy(list.begin(), list.end(), this->begin()); // NOSONAR: std::ranges overload is C++20.

        m_size = static_cast<int>(list.size());

        doCheckInvariants();
    }

    void resize(size_t newSize)
    {
        if (newSize > static_cast<size_t>(N))
        {
            throw std::bad_alloc();
        }

        // making it smaller?
        if (newSize < this->size())
        {
            std::destroy(this->begin() + newSize, this->end());
        }
        // making it larger
        else if (newSize > this->size())
        {
            if constexpr (std::is_default_constructible_v<T>)
            {
                std::uninitialized_default_construct(this->end(), this->begin() + newSize);
            }
            else
            {
                throw StaticVectorError("Can't create non-default-constructible type");
            }
        }

        m_size = static_cast<int>(newSize);

        doCheckInvariants();
    }

    void push_back(T item)
    {
        if (m_size >= N)
        {
            throw std::bad_alloc();
        }

        std::uninitialized_move(&item, &item + 1, this->end());
        ++m_size;

        doCheckInvariants();
    }

    void pop_back()
    {
        NVCV_ASSERT(m_size > 0);

        std::destroy_at(this->end() - 1);
        --m_size;

        doCheckInvariants();
    }

    template<class... ARGS>
    void emplace_back(ARGS &&...args)
    {
        if (m_size >= N)
        {
            throw std::bad_alloc();
        }

        new (this->end()) T(std::forward<ARGS>(args)...);
        ++m_size;

        doCheckInvariants();
    }

    size_t size() const
    {
        return m_size;
    }

    static constexpr size_t capacity()
    {
        return N;
    }

    bool empty() const
    {
        return m_size == 0;
    }

    void clear()
    {
        std::destroy(this->begin(), this->end()); // NOSONAR: std::ranges overload is C++20.
        m_size = 0;

        doCheckInvariants();
    }

    friend void swap(StaticVectorHelper &a, StaticVectorHelper &b) noexcept
    {
        using std::swap;
        if constexpr (std::is_trivially_copyable_v<T>)
        {
            swap(a.m_arena, b.m_arena);
        }
        else
        {
            size_t commonSize = std::min(a.size(), b.size());
            std::swap_ranges(a.begin(), a.begin() + commonSize, b.begin());
            if (a.size() < b.size())
            {
                std::uninitialized_move(b.begin() + commonSize, b.end(), a.end());
            }
            else if (a.size() > b.size())
            {
                std::uninitialized_move(a.begin() + commonSize, a.end(), b.end());
            }
        }

        swap(a.m_size, b.m_size);

        a.doCheckInvariants();
        b.doCheckInvariants();
    }

    using iterator       = T *;
    using const_iterator = const T *;

    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator erase(const_iterator pos)
    {
        return erase(pos, std::next(pos));
    }

    iterator erase(const_iterator beg, const_iterator end)
    {
        NVCV_ASSERT(beg >= this->begin());
        NVCV_ASSERT(beg <= this->end());
        NVCV_ASSERT(end <= this->end());
        NVCV_ASSERT(beg <= end);

        auto rangeLength = std::distance(beg, end);
        NVCV_ASSERT(rangeLength >= 0);

        std::swap_ranges(this->begin() + std::distance(this->cbegin(), end), this->end(),
                         this->begin() + std::distance(this->cbegin(), beg));
        this->resize(this->size() - static_cast<size_type>(rangeLength));

        // must return the iterator following the last removed element. If the
        // iterator pos refers to the last element, the end() iterator is
        // returned.
        return this->begin() + std::distance(this->cbegin(), beg); // this should be enough to satisfy that
    }

    iterator begin()
    {
#if __cpp_lib_launder >= 201606
        return std::launder(reinterpret_cast<T *>(m_arena));
#else
        return reinterpret_cast<T *>(m_arena);
#endif
    }

    const_iterator begin() const
    {
#if __cpp_lib_launder >= 201606
        return std::launder(reinterpret_cast<const T *>(m_arena));
#else
        return reinterpret_cast<const T *>(m_arena);
#endif
    }

    const_iterator cbegin() const
    {
        return this->begin();
    }

    iterator end()
    {
        return this->begin() + m_size;
    }

    const_iterator end() const
    {
        return this->begin() + m_size;
    }

    const_iterator cend() const
    {
        return this->end();
    }

    reverse_iterator rbegin()
    {
        return std::make_reverse_iterator(this->end());
    }

    const_reverse_iterator rbegin() const
    {
        return std::make_reverse_iterator(this->end());
    }

    const_reverse_iterator crbegin() const
    {
        return std::make_reverse_iterator(this->cend());
    }

    reverse_iterator rend()
    {
        return std::make_reverse_iterator(this->begin());
    }

    const_reverse_iterator rend() const
    {
        return std::make_reverse_iterator(this->begin());
    }

    const_reverse_iterator crend() const
    {
        return std::make_reverse_iterator(this->cbegin());
    }

    reference front()
    {
        NVCV_ASSERT(m_size > 0);
        return *this->begin();
    }

    const_reference front() const
    {
        NVCV_ASSERT(m_size > 0);
        return *this->begin();
    }

    reference back()
    {
        NVCV_ASSERT(m_size > 0);
        return this->end()[-1];
    }

    const_reference back() const
    {
        NVCV_ASSERT(m_size > 0);
        return this->end()[-1];
    }

    reference operator[](int i)
    {
        NVCV_ASSERT(i >= 0);
        NVCV_ASSERT(i < m_size);
        return this->begin()[i];
    }

    const_reference operator[](int i) const
    {
        NVCV_ASSERT(i >= 0);
        NVCV_ASSERT(i < m_size);
        return this->begin()[i];
    }

    reference at(int i)
    {
        if (i < 0 || i >= m_size)
        {
            throw std::out_of_range("i");
        }

        return this->begin()[i];
    }

    const_reference at(int i) const
    {
        if (i < 0 || i >= m_size)
        {
            throw std::out_of_range("i");
        }

        return this->begin()[i];
    }

private:
    void doCheckInvariants() const
    {
        NVCV_ASSERT(m_size <= N);
    }

    int m_size;

    // our memory buffer
    std::aligned_storage_t<sizeof(T), alignof(T)> m_arena[N]; // NOSONAR: raw storage for placement-new elements.

    // our partial specializations will have initialize m_size
    template<class, int, bool, bool>
    friend class StaticVectorHelper;
};

// Partial specialization, class is copy constructible but not copy assignable
template<class T, int N>
class StaticVectorHelper<T, N, true, false> : public StaticVectorHelper<T, N, false, false>
{
    using Base = StaticVectorHelper<T, N, false, false>;

public:
    using Base::Base;

    StaticVectorHelper(const StaticVectorHelper &that)
    {
        Base::m_size = that.size();
        std::uninitialized_copy(that.begin(), that.end(), this->begin()); // NOSONAR: std::ranges overload is C++20.
    }

    StaticVectorHelper(StaticVectorHelper &&that) noexcept
        : Base(std::move(that))
    {
    }

    StaticVectorHelper &operator=(const StaticVectorHelper &that) = default;

    StaticVectorHelper &operator=(StaticVectorHelper &&that) noexcept
    {
        Base::operator=(std::move(that));
        return *this;
    }
};

// Partial specialization, class is NOT copy constructible but IS copy assignable
template<class T, int N>
class StaticVectorHelper<T, N, false, true> : public StaticVectorHelper<T, N, false, false>
{
    using Base = StaticVectorHelper<T, N, false, false>;

public:
    using Base::Base;

    ~StaticVectorHelper()
    {
        this->clear();
    }

    StaticVectorHelper &operator=(const StaticVectorHelper &that) = delete;

protected:
    void copyAssignFrom(const StaticVectorHelper &that) noexcept(
        std::is_nothrow_copy_assignable_v<T> &&std::is_nothrow_copy_constructible_v<T>)
    {
        if (this != &that)
        {
            size_t commonSize = std::min(this->size(), that.size());
            std::copy(that.begin(), that.begin() + commonSize, this->begin());

            if (that.size() > this->size())
            {
                std::uninitialized_copy(that.begin() + commonSize, that.end(), this->begin() + commonSize);
            }
            else if (that.size() < this->size())
            {
                std::destroy(this->begin() + commonSize, this->end());
            }

            Base::m_size = that.m_size;
        }
    }

public:
    StaticVectorHelper(const StaticVectorHelper &that) = delete;

    StaticVectorHelper(StaticVectorHelper &&that) noexcept
        : Base(std::move(that))
    {
    }

    StaticVectorHelper &operator=(StaticVectorHelper &&that) noexcept
    {
        Base::operator=(std::move(that));
        return *this;
    }
};

// Partial specialization, class is BOTH copy constructible and assignable
template<class T, int N>
class StaticVectorHelper<T, N, true, true> : public StaticVectorHelper<T, N, false, true>
{
    using Base = StaticVectorHelper<T, N, false, true>;

public:
    using Base::Base;

    ~StaticVectorHelper()
    {
        this->clear();
    }

    StaticVectorHelper(const StaticVectorHelper &that) noexcept(std::is_nothrow_copy_constructible_v<T>)
    {
        Base::m_size = static_cast<int>(that.size());
        std::uninitialized_copy(that.begin(), that.end(), this->begin()); // NOSONAR: std::ranges overload is C++20.
    }

    StaticVectorHelper(StaticVectorHelper &&that) noexcept
        : Base(std::move(that))
    {
    }

    StaticVectorHelper &operator=(const StaticVectorHelper &that) noexcept(
        std::is_nothrow_copy_assignable_v<T> &&std::is_nothrow_copy_constructible_v<T>)
    {
        this->copyAssignFrom(that);
        return *this;
    }

    StaticVectorHelper &operator=(StaticVectorHelper &&that) noexcept
    {
        Base::operator=(std::move(that));
        return *this;
    }

private:
    using Base::m_size;
};

template<class T, int N>
class StaticVector : public StaticVectorHelper<T, N>
{
public:
    using StaticVectorHelper<T, N>::StaticVectorHelper;
};

} // namespace nvcv::util

#endif // NVCV_UTIL_STATICVECTOR_HPP
