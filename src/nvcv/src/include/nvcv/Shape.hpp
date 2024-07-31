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

#ifndef NVCV_SHAPE_HPP
#define NVCV_SHAPE_HPP

#include "Exception.hpp"
#include "Status.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <iostream>

namespace nvcv {

/**
 * @brief Template class representing an N-dimensional shape.
 *
 * This class is designed to encapsulate the shape of an N-dimensional tensor,
 * where the size in each dimension is of type T.
 *
 * @tparam T The type of the size in each dimension (e.g., int, size_t).
 * @tparam N The maximum number of dimensions this shape can represent.
 */
template<class T, int N>
class Shape
{
    using Data = std::array<T, N>;

public:
    using value_type      = typename Data::value_type;
    using size_type       = int;
    using reference       = typename Data::reference;
    using const_reference = typename Data::const_reference;
    using iterator        = typename Data::iterator;
    using const_iterator  = typename Data::const_iterator;

    constexpr static int MAX_RANK = N;

    // Constructors
    Shape();
    Shape(const Shape &that);

    /**
     * @brief Construct with a given rank, sizes default to 0.
     *
     * @param size The rank of the shape.
     */
    explicit Shape(size_type size);

    /**
     * @brief Constructor using a buffer.
     *
     * Constructs a shape by copying 'n' elements from the buffer pointed by 'data'.
     *
     * @param data Pointer to the buffer.
     * @param n Number of elements to copy from the buffer.
     */
    Shape(const T *data, size_t n);

    Shape(std::initializer_list<value_type> shape);

    reference       operator[](int i); ///< Access the i-th dimension.
    const_reference operator[](int i) const;

    size_type rank() const;  ///< Get the rank (number of dimensions) of the shape.
    size_type size() const;  ///< Get the total size represented by the shape.
    bool      empty() const; ///< Check if the shape is empty.

    // iterators
    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    const_iterator cbegin() const;
    const_iterator cend() const;

    // Comparison operators
    bool operator==(const Shape &that) const;
    bool operator!=(const Shape &that) const;

    bool operator<(const Shape &that) const;

private:
    Data      m_data;
    size_type m_size;
};

// Implementation

template<class T, int N>
Shape<T, N>::Shape()
    : m_size(0)
{
}

template<class T, int N>
Shape<T, N>::Shape(int size)
    : m_size(size)
{
    std::fill(this->begin(), this->end(), 0);
}

template<class T, int N>
Shape<T, N>::Shape(const Shape &that)
    : m_size(that.m_size)
{
    std::copy(that.begin(), that.end(), m_data.begin());
}

template<class T, int N>
Shape<T, N>::Shape(std::initializer_list<value_type> shape)
    : Shape(shape.begin(), shape.size())
{
}

template<class T, int N>
Shape<T, N>::Shape(const T *data, size_t n)
    : m_size(n)
{
    if (data == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Shape data must not be NULL");
    }

    if (n > m_data.size())
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Shape ranks is too big");
    }

    std::copy_n(data, n, m_data.begin());
}

template<class T, int N>
auto Shape<T, N>::operator[](int i) -> reference
{
    assert(0 <= i && i < m_size);
    return m_data[i];
}

template<class T, int N>
auto Shape<T, N>::operator[](int i) const -> const_reference
{
    assert(0 <= i && i < m_size);
    return m_data[i];
}

template<class T, int N>
bool Shape<T, N>::operator==(const Shape &that) const
{
    if (m_size == that.m_size)
    {
        return std::equal(this->begin(), this->end(), that.begin());
    }
    else
    {
        return false;
    }
}

template<class T, int N>
bool Shape<T, N>::operator!=(const Shape &that) const
{
    return !operator==(that);
}

template<class T, int N>
bool Shape<T, N>::operator<(const Shape &that) const
{
    return std::lexicographical_compare(this->begin(), this->end(), that.begin(), that.end());
}

template<class T, int N>
auto Shape<T, N>::rank() const -> size_type
{
    return m_size;
}

template<class T, int N>
auto Shape<T, N>::size() const -> size_type
{
    return m_size;
}

template<class T, int N>
bool Shape<T, N>::empty() const
{
    return m_size == 0;
}

template<class T, int N>
auto Shape<T, N>::begin() -> iterator
{
    return m_data.begin();
}

template<class T, int N>
auto Shape<T, N>::end() -> iterator
{
    return m_data.begin() + m_size;
}

template<class T, int N>
auto Shape<T, N>::begin() const -> const_iterator
{
    return m_data.begin();
}

template<class T, int N>
auto Shape<T, N>::end() const -> const_iterator
{
    return m_data.begin() + m_size;
}

template<class T, int N>
auto Shape<T, N>::cbegin() const -> const_iterator
{
    return m_data.cbegin();
}

template<class T, int N>
auto Shape<T, N>::cend() const -> const_iterator
{
    return m_data.cend() + m_size;
}

template<class T, int N>
std::ostream &operator<<(std::ostream &out, const Shape<T, N> &shape)
{
    if (shape.empty())
    {
        return out << "(empty)";
    }
    else
    {
        out << shape[0];
        for (int i = 1; i < shape.size(); ++i)
        {
            out << 'x' << shape[i];
        }
        return out;
    }
}

} // namespace nvcv

#endif // NVCV_SHAPE_HPP
