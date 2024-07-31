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
 * @file LinAlg.hpp
 *
 * @brief Defines linear algebra classes and functions.
 */

#ifndef NVCV_CUDA_MATH_LINALG_HPP
#define NVCV_CUDA_MATH_LINALG_HPP

#include <cvcuda/cuda_tools/MathWrappers.hpp> // for cuda::max, etc.
#include <cvcuda/cuda_tools/TypeTraits.hpp>   // for cuda::Require, etc.

#include <algorithm>        // for std::swap, etc.
#include <cassert>          // for assert, etc.
#include <cmath>            // for std::pow, etc.
#include <cstdlib>          // for std::size_t, etc.
#include <initializer_list> // for std::initializer_list, etc.
#include <ostream>          // for std::ostream, etc.
#include <vector>           // for std::vector, etc.

namespace nvcv::cuda::math {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_LINALG Linear algebra
 * @{
 */

/**
 * Vector class to represent small vectors.
 *
 * @tparam T Vector value type.
 * @tparam N Number of elements.
 */
template<class T, int N>
class Vector
{
public:
    // Type of values in this vector.
    using Type = T;

    /**
     * Load values from a C-array into this vector.
     *
     * @param[in] inVector Input C-array vector to load values from.
     */
    constexpr __host__ __device__ void load(const T *inVector)
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            m_data[i] = inVector[i];
        }
    }

    /**
     * Load values from a C++ initiliazer list into this vector.
     *
     * @param[in] l Input C++ initializer list to load values from.
     */
    constexpr __host__ __device__ void load(std::initializer_list<T> l)
    {
        load(std::data(l));
    }

    /**
     * Store values to a C-array from this vector.
     *
     * @param[out] outVector Output C-array vector to store values to.
     */
    constexpr __host__ __device__ void store(T *outVector) const
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            outVector[i] = m_data[i];
        }
    }

    /**
     * @brief Get size (number of elements) of this vector.
     *
     * @return Vector size.
     */
    constexpr __host__ __device__ int size() const
    {
        return N;
    }

    /**
     * Subscript operator for read-only access.
     *
     * @param[in] i Position to access.
     *
     * @return Value (constant reference) at given position.
     */
    constexpr const __host__ __device__ T &operator[](int i) const
    {
        assert(i >= 0 && i < size());
        return m_data[i];
    }

    /**
     * Subscript operator for read-and-write access.
     *
     * @param[in] i Position to access.
     *
     * @return Value (reference) at given position.
     */
    constexpr __host__ __device__ T &operator[](int i)
    {
        assert(i >= 0 && i < size());
        return m_data[i];
    }

    /**
     * Pointer-access operator (constant).
     *
     * @return Pointer to the first element of this vector.
     */
    constexpr __host__ __device__ operator const T *() const
    {
        return &m_data[0];
    }

    /**
     * Pointer-access operator.
     *
     * @return Pointer to the first element of this vector.
     */
    constexpr __host__ __device__ operator T *()
    {
        return &m_data[0];
    }

    /**
     * Begin (constant) pointer access.
     *
     * @return Pointer to the first element of this vector.
     */
    constexpr const __host__ __device__ T *cbegin() const
    {
        return &m_data[0];
    }

    /**
     * Begin pointer access.
     *
     * @return Pointer to the first element of this vector.
     */
    constexpr __host__ __device__ T *begin()
    {
        return &m_data[0];
    }

    /**
     * End (constant) pointer access.
     *
     * @return Pointer to the one-past-last element of this vector.
     */
    constexpr const __host__ __device__ T *cend() const
    {
        return &m_data[0] + size();
    }

    /**
     * End pointer access.
     *
     * @return Pointer to the one-past-last element of this vector.
     */
    constexpr __host__ __device__ T *end()
    {
        return &m_data[0] + size();
    }

    /**
     * Convert a vector of this class to an stl vector.
     *
     * @return STL std::vector.
     */
    std::vector<T> to_vector() const
    {
        return std::vector<T>(cbegin(), cend());
    }

    /**
     * Get a sub-vector of this vector.
     *
     * @param[in] beg Position to start getting values from this vector.
     *
     * @return Vector of value from given index to the end.
     *
     * @tparam R Size of the sub-vector to be returned.
     */
    template<int R>
    constexpr __host__ __device__ Vector<T, R> subv(int beg) const
    {
        Vector<T, R> v;
        for (int i = beg; i < beg + R; ++i)
        {
            v[i - beg] = m_data[i];
        }
        return v;
    }

    // On-purpose public data to allow POD-class direct initialization.
    T m_data[N];
};

/**
 * Matrix class to represent small matrices.
 *
 * It uses the Vector class to stores each row, storing elements in row-major order, i.e. it has M row vectors
 * where each vector has N elements.
 *
 * @tparam T Matrix value type.
 * @tparam M Number of rows.
 * @tparam N Number of columns. Default is M (a square matrix).
 */
template<class T, int M, int N = M>
class Matrix
{
public:
    // Type of values in this matrix.
    using Type = T;

    /**
     * Load values from a flatten array into this matrix.
     *
     * @param[in] inFlattenMatrix Input flatten matrix to load values from.
     */
    constexpr __host__ __device__ void load(const T *inFlattenMatrix)
    {
        int idx = 0;
#pragma unroll
        for (int i = 0; i < M; ++i)
        {
#pragma unroll
            for (int j = 0; j < N; ++j)
            {
                m_data[i][j] = inFlattenMatrix[idx++];
            }
        }
    }

    /**
     * Load values from a C++ initiliazer list into this matrix.
     *
     * @param[in] l Input C++ initializer list to load values from.
     */
    constexpr __host__ __device__ void load(std::initializer_list<T> l)
    {
        load(std::data(l));
    }

    /**
     * Store values to a flatten array from this matrix.
     *
     * @param[out] outFlattenMatrix Output flatten matrix to store values to.
     */
    constexpr __host__ __device__ void store(T *outFlattenMatrix) const
    {
        int idx = 0;
#pragma unroll
        for (int i = 0; i < M; ++i)
        {
#pragma unroll
            for (int j = 0; j < N; ++j)
            {
                outFlattenMatrix[idx++] = m_data[i][j];
            }
        }
    }

    /**
     * @brief Get number of rows of this matrix.
     *
     * @return Number of rows.
     */
    constexpr __host__ __device__ int rows() const
    {
        return M;
    }

    /**
     * Get number of columns of this matrix.
     *
     * @return Number of columns.
     */
    constexpr __host__ __device__ int cols() const
    {
        return N;
    }

    /**
     * Subscript operator for read-only access.
     *
     * @param[in] i Row of the matrix to access.
     *
     * @return Vector (constant reference) of the corresponding row.
     */
    constexpr const __host__ __device__ Vector<T, N> &operator[](int i) const
    {
        assert(i >= 0 && i < rows());
        return m_data[i];
    }

    /**
     * Subscript operator for read-and-write access.
     *
     * @param[in] i Row of the matrix to access.
     *
     * @return Vector (reference) of the corresponding row.
     */
    constexpr __host__ __device__ Vector<T, N> &operator[](int i)
    {
        assert(i >= 0 && i < rows());
        return m_data[i];
    }

    /**
     * Subscript operator for read-only access of matrix elements.
     *
     * @param[in] c Coordinates (y row and x column) of the matrix element to access.
     *
     * @return Element (constant reference) of the corresponding row and column.
     */
    constexpr const __host__ __device__ T &operator[](int2 c) const
    {
        assert(c.y >= 0 && c.y < rows());
        assert(c.x >= 0 && c.x < cols());
        return m_data[c.y][c.x];
    }

    /**
     * Subscript operator for read-and-write access of matrix elements.
     *
     * @param[in] c Coordinates (y row and x column) of the matrix element to access.
     *
     * @return Element (reference) of the corresponding row and column.
     */
    constexpr __host__ __device__ T &operator[](int2 c)
    {
        assert(c.y >= 0 && c.y < rows());
        assert(c.x >= 0 && c.x < cols());
        return m_data[c.y][c.x];
    }

    /**
     * Get column j of this matrix.
     *
     * @param[in] j Index of column to get.
     *
     * @return Column j (copied) as a vector.
     */
    constexpr __host__ __device__ Vector<T, M> col(int j) const
    {
        Vector<T, M> c;
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            c[i] = m_data[i][j];
        }
        return c;
    }

    /**
     * Set column j of this matrix.
     *
     * @param[in] j Index of column to set.
     * @param[in] v Value to place in matrix column.
     */
    constexpr __host__ __device__ void set_col(int j, T v)
    {
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            m_data[i][j] = v;
        }
    }

    /**
     * Set column j of this matrix.
     *
     * @param[in] j Index of column to set.
     * @param[in] c Vector to place in matrix column.
     */
    constexpr __host__ __device__ void set_col(int j, const Vector<T, M> &c)
    {
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            m_data[i][j] = c[i];
        }
    }

    // @overload void set_col(int j, const T *c)
    constexpr __host__ __device__ void set_col(int j, const T *c)
    {
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            m_data[i][j] = c[i];
        }
    }

    /**
     * Get a sub-matrix of this matrix.
     *
     * @param[in] skip_i Row to skip when getting values from this matrix.
     * @param[in] skip_j Column to skip when getting values from this matrix.
     *
     * @return Matrix with one less row and one less column.
     */
    constexpr __host__ __device__ Matrix<T, M - 1, N - 1> subm(int skip_i, int skip_j) const
    {
        Matrix<T, M - 1, N - 1> ret;
        int                     ri = 0;
        for (int i = 0; i < rows(); ++i)
        {
            if (i == skip_i)
            {
                continue;
            }
            int rj = 0;
            for (int j = 0; j < cols(); ++j)
            {
                if (j == skip_j)
                {
                    continue;
                }
                ret[ri][rj] = (*this)[i][j];
                ++rj;
            }
            ++ri;
        }
        return ret;
    }

    // On-purpose public data to allow POD-class direct initialization.
    Vector<T, N> m_data[M];
};

namespace detail {

template<class T>
constexpr __host__ __device__ void swap(T &a, T &b)
{
#ifdef __CUDA_ARCH__
    T c = a;
    a   = b;
    b   = c;
#else
    std::swap(a, b);
#endif
}

} // namespace detail

// Vector-based operations -----------------------------------------------------

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator+=(Vector<T, N> &lhs, const Vector<T, N> &rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] += rhs[j];
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator+(const Vector<T, N> &a, const Vector<T, N> &b)
{
    Vector<T, N> r(a);
    return r += b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator+=(Vector<T, N> &lhs, T rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] += rhs;
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator+(const Vector<T, N> &a, T b)
{
    Vector<T, N> r(a);
    return r += b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator+(T a, const Vector<T, N> &b)
{
    Vector<T, N> r(b);
    return r += a;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator-=(Vector<T, N> &lhs, const Vector<T, N> &rhs)
{
    return lhs += -rhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator-(const Vector<T, N> &a, const Vector<T, N> &b)
{
    Vector<T, N> r(a);
    return r -= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator-=(Vector<T, N> &lhs, T rhs)
{
    return lhs += static_cast<T>(-rhs);
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator-(const Vector<T, N> &a, T b)
{
    Vector<T, N> r(a);
    return r -= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator-(T a, const Vector<T, N> &b)
{
    Vector<T, N> r(-b);
    return r += a;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator*=(Vector<T, N> &lhs, const T &rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] *= rhs;
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator*(const Vector<T, N> &a, const T &b)
{
    Vector<T, N> r(a);
    return r *= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator*(const T &a, const Vector<T, N> &b)
{
    return b * a;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator*=(Vector<T, N> &lhs, const Vector<T, N> &rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] *= rhs[j];
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator*(const Vector<T, N> &a, const Vector<T, N> &b)
{
    Vector<T, N> r(a);
    return r *= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator/=(Vector<T, N> &lhs, const T &rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] /= rhs;
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator/(const Vector<T, N> &a, const T &b)
{
    Vector<T, N> r(a);
    return r /= b;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator/(T a, const Vector<T, N> &b)
{
    Vector<T, N> r(b);
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        r[j] = a / r[j];
    }
    return r;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> &operator/=(Vector<T, N> &lhs, const Vector<T, N> &rhs)
{
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        lhs[j] /= rhs[j];
    }
    return lhs;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator/(const Vector<T, N> &a, const Vector<T, N> &b)
{
    Vector<T, N> r(a);
    return r /= b;
}

template<class T, int N>
std::ostream &operator<<(std::ostream &out, const Vector<T, N> &v)
{
    out << '[';
    for (int i = 0; i < N; ++i)
    {
        out << v[i];
        if (i < N - 1)
        {
            out << ' ';
        }
    }
    return out << ']';
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> operator-(const Vector<T, N> &v)
{
    Vector<T, N> r;
    for (int i = 0; i < N; ++i)
    {
        r[i] = -v[i];
    }
    return r;
}

template<class T, int N>
constexpr __host__ __device__ bool operator==(const Vector<T, N> &a, const Vector<T, N> &b)
{
    for (int i = 0; i < N; ++i)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}

template<class T, int N>
constexpr __host__ __device__ bool operator==(const T &a, const Vector<T, N> &b)
{
    for (int i = 0; i < N; ++i)
    {
        if (a != b[i])
        {
            return false;
        }
    }
    return true;
}

template<class T, int N>
constexpr __host__ __device__ bool operator==(const Vector<T, N> &a, const T &b)
{
    for (int i = 0; i < N; ++i)
    {
        if (a[i] != b)
        {
            return false;
        }
    }
    return true;
}

template<class T, int N>
constexpr __host__ __device__ bool operator<(const Vector<T, N> &a, const Vector<T, N> &b)
{
    for (int i = 0; i < N; ++i)
    {
        if (a[i] != b[i])
        {
            return a[i] < b[i];
        }
    }
    return false;
}

// Matrix-based operations -----------------------------------------------------

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator*(const Matrix<T, M, N> &m, T val)
{
    Matrix<T, M, N> r(m);
    return r *= val;
}

template<class T, int M, int N>
std::ostream &operator<<(std::ostream &out, const Matrix<T, M, N> &m)
{
    out << '[';
    for (int i = 0; i < m.rows(); ++i)
    {
        for (int j = 0; j < m.cols(); ++j)
        {
            out << m[i][j];
            if (j < m.cols() - 1)
            {
                out << ' ';
            }
        }
        if (i < m.rows() - 1)
        {
            out << "\n";
        }
    }
    return out << ']';
}

template<class T, int M, int N, int P>
constexpr __host__ __device__ Matrix<T, M, P> operator*(const Matrix<T, M, N> &a, const Matrix<T, N, P> &b)
{
    Matrix<T, M, P> r;
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < P; ++j)
        {
            r[i][j] = a[i][0] * b[0][j];
#pragma unroll
            for (int k = 1; k < N; ++k)
            {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return r;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> &operator*=(Matrix<T, M, N> &lhs, T rhs)
{
#pragma unroll
    for (int i = 0; i < lhs.rows(); ++i)
    {
#pragma unroll
        for (int j = 0; j < lhs.cols(); ++j)
        {
            lhs[i][j] *= rhs;
        }
    }
    return lhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator*(T val, const Matrix<T, M, N> &m)
{
    return m * val;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> &operator+=(Matrix<T, M, N> &lhs, const Matrix<T, M, N> &rhs)
{
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            lhs[i][j] += rhs[i][j];
        }
    }
    return lhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator+(const Matrix<T, M, N> &lhs, const Matrix<T, M, N> &rhs)
{
    Matrix<T, M, N> r(lhs);
    return r += rhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> &operator-=(Matrix<T, M, N> &lhs, const Matrix<T, M, N> &rhs)
{
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            lhs[i][j] -= rhs[i][j];
        }
    }
    return lhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator-(const Matrix<T, M, N> &a, const Matrix<T, M, N> &b)
{
    Matrix<T, M, N> r(a);
    return r -= b;
}

template<class T, int M, int N>
constexpr __host__ __device__ Vector<T, N> operator*(const Vector<T, M> &v, const Matrix<T, M, N> &m)
{
    Vector<T, N> r;

#pragma unroll
    for (int j = 0; j < m.cols(); ++j)
    {
        r[j] = v[0] * m[0][j];
#pragma unroll
        for (int i = 1; i < m.rows(); ++i)
        {
            r[j] += v[i] * m[i][j];
        }
    }

    return r;
}

template<class T, int M, int N>
constexpr __host__ __device__ Vector<T, M> operator*(const Matrix<T, M, N> &m, const Vector<T, N> &v)
{
    Vector<T, M> r;

#pragma unroll
    for (int i = 0; i < m.rows(); ++i)
    {
        r[i] = m[i][0] * v[0];
#pragma unroll
        for (int j = 1; j < m.cols(); ++j)
        {
            r[i] += m[i][j] * v[j];
        }
    }

    return r;
}

template<class T, int M, int N, class = cuda::Require<(M == N && N > 1)>>
constexpr __host__ __device__ Matrix<T, M, N> operator*(const Matrix<T, M, 1> &m, const Vector<T, N> &v)
{
    Matrix<T, M, N> r;

#pragma unroll
    for (int i = 0; i < r.rows(); ++i)
    {
#pragma unroll
        for (int j = 0; j < r.cols(); ++j)
        {
            r[i][j] = m[i][0] * v[j];
        }
    }

    return r;
}

template<class T, int M, int N>
constexpr __host__ __device__ Vector<T, N> &operator*=(Vector<T, M> &v, const Matrix<T, M, M> &m)
{
    return v = v * m;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> &operator*=(Matrix<T, M, N> &lhs, const Matrix<T, N, N> &rhs)
{
    return lhs = lhs * rhs;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> operator-(const Matrix<T, M, N> &m)
{
    Matrix<T, M, N> r;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            r[i][j] = -m[i][j];
        }
    }
    return r;
}

template<class T, int M, int N>
constexpr __host__ __device__ bool operator==(const Matrix<T, M, N> &a, const Matrix<T, M, N> &b)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }
    }
    return true;
}

template<class T, int M, int N>
constexpr __host__ __device__ bool operator==(const T &a, const Matrix<T, M, N> &b)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (a != b[i])
            {
                return false;
            }
        }
    }
    return true;
}

template<class T, int M, int N>
constexpr __host__ __device__ bool operator==(const Matrix<T, M, N> &a, const T &b)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (a[i] != b)
            {
                return false;
            }
        }
    }
    return true;
}

template<class T, int M, int N>
constexpr __host__ __device__ bool operator<(const Matrix<T, M, N> &a, const Matrix<T, M, N> &b)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (a[i] != b[i])
            {
                return a[i] < b[i];
            }
        }
    }

    // if we reach here, it means that a==b
    return false;
}

template<typename T, int N, int M>
constexpr Matrix<T, N, M> as_matrix(const T (&values)[N][M])
{
    Matrix<T, N, M> m;
#pragma unroll
    for (int i = 0; i < N; i++)
    {
#pragma unroll
        for (int j = 0; j < M; j++)
        {
            m[i][j] = values[i][j];
        }
    }
    return m;
}

// Special matrices ------------------------------------------------------------

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> zeros()
{
    Vector<T, N> v = {};
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        v[j] = T{0};
    }
    return v;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> zeros()
{
    Matrix<T, M, N> m = {};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            m[i][j] = T{0};
        }
    }
    return m;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> ones()
{
    Vector<T, N> v = {};
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        v[j] = T{1};
    }
    return v;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> ones()
{
    Matrix<T, M, N> m = {};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            m[i][j] = T{1};
        }
    }
    return m;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> identity()
{
    Matrix<T, M, N> m = {};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            m[i][j] = i == j ? 1 : 0;
        }
    }
    return m;
}

template<class T, int M>
constexpr __host__ __device__ Matrix<T, M, M> vander(const Vector<T, M> &v)
{
    Matrix<T, M, M> m = {};
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            m[i][j] = cuda::pow(v[j], i);
        }
    }
    return m;
}

template<class T, int R>
constexpr __host__ __device__ Matrix<T, R, R> compan(const Vector<T, R> &a)
{
    Matrix<T, R, R> m;
    for (int i = 0; i < R; ++i)
    {
        for (int j = 0; j < R; ++j)
        {
            if (j == R - 1)
            {
                m[i][j] = -a[i];
            }
            else
            {
                m[i][j] = j == i - 1 ? 1 : 0;
            }
        }
    }
    return m;
}

template<class T, int M>
constexpr __host__ __device__ Matrix<T, M, M> diag(const Vector<T, M> &v)
{
    Matrix<T, M, M> m = {};
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            m[i][j] = i == j ? v[i] : 0;
        }
    }
    return m;
}

// Basic operations ------------------------------------------------------------

template<class T, int N>
constexpr __host__ __device__ T dot(const Vector<T, N> &a, const Vector<T, N> &b)
{
    T d = a[0] * b[0];
#pragma unroll
    for (int j = 1; j < N; ++j)
    {
        d += a[j] * b[j];
    }
    return d;
}

template<class T, int N>
constexpr __host__ __device__ Vector<T, N> reverse(const Vector<T, N> &a)
{
    Vector<T, N> r = {};
#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        r[j] = a[N - 1 - j];
    }
    return r;
}

// Transformations -------------------------------------------------------------

template<class T, int M>
constexpr __host__ __device__ Matrix<T, M, M> &transp_inplace(Matrix<T, M, M> &m)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = i + 1; j < M; ++j)
        {
            detail::swap(m[i][j], m[j][i]);
        }
    }
    return m;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, N, M> transp(const Matrix<T, M, N> &m)
{
    Matrix<T, N, M> tm = {};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            tm[j][i] = m[i][j];
        }
    }
    return tm;
}

template<class T, int N>
constexpr __host__ __device__ Matrix<T, N, 1> transp(const Vector<T, N> &v)
{
    Matrix<T, N, 1> tv = {};
    tv.set_col(0, v);
    return tv;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> flip_rows(const Matrix<T, M, N> &m)
{
    Matrix<T, M, N> f = {};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            f[i][j] = m[M - 1 - i][j];
        }
    }
    return f;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> flip_cols(const Matrix<T, M, N> &m)
{
    Matrix<T, M, N> f = {};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            f[i][j] = m[i][N - 1 - j];
        }
    }
    return f;
}

template<class T, int M, int N>
constexpr __host__ __device__ Matrix<T, M, N> flip(const Matrix<T, M, N> &m)
{
    Matrix<T, M, N> f = {};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            f[i][j] = m[M - 1 - i][N - 1 - j];
        }
    }
    return f;
}

template<int R, typename T, int M, int N, class = cuda::Require<R <= M>>
constexpr __host__ __device__ Matrix<T, R, N> head(const Matrix<T, M, N> &m)
{
    Matrix<T, R, N> h;

#pragma unroll
    for (int i = 0; i < R; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            h[i][j] = m[i][j];
        }
    }

    return h;
}

template<int R, typename T, int M, int N, class = cuda::Require<R <= M>>
constexpr __host__ __device__ Matrix<T, R, N> tail(const Matrix<T, M, N> &m)
{
    Matrix<T, R, N> t;

#pragma unroll
    for (int i = 0; i < R; ++i)
    {
#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            t[i][j] = m[M - R + i][j];
        }
    }

    return t;
}

// Advanced operations ---------------------------------------------------------

// Linear-time invariant (LTI) filtering is a fundamental operation in signal and image processing.  Many
// applications use LTI filters that can be expressed as linear, constant-coefficient difference equations.

// Functions below implement a convolution pass, i.e. finite impulse response (FIR), and a causal/anticausal
// combination of recursive filter passes, i.e. infinite impulse response (IIR), both defined by a set of weights.

// Definitions: input single element x, block b, length N; filter weights w, order R; prologue p; epilogue e.
// Illustrative example of N=11 and R=3 showing a block b in between previous and next blocks.

//                  |----------------- b -----------------|
// <previous block> | [ e0 e1 e2 ] x x x x x [ p0 p1 p2 ] | <next block>

// FIR + IIR filtering with causal combination is called forward; with anticausal combination is called reverse.

// Forward (fwd): y = w[0] * x - w[1] * p2 - w[2] * p1 - w[3] * p0
// The y passed in is considered to be: y = w[0] * x

// Reverse (rev): z = w[0] * y - w[1] * e0 - w[2] * e1 - w[3] * e2
// The z passed in is considered to be: z = w[0] * y

// Forward pass in a single element, updating prologue accordingly and returning result
template<typename T, int R>
constexpr __host__ __device__ T fwd1(Vector<T, R> &p, T y, const Vector<T, R + 1> &w)
{
    y = y - p[R - 1] * w[1];

#pragma unroll
    for (int k = R - 1; k >= 1; --k)
    {
        y = y - p[R - 1 - k] * w[k + 1];

        p[R - 1 - k] = p[R - 1 - k + 1];
    }

    p[R - 1] = y;

    return y;
}

// Forward pass in a block of N elements, updating prologue accordingly and in-place
template<typename T, int N, int R>
constexpr __host__ __device__ void fwdN(Vector<T, R> &p, Vector<T, N> &b, const Vector<T, R + 1> &w)
{
#pragma unroll
    for (int k = 0; k < N; ++k)
    {
        b[k] = fwd1(p, w[0] * b[k], w);
    }
}

// Forward-transpose pass over rows of a block of MxN elements, updating prologue accordingly and in-place
template<typename T, int M, int N, int R>
constexpr __host__ void fwdT(Matrix<T, M, R> &p, Matrix<T, M, N> &b, const Vector<T, R + 1> &w)
{
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
        fwdN(p[i], b[i], w);
    }
}

// Forward pass over columns of a block of MxN elements, returning result
template<typename T, int M, int N, int R>
constexpr __host__ Matrix<T, M, N> fwd(const Matrix<T, R, N> &p, const Matrix<T, M, N> &b, const Vector<T, R + 1> &w)
{
    Matrix<T, M, N> bout;

#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        Vector<T, R> pT = p.col(j);

#pragma unroll
        for (int i = 0; i < M; ++i)
        {
            bout[i][j] = fwd1(pT, b[i][j] * w[0], w);
        }
    }

    return bout;
}

// Reverse pass in a single element, updating epilogue accordingly and returning result
template<class T, int R>
constexpr __host__ __device__ T rev1(T z, Vector<T, R> &e, const Vector<T, R + 1> &w)
{
    z = z - e[0] * w[1];

#pragma unroll
    for (int k = R - 1; k >= 1; --k)
    {
        z = z - e[k] * w[k + 1];

        e[k] = e[k - 1];
    }

    e[0] = z;

    return z;
}

// Reverse pass in a block of N elements, updating prologue accordingly and in-place
template<typename T, int N, int R>
constexpr __host__ __device__ void revN(Vector<T, N> &b, Vector<T, R> &e, const Vector<T, R + 1> &w)
{
#pragma unroll
    for (int k = N - 1; k >= 0; --k)
    {
        b[k] = rev1(w[0] * b[k], e, w);
    }
}

// Reverse-transpose pass over rows of a block of MxN elements, updating prologue accordingly and in-place
template<typename T, int M, int N, int R>
constexpr __host__ void revT(Matrix<T, M, N> &b, Matrix<T, M, R> &e, const Vector<T, R + 1> &w)
{
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
        revN(b[i], e[i], w);
    }
}

// Reverse pass over columns of a block of MxN elements, returning result
template<typename T, int M, int N, int R>
constexpr __host__ Matrix<T, M, N> rev(const Matrix<T, M, N> &b, const Matrix<T, R, N> &e, const Vector<T, R + 1> &w)
{
    Matrix<T, M, N> bout;

#pragma unroll
    for (int j = 0; j < N; ++j)
    {
        Vector<T, R> eT = e.col(j);

#pragma unroll
        for (int i = M - 1; i >= 0; --i)
        {
            bout[i][j] = rev1(b[i][j] * w[0], eT, w);
        }
    }

    return bout;
}

// Determinant -----------------------------------------------------------------

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 0, 0> &m)
{
    return T{1};
}

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 1, 1> &m)
{
    return m[0][0];
}

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 2, 2> &m)
{
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 3, 3> &m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) + m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2])
         + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

template<class T, int M>
constexpr __host__ __device__ T det(const Matrix<T, M, M> &m)
{
    T d = T{0};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
        d += ((i % 2 == 0 ? 1 : -1) * m[0][i] * det(m.subm(0, i)));
    }
    return d;
}

// LU decomposition & solve ----------------------------------------------------

// Do LU decomposition of matrix m using auxiliary pivot vector p and working type F (defaults to float)
template<class F = float, class T, int N>
constexpr __host__ __device__ bool lu_inplace(Matrix<T, N, N> &m, Vector<int, N> &p)
{
    Vector<F, N> v = {};

#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        F big = 0;

#pragma unroll
        for (int j = 0; j < N; ++j)
        {
            big = cuda::max<F>(big, cuda::abs(m[i][j]));
        }

        if (big == 0)
        {
            return false;
        }

        v[i] = 1.0 / big;
    }

#pragma unroll
    for (int k = 0; k < N; ++k)
    {
        F   big  = 0;
        int imax = k;

#pragma unroll
        for (int i = k; i < N; ++i)
        {
            F aux = v[i] * cuda::abs(m[i][k]);

            if (aux > big)
            {
                big  = aux;
                imax = i;
            }
        }

        if (k != imax)
        {
            detail::swap(m[imax], m[k]);

            v[imax] = v[k];
        }

        p[k] = imax;

        if (m[k][k] == 0)
        {
            return false;
        }

#pragma unroll
        for (int i = k + 1; i < N; ++i)
        {
            T aux = m[i][k] /= m[k][k];

#pragma unroll
            for (int j = k + 1; j < N; ++j)
            {
                m[i][j] -= aux * m[k][j];
            }
        }
    }

    return true;
}

// Solve in-place using given LU decomposition lu and pivot p, the result x is returned in b
template<class T, int N>
constexpr __host__ __device__ void solve_inplace(const Matrix<T, N, N> &lu, const Vector<int, N> &p, Vector<T, N> &b)
{
    int ii = -1;

#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        int ip  = p[i];
        T   sum = b[ip];
        b[ip]   = b[i];

        if (ii >= 0)
        {
#pragma unroll
            for (int j = ii; j < i; ++j)
            {
                sum -= lu[i][j] * b[j];
            }
        }
        else if (sum != 0)
        {
            ii = i;
        }

        b[i] = sum;
    }

#pragma unroll
    for (int i = N - 1; i >= 0; --i)
    {
        T sum = b[i];

#pragma unroll
        for (int j = i + 1; j < N; ++j)
        {
            sum -= lu[i][j] * b[j];
        }

        b[i] = sum / lu[i][i];
    }
}

// Solve in-place m * x = b, where x is returned in b
template<class T, int N>
constexpr __host__ __device__ bool solve_inplace(const Matrix<T, N, N> &m, Vector<T, N> &b)
{
    Vector<int, N>  p  = {};
    Matrix<T, N, N> LU = m;

    if (!lu_inplace(LU, p))
    {
        return false;
    }

    solve_inplace(LU, p, b);

    return true;
}

// Matrix Inverse --------------------------------------------------------------

namespace detail {

// In this detail, all inverse (and in-place) functions use determinant d of the input matrix m
template<class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 1, 1> &m, const T &d)
{
    m[0][0] = T{1} / d;
}

template<class T>
constexpr __host__ __device__ Matrix<T, 1, 1> inv(const Matrix<T, 1, 1> &m, const T &d)
{
    Matrix<T, 1, 1> A;
    inv_inplace(A, d);
    return A;
}

template<class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 2, 2> &m, const T &d)
{
    detail::swap(m[0][0], m[1][1]);
    m[0][0] /= d;
    m[1][1] /= d;

    m[0][1] = -m[0][1] / d;
    m[1][0] = -m[1][0] / d;
}

template<class T>
constexpr __host__ __device__ Matrix<T, 2, 2> inv(const Matrix<T, 2, 2> &m, const T &d)
{
    Matrix<T, 2, 2> A = m;
    inv_inplace(A, d);

    return A;
}

template<class T>
constexpr __host__ __device__ Matrix<T, 3, 3> inv(const Matrix<T, 3, 3> &m, const T &d)
{
    Matrix<T, 3, 3> A;
    A[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
    A[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / d;
    A[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    A[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / d;
    A[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
    A[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / d;
    A[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
    A[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / d;
    A[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;

    return A;
}

template<class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 3, 3> &m, const T &d)
{
    m = inv(m, d);
}

} // namespace detail

// Do inverse of matrix m asserting its success
template<class T, int N, class = cuda::Require<(N < 4)>>
constexpr __host__ __device__ Matrix<T, N, N> inv(const Matrix<T, N, N> &m)
{
    T d = det(m);
    assert(d != 0);
    return detail::inv(m, d);
}

// Do inverse in-place of matrix m returning true if succeeded (m has determinant)
template<class T, int N, class = cuda::Require<(N < 4)>>
constexpr __host__ __device__ bool inv_inplace(Matrix<T, N, N> &m)
{
    T d = det(m);

    if (d == 0)
    {
        return false;
    }

    detail::inv_inplace(m, d);

    return true;
}

// Do inverse in-place of matrix m returning out using LU decomposition written to m
template<class T, int M>
constexpr __host__ __device__ void inv_lu_inplace(Matrix<T, M, M> &out, Matrix<T, M, M> &m)
{
    Vector<int, M> p = {};

    bool validResult = lu_inplace(m, p);
    assert(validResult);
    if (!validResult)
    {
        return;
    }

    out = identity<T, M, M>();

#pragma unroll
    for (int i = 0; i < M; ++i)
    {
        solve_inplace(m, p, out[i]);
    }

    transp_inplace(out);
}

// Do inverse in-place of matrix m using LU decomposition
template<class T, int M>
constexpr __host__ __device__ void inv_lu_inplace(Matrix<T, M, M> &m)
{
    Matrix<T, M, M> res;
    inv_lu_inplace(res, m);
    m = res;
}

// Do inverse using LU decomposition
template<class T, int M>
constexpr __host__ __device__ Matrix<T, M, M> inv_lu(Matrix<T, M, M> m)
{
    inv_lu_inplace(m);
    return m;
}

/**@}*/

} // namespace nvcv::cuda::math

#endif // NVCV_CUDA_MATH_LINALG_HPP
